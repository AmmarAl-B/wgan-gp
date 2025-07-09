import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from tqdm import tqdm
import os
import joblib


class Config:
    DATA_FILE = './data/smartattackdata_original_foldcauchy.csv'
    OUTPUT_DIR = 'tabddpm_output'
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'tabddpm_model.pth')
    DATA_HANDLER_STATE_PATH = os.path.join(OUTPUT_DIR, 'data_handler_state.joblib')
    SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, 'synthetic_data_tabddpm.csv')
    N_TIMESTEPS = 1000
    BETA_SCHEDULE = 'linear'
    MLP_HIDDEN_DIMS = (512, 1024, 512)
    BATCH_SIZE = 256
    NUM_EPOCHS = 800
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataHandler:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.num_transformer = None
        self.label_encoders = {}
        self.feature_order = []

    def load_and_preprocess(self):
        print(f"Loading data from {self.cfg.DATA_FILE}...")
        data_dir = os.path.dirname(self.cfg.DATA_FILE)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)

        df = pd.read_csv(self.cfg.DATA_FILE)
        df = df.drop(columns=['Unnamed: 0', 'Label_Desc', 'filename'], errors='ignore')

        benign_df = df[df['Label_code'] == 0].copy()
        benign_df = benign_df.drop(columns=['Label_code'])

        benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in benign_df.select_dtypes(include=np.number).columns:
            if benign_df[col].isnull().any():
                benign_df[col] = benign_df[col].fillna(benign_df[col].median())

        self.numerical_cols = benign_df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = benign_df.select_dtypes(include='object').columns.tolist()
        self.feature_order = self.numerical_cols + self.categorical_cols
        self.df = benign_df[self.feature_order].reset_index(drop=True)

        print(
            f"Preprocessing data... Found {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical features.")

        self.num_transformer = QuantileTransformer(output_distribution='normal',
                                                   n_quantiles=max(min(len(self.df) // 30, 1000), 10), random_state=42)
        num_data_scaled = self.num_transformer.fit_transform(self.df[self.numerical_cols])

        cat_data_encoded_list = []
        for col in self.categorical_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            cat_data_encoded_list.append(encoded.reshape(-1, 1))

        if self.categorical_cols:
            cat_data_encoded = np.hstack(cat_data_encoded_list)
            full_data = np.hstack([num_data_scaled, cat_data_encoded])
        else:
            full_data = num_data_scaled

        return torch.tensor(full_data, dtype=torch.float32)

    def get_dataloader(self, tensor_data):
        dataset = TensorDataset(tensor_data)
        return DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, drop_last=True)

    def save_state(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        state = {
            'num_transformer': self.num_transformer,
            'label_encoders': self.label_encoders,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'feature_order': self.feature_order
        }
        joblib.dump(state, self.cfg.DATA_HANDLER_STATE_PATH)
        print(f"Saved data handler state to {self.cfg.DATA_HANDLER_STATE_PATH}")

    def load_state(self):
        state = joblib.load(self.cfg.DATA_HANDLER_STATE_PATH)
        self.num_transformer = state['num_transformer']
        self.label_encoders = state['label_encoders']
        self.numerical_cols = state['numerical_cols']
        self.categorical_cols = state['categorical_cols']
        self.feature_order = state['feature_order']
        print(f"Loaded data handler state from {self.cfg.DATA_HANDLER_STATE_PATH}")

    def inverse_transform(self, synth_data_tensor: torch.Tensor):
        synth_data = synth_data_tensor.cpu().numpy()
        num_synth = synth_data[:, :len(self.numerical_cols)]
        cat_synth = synth_data[:, len(self.numerical_cols):]

        num_unscaled = self.num_transformer.inverse_transform(num_synth)
        df_synth = pd.DataFrame(num_unscaled, columns=self.numerical_cols)

        for i, col in enumerate(self.categorical_cols):
            le = self.label_encoders[col]
            cat_labels = np.round(cat_synth[:, i]).astype(int)
            cat_labels = np.clip(cat_labels, 0, len(le.classes_) - 1)
            df_synth[col] = le.inverse_transform(cat_labels)

        return df_synth[self.feature_order]


class GaussianMultinomialDiffusion(nn.Module):
    def __init__(self, num_numerical_features, cat_cardinalities, cfg: Config):
        super().__init__()
        self.num_numerical = num_numerical_features
        self.cat_cardinalities = cat_cardinalities
        self.T = cfg.N_TIMESTEPS
        self.device = cfg.DEVICE

        if cfg.BETA_SCHEDULE == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, self.T, device=self.device)
        elif cfg.BETA_SCHEDULE == 'cosine':
            s = 0.008
            steps = torch.arange(self.T + 1, device=self.device, dtype=torch.float32) / self.T
            alpha_bar = torch.cos(((steps + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {cfg.BETA_SCHEDULE}")

        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def q_sample(self, x0, t):
        x0_num = x0[:, :self.num_numerical]
        noise = torch.randn_like(x0_num, device=self.device)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        xt_num = sqrt_alpha_bar_t * x0_num + sqrt_one_minus_alpha_bar_t * noise
        xt_cat = x0[:, self.num_numerical:]
        xt = torch.cat([xt_num, xt_cat], dim=1)
        return xt, noise

    @torch.no_grad()
    def p_sample(self, model, xt, t_tensor):
        t_int = t_tensor[0].item()
        pred_noise_num, pred_cat_logits = model(xt, t_tensor)
        beta_t = self.betas[t_int]
        alpha_t = self.alphas[t_int]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_int]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        x_prev_num = (1 / sqrt_alpha_t) * (
                    xt[:, :self.num_numerical] - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise_num)
        if t_int > 0:
            noise = torch.randn_like(x_prev_num)
            x_prev_num += torch.sqrt(beta_t) * noise
        x_prev_cat_list = []
        if self.cat_cardinalities:
            prob_idx = 0
            for k in self.cat_cardinalities:
                cat_logits = pred_cat_logits[:, prob_idx:prob_idx + k]
                cat_dist = torch.distributions.Categorical(logits=cat_logits)
                x_prev_cat_list.append(cat_dist.sample().unsqueeze(1))
                prob_idx += k
        x_prev_cat = torch.cat(x_prev_cat_list, dim=1).float() if self.cat_cardinalities else torch.empty(xt.shape[0],
                                                                                                          0,
                                                                                                          device=self.device)
        return torch.cat([x_prev_num, x_prev_cat], dim=1)


class MLPDenoiser(nn.Module):
    def __init__(self, num_features, num_numerical, cat_cardinalities, cfg: Config):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.device = cfg.DEVICE
        time_emb_dim = 128
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        input_dim = num_features + time_emb_dim
        hidden_dims = [input_dim] + list(cfg.MLP_HIDDEN_DIMS)
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.Mish()
            ])
        self.mlp = nn.Sequential(*layers)
        self.out_num = nn.Linear(cfg.MLP_HIDDEN_DIMS[-1], num_numerical)
        if cat_cardinalities:
            self.out_cat = nn.Linear(cfg.MLP_HIDDEN_DIMS[-1], sum(cat_cardinalities))
        else:
            self.out_cat = None

    def _sinusoidal_embedding(self, t, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t):
        t_emb = self._sinusoidal_embedding(t, 128)
        t_emb = self.time_emb(t_emb)
        x_t = torch.cat([x, t_emb], dim=1)
        h = self.mlp(x_t)
        pred_noise_num = self.out_num(h)
        pred_cat_logits = self.out_cat(h) if self.out_cat else None
        return pred_noise_num, pred_cat_logits


def train(cfg: Config):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    data_handler = DataHandler(cfg)
    data_tensor = data_handler.load_and_preprocess()
    dataloader = data_handler.get_dataloader(data_tensor)
    data_handler.save_state()
    num_features = data_tensor.shape[1]
    num_numerical = len(data_handler.numerical_cols)
    cat_cardinalities = [len(le.classes_) for le in data_handler.label_encoders.values()]
    model = MLPDenoiser(num_features, num_numerical, cat_cardinalities, cfg).to(cfg.DEVICE)
    diffusion = GaussianMultinomialDiffusion(num_numerical, cat_cardinalities, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    print("--- Starting TabDDPM Training ---")
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        for (batch,) in progress_bar:
            optimizer.zero_grad()
            x0 = batch.to(cfg.DEVICE)
            t = torch.randint(0, cfg.N_TIMESTEPS, (x0.shape[0],), device=cfg.DEVICE)
            xt, noise_gt_num = diffusion.q_sample(x0, t)
            pred_noise_num, pred_cat_logits = model(xt, t)
            loss_num = F.mse_loss(pred_noise_num, noise_gt_num)
            loss_cat = 0
            if cat_cardinalities:
                x0_cat = x0[:, num_numerical:].long()
                start_idx = 0
                for i, k in enumerate(cat_cardinalities):
                    target_cat = x0_cat[:, i]
                    pred_logits_cat = pred_cat_logits[:, start_idx: start_idx + k]
                    loss_cat += F.cross_entropy(pred_logits_cat, target_cat)
                    start_idx += k
                loss_cat = loss_cat / len(cat_cardinalities)
            loss = loss_num + loss_cat if cat_cardinalities else loss_num
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": total_loss / (progress_bar.n + 1)})
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"--- Training Finished. Model saved to {cfg.MODEL_PATH} ---")


def sample(cfg: Config, num_samples: int):
    print("\n--- Generating Synthetic Data ---")
    data_handler = DataHandler(cfg)
    data_handler.load_state()

    num_features = len(data_handler.feature_order)
    num_numerical = len(data_handler.numerical_cols)
    cat_cardinalities = [len(le.classes_) for le in data_handler.label_encoders.values()]

    model = MLPDenoiser(num_features, num_numerical, cat_cardinalities, cfg).to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE, weights_only=True))
    model.eval()

    diffusion = GaussianMultinomialDiffusion(num_numerical, cat_cardinalities, cfg)

    xt_num = torch.randn(num_samples, num_numerical, device=cfg.DEVICE)
    xt_cat = torch.zeros(num_samples, len(cat_cardinalities), device=cfg.DEVICE)
    xt = torch.cat([xt_num, xt_cat], dim=1)

    for t in tqdm(reversed(range(cfg.N_TIMESTEPS)), desc="Sampling"):
        time_tensor = torch.full((num_samples,), t, device=cfg.DEVICE, dtype=torch.long)
        xt = diffusion.p_sample(model, xt, time_tensor)

    synthetic_df = data_handler.inverse_transform(xt)
    synthetic_df.to_csv(cfg.SYNTHETIC_DATA_PATH, index=False)
    print(f"Generated {num_samples} samples and saved to {cfg.SYNTHETIC_DATA_PATH}")
    print("\nFirst 5 rows of synthetic data:")
    print(synthetic_df.head())


if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Data file not found at {config.DATA_FILE}")
        print("Please ensure the CSV file is in the correct directory.")
    else:
        train(config)
        real_df = pd.read_csv(config.DATA_FILE)
        num_benign_samples = len(real_df[real_df['Label_code'] == 0])
        sample(config, num_samples=num_benign_samples)