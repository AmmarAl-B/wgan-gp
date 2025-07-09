"""
Tabular Self-Attention Generative Adversarial Network (Tab-SAGAN) v3
------------------------------------------------------------------
This script implements a revised and stabilized Self-Attention GAN for
synthesizing high-fidelity tabular data. It incorporates Quantile
Transformation, Spectral Normalization, a corrected Self-Attention mechanism
that operates on features, and batched generation to prevent memory errors.

The pipeline includes:
1. Data loading with QuantileTransformer preprocessing.
2. PyTorch implementation of a Generator and Discriminator with a corrected
   Self-Attention layer and Spectral Normalization.
3. A robust WGAN-GP training loop.
4. A comprehensive evaluation suite with a Real-on-Real baseline for context.
"""

import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class Config:
    """Configuration class for hyperparameters and file paths."""

    DATA_FILE = './data/smartattackdata_original_foldcauchy.csv'

    OUTPUT_DIR = 'sagan_results_v3'
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    REPORT_DIR = os.path.join(OUTPUT_DIR, 'reports')

    GENERATOR_PATH = os.path.join(MODEL_DIR, 'generator.pth')
    DISCRIMINATOR_PATH = os.path.join(MODEL_DIR, 'discriminator.pth')
    SCALER_PATH = os.path.join(MODEL_DIR, 'quantile_transformer.joblib')
    SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, 'synthetic_data.csv')

    LATENT_DIM = 128
    HIDDEN_DIM = 256
    EPOCHS = 350
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    BETA_1 = 0.5
    BETA_2 = 0.9
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    FEATURE_COLUMNS = [
        'fl_dur', 'tot_fw_pk', 'tot_bw_pk', 'tot_l_fw_pkt', 'fw_pkt_l_max',
        'fw_pkt_l_min', 'fw_pkt_l_avg', 'fw_pkt_l_std', 'bw_pkt_l_max',
        'bw_pkt_l_min', 'bw_pkt_l_mean', 'bw_pkt_l_std', 'fw_fl_byt_s',
        'bw_fl_byt_s', 'fw_fl_pkt_s', 'bw_fl_pkt_s', 'fw_iat_tot',
        'fw_iat_avg', 'fw_iat_std', 'fw_iat_max', 'fw_iat_min',
        'bw_iat_tot', 'bw_iat_avg', 'bw_iat_std', 'bw_iat_max',
        'bw_iat_min', 'fw_pkt_s', 'bw_pkt_s', 'pkt_size_avg'
    ]
    LABEL_COL = 'Label_code'
    BENIGN_CLASS = 0
    ATTACK_CLASS = 1


class DataHandler:
    """Handles loading, preprocessing, and saving of data and scaler."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = None
        self.feature_names = None

    def get_training_dataloader(self) -> DataLoader:
        """Loads and prepares the benign data for GAN training."""
        print("--- Loading and Preparing Data for Training ---")
        if not os.path.exists(self.cfg.DATA_FILE):
            raise FileNotFoundError(f"Data file not found at {self.cfg.DATA_FILE}")

        df = pd.read_csv(self.cfg.DATA_FILE)
        benign_df = df[df[self.cfg.LABEL_COL] == self.cfg.BENIGN_CLASS][self.cfg.FEATURE_COLUMNS]
        self.feature_names = benign_df.columns.tolist()

        benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        benign_df.fillna(benign_df.median(), inplace=True)

        self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=max(min(len(benign_df) // 10, 1000), 10))
        data_scaled = self.scaler.fit_transform(benign_df)
        joblib.dump(self.scaler, self.cfg.SCALER_PATH)
        print(f"QuantileTransformer fitted and saved to {self.cfg.SCALER_PATH}")

        tensor_data = torch.FloatTensor(data_scaled)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True, drop_last=True)

        print(f"Prepared {len(benign_df)} benign samples for training.")
        return dataloader

    def get_data_for_evaluation(self) -> (pd.DataFrame, pd.DataFrame):
        """Loads the full real dataset for evaluation purposes."""
        df = pd.read_csv(self.cfg.DATA_FILE)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        real_benign = df[df[self.cfg.LABEL_COL] == self.cfg.BENIGN_CLASS][self.cfg.FEATURE_COLUMNS]
        real_full = df[self.cfg.FEATURE_COLUMNS + [self.cfg.LABEL_COL]]
        return real_benign, real_full


class SelfAttention(nn.Module):
    """Self-attention layer that attends over the feature dimension using Conv1d."""
    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim

        self.query_conv = spectral_norm(nn.Conv1d(in_channels=1, out_channels=self.in_dim // 8, kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv1d(in_channels=1, out_channels=self.in_dim // 8, kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1))

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, feat_dim = x.size()

        x_reshaped = x.unsqueeze(1)

        proj_query = self.query_conv(x_reshaped).permute(0, 2, 1)
        proj_key = self.key_conv(x_reshaped)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(x_reshaped)

        # Corrected the order of matrix multiplication to resolve the runtime error
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.squeeze(1)

        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    """Generator model with Spectral Norm and Self-Attention."""
    def __init__(self, latent_dim: int, feature_dim: int, hidden_dim: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, hidden_dim)),
            nn.ReLU(inplace=True),

            spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.ReLU(inplace=True),

            SelfAttention(hidden_dim * 2),

            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim * 2)),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator model with Spectral Norm and Self-Attention."""
    def __init__(self, feature_dim: int, hidden_dim: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim, hidden_dim * 2)),
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(hidden_dim * 2),

            spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compute_gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculates the gradient penalty loss for WGAN GP."""
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), 1, requires_grad=False, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Trainer:
    """Orchestrates the training process for the Tab-SAGAN."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        self.generator = Generator(cfg.LATENT_DIM, len(cfg.FEATURE_COLUMNS), cfg.HIDDEN_DIM).to(self.device)
        self.discriminator = Discriminator(len(cfg.FEATURE_COLUMNS), cfg.HIDDEN_DIM).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, cfg.BETA_2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, cfg.BETA_2))

        self.data_handler = DataHandler(cfg)

    def train(self):
        """Executes the main training loop."""
        dataloader = self.data_handler.get_training_dataloader()

        print("\n--- Starting Stabilized Tab-SAGAN Training ---")
        for epoch in range(self.cfg.EPOCHS):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{self.cfg.EPOCHS}")
            for i, (real_data_batch,) in progress_bar:
                real_data = real_data_batch.to(self.device)

                self.optimizer_D.zero_grad()

                z = torch.randn(real_data.size(0), self.cfg.LATENT_DIM, device=self.device)
                fake_data = self.generator(z)

                real_validity = self.discriminator(real_data)
                fake_validity = self.discriminator(fake_data.detach())

                gradient_penalty = compute_gradient_penalty(self.discriminator, real_data, fake_data, self.device)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.cfg.LAMBDA_GP * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                if i % self.cfg.CRITIC_ITERATIONS == 0:
                    self.optimizer_G.zero_grad()

                    z = torch.randn(real_data.size(0), self.cfg.LATENT_DIM, device=self.device)
                    gen_data = self.generator(z)

                    g_loss = -torch.mean(self.discriminator(gen_data))
                    g_loss.backward()
                    self.optimizer_G.step()

                    progress_bar.set_postfix({
                        "D Loss": f"{d_loss.item():.4f}",
                        "G Loss": f"{g_loss.item():.4f}"
                    })

        print("--- Training Finished ---")
        torch.save(self.generator.state_dict(), self.cfg.GENERATOR_PATH)
        torch.save(self.discriminator.state_dict(), self.cfg.DISCRIMINATOR_PATH)
        print(f"Models saved to {self.cfg.MODEL_DIR}")

    def generate_synthetic_data(self, num_samples: int) -> pd.DataFrame:
        """Generates synthetic data using the trained generator in batches."""
        print("\n--- Generating Synthetic Data ---")
        generator = Generator(self.cfg.LATENT_DIM, len(self.cfg.FEATURE_COLUMNS), self.cfg.HIDDEN_DIM).to(self.device)
        generator.load_state_dict(torch.load(self.cfg.GENERATOR_PATH, map_location=self.device))
        generator.eval()

        scaler = joblib.load(self.cfg.SCALER_PATH)

        all_synthetic_data = []
        generation_batch_size = self.cfg.BATCH_SIZE

        with torch.no_grad():
            for _ in tqdm(range(0, num_samples, generation_batch_size), desc="Generating Batches"):
                current_batch_size = min(generation_batch_size, num_samples - len(all_synthetic_data) * generation_batch_size)
                if current_batch_size <= 0:
                    break

                z = torch.randn(current_batch_size, self.cfg.LATENT_DIM, device=self.device)
                synthetic_batch_scaled = generator(z).cpu().numpy()
                all_synthetic_data.append(synthetic_batch_scaled)

        synthetic_data_scaled = np.concatenate(all_synthetic_data, axis=0)

        synthetic_data_unscaled = scaler.inverse_transform(synthetic_data_scaled)
        synthetic_df = pd.DataFrame(synthetic_data_unscaled, columns=self.data_handler.feature_names)

        synthetic_df.to_csv(self.cfg.SYNTHETIC_DATA_PATH, index=False)
        print(f"Generated {len(synthetic_df)} samples and saved to {self.cfg.SYNTHETIC_DATA_PATH}")
        return synthetic_df


class Evaluator:
    """Runs a comprehensive evaluation suite."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_all_evaluations(self, real_benign: pd.DataFrame, real_full: pd.DataFrame, synthetic_benign: pd.DataFrame):
        """Executes all evaluation components and saves reports."""
        print("\n--- Starting Full Evaluation Suite ---")
        self.evaluate_statistical_fidelity(real_benign, synthetic_benign)
        self.evaluate_ml_utility(real_full, synthetic_benign)
        self.perform_visual_diagnostics(real_benign, synthetic_benign)
        print("--- Evaluation Complete ---")

    def evaluate_statistical_fidelity(self, real_benign: pd.DataFrame, synthetic_benign: pd.DataFrame):
        """Performs and reports the Kolmogorov-Smirnov test results."""
        print("\n1. Evaluating Statistical Fidelity (K-S Test)...")
        ks_results = {}
        for col in real_benign.columns:
            stat, p_value = ks_2samp(real_benign[col], synthetic_benign[col])
            ks_results[col] = p_value

        ks_df = pd.DataFrame(list(ks_results.items()), columns=['Feature', 'P-Value'])
        ks_df = ks_df.sort_values(by='P-Value', ascending=False)

        report_path = os.path.join(self.cfg.REPORT_DIR, 'ks_test_report.csv')
        ks_df.to_csv(report_path, index=False)
        print(f"K-S test results saved to {report_path}")
        print("Top 5 K-S Test Results:")
        print(ks_df.head())

    def evaluate_ml_utility(self, real_full: pd.DataFrame, synthetic_benign: pd.DataFrame):
        """Performs Train-on-Synthetic vs. Train-on-Real evaluation."""
        print("\n2. Evaluating Machine Learning Utility...")

        X_real_full = real_full[self.cfg.FEATURE_COLUMNS]
        y_real_full = (real_full[self.cfg.LABEL_COL] == self.cfg.ATTACK_CLASS).astype(int)

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real_full, y_real_full, test_size=0.3, random_state=42, stratify=y_real_full
        )

        model_real = LGBMClassifier(random_state=42)
        model_real.fit(X_train_real, y_train_real)
        y_pred_real = model_real.predict(X_test_real)

        acc_real = accuracy_score(y_test_real, y_pred_real)
        f1_real = f1_score(y_test_real, y_pred_real)

        real_attack = real_full[real_full[self.cfg.LABEL_COL] == self.cfg.ATTACK_CLASS]
        X_synth_train = pd.concat([synthetic_benign, real_attack[self.cfg.FEATURE_COLUMNS]], ignore_index=True)
        y_synth_train = np.concatenate([np.zeros(len(synthetic_benign)), np.ones(len(real_attack))])

        model_synth = LGBMClassifier(random_state=42)
        model_synth.fit(X_synth_train, y_synth_train)
        y_pred_synth = model_synth.predict(X_test_real)

        acc_synth = accuracy_score(y_test_real, y_pred_synth)
        f1_synth = f1_score(y_test_real, y_pred_synth)

        report_text = f"""
        Machine Learning Utility Report
        ===============================
        Model: LGBMClassifier

        This report compares a model trained on real data (baseline) against a
        model trained on synthetic data. Both are tested on a held-out set
        of real data.

        Baseline (Train on Real, Test on Real):
        - Accuracy:  {acc_real:.4f}
        - F1-Score:  {f1_real:.4f}

        TSTR (Train on Synthetic, Test on Real):
        - Accuracy:  {acc_synth:.4f}
        - F1-Score:  {f1_synth:.4f}
        """

        report_path = os.path.join(self.cfg.REPORT_DIR, 'ml_utility_report.txt')
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"ML utility report saved to {report_path}")
        print(f"TSTR Accuracy: {acc_synth:.4f} (Baseline Real-on-Real Accuracy: {acc_real:.4f})")

    def perform_visual_diagnostics(self, real_benign: pd.DataFrame, synthetic_benign: pd.DataFrame):
        """Generates and saves visual comparison plots."""
        print("\n3. Performing Visual Diagnostics...")
        plt.style.use('seaborn-v0_8-whitegrid')

        num_features = len(self.cfg.FEATURE_COLUMNS)
        cols = 5
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = axes.flatten()
        for i, col in enumerate(self.cfg.FEATURE_COLUMNS):
            sns.kdeplot(real_benign[col], ax=axes[i], label='Real', fill=True, color='blue', alpha=0.5, clip=(real_benign[col].min(), real_benign[col].max()))
            sns.kdeplot(synthetic_benign[col], ax=axes[i], label='Synthetic', fill=True, color='orange', alpha=0.5, clip=(real_benign[col].min(), real_benign[col].max()))
            axes[i].set_title(col, fontsize=10)
            axes[i].legend()
            axes[i].tick_params(axis='x', labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.0)
        plt.suptitle('Univariate Distribution Comparison', fontsize=16, y=1.02)
        dist_path = os.path.join(self.cfg.REPORT_DIR, 'distribution_comparison.png')
        plt.savefig(dist_path)
        plt.close()
        print(f"Distribution comparison plot saved to {dist_path}")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        real_corr = real_benign.corr()
        sns.heatmap(real_corr, ax=axes[0], cmap='viridis').set_title('Real Data Correlation')
        synthetic_corr = synthetic_benign.corr()
        sns.heatmap(synthetic_corr, ax=axes[1], cmap='viridis').set_title('Synthetic Data Correlation')
        plt.tight_layout()
        corr_path = os.path.join(self.cfg.REPORT_DIR, 'correlation_comparison.png')
        plt.savefig(corr_path)
        plt.close()
        print(f"Correlation heatmap saved to {corr_path}")

        pca = PCA(n_components=2)
        scaler_pca = QuantileTransformer(output_distribution='normal')
        real_benign_scaled = scaler_pca.fit_transform(real_benign)
        synthetic_benign_scaled = scaler_pca.transform(synthetic_benign)

        real_pca = pca.fit_transform(real_benign_scaled)
        synthetic_pca = pca.transform(synthetic_benign_scaled)

        plt.figure(figsize=(10, 7))
        plt.scatter(real_pca[:, 0], real_pca[:, 1], s=15, alpha=0.6, label='Real Data', c='blue')
        plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], s=15, alpha=0.6, label='Synthetic Data', c='orange')
        plt.title('PCA Projection of Real vs. Synthetic Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        pca_path = os.path.join(self.cfg.REPORT_DIR, 'pca_comparison.png')
        plt.savefig(pca_path)
        plt.close()
        print(f"PCA comparison plot saved to {pca_path}")


if __name__ == '__main__':
    config = Config()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    trainer = Trainer(config)
    trainer.train()

    data_handler = DataHandler(config)
    real_benign_eval, real_full_eval = data_handler.get_data_for_evaluation()
    num_samples_to_gen = len(real_benign_eval)

    synthetic_data = trainer.generate_synthetic_data(num_samples=num_samples_to_gen)

    evaluator = Evaluator(config)
    evaluator.run_all_evaluations(real_benign_eval, real_full_eval, synthetic_data)