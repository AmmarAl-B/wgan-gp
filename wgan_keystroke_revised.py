import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer
import joblib
import logging
import keras
from keras import layers
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class Config:
    DATA_FILE = './data/smartattackdata_original_foldcauchy.csv'
    SCALER_PATH = './models/quantile_transformer.joblib'
    MODEL_DIR = './models/'
    OUTPUT_DIR = './output/'
    SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, 'synthetic_keystroke_data_revised.csv')
    LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'wgan_gp_training_loss_revised.png')

    EPOCHS = 300
    BATCH_SIZE = 128
    LATENT_DIM = 128
    N_CRITIC = 5
    GP_WEIGHT = 10.0
    LEARNING_RATE = 2e-4
    BETA_1 = 0.5
    BETA_2 = 0.9


class DataHandler:
    def __init__(self, file_path: str, target_col: str = 'Label_code', benign_label: int = 0, n_quantiles: int = 1000):
        self.file_path = file_path
        self.target_col = target_col
        self.benign_label = benign_label
        self.n_quantiles = n_quantiles
        self.scaler = None
        self.feature_names = None

    def _load_and_clean_data(self) -> pd.DataFrame:
        logging.info(f"Loading data from {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            logging.error(f"FATAL: Data file not found at '{self.file_path}'.")
            raise

        df = df.drop(columns=['Unnamed: 0', 'Label_Desc', 'filename'], errors='ignore')

        benign_df = df[df[self.target_col] == self.benign_label].copy()
        benign_df = benign_df.drop(columns=[self.target_col])

        numeric_cols = benign_df.select_dtypes(include=np.number).columns
        benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in numeric_cols:
            if benign_df[col].isnull().any():
                median_val = benign_df[col].median()
                benign_df[col].fillna(median_val, inplace=True)
                logging.info(f"  - Filled NaN values in column '{col}' with median {median_val:.4f}")

        self.feature_names = benign_df.columns.tolist()
        logging.info(f"Found {len(benign_df)} benign samples with {len(self.feature_names)} features.")
        return benign_df

    def preprocess_and_save_scaler(self, scaler_path: str = 'quantile_transformer.joblib'):
        benign_data = self._load_and_clean_data()

        num_samples = len(benign_data)
        effective_quantiles = min(num_samples, self.n_quantiles)
        logging.info(
            f"Initializing QuantileTransformer with output_distribution='normal' and n_quantiles={effective_quantiles}.")

        self.scaler = QuantileTransformer(output_distribution='normal', n_quantiles=effective_quantiles,
                                          random_state=42)
        self.scaler.fit(benign_data)

        joblib.dump(self.scaler, scaler_path)
        logging.info(f"✅ QuantileTransformer fitted and saved to '{scaler_path}'.")

    def get_training_dataset(self, batch_size: int, scaler_path: str = 'quantile_transformer.joblib') -> tuple[
        tf.data.Dataset, tuple]:
        benign_data = self._load_and_clean_data()

        try:
            self.scaler = joblib.load(scaler_path)
            logging.info(f"Loaded pre-fitted QuantileTransformer from '{scaler_path}'.")
        except FileNotFoundError:
            logging.error(
                f"FATAL: Scaler file '{scaler_path}' not found. Please run `preprocess_and_save_scaler` first.")
            raise

        scaled_data = self.scaler.transform(benign_data)

        logging.info("Creating tf.data.Dataset for training...")
        dataset = tf.data.Dataset.from_tensor_slices(scaled_data.astype(np.float32))
        dataset = dataset.shuffle(buffer_size=len(scaled_data)).batch(batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE)

        return dataset, scaled_data.shape


def build_generator(latent_dim, num_features):
    model = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(1024),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(num_features, activation="tanh"),
        ],
        name="generator",
    )
    return model


def build_critic(num_features):
    model = keras.Sequential(
        [
            keras.Input(shape=(num_features,)),
            layers.Dense(1024),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(512),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dense(1),
        ],
        name="critic",
    )
    return model


class WGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gp_metric = keras.metrics.Mean(name="gp")

    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    @property
    def metrics(self):
        return [self.c_loss_metric, self.g_loss_metric, self.gp_metric]

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        interpolated_samples = real_samples + alpha * (fake_samples - real_samples)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            pred = self.critic(interpolated_samples, training=True)

        grads = tape.gradient(pred, interpolated_samples)[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]

        for _ in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_samples = self.generator(random_latent_vectors, training=True)
                real_output = self.critic(real_samples, training=True)
                fake_output = self.critic(fake_samples, training=True)

                c_cost = self.c_loss_fn(real_output, fake_output)
                gp = self.gradient_penalty(batch_size, real_samples, fake_samples)
                c_loss = c_cost + gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_latent_vectors, training=True)
            gen_output = self.critic(generated_samples, training=True)
            g_loss = self.g_loss_fn(gen_output)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        self.gp_metric.update_state(gp)
        return {"c_loss": self.c_loss_metric.result(), "g_loss": self.g_loss_metric.result(),
                "gp": self.gp_metric.result()}


def critic_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


if __name__ == '__main__':
    cfg = Config()

    # Create necessary directories if they don't exist
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.DATA_FILE), exist_ok=True)

    # Create a dummy data file for demonstration if it doesn't exist
    if not os.path.exists(cfg.DATA_FILE):
        logging.warning(f"Data file not found. Creating a dummy file at {cfg.DATA_FILE}")
        num_samples = 2000
        num_features = 29
        data = np.random.standard_cauchy((num_samples, num_features))
        columns = ['fl_dur', 'tot_fw_pk', 'tot_bw_pk', 'tot_l_fw_pkt', 'fw_pkt_l_max', 'fw_pkt_l_min', 'fw_pkt_l_avg',
                   'fw_pkt_l_std', 'bw_pkt_l_max', 'bw_pkt_l_min', 'bw_pkt_l_mean', 'bw_pkt_l_std', 'fw_fl_byt_s',
                   'bw_fl_byt_s', 'fw_fl_pkt_s', 'bw_fl_pkt_s', 'fw_iat_tot', 'fw_iat_avg', 'fw_iat_std', 'fw_iat_max',
                   'fw_iat_min', 'bw_iat_tot', 'bw_iat_avg', 'bw_iat_std', 'bw_iat_max', 'bw_iat_min', 'fw_pkt_s',
                   'bw_pkt_s', 'pkt_size_avg']
        dummy_df = pd.DataFrame(data, columns=columns)
        dummy_df['Unnamed: 0'] = range(num_samples)
        dummy_df['Label_code'] = np.random.randint(0, 2, size=num_samples)
        dummy_df['Label_Desc'] = 'Dummy'
        dummy_df['filename'] = 'dummy_file'
        dummy_df.to_csv(cfg.DATA_FILE, index=False)

    data_handler = DataHandler(file_path=cfg.DATA_FILE, n_quantiles=1000)

    if not os.path.exists(cfg.SCALER_PATH):
        logging.info("Scaler not found. Fitting and saving a new one...")
        data_handler.preprocess_and_save_scaler(scaler_path=cfg.SCALER_PATH)

    dataset, data_shape = data_handler.get_training_dataset(batch_size=cfg.BATCH_SIZE, scaler_path=cfg.SCALER_PATH)
    num_features = data_shape[1]
    num_samples = data_shape[0]

    generator = build_generator(cfg.LATENT_DIM, num_features)
    critic = build_critic(num_features)
    logging.info("\nGenerator Architecture:")
    generator.summary()
    logging.info("\nCritic Architecture:")
    critic.summary()

    wgan = WGAN_GP(
        critic=critic,
        generator=generator,
        latent_dim=cfg.LATENT_DIM,
        critic_steps=cfg.N_CRITIC,
        gp_weight=cfg.GP_WEIGHT
    )

    generator_optimizer = keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2)
    critic_optimizer = keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2)

    wgan.compile(
        c_optimizer=critic_optimizer,
        g_optimizer=generator_optimizer,
        c_loss_fn=critic_loss,
        g_loss_fn=generator_loss,
    )

    logging.info("\nStarting WGAN-GP training...")
    history = wgan.fit(dataset, epochs=cfg.EPOCHS)
    logging.info("✅ Training finished.")

    generator.save(os.path.join(cfg.MODEL_DIR, 'generator_model.h5'))
    logging.info(f"✅ Generator model saved to '{cfg.MODEL_DIR}'")

    logging.info("\nGenerating synthetic data...")
    random_latent_vectors = tf.random.normal(shape=(num_samples, cfg.LATENT_DIM))
    synthetic_data_scaled = generator.predict(random_latent_vectors)

    scaler = joblib.load(cfg.SCALER_PATH)
    synthetic_data_unscaled = scaler.inverse_transform(synthetic_data_scaled)

    synthetic_df = pd.DataFrame(synthetic_data_unscaled, columns=data_handler.feature_names)
    synthetic_df.to_csv(cfg.SYNTHETIC_DATA_PATH, index=False)
    logging.info(f"✅ Successfully generated and saved {num_samples} synthetic samples to '{cfg.SYNTHETIC_DATA_PATH}'")
    logging.info("\nFirst 5 rows of the generated synthetic data:")
    print(synthetic_df.head())

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['c_loss'], label='Critic Loss')
    plt.plot(history.history['g_loss'], label='Generator Loss')
    plt.title('WGAN-GP Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Wasserstein Distance Estimate)')
    plt.legend()
    plt.grid(True)
    plt.savefig(cfg.LOSS_PLOT_PATH)
    logging.info(f"\n✅ Saved training loss plot to '{cfg.LOSS_PLOT_PATH}'")