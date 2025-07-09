import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 128
EPOCHS = 300
LATENT_DIM = 128
N_CRITIC = 5
GP_WEIGHT = 10.0
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9


def load_and_preprocess_data(file_path):
    """
    Loads the keystroke dataset, cleans invalid values (NaN/Infinity), filters
    for benign samples, and scales the features to the [-1, 1] range.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Could not read as Excel: {e}. Trying as CSV...")
        try:
            df = pd.read_csv(file_path)
        except Exception as e_csv:
            print(f"Could not read as CSV: {e_csv}. Please check the file format.")
            return None, None, None

    df = df.drop(columns=['Unnamed: 0', 'Label_Desc', 'filename'], errors='ignore')

    benign_df = df[df['Label_code'] == 0].copy()
    benign_df = benign_df.drop(columns=['Label_code'])

    # --- Data Cleaning Step ---
    numeric_cols = benign_df.select_dtypes(include=np.number).columns

    if np.isinf(benign_df[numeric_cols]).any().any() or benign_df[numeric_cols].isnull().any().any():
        print("Dataset contains NaN or Infinity values. Performing automatic cleaning...")
        # Replace infinity with NaN
        benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Impute NaN with the median of each column
        for col in numeric_cols:
            if benign_df[col].isnull().any():
                median_val = benign_df[col].median()
                benign_df[col].fillna(median_val, inplace=True)
                print(f"  - Filled NaN values in column '{col}' with median value {median_val:.4f}")
        print("Cleaning complete.")

    feature_names = benign_df.columns.tolist()
    print(f"Found {len(benign_df)} benign samples.")
    print(f"Features being used ({len(feature_names)}): {feature_names}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(benign_df)

    return scaled_data, scaler, feature_names


def build_generator(latent_dim, num_features):
    """
    Builds the Generator model which takes a random noise vector as input and
    outputs a synthetic data sample.
    """
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(num_features, activation="tanh"),
    ], name="generator")
    return model


def build_critic(num_features):
    """
    Builds the Critic model which takes a data sample (real or fake) and outputs
    a scalar score of its "realness".
    """
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation="linear"),
    ], name="critic")
    return model


class WGAN_GP(keras.Model):
    """
    A Keras Model class that encapsulates the WGAN-GP architecture and custom
    training logic.
    """

    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        """Configures the model for training."""
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    @property
    def metrics(self):
        """Returns the list of metrics to display."""
        return [self.c_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        """
        Calculates the gradient penalty term, the core of WGAN-GP for enforcing
        the Lipschitz constraint.
        """
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        interpolated_samples = real_samples + alpha * (fake_samples - real_samples)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_samples)
            pred = self.critic(interpolated_samples, training=True)

        grads = tape.gradient(pred, [interpolated_samples])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        """
        Defines the logic for a single training step, including asymmetric updates
        and loss calculation.
        """
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
        return {"c_loss": self.c_loss_metric.result(), "g_loss": self.g_loss_metric.result()}


def critic_loss(real_output, fake_output):
    """Wasserstein loss for the critic."""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss(fake_output):
    """Wasserstein loss for the generator."""
    return -tf.reduce_mean(fake_output)


if __name__ == '__main__':
    DATA_FILE = './data/smartattackdata_original_foldcauchy.csv'
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found.")
        print("Please ensure the Excel or CSV file with your keystroke data is in the same directory.")
    else:
        real_data, scaler, feature_names = load_and_preprocess_data(DATA_FILE)

        if real_data is not None:
            num_features = real_data.shape[1]
            print(f"Data loaded and preprocessed. Number of features: {num_features}")

            generator = build_generator(LATENT_DIM, num_features)
            critic = build_critic(num_features)

            print("\nGenerator Architecture:")
            generator.summary()
            print("\nCritic Architecture:")
            critic.summary()

            wgan = WGAN_GP(
                critic=critic,
                generator=generator,
                latent_dim=LATENT_DIM,
                critic_steps=N_CRITIC,
                gp_weight=GP_WEIGHT
            )

            generator_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)
            critic_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2)

            wgan.compile(
                c_optimizer=critic_optimizer,
                g_optimizer=generator_optimizer,
                c_loss_fn=critic_loss,
                g_loss_fn=generator_loss,
            )

            print("\nStarting WGAN-GP training...")
            dataset = tf.data.Dataset.from_tensor_slices(real_data.astype(np.float32))
            dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

            history = wgan.fit(dataset, epochs=EPOCHS)
            print("Training finished.")

            print("\nGenerating synthetic data...")
            num_samples_to_generate = len(real_data)
            random_latent_vectors = tf.random.normal(shape=(num_samples_to_generate, LATENT_DIM))
            synthetic_data_scaled = generator.predict(random_latent_vectors)

            synthetic_data = scaler.inverse_transform(synthetic_data_scaled)

            synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)

            synthetic_df.to_csv('synthetic_keystroke_data.csv', index=False)
            print(
                f"Successfully generated and saved {num_samples_to_generate} synthetic samples to 'synthetic_keystroke_data.csv'")
            print("\nFirst 5 rows of the generated synthetic data:")
            print(synthetic_df.head())

            plt.figure(figsize=(10, 5))
            plt.plot(history.history['c_loss'], label='Critic Loss')
            plt.plot(history.history['g_loss'], label='Generator Loss')
            plt.title('WGAN-GP Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (Wasserstein Distance Estimate)')
            plt.legend()
            plt.grid(True)
            plt.savefig('wgan_gp_training_loss.png')
            print("\nSaved training loss plot to 'wgan_gp_training_loss.png'")
