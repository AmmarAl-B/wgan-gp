import pandas as pd
import numpy as np
import os
import logging
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import get_column_plot
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class Config:
    """Configuration class for file paths and model hyperparameters."""
    DATA_DIR = './data/'
    MODEL_DIR = './models/'
    OUTPUT_DIR = './output/'

    REAL_DATA_FILE = os.path.join(DATA_DIR, 'smartattackdata_original_foldcauchy.csv')
    SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, 'ctgan_synthetic_keystroke_data.csv')
    MODEL_PATH = os.path.join(MODEL_DIR, 'ctgan_synthesizer.pkl')

    TARGET_COL = 'Label_code'
    BENIGN_LABEL = 0
    COLS_TO_DROP = ['Unnamed: 0', 'Label_Desc', 'filename']

    EPOCHS = 50
    BATCH_SIZE = 60
    GENERATOR_LR = 2e-4
    DISCRIMINATOR_LR = 2e-4
    EMBEDDING_DIM = 32
    GENERATOR_DIM = (32, 32)
    DISCRIMINATOR_DIM = (32, 32)


def load_and_prepare_data(cfg: Config) -> pd.DataFrame:
    """Loads, cleans, and filters the data for benign samples."""
    logging.info(f"Loading data from {cfg.REAL_DATA_FILE}...")
    try:
        df = pd.read_csv(cfg.REAL_DATA_FILE)
    except FileNotFoundError:
        logging.error(f"FATAL: Data file not found at '{cfg.REAL_DATA_FILE}'.")
        raise

    df = df.drop(columns=cfg.COLS_TO_DROP, errors='ignore')

    benign_df = df[df[cfg.TARGET_COL] == cfg.BENIGN_LABEL].copy()
    benign_df = benign_df.drop(columns=[cfg.TARGET_COL])

    numeric_cols = benign_df.select_dtypes(include=np.number).columns
    benign_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in numeric_cols:
        if benign_df[col].isnull().any():
            median_val = benign_df[col].median()
            benign_df[col] = benign_df[col].fillna(median_val)
            logging.info(f"  - Filled NaN values in column '{col}' with median {median_val:.4f}")

    logging.info(f"Prepared {len(benign_df)} benign samples with {len(benign_df.columns)} features.")
    return benign_df


def train_and_generate(real_data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Trains the CTGAN synthesizer and generates synthetic data."""
    logging.info("Inferring metadata from the real data...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    for column in metadata.columns:
        metadata.update_column(column_name=column, sdtype='numerical')

    logging.info("Updated all columns to 'numerical' sdtype.")

    logging.info("Initializing CTGANSynthesizer with custom hyperparameters...")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        generator_lr=cfg.GENERATOR_LR,
        discriminator_lr=cfg.DISCRIMINATOR_LR,
        embedding_dim=cfg.EMBEDDING_DIM,
        generator_dim=cfg.GENERATOR_DIM,
        discriminator_dim=cfg.DISCRIMINATOR_DIM,
        verbose=True
    )

    logging.info(f"Starting CTGAN training for {cfg.EPOCHS} epochs...")
    synthesizer.fit(real_data)
    logging.info("✅ Training finished.")

    synthesizer.save(filepath=cfg.MODEL_PATH)
    logging.info(f"✅ Synthesizer model saved to '{cfg.MODEL_PATH}'.")

    logging.info(f"Generating {len(real_data)} synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=len(real_data))

    synthetic_data.to_csv(cfg.SYNTHETIC_DATA_PATH, index=False)
    logging.info(f"✅ Synthetic data saved to '{cfg.SYNTHETIC_DATA_PATH}'.")

    logging.info("\nFirst 5 rows of the generated synthetic data:")
    print(synthetic_data.head())

    return synthetic_data


def perform_evaluation(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, cfg: Config) -> None:
    """Performs and visualizes a comprehensive evaluation of the synthetic data."""
    logging.info("\n--- Starting Evaluation ---")

    logging.info("Performing Kolmogorov-Smirnov (K-S) tests for all features...")
    ks_results = []
    for col in real_data.columns:
        stat, p_value = ks_2samp(real_data[col], synthetic_data[col])
        ks_results.append({'Feature': col, 'P-Value': p_value})

    ks_df = pd.DataFrame(ks_results).sort_values(by='P-Value')
    logging.info("K-S Test Results (Real vs. CTGAN-Synthetic):")
    print(ks_df.to_string())

    logging.info("Generating univariate distribution comparison plots...")
    plot_cols = ['fl_dur', 'fw_pkt_l_avg', 'fw_iat_avg', 'bw_iat_avg']
    for col in plot_cols:
        if col in real_data.columns:
            try:
                fig = get_column_plot(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    column_name=col
                )
                plot_path = os.path.join(cfg.OUTPUT_DIR, f'dist_plot_{col}.png')
                fig.savefig(plot_path)
                logging.info(f"  - Saved distribution plot for '{col}' to {plot_path}")
                plt.close(fig)
            except Exception as e:
                logging.warning(f"Could not generate plot for column '{col}': {e}")

    logging.info("Generating correlation matrix heatmaps...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plt.suptitle('Correlation Matrix Comparison', fontsize=16)

        real_corr = real_data.corr()
        sns.heatmap(real_corr, ax=axes[0], cmap='viridis')
        axes[0].set_title('Real Data Correlation Matrix')

        synthetic_corr = synthetic_data.corr()
        sns.heatmap(synthetic_corr, ax=axes[1], cmap='viridis')
        axes[1].set_title('Synthetic Data Correlation Matrix')

        corr_plot_path = os.path.join(cfg.OUTPUT_DIR, 'correlation_comparison.png')
        plt.savefig(corr_plot_path)
        logging.info(f"  - Saved correlation heatmap comparison to {corr_plot_path}")
        plt.close(fig)
    except Exception as e:
        logging.error(f"Could not generate correlation heatmaps: {e}")

    logging.info("✅ Evaluation complete.")


if __name__ == '__main__':
    cfg = Config()

    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(cfg.REAL_DATA_FILE):
        logging.error(f"Data file '{cfg.REAL_DATA_FILE}' not found. Please place it in the '{cfg.DATA_DIR}' directory.")
    else:
        real_benign_data = load_and_prepare_data(cfg)
        synthetic_data = train_and_generate(real_benign_data, cfg)
        perform_evaluation(real_benign_data, synthetic_data, cfg)