import pandas as pd
import numpy as np
import warnings
import os
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def define_columns():
    all_columns = [
        'Unnamed: 0', 'fl_dur', 'tot_fw_pk', 'tot_bw_pk', 'tot_l_fw_pkt',
        'fw_pkt_l_max', 'fw_pkt_l_min', 'fw_pkt_l_avg', 'fw_pkt_l_std',
        'bw_pkt_l_max', 'bw_pkt_l_min', 'bw_pkt_l_mean', 'bw_pkt_l_std',
        'fw_fl_byt_s', 'bw_fl_byt_s', 'fw_fl_pkt_s', 'bw_fl_pkt_s',
        'fw_iat_tot', 'fw_iat_avg', 'fw_iat_std', 'fw_iat_max', 'fw_iat_min',
        'bw_iat_tot', 'bw_iat_avg', 'bw_iat_std', 'bw_iat_max', 'bw_iat_min',
        'fw_pkt_s', 'bw_pkt_s', 'pkt_size_avg', 'Label_code', 'Label_Desc', 'filename'
    ]

    cols_to_drop = ['Unnamed: 0', 'Label_Desc', 'filename', 'Label_code']

    features_to_use = [col for col in all_columns if col not in cols_to_drop and col != 'Label_code']

    return all_columns, features_to_use


def load_and_prepare_data(file_path, all_columns, features_to_use):
    print("--- Loading and Preparing Data ---")
    try:
        real_data_full = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Dataset '{file_path}' not found.")
        print("Creating a dummy dataframe for demonstration purposes.")
        real_data_full = pd.DataFrame(np.random.rand(100, len(all_columns)), columns=all_columns)
        real_data_full['Label_code'] = np.random.randint(0, 2, 100)

    benign_data = real_data_full[real_data_full['Label_code'] == 0][features_to_use].copy()

    for col in benign_data.select_dtypes(include=np.number).columns:
        if np.isinf(benign_data[col]).any() or benign_data[col].isnull().any():
            benign_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            median_val = benign_data[col].median()
            benign_data[col].fillna(median_val, inplace=True)
            print(f"Cleaned NaNs/Infs in column: {col}")

    print(f"Loaded {len(real_data_full)} total samples.")
    print(f"Prepared {len(benign_data)} benign samples for training.\n")
    return benign_data, real_data_full


def train_tvae_and_generate(benign_data, output_path):
    print("--- Training TVAE and Generating Synthetic Data ---")

    constant_columns = {}
    variable_columns = []
    for col in benign_data.columns:
        if benign_data[col].nunique() == 1:
            constant_columns[col] = benign_data[col].iloc[0]
        else:
            variable_columns.append(col)

    if constant_columns:
        print("Found constant value columns. They will be handled separately.")
        for col, val in constant_columns.items():
            print(f"  - Column '{col}' has a constant value of {val}")
        print("\nTraining TVAE only on variable columns...")

    data_for_training = benign_data[variable_columns]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data_for_training)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("WARNING: CUDA not available. Training on CPU. This may be slow.")

    synthesizer = TVAESynthesizer(
        metadata,
        epochs=300,
        batch_size=500,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        cuda=use_cuda
    )

    print("Starting TVAE model training...")
    synthesizer.fit(data_for_training)
    print("Training complete.")

    num_synthetic_samples = len(benign_data)
    synthetic_data = synthesizer.sample(num_rows=num_synthetic_samples)

    if constant_columns:
        print("\nRe-inserting constant value columns into synthetic data...")
        for col, val in constant_columns.items():
            synthetic_data[col] = val

    synthetic_data = synthetic_data[benign_data.columns]

    synthetic_data.to_csv(output_path, index=False)
    print(f"Successfully generated and saved {num_synthetic_samples} synthetic samples to '{output_path}'.")
    print("\nGenerated Data Head:")
    print(synthetic_data.head())
    print("\n")
    return synthetic_data


def evaluate_statistical_fidelity(real_benign, synthetic_benign):
    print("--- Running Statistical Fidelity Evaluation (K-S Test) ---")
    ks_results = {}
    for column in real_benign.columns:
        if column in synthetic_benign.columns:
            if real_benign[column].nunique() > 1:
                ks_stat, p_value = ks_2samp(real_benign[column], synthetic_benign[column])
                ks_results[column] = p_value
            else:
                ks_results[column] = 1.0
        else:
            ks_results[column] = 0.0

    ks_df = pd.DataFrame(list(ks_results.items()), columns=['Feature', 'TVAE P-Value']).sort_values('TVAE P-Value')
    print("Kolmogorov-Smirnov Test Results:")
    print(ks_df)
    print("\n")


def evaluate_ml_utility(real_data_full, synthetic_benign, features_to_use):
    print("--- Running Machine Learning Utility Evaluation (TSTL) ---")
    real_benign = real_data_full[real_data_full['Label_code'] == 0][features_to_use]
    real_attack = real_data_full[real_data_full['Label_code'] == 1][features_to_use]

    X_synthetic_train = pd.concat([synthetic_benign, real_attack], ignore_index=True)
    y_synthetic_train = pd.Series([0] * len(synthetic_benign) + [1] * len(real_attack))

    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        real_data_full[features_to_use], real_data_full['Label_code'],
        test_size=0.3, random_state=42, stratify=real_data_full['Label_code']
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_synthetic_train, y_synthetic_train)

    y_pred = classifier.predict(X_real_test)
    accuracy = accuracy_score(y_real_test, y_pred)
    f1 = f1_score(y_real_test, y_pred)

    print("Classifier trained on TVAE synthetic data achieved:")
    print(f"  Accuracy on real test set: {accuracy * 100:.2f}%")
    print(f"  F1-Score on real test set: {f1 * 100:.2f}%")

    if accuracy < 0.1:
        print("\nCRITICAL WARNING: ML Utility accuracy is near zero.")
        print("This strongly suggests the generative model failed to train correctly.")
        print("Please check for errors during the training phase.\n")

    classifier_real = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier_real.fit(X_real_train, y_real_train)
    y_pred_real = classifier_real.predict(X_real_test)
    accuracy_real = accuracy_score(y_real_test, y_pred_real)
    f1_real = f1_score(y_real_test, y_pred_real)

    print("\nBaseline classifier trained on real data achieved:")
    print(f"  Accuracy on real test set: {accuracy_real * 100:.2f}%")
    print(f"  F1-Score on real test set: {f1_real * 100:.2f}%")
    print("\n")


def perform_visual_diagnostics(real_benign, synthetic_benign, output_dir):
    print("--- Performing Visual Diagnostics ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    sns.heatmap(real_benign.corr(), ax=axes[0], cmap='viridis')
    axes[0].set_title('Correlation Matrix - Real Benign Data', fontsize=14)
    sns.heatmap(synthetic_benign.corr(), ax=axes[1], cmap='viridis')
    axes[1].set_title('Correlation Matrix - TVAE Synthetic Data', fontsize=14)
    fig.suptitle('Comparison of Feature Correlation Matrices', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'))
    plt.close()
    print("Saved correlation comparison plot.")

    key_features = ['fw_fl_byt_s', 'fw_pkt_s', 'fw_iat_avg', 'bw_iat_avg', 'pkt_size_avg']
    variable_key_features = [f for f in key_features if
                             f in synthetic_benign.columns and synthetic_benign[f].nunique() > 1]

    if variable_key_features:
        fig, axes = plt.subplots(1, len(variable_key_features), figsize=(5 * len(variable_key_features), 5))
        if len(variable_key_features) == 1:
            axes = [axes]
        for i, feature in enumerate(variable_key_features):
            sns.kdeplot(real_benign[feature], ax=axes[i], label='Real', color='blue', fill=True, alpha=0.3)
            sns.kdeplot(synthetic_benign[feature], ax=axes[i], label='TVAE Synthetic', color='red', linestyle='--')
            axes[i].set_title(f'Distribution of {feature}', fontsize=12)
            axes[i].legend()
        fig.suptitle('Comparison of 1D Marginal Distributions', fontsize=18, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
        plt.close()
        print("Saved distribution comparison plot.")

    feature1 = 'fw_iat_avg'
    feature2 = 'fw_pkt_l_avg'
    if feature1 in synthetic_benign.columns and feature2 in synthetic_benign.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        axes[0].scatter(real_benign[feature1], real_benign[feature2], alpha=0.3, s=10)
        axes[0].set_title('Joint Distribution - Real Data', fontsize=14)
        axes[0].set_xlabel(feature1)
        axes[0].set_ylabel(feature2)
        axes[1].scatter(synthetic_benign[feature1], synthetic_benign[feature2], alpha=0.3, s=10, color='red')
        axes[1].set_title('Joint Distribution - TVAE Synthetic Data', fontsize=14)
        axes[1].set_xlabel(feature1)
        fig.suptitle(f'Comparison of 2D Joint Distribution: {feature1} vs. {feature2}', fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_distribution_comparison.png'))
        plt.close()
        print("Saved joint distribution comparison plot.")
    print("\nVisual diagnostics complete.")


def main():
    REAL_DATA_FILE = 'smartattackdata_original_foldcauchy.csv'
    SYNTHETIC_DATA_FILE = 'synthetic_data_tvae.csv'
    VISUALIZATION_DIR = 'tvae_evaluation_report'

    all_columns, features_to_use = define_columns()

    real_benign_data, real_data_full = load_and_prepare_data(REAL_DATA_FILE, all_columns, features_to_use)

    synthetic_data_tvae = train_tvae_and_generate(real_benign_data, SYNTHETIC_DATA_FILE)

    evaluate_statistical_fidelity(real_benign_data, synthetic_data_tvae)

    evaluate_ml_utility(real_data_full, synthetic_data_tvae, features_to_use)

    perform_visual_diagnostics(real_benign_data, synthetic_data_tvae, VISUALIZATION_DIR)

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()