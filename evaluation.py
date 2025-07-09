import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp
import os


def run_evaluation(real_data_path, synthetic_data_path, report_dir="tabDDPM_evaluation_report"):
    """
    Runs a comprehensive evaluation suite comparing real and synthetic data, including
    robust data cleaning and visualization.
    """
    print("--- Starting Synthetic Data Evaluation Suite ---")

    if not os.path.exists(real_data_path):
        print(f"Error: Real data file not found at {real_data_path}")
        return
    if not os.path.exists(synthetic_data_path):
        print(f"Error: Synthetic data file not found at {synthetic_data_path}")
        return

    try:
        real_df_full = pd.read_excel(real_data_path, engine='openpyxl')
    except Exception:
        real_df_full = pd.read_csv(real_data_path)

    real_df = real_df_full[real_df_full['Label_code'] == 0].copy()
    real_df = real_df.drop(columns=['Unnamed: 0', 'Label_Desc', 'filename', 'Label_code'], errors='ignore')

    synthetic_df = pd.read_csv(synthetic_data_path)

    print("Initial real data loaded. Applying cleaning logic...")
    numeric_cols = real_df.select_dtypes(include=np.number).columns
    if real_df[numeric_cols].isnull().any().any() or np.isinf(real_df[numeric_cols]).any().any():
        print("Real dataset contains NaN or Infinity values. Performing automatic cleaning...")
        real_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in numeric_cols:
            if real_df[col].isnull().any():
                median_val = real_df[col].median()
                real_df[col].fillna(median_val, inplace=True)
        print("Cleaning complete.")

    synthetic_df = synthetic_df[real_df.columns]

    print(f"Loaded and cleaned {len(real_df)} real samples and {len(synthetic_df)} synthetic samples.")

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    print(f"Saving evaluation results to '{report_dir}/'")

    print("\n1. Comparing column distributions (Kolmogorov-Smirnov Test)...")
    ks_results = {}
    for col in real_df.columns:
        stat, p_value = ks_2samp(real_df[col], synthetic_df[col])
        ks_results[col] = p_value

    ks_df = pd.DataFrame(list(ks_results.items()), columns=['Feature', 'P-Value'])
    ks_df = ks_df.sort_values(by='P-Value', ascending=False)
    ks_df.to_csv(f"{report_dir}/ks_test_results.csv", index=False)
    print("  - K-S test results saved.")

    print("2. Generating visual distribution comparison plots...")
    num_features = len(real_df.columns)
    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(real_df.columns):
        ax = axes[i]
        if real_df[col].nunique() > 1:
            sns.kdeplot(real_df[col], ax=ax, label='Real', fill=True, warn_singular=False)
            sns.kdeplot(synthetic_df[col], ax=ax, label='Synthetic', fill=True, warn_singular=False)
        else:
            ax.hist(real_df[col], label='Real', color='blue', alpha=0.6, bins=1)
            ax.hist(synthetic_df[col], label='Synthetic', color='orange', alpha=0.6, bins=10)
            ax.set_title(f"{col} (Constant Value)")

        if real_df[col].nunique() > 1:
            ax.set_title(col)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{report_dir}/distribution_comparison.png")
    plt.close()
    print("  - Distribution comparison plot saved.")

    print("3. Comparing correlation matrices...")
    real_corr = real_df.corr()
    synthetic_corr = synthetic_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(real_corr, ax=axes[0], cmap='viridis').set_title('Real Data Correlation')
    sns.heatmap(synthetic_corr, ax=axes[1], cmap='viridis').set_title('Synthetic Data Correlation')
    plt.tight_layout()
    plt.savefig(f"{report_dir}/correlation_comparison.png")
    plt.close()
    print("  - Correlation heatmap saved.")

    print("4. Comparing data in 2D PCA space...")
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_df)
    synthetic_pca = pca.transform(synthetic_df)

    plt.figure(figsize=(10, 7))
    plt.scatter(real_pca[:, 0], real_pca[:, 1], s=10, alpha=0.5, label='Real Data')
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], s=10, alpha=0.5, label='Synthetic Data')
    plt.title('PCA Projection of Real vs. Synthetic Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_dir}/pca_comparison.png")
    plt.close()
    print("  - PCA comparison plot saved.")

    print("5. Evaluating ML utility (Train-Synthetic, Test-Real)...")
    detector = IsolationForest(contamination='auto', random_state=42)
    detector.fit(synthetic_df)
    real_data_predictions = detector.predict(real_df)
    accuracy = (real_data_predictions == 1).mean() * 100

    report_text = f"""
    Machine Learning Utility Report (Train-Synthetic, Test-Real)
    ============================================================
    Model: Isolation Forest (Anomaly Detector)

    1. The model was trained ONLY on the synthetic data.
    2. The model was then tested on the REAL benign data.

    Result:
    -------
    - Accuracy: {accuracy:.2f}% of the REAL samples were correctly identified as 'normal' by the model.

    Interpretation:
    - A high accuracy (>90-95%) suggests that the synthetic data is a good substitute for the real data,
      as the model learned the underlying patterns of normal behavior from it.
    - A low accuracy suggests the synthetic data's distribution is significantly different from the real data.
    """

    with open(f"{report_dir}/ml_utility_report.txt", "w") as f:
        f.write(report_text)

    print(f"  - ML utility report saved. Accuracy: {accuracy:.2f}%")

    print("\n--- Evaluation Complete ---")


if __name__ == '__main__':
    REAL_DATA_FILE = './data/smartattackdata_original_foldcauchy.csv'
    SYNTHETIC_DATA_FILE = './tabddpm_output/synthetic_data_tabddpm.csv'

    run_evaluation(REAL_DATA_FILE, SYNTHETIC_DATA_FILE)