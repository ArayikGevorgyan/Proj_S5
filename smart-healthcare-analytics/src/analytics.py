import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .patient_database import get_prediction_history


PLOTS_DIR = "docs/plots"


def _ensure_plots_dir() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)


def get_patient_history(patient_id: str) -> pd.DataFrame:
    """Fetch chronological prediction history for a patient."""
    if not patient_id:
        return pd.DataFrame()
    return get_prediction_history(patient_id)


def plot_risk_trend(patient_logs: pd.DataFrame,
                    patient_id: str,
                    patient_name: Optional[str] = None) -> Optional[str]:
    """Plot risk score trend for a patient and return image path."""
    if patient_logs is None or patient_logs.empty:
        return None

    _ensure_plots_dir()
    logs = patient_logs.copy()
    logs["Timestamp"] = pd.to_datetime(logs["Timestamp"])
    logs.sort_values("Timestamp", inplace=True)

    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=logs,
        x="Timestamp",
        y="RiskScore",
        hue="DiseaseType",
        marker="o"
    )
    title = f"Risk Trend — {patient_name or patient_id}"
    plt.title(title)
    plt.ylabel("Risk Score")
    plt.xlabel("Timestamp")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    filename = f"risk_trend_{patient_id}.png"
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path


def correlation_over_time(df: pd.DataFrame,
                          upto_records: Optional[int] = None,
                          output_name: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    """Compute and plot correlation matrix for the first N records."""
    if df is None or df.empty:
        return pd.DataFrame(), None

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame(), None

    upto = upto_records or len(numeric_df)
    upto = max(2, min(len(numeric_df), upto))
    subset = numeric_df.iloc[:upto]
    corr = subset.corr()

    _ensure_plots_dir()
    filename = output_name or f"correlation_upto_{upto}.png"
    path = os.path.join(PLOTS_DIR, filename)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap (first {upto} records)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return corr, path


def cluster_patients(df: pd.DataFrame,
                     n_clusters: int = 3,
                     output_name: Optional[str] = None) -> dict:
    """Run K-Means clustering and generate PCA scatter plot."""
    if df is None or df.empty:
        return {}

    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.drop(columns=["Disease"], errors="ignore")
    if numeric_df.shape[1] < 2:
        return {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    comp_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    comp_df["Cluster"] = labels

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_df.columns)

    _ensure_plots_dir()
    filename = output_name or f"cluster_scatter_{n_clusters}.png"
    path = os.path.join(PLOTS_DIR, filename)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=comp_df, x="PC1", y="PC2", hue="Cluster", palette="viridis", s=60)
    plt.title(f"K-Means Clusters (k={n_clusters})")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    summary = pd.Series(labels).value_counts().reset_index()
    summary.columns = ["Cluster", "Count"]

    return {
        "labels": labels,
        "plot_path": path,
        "summary": summary,
        "centroids": centroids
    }
