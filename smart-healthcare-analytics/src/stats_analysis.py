import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

def descriptive_statistics(df):
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns

    desc = {}
    desc["mean"] = numeric_df.mean()
    desc["median"] = numeric_df.median()
    desc["std"] = numeric_df.std()
    desc["variance"] = numeric_df.var()
    desc["min"] = numeric_df.min()
    desc["max"] = numeric_df.max()

    return desc

def correlation_matrix(df: pd.DataFrame, output_path="docs/plots/correlation_heatmap.png"):
    """Plot and save correlation heatmap."""
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return corr

def distribution_plots(df: pd.DataFrame, columns, output_dir="docs/plots/"):
    """Generate histograms and boxplots for selected columns."""
    os.makedirs(output_dir, exist_ok=True)
    for col in columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{output_dir}{col}_hist.png")
        plt.close()

        plt.figure(figsize=(4,4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{output_dir}{col}_box.png")
        plt.close()

def hypothesis_tests(df: pd.DataFrame):
    """Example: compare mean glucose between male and female patients."""
    if "Gender" in df.columns and "Glucose" in df.columns:
        males = df[df["Gender"] == "Male"]["Glucose"]
        females = df[df["Gender"] == "Female"]["Glucose"]
        t_stat, p_val = stats.ttest_ind(males, females, equal_var=False)
        return {"t_statistic": round(t_stat,2), "p_value": round(p_val,4)}
    return {}
