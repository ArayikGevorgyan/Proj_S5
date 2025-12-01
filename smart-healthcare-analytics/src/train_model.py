import os
import pandas as pd
from src.ml_model import train_models, train_multi_disease_models

# ---------------- CONFIG ----------------
DATA_DIR = "data/"
DEFAULT_DATA = os.path.join(DATA_DIR, "processed_data.csv")

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically encode categorical columns into numeric codes."""
    return df.apply(lambda col: col.astype('category').cat.codes if col.dtypes == 'object' else col)


def single_disease_training():
    """Train models on a single dataset (default)."""
    if not os.path.exists(DEFAULT_DATA):
        print(f"❌ Dataset not found at {DEFAULT_DATA}")
        return

    df = pd.read_csv(DEFAULT_DATA)
    df = encode_categoricals(df)

    print(f"📂 Loaded dataset ({len(df)} rows, {len(df.columns)} columns)")
    print("\n🚀 Training models...")

    results, best_model = train_models(df, target_column="Disease")

    print("\n✅ Training complete. Best model saved to models/model_bundle.pkl")
    print("\n📊 Model Performance Summary:")
    for name, metrics in results.items():
        print(f"• {name:<20} | Accuracy: {metrics['accuracy']*100:.2f}% | AUC: {metrics['roc_auc']:.2f}")
    print("\nROC curves saved to docs/plots/")


def multi_disease_training():
    """Train models for multiple diseases if datasets are available."""
    datasets = {}
    possible = ["diabetes", "heart", "stroke"]
    for name in possible:
        path = os.path.join(DATA_DIR, f"{name}_data.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = encode_categoricals(df)
            datasets[name] = df

    if not datasets:
        print("⚠️  No multi-disease datasets found. Using single-disease mode instead.\n")
        single_disease_training()
        return

    print("🧠 Multi-Disease Training Mode Enabled")
    print(f"📊 Datasets found: {', '.join(datasets.keys())}")
    print("\n🚀 Training models for each disease...")

    results = train_multi_disease_models(datasets)

    print("\n✅ Multi-Disease Training Complete.")
    for disease, acc in results.items():
        print(f"• {disease.title():<10} → Accuracy: {acc*100:.2f}%")
    print("\nAll models saved under /models/ directory.")


def main():
    """Main entry: choose mode automatically."""
    print("🔧 Smart Healthcare Analytics — Model Trainer\n")
    if any(os.path.exists(os.path.join(DATA_DIR, f"{d}_data.csv")) for d in ["diabetes", "heart", "stroke"]):
        multi_disease_training()
    else:
        single_disease_training()


if __name__ == "__main__":
    main()
