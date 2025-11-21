import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# File paths anchored to project root to stay robust across working dirs
BASE_DIR = Path(__file__).resolve().parents[1]


def project_path(*parts) -> Path:
    return BASE_DIR.joinpath(*parts)


BUNDLE_PATH = project_path("models", "model_bundle.pkl")
MODELS_DIR = project_path("models")
PLOTS_DIR = project_path("docs", "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Utility helpers
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to categorical codes for model training."""
    return df.apply(
        lambda col: col.astype("category").cat.codes if col.dtypes == "object" else col
    )


def _train_and_save_bundle(df: pd.DataFrame, bundle_path,
                           target_column: str = "Disease") -> float:
    """Train a RandomForest model on df and persist bundle, returning accuracy."""
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    X, y = prepare_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": list(X.columns)
    }
    joblib.dump(bundle, bundle_path)
    return float(acc)


# -----------------------------
# DATA PREPARATION
# -----------------------------
def prepare_data(df: pd.DataFrame, target_column: str):
    """Split dataset into features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# -----------------------------
# MODEL TRAINING & EVALUATION
# -----------------------------
def train_models(df: pd.DataFrame, target_column="Disease"):
    """Train multiple models, evaluate them, and save the best one."""
    df = encode_categoricals(df)
    X, y = prepare_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = {}
    best_model = None
    best_acc = 0.0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Save ROC plot
        roc_path = PLOTS_DIR / f"{name.replace(' ', '_')}_ROC.png"
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        results[name] = {
            "accuracy": round(acc, 3),
            "roc_auc": round(roc_auc, 3),
            "roc_curve": str(roc_path)
        }

        if acc > best_acc:
            best_acc = acc
            best_model = model

    # Save best model bundle
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "features": list(X.columns),
        "classes": list(y.unique())
    }
    joblib.dump(bundle, BUNDLE_PATH)

    return results, best_model


# -----------------------------
# MULTI-DISEASE TRAINING
# -----------------------------
def train_multi_disease_models(df_dict: dict):
    """
    Train models for multiple diseases (e.g., diabetes, heart, stroke).
    Example input:
    df_dict = {'diabetes': df1, 'heart': df2, 'stroke': df3}
    """
    results = {}
    for disease, df in df_dict.items():
        print(f"\nTraining model for {disease.upper()}...")
        disease_path = MODELS_DIR / f"{disease}_bundle.pkl"

        df_encoded = encode_categoricals(df)
        acc = _train_and_save_bundle(df_encoded, disease_path, target_column="Disease")
        results[disease] = round(acc, 3)

    return results


def load_multi_model(disease: str):
    """
    Load model bundle for specific disease.
    If the bundle is missing, attempt to auto-train using available datasets.
    """
    disease = disease.lower()
    path = MODELS_DIR / f"{disease}_bundle.pkl"
    if not path.exists():
        # Try disease-specific dataset (e.g., data/diabetes_data.csv), otherwise fall back.
        candidate_paths = [
            project_path("data", f"{disease}_data.csv"),
            project_path("data", "processed_data.csv")
        ]
        for data_path in candidate_paths:
            data_path = Path(data_path)
            if not data_path.exists():
                continue
            try:
                df = pd.read_csv(data_path)
                df = encode_categoricals(df)
                _train_and_save_bundle(df, path, target_column="Disease")
                break
            except Exception:
                continue

    if not path.exists():
        return None, None

    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"]


# -----------------------------
# PREDICTION
# -----------------------------
def load_model():
    """Load the default (best) model."""
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("No trained model found. Train the model first.")
    bundle = joblib.load(BUNDLE_PATH)
    return bundle["model"], bundle["scaler"]



def predict_risk(model, scaler, input_data: dict, return_probabilities: bool = False):
    """
    Predict disease risk for a single patient.
    If return_probabilities=True, include class probabilities when available.
    """
    import pandas as pd
    df = pd.DataFrame([input_data])
    X_scaled = scaler.transform(df)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        pos = float(proba[1])
        neg = float(proba[0])
    else:
        # Fall back to decision function or predict
        pred = model.predict(X_scaled)[0]
        pos = float(pred)
        neg = 1.0 - pos

    prediction = int(pos > 0.6)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    result = {
        "prediction": prediction,
        "risk_score": round(pos, 2),
        "result": "High Risk" if pos > 0.6 else "Low Risk",
        "timestamp": timestamp,
        "model_name": type(model).__name__,
        "input_data": input_data,
        "feature_order": list(input_data.keys()),
    }
    if return_probabilities:
        result["proba_0"] = round(neg, 4)
        result["proba_1"] = round(pos, 4)
    return result


def get_feature_importance(model,
                           feature_names: List[str],
                           sample_data: Optional[pd.DataFrame] = None) -> dict:
    """Return RandomForest importances and optional SHAP global values."""
    importances = []
    shap_values = None

    if hasattr(model, "feature_importances_") and feature_names:
        importances = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )

    if sample_data is not None and not sample_data.empty:
        try:
            explainer = shap.Explainer(model, sample_data)
            shap_raw = explainer(sample_data)
            shap_mean = shap_raw.abs.mean(axis=0).values
            shap_values = sorted(
                zip(feature_names, shap_mean),
                key=lambda x: x[1],
                reverse=True
            )
        except Exception:
            shap_values = None

    return {
        "model_importance": importances,
        "shap_importance": shap_values
    }



# -----------------------------
# EXPLAINABLE AI (SHAP)
# -----------------------------
def explain_prediction(model, scaler, input_data: dict,
                       output_path=os.path.join(PLOTS_DIR, "shap_explanation.png")):
    """Generate a SHAP explanation plot for one prediction."""
    df = pd.DataFrame([input_data])
    X_scaled = scaler.transform(df)
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, df, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# -----------------------------
# MAIN (for manual testing)
# -----------------------------
if __name__ == "__main__":
    from src.data_preparation import load_data, clean_data

    print("🔹 Loading data...")
    df = clean_data(load_data("data/sample_patients.csv"))

    print("🔹 Training models...")
    results, model = train_models(df, target_column="Disease")

    print("\n--- Model Results ---")
    for name, info in results.items():
        print(f"{name}: Accuracy={info['accuracy']} | AUC={info['roc_auc']}")

    print(f"\n✅ Best model saved to {BUNDLE_PATH}")
