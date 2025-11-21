import os
from datetime import datetime
from functools import reduce
from pathlib import Path

import pandas as pd
import re

# ---- Pure functional helpers ----
def to_float(value):
    """Convert string to float safely."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def validate_field(value, pattern):
    """Validate a single value using regex."""
    return bool(re.match(pattern, str(value)))

# ---- Main pipeline functions ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(filepath: str | os.PathLike) -> Path:
    """Return absolute path within the project for robust file access."""
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file into immutable Pandas DataFrame."""
    file_path = _resolve_path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    return df.copy()  # immutability: never mutate the original

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize patient data."""
    numeric_cols = ["Age", "BMI", "Glucose", "BloodPressure", "Cholesterol"]

    # Functional: apply to_float to numeric columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.map(to_float))

    # Fill missing categorical values with mode
    for col in ["Gender", "Smoking", "PhysicalActivity"]:
        if col in df.columns:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_value)

    return df.copy()

def validate_patient_record(record: dict) -> bool:
    """Validate a single patient record using regex (Automata)."""
    patterns = {
        "Age": r"^[1-9][0-9]?$",
        "Gender": r"^(Male|Female)$",
        "Glucose": r"^[0-9]{2,3}$",
        "BMI": r"^[0-9]{1,2}(\.[0-9])?$"
    }

    return all(validate_field(record.get(k, ""), v) for k, v in patterns.items())

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Functional transformations (e.g., feature engineering)."""
    # Example: Compute risk_index = weighted combination
    df["RiskIndex"] = list(
        map(lambda x: (0.3 * x["BMI"]) + (0.4 * x["Glucose"]/100) + (0.3 * x["Cholesterol"]/200),
            df.to_dict(orient="records"))
    )
    return df.copy()

def summarize_data(df: pd.DataFrame) -> dict:
    """Return summary statistics using reduce()."""
    numeric_cols = ["Age", "BMI", "Glucose", "BloodPressure", "Cholesterol"]
    stats = {}

    for col in numeric_cols:
        stats[col] = {
            "mean": round(df[col].mean(), 2),
            "min": df[col].min(),
            "max": df[col].max(),
        }

    # Use reduce to compute overall average
    overall_avg = reduce(lambda a, b: a + b, [stats[c]["mean"] for c in numeric_cols]) / len(numeric_cols)
    stats["overall_avg"] = round(overall_avg, 2)
    return stats


def clean_input_data(data):
    """Validate and clean input dictionary of patient metrics."""
    df = pd.DataFrame([data])
    numeric_cols = ["age", "bmi", "glucose", "blood_pressure", "cholesterol"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def merge_uploaded_dataset(uploaded_file,
                           base_dataset: str = "data/sample_patients.csv",
                           upload_dir: str = "data/uploads") -> pd.DataFrame:
    """
    Save an uploaded CSV and merge it into the primary dataset.
    Returns the updated dataframe used throughout the app.
    """
    upload_dir_path = _resolve_path(upload_dir)
    os.makedirs(upload_dir_path, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    saved_path = upload_dir_path / f"patients_upload_{timestamp}.csv"

    new_df = pd.read_csv(uploaded_file)
    new_df_clean = clean_data(new_df)
    new_df_clean.to_csv(saved_path, index=False)

    base_dataset_path = _resolve_path(base_dataset)
    if base_dataset_path.exists():
        base_df = load_data(base_dataset_path)
    else:
        base_df = pd.DataFrame()

    updated_df = pd.concat([base_df, new_df_clean], ignore_index=True)
    updated_df.to_csv(base_dataset_path, index=False)
    return updated_df
