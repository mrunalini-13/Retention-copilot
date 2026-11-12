
from pathlib import Path
import json
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "models" / "artifacts"

BINARY_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

DUMMY_COLUMNS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
]

CONTRACT_MAPPING = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}


def preprocess_raw_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and encode raw telecom customer data."""
    df = data.copy()

    # Remove customer identifier from model inputs.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean text fields.
    text_columns = df.select_dtypes(include=["object", "str"]).columns
    for col in text_columns:
        df[col] = df[col].str.strip()

    # Convert charges to numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # Remove rows that cannot be used because TotalCharges is missing.
    if "TotalCharges" in df.columns:
        df = df.dropna(subset=["TotalCharges"]).copy()

    # Encode binary columns.
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map({
                    "Yes": 1,
                    "No": 0,
                    "1": 1,
                    "0": 0,
                    "True": 1,
                    "False": 0,
                })
                .fillna(0)
                .astype(int)
            )

    # Encode target when present.
    if "Churn" in df.columns:
        df["Churn"] = (
            df["Churn"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0})
        )

    # Encode contract type.
    if "Contract" in df.columns:
        df["Contract"] = (
            df["Contract"]
            .map(CONTRACT_MAPPING)
            .fillna(0)
            .astype(int)
        )

    # One-hot encode categorical features.
    existing = [c for c in DUMMY_COLUMNS if c in df.columns]
    df = pd.get_dummies(
        df,
        columns=existing,
        drop_first=False,
        dtype=int,
    )

    # Load expected feature order.
    feature_path = ARTIFACTS_DIR / "feature_columns.json"
    with open(feature_path, encoding="utf-8") as f:
        feature_columns = json.load(f)

    # Ensure all expected model input columns exist.
    for col in feature_columns:
        if col not in df.columns and col != "Churn":
            df[col] = 0

    # Remove unexpected columns and align order.
    if "Churn" in df.columns:
        return df[feature_columns + ["Churn"]]

    return df[feature_columns]


def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    """Scale model input features using the saved training scaler."""
    df = data.copy()

    scaler_path = ARTIFACTS_DIR / "standard_scaler.joblib"
    scaler = joblib.load(scaler_path)

    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    return df
