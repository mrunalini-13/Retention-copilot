
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from utils import preprocess_raw_data, scale_features
from mangum import Mangum

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "models" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "Churn.csv"

THRESHOLD = 0.4

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Retention Copilot API",
    description="Predict telecom customer churn risk.",
    version="1.0.0",
)
handler = Mangum(app)

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def predict_customers(customers: list[dict]) -> list[dict]:
    """Preprocess raw customer records and predict churn risk."""
    if not customers:
        raise ValueError("Provide at least one customer.")

    raw_df = pd.DataFrame(customers)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in raw_df.columns
    ]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Do not silently lose rows during preprocessing.
    total_charges = pd.to_numeric(
        raw_df["TotalCharges"], errors="coerce"
    )
    if total_charges.isna().any():
        raise ValueError(
            "TotalCharges must be a valid number for every customer."
        )

    customer_ids = (
        raw_df["customerID"].tolist()
        if "customerID" in raw_df.columns
        else [None] * len(raw_df)
    )

    processed = preprocess_raw_data(raw_df)

    if "Churn" in processed.columns:
        processed = processed.drop(columns=["Churn"])

    scaled = scale_features(processed)

    probabilities = model.predict_proba(scaled)[:, 1]

    results = []
    for customer_id, probability in zip(
        customer_ids, probabilities
    ):
        predicted_churn = int(probability >= THRESHOLD)

        results.append({
            "customerID": customer_id,
            "churn_probability": round(float(probability), 6),
            "churn_risk_percentage": round(
                float(probability) * 100, 2
            ),
            "predicted_churn": predicted_churn,
            "risk_category": (
                "At Risk" if predicted_churn else "Low Risk"
            ),
        })

    return results


@app.get("/")
def home():
    return {
        "message": "Retention Copilot API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(payload: dict | list[dict]):
    try:
        customers = payload if isinstance(payload, list) else [payload]
        return {"predictions": predict_customers(customers)}
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc


@app.get("/customers/{customer_id}/predict")
def predict_existing_customer(customer_id: str):
    """Find a customer in the raw dataset and predict their churn risk."""
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500, detail="Customer dataset not found."
        )

    customers = pd.read_csv(DATA_PATH, dtype={"customerID": str})
    match = customers.loc[customers["customerID"] == customer_id]

    if match.empty:
        raise HTTPException(
            status_code=404, detail="Customer ID not found."
        )

    customer = match.iloc[0].to_dict()

    try:
        return {
            "prediction": predict_customers([customer])[0]
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422, detail=str(exc)
        ) from exc
