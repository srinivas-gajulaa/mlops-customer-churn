from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging
import boto3
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("churn_api")

# --- FastAPI App ---
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to Churn Prediction API!"}

# --- Download Model from S3 ---
BUCKET_NAME = "mlflow-churn-artifacts"
S3_KEY = "model.pkl"  # or full S3 path if under subfolder
LOCAL_MODEL_PATH = "models/churn_model.pkl"

if not os.path.exists(LOCAL_MODEL_PATH):
    logger.info("Model not found locally. Downloading from S3...")
    os.makedirs("models", exist_ok=True)

    s3 = boto3.client("s3")
    try:
        s3.download_file(BUCKET_NAME, S3_KEY, LOCAL_MODEL_PATH)
        logger.info("✅ Model downloaded from S3.")
    except Exception as e:
        logger.error(f"❌ Failed to download model from S3: {e}")
        raise FileNotFoundError(f"Model could not be downloaded: {e}")

# --- Load Trained Model ---
model = joblib.load(LOCAL_MODEL_PATH)
logger.info(f"Model loaded from {LOCAL_MODEL_PATH}")

# --- Input Schema ---
class InputData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.5
            }
        }

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        df = pd.get_dummies(df)

        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0

        df = df[model.feature_names_in_]
        logger.info(f"Input columns: {df.columns.tolist()}")

        prediction = model.predict(df)
        logger.info(f"Prediction: {prediction[0]}")

        return {"churn_prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
