import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("train")

logger.info("🚀 Starting training script...")

# --- Load Dataset ---
data_path = os.path.join("data", "telco_churn.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
logger.info(f"✅ Loaded dataset from {data_path} with shape {df.shape}")

# --- Preprocessing ---
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Split into features and label
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode categorical features
X = pd.get_dummies(X)
logger.info(f"🧹 After one-hot encoding: {X.shape[1]} features")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("random_state", 42)
    
    # --- Model Training ---
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    logger.info("✅ Model training complete.")

    # --- Evaluation ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"📊 Accuracy: {acc:.4f}")
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    
    # --- Save Model ---
    clf.feature_names_in_ = X_train.columns  # Attach input feature names
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "churn_model.pkl")
    joblib.dump(clf, model_path)
    logger.info(f"💾 Model saved to {model_path}")

    # Log model to MLflow
    mlflow.sklearn.log_model(
        clf, 
        artifact_path="s3://mlflow-churn-artifacts", 
        registered_model_name="churn_rf_model"
    )

logger.info("🏁 Training complete. MLflow logs stored in S3.")
