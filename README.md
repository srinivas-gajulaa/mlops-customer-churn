# 🔁 MLOps-Powered Customer Churn Prediction

A production-ready MLOps pipeline that predicts customer churn using machine learning (XGBoost), with full experiment tracking, reproducibility, and model management using MLflow.

---

## 🚀 Features

* Customer churn prediction using XGBoost & Logistic Regression
* Experiment tracking and versioning via MLflow
* Modular pipeline for preprocessing, training, evaluation
* MLflow UI to visualize metrics, parameters, and artifacts
* Clean project structure following MLOps principles

---

## 📦 Tech Stack

* Python 3.12
* scikit-learn + pandas
* XGBoost
* MLflow
* Jupyter Notebook
* (Optional): FastAPI, Docker, GitHub Actions

---

## 📁 Project Structure

```
mlops-customer-churn/
├── data/                  # Raw and processed data
├── notebooks/             # EDA and model experiments
│   └── train_model.ipynb
├── src/                  # Python modules for pipeline steps
│   ├── data_prep.py      # Data cleaning, encoding, splitting
│   ├── model.py          # Model training and evaluation
│   └── utils.py          # Helper functions
├── models/               # Saved models (pickle or MLflow artifacts)
├── mlruns/               # MLflow tracking data
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧪 Sample Run Flow

```bash
# Create virtual environment and install requirements
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run training pipeline from notebook or script
jupyter notebook notebooks/train_model.ipynb

# Start MLflow tracking UI
mlflow ui
```

Then visit: [http://localhost:5000](http://localhost:5000) to explore runs, metrics, models.

---

## 🛠️ Setup Instructions

```bash
# Clone repo and setup venv
git clone https://github.com/srinivas-gajulaa/mlops-customer-churn.git
cd mlops-customer-churn
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (Optional) Run API if deployed with FastAPI
uvicorn src.api:app --reload
```

---

## 🎤 1-Minute Interview Demo Script

"This is a customer churn prediction pipeline built using core MLOps concepts. I trained models using XGBoost and Logistic Regression, and integrated MLflow for tracking experiments, parameters, and metrics."

"The project includes data preprocessing, model training, evaluation, and artifact logging — all in a modular way so that the code is reusable and easy to maintain. MLflow's UI helps track performance over time and manage multiple runs."

"This project shows my ability to structure real-world ML pipelines using MLOps tools, and to build scalable workflows for deployment and monitoring."

---

## 📬 Contact

**Author:** Sreeni Gajula
**LinkedIn:** [linkedin.com/in/srinivas-gajula](https://linkedin.com/in/srinivas-gajula)

---

## ✅ Pro Tip

Yes, absolutely put this project on GitHub!

* It proves hands-on MLOps understanding
* Demonstrates MLflow integration and experiment reproducibility
* Highlights ability to productionize and track models like in real teams

Let me know if you'd like to containerize this or set up a CI/CD flow next!
