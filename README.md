# 🚀 Fraud Detection ML API (Production-Ready)

A production-style machine learning system for credit card fraud detection, built using XGBoost and deployed as a FastAPI service with monitoring and Docker.

This project demonstrates the complete ML lifecycle:

Data → Model Training → Threshold Tuning → API Serving → Logging → Drift Monitoring → Containerization → Cloud Deployment

---

## 🔗 Live Demo

Swagger UI:  
👉 https://fraud-ml-engineering.onrender.com/docs

---

## 🧠 Problem

Credit card fraud detection is a highly imbalanced classification problem (~0.17% fraud cases).

Accuracy alone is misleading.  
This project focuses on:

- Precision  
- Recall  
- ROC-AUC  
- Threshold tuning for fraud sensitivity  

---

## ⚙️ Features

### Model Training
- XGBoost classifier  
- Imbalance handling using `scale_pos_weight`  
- Stratified train/test split  
- Precision, Recall, ROC-AUC evaluation  
- Threshold tuning for decision control  
- Model artifact persistence using `joblib`  

### API Layer
- FastAPI inference service  
- `/predict` endpoint  
- Input validation using Pydantic  
- Returns fraud probability + threshold-based label  

### Structured Logging
Each prediction generates structured logs:

```json
{
  "ts": "2026-02-22T19:13:05.605249+00:00",
  "fraud_probability": 0.00137,
  "threshold": 0.4,
  "label": 0
}
```

Logs are streamed to cloud logs (Render) to simulate production observability.

### Monitoring & Drift Detection
A monitoring script compares live prediction distribution against the training baseline:

- Mean probability  
- 95th percentile probability  
- Relative change detection  
- Drift alert warnings  

---

## 🏗 Architecture

```
User Request
      ↓
FastAPI (/predict)
      ↓
XGBoost Model
      ↓
Fraud Probability
      ↓
Threshold Decision
      ↓
Structured Logging
      ↓
Monitoring Script (Drift Checks)
```

---

## 📂 Project Structure

```
fraud-ml-engineering/
│
├── app/                # FastAPI service
├── src/                # Training & monitoring scripts
├── artifacts/          # Model + threshold + baseline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
uvicorn app.main:app --reload
```

Visit:
http://127.0.0.1:8000/docs

---

## 🐳 Docker

Build image:

```bash
docker build -t fraud-ml-api:1.0 .
```

Run container:

```bash
docker run -p 8000:8000 fraud-ml-api:1.0
```

---

## 🛠 Tech Stack

- Python 3.11  
- XGBoost  
- Scikit-learn  
- FastAPI  
- Docker  
- Render  

---

## 🎯 Key Focus

This project emphasizes productionizing ML systems, not just training models in notebooks.

It demonstrates:

- Handling imbalanced data  
- Threshold control  
- Structured logging  
- Baseline drift monitoring  
- Containerized deployment  