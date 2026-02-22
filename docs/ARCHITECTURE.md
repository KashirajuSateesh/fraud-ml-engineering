# 🧠 Fraud Detection ML System Architecture

This document explains how the fraud detection system works end-to-end.

---

## 🔄 High-Level Flow

```
Credit Card Dataset (creditcard.csv)
        ↓
Training Script (src/train.py)
        ↓
Saved Artifacts (artifacts/)
  - xgb_model.joblib
  - threshold.joblib
  - baseline.json
        ↓
FastAPI Service (app/main.py)
        ↓
User Request (POST /predict)
        ↓
Prediction Pipeline
  - Validate JSON (Pydantic)
  - Convert to DataFrame
  - model.predict_proba()
  - Apply threshold → label
        ↓
Structured Logging (stdout + jsonl)
        ↓
Monitoring Script (src/monitoring.py)
  - Compare live vs baseline
  - Drift alerts
        ↓
Docker Container
        ↓
Render Deployment
        ↓
Public API URL
```

---

## 🟢 1) Offline Training Layer

- Load dataset and separate `X` (features) and `y` (Class)
- Split into train/test using stratification (keeps fraud ratio consistent)
- Train XGBoost and handle class imbalance (`scale_pos_weight`)
- Evaluate Precision / Recall / ROC-AUC
- Tune decision threshold
- Save model + threshold + baseline distribution stats

---

## 🔵 2) Inference Layer (Production API)

On startup:
- Loads the saved model, threshold, and baseline

On `/predict`:
- Validates incoming JSON
- Converts features into a 1-row DataFrame
- Returns:
  - `fraud_probability`
  - `threshold`
  - `label` (0 or 1)

---

## 🟣 3) Monitoring Layer

- Reads logged predictions
- Computes live probability distribution (mean + p95)
- Compares against training baseline
- Prints drift warnings if distribution shifts significantly

---

## 🐳 4) Deployment Layer

- Docker packages the API + dependencies + artifacts
- Render builds and runs the container
