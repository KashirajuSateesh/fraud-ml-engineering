import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "artifacts/xgb_model.joblib"
BASELINE_PATH = "artifacts/baseline.json"

df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(MODEL_PATH)

proba = model.predict_proba(X_test)[:, 1]

baseline = {
    "mean_probability": float(proba.mean()),
    "p95_probability": float(np.percentile(proba, 95)),
    "count": int(len(proba))
}

with open(BASELINE_PATH, "w") as f:
    json.dump(baseline, f, indent=2)

print("Saved baseline to: ", BASELINE_PATH)
print(baseline)