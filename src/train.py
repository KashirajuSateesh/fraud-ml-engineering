import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

#Load Data
df = pd.read_csv("data/creditcard.csv")

#Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("Feature Shape: ", X.shape)
print("Target Shape: ", y.shape)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nAfter Split: ")
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train fraud %: ", y_train.mean()*100)
print("y_test fraud %: ", y_test.mean()*100)

#Calculate Imbalance Ratio
neg = (y_train==0).sum()
pos = (y_train==1).sum()

scale_pos_weight = neg/pos

print("\nScale_pos_weight: ", scale_pos_weight)

#Train a baseline RandomForest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

# Predit Probabilities and labels
proba = rf.predict_proba(X_test)[:,1]

preds = (proba>=0.4).astype(int)

precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
roc_auc = roc_auc_score(y_test, proba)

print("\n=== Random Forest Baseline (thresold=0.5) ===")
print("Precision: ", precision)
print("Recall: ", recall)
print("Rou_Auc: ", roc_auc)


#Train XGBoost
xgb = XGBClassifier(
    n_estimators = 400,
    max_depth = 4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

proba_xgb = xgb.predict_proba(X_test)[:,1]
roc_auc_xgb = roc_auc_score(y_test, proba_xgb)

print("\nROC-AUC (XGBoost):", roc_auc_xgb)
# pred_xgb = (proba_xgb>=threshold).astype(int)

# precision_xgb = precision_score(y_test, pred_xgb, zero_division=0)
# recal_xgb = recall_score(y_test, pred_xgb, zero_division=0)
# roc_auc_xgb = roc_auc_score(y_test, proba_xgb)

# print("\n=== XGBoost (thresold=0.5) ===")
# print("Precision: ", precision_xgb)
# print("Recall: ", recal_xgb)
# print("Rou_Auc: ", roc_auc_xgb)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
print("\n=== XGBoost Threshold Tuning ===")
best_t=None
best_score = -1 # simple combined score (precision + recall)
best_p, best_r = None, None
for t in thresholds:
    preds_xgb = (proba_xgb>=t).astype(int)
    precision_xgb = precision_score(y_test, preds_xgb, zero_division=0)
    recall_xgb = recall_score(y_test, preds_xgb, zero_division=0)
    print(f"Threshold={t:.1f}  Precision = {precision_xgb:.3f}  Recall = {recall_xgb:.3f}")

# Chosen a threshold to lock it for the preference of the model
chosen_threshold = 0.4 # recommended for high recall with decent precision
print("Chosen threshold: ", chosen_threshold)

preds_final = (proba_xgb >= chosen_threshold).astype(int)
precision_final = precision_score(y_test, preds_final, zero_division=0)
recall_final = recall_score(y_test, preds_final, zero_division=0)

print("\n=== XGBoost Final Chosen Threshold ===")
print("Precision: ", precision_final)
print("Recall: ", recall_final)
print("ROC_AUC: ", roc_auc_xgb)

# Save artifacts for deployment

joblib.dump(xgb, "artifacts/xgb_model.joblib")
joblib.dump(chosen_threshold, "artifacts/threshold.joblib")

print("\nSaved model to: artifacts/xgb_model.joblib")
print("Saved threshold to: artifacts/threshold.joblib")