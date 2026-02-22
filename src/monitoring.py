import json
import numpy as np

LOG_PATH = "logs/predictions.jsonl"
N_LAST = 1000
BASELINE_PATH = "artifacts/baseline.json"

probs = []

with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        probs.append(float(record["fraud_probability"]))

probs_np = np.array(probs, dtype=float)
probs_np = probs_np[-N_LAST:]

with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

baseline_mean = float(baseline["mean_probability"])
baseline_p95 = float(baseline["p95_probability"])

print("total_predictions_logged:", probs_np.size)
print("mean_probability:", probs_np.mean())
print("p95_probability:", np.percentile(probs_np, 95))

print("\n=== Baseline (from training test set) ===")
print("baseline_mean_probability:", baseline_mean)
print("baseline_p95_probability:", baseline_p95)

delta_mean = probs_np.mean() - baseline_mean
delta_p95 = np.percentile(probs_np, 95) - baseline_p95

relative_mean_change = abs(delta_mean) / baseline_mean if baseline_mean > 0 else 0
relative_p95_change = abs(delta_p95) / baseline_p95 if baseline_p95 > 0 else 0

print("\n=== Relative Change ===")
print("relative_mean_change:", relative_mean_change)
print("relative_p95_change:", relative_p95_change)

print("\n=== Drift Alerts ===")

if relative_mean_change > 1.0:
    print("🚨 CRITICAL: Mean probability drift > 100%")
elif relative_mean_change > 0.5:
    print("⚠️ WARNING: Mean probability drift > 50%")

if relative_p95_change > 1.0:
    print("🚨 CRITICAL: p95 probability drift > 100%")
elif relative_p95_change > 0.5:
    print("⚠️ WARNING: p95 probability drift > 50%")

print("\n=== Difference (live - baseline) ===")
print("delta_mean:", delta_mean)
print("delta_p95:", delta_p95)

print("\n=== Drift Checks ===")
if(probs_np.mean() > 0.05):
    print("⚠️ Drift warning: Mean Probability unsually high")

if np.percentile(probs_np, 95) > 0.5:
    print("⚠️ Drift Warning: Many high-risk predictions detected")