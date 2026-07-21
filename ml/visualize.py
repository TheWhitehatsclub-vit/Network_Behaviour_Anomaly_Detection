import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

model = joblib.load("model.pkl")

if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    if expected != FEATURES:
        raise ValueError(
            f"FEATURE ORDER MISMATCH!\n"
            f"Model was trained on: {expected}\n"
            f"This file is using:   {FEATURES}\n"
            f"These must match exactly, in the same order."
        )
    
normal = pd.read_csv("normal_holdout.csv", usecols=FEATURES).dropna()

attack_files = {
    "Aggressive Scan": "Aggressive_Scan_Standardized.csv",
    "Full Port Scan": "Full_Port_Scan_Standardized.csv",
    "HTTP Burst": "HTTP_Burst_Standardized.csv",
    "Large Download": "Large_Download_Standardized.csv"
}

attacks = {name: pd.read_csv(file, usecols=FEATURES).dropna() for name, file in attack_files.items()}

# ---------- Plot 1: Score distribution ----------
normal_scores = model.decision_function(normal[FEATURES])

plt.figure(figsize=(10, 6))
plt.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="steelblue", density=True)

colors = ["indianred", "darkorange", "purple", "seagreen"]
for (name, df), color in zip(attacks.items(), colors):
    scores = model.decision_function(df[FEATURES])
    plt.hist(scores, bins=50, alpha=0.5, label=name, color=color, density=True)

plt.axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary (0)")
plt.xlabel("Anomaly Score (more negative = more anomalous)")
plt.ylabel("Density")
plt.title("Anomaly Score Distribution: Normal vs Attack Traffic")
plt.legend()
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=150)
plt.close()
print("Saved score_distribution.png")

# ---------- Plot 2: Confusion matrix ----------
all_true = []
all_pred = []

normal_preds = model.predict(normal[FEATURES])
all_true += [1] * len(normal)
all_pred += list(normal_preds)

for name, df in attacks.items():
    preds = model.predict(df[FEATURES])
    all_true += [-1] * len(df)
    all_pred += list(preds)

cm = confusion_matrix(all_true, all_pred, labels=[-1, 1])

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_labels = ["Anomaly", "Normal"]
plt.xticks([0, 1], tick_labels)
plt.yticks([0, 1], tick_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")