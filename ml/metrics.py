from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

model = joblib.load("model.pkl")

normal_sample = pd.read_csv("normal_holdout.csv", usecols=FEATURES).dropna()

attack_files = {
    "Aggressive Scan": "Aggressive_Scan_Standardized.csv",
    "Full Port Scan": "Full_Port_Scan_Standardized.csv",
    "HTTP Burst": "HTTP_Burst_Standardized.csv",
    "Large Download": "Large_Download_Standardized.csv"
}

all_true = []
all_pred = []

normal_preds = model.predict(normal_sample[FEATURES])
all_true += [1] * len(normal_sample)
all_pred += list(normal_preds)

for name, file in attack_files.items():
    df = pd.read_csv(file, usecols=FEATURES).dropna()
    preds = model.predict(df[FEATURES])
    all_true += [-1] * len(df)
    all_pred += list(preds)
    detected = (preds == -1).sum()
    print(f"{name}: {detected}/{len(df)} detected ({detected/len(df)*100:.1f}%)")

print()
print(classification_report(all_true, all_pred, target_names=["Anomaly", "Normal"]))
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_pred))