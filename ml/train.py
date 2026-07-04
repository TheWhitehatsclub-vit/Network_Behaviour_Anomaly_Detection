from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

normal = pd.read_csv("Regular_Network_Traffic_Standardized.csv", usecols=FEATURES)
normal = normal.dropna()

model = IsolationForest(contamination=0.17, random_state=42)
model.fit(normal)
print(f"Training done. Rows used: {len(normal)}")

attack_files = [
    "Aggressive_Scan_Standardized.csv",
    "Full_Port_Scan_Standardized.csv",
    "HTTP_Burst_Standardized.csv",
    "Large_Download_Standardized.csv"
]

for filename in attack_files:
    attack = pd.read_csv(filename, usecols=FEATURES)
    attack = attack.dropna()
    predictions = model.predict(attack)
    anomaly_count = (predictions == -1).sum()
    print(f"{filename}: {anomaly_count}/{len(attack)} flagged as ANOMALY")

normal_sample = normal.sample(20, random_state=42)
normal_preds = model.predict(normal_sample)
normal_correct = (normal_preds == 1).sum()
print(f"\nNormal traffic check: {normal_correct}/20 correctly identified as Normal")

joblib.dump(model, "model.pkl")
print("Model saved.")

feature_means = normal[FEATURES].mean().to_dict()
joblib.dump(feature_means, "feature_means.pkl")