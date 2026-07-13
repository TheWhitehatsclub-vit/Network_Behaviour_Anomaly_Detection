from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

full_normal = pd.read_csv("Regular_Network_Traffic_Standardized.csv", usecols=FEATURES)
full_normal = full_normal.dropna()

holdout = full_normal.sample(frac=0.2, random_state=42)
train_data = full_normal.drop(holdout.index)

holdout.to_csv("normal_holdout.csv", index=False)
print(f"Training rows: {len(train_data)}  |  Held-out test rows: {len(holdout)}")

model = IsolationForest(contamination=0.17, random_state=42)
model.fit(train_data[FEATURES])
print("Training done.")

attack_files = [
    "Aggressive_Scan_Standardized.csv",
    "Full_Port_Scan_Standardized.csv",
    "HTTP_Burst_Standardized.csv",
    "Large_Download_Standardized.csv"
]

for filename in attack_files:
    attack = pd.read_csv(filename, usecols=FEATURES)
    attack = attack.dropna()
    predictions = model.predict(attack[FEATURES])
    anomaly_count = (predictions == -1).sum()
    print(f"{filename}: {anomaly_count}/{len(attack)} flagged as ANOMALY")

holdout_preds = model.predict(holdout[FEATURES])
holdout_correct = (holdout_preds == 1).sum()
print(f"\nHeld-out normal traffic check: {holdout_correct}/{len(holdout)} correctly identified as Normal")

joblib.dump(model, "model.pkl")
print("Model saved.")

feature_means = train_data[FEATURES].mean().to_dict()
joblib.dump(feature_means, "feature_means.pkl")