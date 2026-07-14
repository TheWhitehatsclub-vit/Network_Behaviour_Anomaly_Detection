import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

model = joblib.load("model.pkl")

normal = pd.read_csv("Regular_Network_Traffic_Standardized.csv", usecols=FEATURES).dropna()
attack_files = [
    "Aggressive_Scan_Standardized.csv",
    "Full_Port_Scan_Standardized.csv",
    "HTTP_Burst_Standardized.csv",
    "Large_Download_Standardized.csv"
]

normal_scores = model.decision_function(normal[FEATURES])
print("Normal traffic scores:")
print(f"  min={normal_scores.min():.4f}  max={normal_scores.max():.4f}  mean={normal_scores.mean():.4f}")

for filename in attack_files:
    attack = pd.read_csv(filename, usecols=FEATURES).dropna()
    scores = model.decision_function(attack[FEATURES])
    print(f"\n{filename} scores:")
    print(f"  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")