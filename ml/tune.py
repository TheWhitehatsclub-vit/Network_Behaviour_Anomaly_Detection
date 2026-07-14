from sklearn.ensemble import IsolationForest
import pandas as pd

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

normal = pd.read_csv("Regular_Network_Traffic_Standardized.csv", usecols=FEATURES).dropna()

attack_files = {
    "Aggressive Scan": "Aggressive_Scan_Standardized.csv",
    "Full Port Scan": "Full_Port_Scan_Standardized.csv",
    "HTTP Burst": "HTTP_Burst_Standardized.csv",
    "Large Download": "Large_Download_Standardized.csv"
}

attacks = {name: pd.read_csv(file, usecols=FEATURES).dropna() for name, file in attack_files.items()}

contamination_values = [round(0.02 + i * 0.005, 3) for i in range(33)]

fp_sample_size = 2000
normal_sample = normal.sample(fp_sample_size, random_state=42)

results = []

for c in contamination_values:
    model = IsolationForest(contamination=c, random_state=42)
    model.fit(normal)

    normal_preds = model.predict(normal_sample)
    fp_count = (normal_preds == -1).sum()
    fp_rate = fp_count / fp_sample_size * 100

    row = {"contamination": c, "fp_rate": fp_rate}
    detection_total = 0
    for name, df in attacks.items():
        preds = model.predict(df)
        detected = (preds == -1).sum()
        rate = detected / len(df) * 100
        row[name] = rate
        detection_total += rate
    row["avg_detection"] = detection_total / len(attacks)

    results.append(row)
    print(f"contamination={c:.3f}  FP={fp_rate:5.2f}%  avg_detection={row['avg_detection']:5.1f}%")

print()
print(f"{'contam':>7} {'FP%':>6} {'AggrScan%':>10} {'FullPort%':>10} {'HTTPBurst%':>11} {'LargeDL%':>9} {'AvgDet%':>8}")
print("-" * 68)
for r in results:
    print(f"{r['contamination']:>7.3f} {r['fp_rate']:>6.2f} "
          f"{r['Aggressive Scan']:>10.1f} {r['Full Port Scan']:>10.1f} "
          f"{r['HTTP Burst']:>11.1f} {r['Large Download']:>9.1f} {r['avg_detection']:>8.1f}")