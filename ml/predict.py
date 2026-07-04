import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

model = joblib.load("model.pkl")
feature_means = joblib.load("feature_means.pkl")

def predict(traffic_row: dict):
    X = pd.DataFrame([traffic_row])[FEATURES]
    score = model.decision_function(X)[0]
    label = model.predict(X)[0]
    severity = "high" if score < -0.15 else "medium" if score < -0.05 else "low"

    deviations = {f: abs(X[f].values[0] - feature_means[f]) for f in FEATURES}
    top_feature = max(deviations, key=deviations.get)

    return {
        "status": "anomaly" if label == -1 else "normal",
        "anomaly_score": round(float(score), 4),
        "severity": severity,
        "triggered_by": top_feature
    }