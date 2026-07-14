import pandas as pd
import joblib

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

def predict(traffic_row: dict):
    model = joblib.load("model.pkl")
    feature_means = joblib.load("feature_means.pkl")

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if expected != FEATURES:
            raise ValueError(
                f"FEATURE ORDER MISMATCH!\n"
                f"Model was trained on: {expected}\n"
                f"This file is using:   {FEATURES}\n"
                f"These must match exactly, in the same order."
            )

    X = pd.DataFrame([traffic_row])[FEATURES]

    if X.isnull().values.any():
        missing = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Missing/NaN values in input for features: {missing}")

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