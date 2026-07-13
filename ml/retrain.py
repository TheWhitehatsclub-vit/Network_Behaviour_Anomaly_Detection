from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import os
from datetime import datetime

FEATURES = ["Inter_Arrival_Time", "Src_Port", "Dst_Port", "Seq", "Ack", "Win", "Payload_Len", "Packet_Length"]

TRAINING_DATA_FILE = "current_training_data.csv"
ORIGINAL_DATA_FILE = "Regular_Network_Traffic_Standardized.csv"


def retrain(false_positive_rows: list):
    if not false_positive_rows:
        return {"status": "skipped", "reason": "no rows provided"}

    for row in false_positive_rows:
        missing = [f for f in FEATURES if f not in row]
        if missing:
            return {"status": "error", "reason": f"missing features: {missing}"}

    if os.path.exists(TRAINING_DATA_FILE):
        current_data = pd.read_csv(TRAINING_DATA_FILE, usecols=FEATURES)
    else:
        current_data = pd.read_csv(ORIGINAL_DATA_FILE, usecols=FEATURES)
    current_data = current_data.dropna()

    new_rows = pd.DataFrame(false_positive_rows)[FEATURES]
    updated_data = pd.concat([current_data, new_rows], ignore_index=True)

    old_model = joblib.load("model.pkl")
    contamination = old_model.contamination
    joblib.dump(old_model, "model_backup.pkl")

    new_model = IsolationForest(contamination=contamination, random_state=42)
    new_model.fit(updated_data)
    joblib.dump(new_model, "model.pkl")

    updated_data.to_csv(TRAINING_DATA_FILE, index=False)

    new_feature_means = updated_data[FEATURES].mean().to_dict()
    joblib.dump(new_feature_means, "feature_means.pkl")

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "rows_added": len(false_positive_rows),
        "total_training_rows": len(updated_data)
    }])
    log_entry.to_csv("retrain_log.csv", mode="a",
                      header=not os.path.exists("retrain_log.csv"), index=False)

    return {
        "status": "success",
        "rows_added": len(false_positive_rows),
        "total_training_rows": len(updated_data)
    }


if __name__ == "__main__":
    fake_false_positive = {
        "Inter_Arrival_Time": 0.02,
        "Src_Port": 0.3,
        "Dst_Port": -0.2,
        "Seq": 0.0,
        "Ack": 0.1,
        "Win": -0.1,
        "Payload_Len": 0.1,
        "Packet_Length": -0.05
    }
    result = retrain([fake_false_positive])
    print(result)