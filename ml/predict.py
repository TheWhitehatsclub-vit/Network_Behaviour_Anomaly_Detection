import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict(connection_count, packet_size, data_sent, duration):
    data = pd.DataFrame({
        "connection_count": [connection_count],
        "packet_size": [packet_size],
        "data_sent": [data_sent],
        "duration": [duration]
    })

    prediction = model.predict(data)[0]
    score = model.decision_function(data)[0]

    label = "ANOMALY" if prediction == -1 else "Normal"

    return {
        "label": label,
        "score": float(score)
    }

if __name__ == "__main__":
    result = predict(700, 70, 80000, 0.3)
    print(result)