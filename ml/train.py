from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import joblib

# fake data for now, real CSV later
normal = pd.DataFrame({
    "connection_count": np.random.randint(10, 50, 200),
    "packet_size": np.random.randint(500, 1500, 200),
    "data_sent": np.random.randint(1000, 5000, 200),
    "duration": np.random.uniform(1, 5, 200)
})

attack = pd.DataFrame({
    "connection_count": np.random.randint(500, 1000, 20),
    "packet_size": np.random.randint(50, 100, 20),
    "data_sent": np.random.randint(50000, 100000, 20),
    "duration": np.random.uniform(0.1, 0.5, 20)
})

FEATURES = ["connection_count", "packet_size", "data_sent", "duration"]
X_train = normal[FEATURES]

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)
print("Training done.")

X_attack = attack[FEATURES]
predictions = model.predict(X_attack)

for i, pred in enumerate(predictions):
    label = "ANOMALY" if pred == -1 else "Normal"
    print(f"  Traffic {i+1}: {label}")

normal_predictions = model.predict(X_train.sample(10))
print("\nNormal traffic check:")
for pred in normal_predictions:
    print("  Normal" if pred == 1 else "  ANOMALY")

joblib.dump(model, "model.pkl")
print("Model saved.")