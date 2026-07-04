import pandas as pd
from predict import predict

FEATURES = ["Inter_Arrival_Time", "Payload_Len", "Packet_Length",
            "Src_Port", "Dst_Port", "Seq", "Ack", "Win"]

normal = pd.read_csv("Regular_Network_Traffic_Standardized.csv", usecols=FEATURES)
attack = pd.read_csv("Aggressive_Scan_Standardized.csv", usecols=FEATURES)

normal_row = normal.iloc[0].to_dict()
attack_row = attack.iloc[0].to_dict()

print("Normal row result:")
print(predict(normal_row))

print("\nAttack row result:")
print(predict(attack_row))