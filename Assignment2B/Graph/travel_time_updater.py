
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, sqrt, atan2
import os

# === CONFIG ===
DATA_PATH = "../Data/Cleaned_dataset.csv"
GRAPH_PATH = "../Graph/scats_graph_from_roads.pkl"
MODEL_PATH = "../Models/lstm_model.h5"
OUTPUT_PATH = "../Graph/scats_graph_live.pkl"
SEQ_LEN = 12

# === Haversine Distance (for heuristic if needed) ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Estimate speed from flow using quadratic formula ===
def estimate_speed(flow):
    a, b = -1.4648375, 93.75
    c = -flow
    D = b ** 2 - 4 * a * c
    if D < 0:
        return 10.0  # default speed
    s1 = (-b + np.sqrt(D)) / (2 * a)
    s2 = (-b - np.sqrt(D)) / (2 * a)
    return max(5.0, min(s1 if s1 > 0 else s2, 60.0))

# === Load latest flow prediction per SCATS site ===
def get_predictions():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(by='Datetime')
    preds = {}
    model = load_model(MODEL_PATH)
    scaler = MinMaxScaler()

    for site_id in df['SCATS Number'].unique():
        site_df = df[df['SCATS Number'] == site_id].copy()
        if len(site_df) < SEQ_LEN:
            continue
        site_df['Norm_Flow'] = scaler.fit_transform(site_df[['Traffic_flow']])
        sequence = site_df['Norm_Flow'].values[-SEQ_LEN:].reshape((1, SEQ_LEN, 1))
        norm_pred = model.predict(sequence)[0][0]
        flow = scaler.inverse_transform([[norm_pred]])[0][0]
        preds[int(site_id)] = flow
    return preds

# === Main: update edge travel times ===
def update_graph():
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    predicted_flows = get_predictions()

    for u, v in G.edges:
        if u in predicted_flows and v in predicted_flows:
            avg_flow = (predicted_flows[u] + predicted_flows[v]) / 2
            speed = estimate_speed(avg_flow)
            dist = G[u][v]['distance']
            travel_time = (dist / speed) * 3600 + 30  # seconds
            G[u][v]['travel_time'] = travel_time

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(G, f)

    print(f"✅ Live graph updated and saved to: {OUTPUT_PATH}")
    print(f"📈 Nodes updated with predicted flow: {len(predicted_flows)}")

if __name__ == "__main__":
    update_graph()
