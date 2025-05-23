
import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import os

# === CONFIG ===
CSV_PATH = "../Data/scats_site_info.csv"
OUTPUT_PATH = "../Graph/scats_graph_from_roads.pkl"
DIST_THRESHOLD_KM = 0.5

# === Haversine Distance ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Load Data ===
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['NB_LATITUDE', 'NB_LONGITUDE', 'MainRoad'])

# === Build Graph ===
G = nx.Graph()

# Add nodes
for _, row in df.iterrows():
    node_id = int(row['SCATS Number'])
    lat, lon = row['NB_LATITUDE'], row['NB_LONGITUDE']
    G.add_node(node_id, pos=(lat, lon), location=row['Location'], road=row['MainRoad'])

# Connect nodes on the same road within distance threshold
grouped = df.groupby('MainRoad')
for road, group in grouped:
    for i, row1 in group.iterrows():
        for j, row2 in group.iterrows():
            if i >= j:
                continue
            dist = haversine(row1['NB_LATITUDE'], row1['NB_LONGITUDE'],
                             row2['NB_LATITUDE'], row2['NB_LONGITUDE'])
            if dist <= DIST_THRESHOLD_KM:
                G.add_edge(int(row1['SCATS Number']), int(row2['SCATS Number']), distance=dist)

# Save graph
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    import pickle
    pickle.dump(G, f)

print(f"✅ Graph saved: {OUTPUT_PATH}")
print(f"🔢 Nodes: {len(G.nodes)}, 🔗 Edges: {len(G.edges)}")
