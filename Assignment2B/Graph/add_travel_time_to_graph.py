
import pickle
import numpy as np

# Load graph
with open("scats_graph_from_roads.pkl", "rb") as f:
    G = pickle.load(f)

# Flow-speed relationship
def estimate_speed_from_flow(flow):
    a = -1.4648375
    b = 93.75
    c = -flow
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 10.0
    root1 = (-b + np.sqrt(discriminant)) / (2*a)
    root2 = (-b - np.sqrt(discriminant)) / (2*a)
    speed = max(root1, root2)
    return min(speed, 60.0)

# Travel time calculator
def calculate_travel_time(distance_km, flow, intersection_delay_sec=30):
    speed_kmph = estimate_speed_from_flow(flow)
    if speed_kmph == 0:
        return float('inf')
    time_hr = distance_km / speed_kmph
    return time_hr * 3600 + intersection_delay_sec

# Dummy flow: use fixed avg flow of 500 (replace with real predictions later)
default_flow = 500

for u, v, data in G.edges(data=True):
    travel_time = calculate_travel_time(data['distance'], default_flow)
    G[u][v]['travel_time'] = travel_time

# Save updated graph
with open("scats_graph_with_travel_time.pkl", "wb") as f:
    pickle.dump(G, f)

print("✅ Travel time added and graph saved.")
