import numpy as np
import pickle


def estimate_speed_from_flow(flow):
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 10.0  # fallback speed if no real root

    root1 = (-b + np.sqrt(discriminant)) / (2*a)
    root2 = (-b - np.sqrt(discriminant)) / (2*a)
    speed = max(root1, root2)
    return min(speed, 60.0)
    
def calculate_travel_time(distance_km, flow, intersection_delay_sec=30):
    speed_kmph = estimate_speed_from_flow(flow)
    if speed_kmph == 0:
        return float('inf')
    time_hr = distance_km / speed_kmph
    return time_hr * 3600 + intersection_delay_sec  # return in seconds

# Example: use the latest known flow per SCATS site
flow_map = df.groupby('SCATS Number')['Traffic_flow'].mean().to_dict()  # or use predicted values if available

# Assign travel time to each edge
for u, v, attrs in G.edges(data=True):
    dest_flow = flow_map.get(v, 500)  # fallback flow if missing
    dist_km = attrs['distance']
    travel_time = calculate_travel_time(dist_km, dest_flow)
    G[u][v]['travel_time'] = travel_time


with open("../Main/scats_graph_with_travel_time.pkl", "rb") as f:
    G = pickle.load(f)

