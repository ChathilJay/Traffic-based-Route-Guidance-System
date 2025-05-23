
import pickle
import folium
import networkx as nx

# === CONFIG ===
GRAPH_PATH = "../Graph/scats_graph_live.pkl"
START_ID = 2000
END_ID = 3685
OUTPUT_HTML = "route_map.html"

# === Load Graph ===
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

# === Heuristic Function ===
from math import radians, sin, cos, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2)**2 + cos(phi1)*cos(phi2)*sin(d_lambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def heuristic(n1, n2):
    lat1, lon1 = G.nodes[n1]['pos']
    lat2, lon2 = G.nodes[n2]['pos']
    return haversine(lat1, lon1, lat2, lon2)

# === Run A* Routing ===
try:
    path = nx.astar_path(G, START_ID, END_ID, heuristic=heuristic, weight='travel_time')
except nx.NetworkXNoPath:
    print("⚠️ No route found between the given SCATS sites.")
    exit()

# === Initialize Map ===
center = G.nodes[START_ID]['pos']
m = folium.Map(location=center, zoom_start=14)

# === Add All Nodes ===
for node in G.nodes:
    lat, lon = G.nodes[node]['pos']
    label = G.nodes[node].get('location', f"SCATS {node}")
    folium.CircleMarker(
        location=(lat, lon),
        radius=4,
        popup=label,
        color="gray" if node not in path else "blue",
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# === Draw Path ===
route_coords = [G.nodes[n]['pos'] for n in path]
folium.PolyLine(
    locations=route_coords,
    color="blue",
    weight=5,
    opacity=0.8,
    tooltip=f"A* route: {path}"
).add_to(m)

# === Save Map ===
m.save(OUTPUT_HTML)
print(f"✅ Route map saved: {OUTPUT_HTML}")
