
import pickle
import networkx as nx
import numpy as np

# Load saved graph
with open("../Graph/scats_graph_live.pkl", "rb") as f:
    G = pickle.load(f)

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# Heuristic for A* (straight-line distance)
def heuristic(n1, n2):
    lat1, lon1 = G.nodes[n1]['pos']
    lat2, lon2 = G.nodes[n2]['pos']
    return haversine(lat1, lon1, lat2, lon2)

# A* route finder using travel_time weights
def a_star_route(start, goal):
    try:
        path = nx.astar_path(G, start, goal, heuristic=heuristic, weight='travel_time')
        travel_time = sum(G[u][v]['travel_time'] for u, v in zip(path[:-1], path[1:]))
        return path, travel_time
    except nx.NetworkXNoPath:
        return None, float('inf')
    
# Example: pick another known SCATS
for node in G.nodes:
    reachable = nx.node_connected_component(G, node)
    if len(reachable) > 1:
        print(f"✅ Node {node} is connected to {len(reachable)} others: {reachable}")


# Example usage
if __name__ == "__main__":
    start_id = 2000  # Replace with your SCATS ID
    end_id = 3685    # Replace with your SCATS ID

    path, total_time = a_star_route(start_id, end_id)
    if path:
        print("🛣️ Best route:", path)
        print(f"⏱️ Estimated travel time: {total_time:.2f} seconds")
    else:
        print("⚠️ No route found between the given SCATS sites.")
