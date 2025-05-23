
# dynamic_route_gui.py
import streamlit as st
import pickle
import networkx as nx
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# === Setup ===
st.set_page_config(layout="wide")
st.title("🚦 SCATS Route Planner with Live Traffic Prediction")

# === Helper: Haversine ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi/2)**2 + cos(phi1)*cos(phi2)*sin(d_lambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Heuristic ===
def heuristic(n1, n2, G):
    lat1, lon1 = G.nodes[n1]['pos']
    lat2, lon2 = G.nodes[n2]['pos']
    return haversine(lat1, lon1, lat2, lon2)

# === Load Graph ===
@st.cache_resource
def load_graph():
    with open("../Graph/scats_graph_from_roads.pkl", "rb") as f:
        return pickle.load(f)

# === Predict Flow Using Model ===
def predict_flow(model_path, df, scats_id, seq_len=12):
    model = load_model(model_path)
    df_site = df[df['SCATS Number'] == scats_id].copy()
    if len(df_site) < seq_len:
        return None
    df_site = df_site.sort_values(by='Datetime')
    scaler = MinMaxScaler()
    df_site['Norm_Flow'] = scaler.fit_transform(df_site[['Traffic_flow']])
    sequence = df_site['Norm_Flow'].values[-seq_len:].reshape(1, seq_len, 1)
    pred = model.predict(sequence)[0][0]
    return scaler.inverse_transform([[pred]])[0][0]

# === Estimate Speed from Flow ===
def estimate_speed(flow):
    a, b = -1.4648375, 93.75
    c = -flow
    D = b**2 - 4*a*c
    if D < 0:
        return 10
    s1 = (-b + sqrt(D)) / (2 * a)
    s2 = (-b - sqrt(D)) / (2 * a)
    return max(5, min(s1 if s1 > 0 else s2, 60))

# === Update Graph Travel Time ===
def update_graph_travel_time(G, model_path, df):
    flows = {}
    for node in G.nodes:
        try:
            flow = predict_flow(model_path, df, node)
            if flow: flows[node] = flow
        except: continue

    for u, v in G.edges:
        if u in flows and v in flows:
            avg_flow = (flows[u] + flows[v]) / 2
            speed = estimate_speed(avg_flow)
            dist = G[u][v]['distance']
            G[u][v]['travel_time'] = (dist / speed) * 3600 + 30
    return G

# === UI Panel ===
st.sidebar.header("🛠️ Controls")
model_choice = st.sidebar.selectbox("Choose Model", [
    "lstm_model.h5", "gru_model.h5", "dnn_model.h5"
])
df = pd.read_csv("../Data/Cleaned_dataset.csv")
G = load_graph()
scats_ids = sorted(list(G.nodes))
start_id = st.sidebar.selectbox("Start SCATS", scats_ids, index=0)
end_id = st.sidebar.selectbox("End SCATS", scats_ids, index=min(1, len(scats_ids)-1))

# === Routing + Map Generation ===
if st.sidebar.button("🔍 Find Route"):
    G = update_graph_travel_time(G, f"Models/{model_choice}", df)
    try:
        path = nx.astar_path(G, start_id, end_id, heuristic=lambda u, v: heuristic(u, v, G), weight='travel_time')
        coords = [G.nodes[n]['pos'] for n in path]
        m = folium.Map(location=G.nodes[start_id]['pos'], zoom_start=14)
        for node in G.nodes:
            folium.CircleMarker(G.nodes[node]['pos'], radius=4, color="gray").add_to(m)
        folium.PolyLine(coords, color="blue", weight=5, opacity=0.8).add_to(m)

        # ⬇️ Store map in session_state
        st.session_state['route_map'] = m
        st.session_state['route_time'] = sum(G[u][v].get('travel_time', 999) for u, v in zip(path[:-1], path[1:]))

    except nx.NetworkXNoPath:
        st.error("No route found between the selected SCATS sites.")
        st.session_state['route_map'] = None

# === Display the Map if Present ===
if 'route_map' in st.session_state and st.session_state['route_map']:
    st.success(f"Route found! Estimated travel time: {st.session_state['route_time']:.2f} sec")
    st_folium(st.session_state['route_map'], width=1000, height=600)
