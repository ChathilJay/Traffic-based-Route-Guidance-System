
import numpy as np
import pandas as pd
import argparse
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

# === CONFIG ===
SEQ_LEN = 12
DATA_PATH = "Cleaned_dataset.csv"

# === Load Data for Specific SCATS Site ===
def load_site_sequence(scats_id):
    df = pd.read_csv(DATA_PATH)
    site_df = df[df['SCATS Number'] == scats_id].copy()
    site_df = site_df.sort_values(by='Datetime')
    scaler = MinMaxScaler()
    site_df['Norm_Flow'] = scaler.fit_transform(site_df[['Traffic_flow']])
    sequence = site_df['Norm_Flow'].values[-SEQ_LEN:]
    return sequence.reshape(1, SEQ_LEN), scaler

# === Predict Next Flow Value ===
def predict_next(model_path, scats_id):
    sequence, scaler = load_site_sequence(scats_id)

    if model_path.endswith(".h5"):
        model = load_model(model_path)
        sequence = sequence.reshape((1, SEQ_LEN, 1))  # for LSTM/GRU
        pred = model.predict(sequence)[0][0]
    elif model_path.endswith(".pkl"):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        pred = model.predict(sequence)[0]
    else:
        raise ValueError("Unsupported model type. Use .h5 for Keras or .pkl for sklearn.")

    # Inverse transform to original scale
    predicted_flow = scaler.inverse_transform([[pred]])[0][0]
    return predicted_flow

# === CLI Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next 15-min flow for a SCATS site.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.h5 or .pkl)")
    parser.add_argument("--scats", type=int, required=True, help="SCATS site number (e.g., 970)")
    args = parser.parse_args()

    flow = predict_next(args.model, args.scats)
    print(f"📈 Predicted next 15-min traffic flow for SCATS {args.scats}: {flow:.2f}")
