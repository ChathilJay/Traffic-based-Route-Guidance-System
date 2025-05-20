import pandas as pd
import numpy as np

#Predicting Volume according to the flow of Time
def predict_volume(df, site_id, model, scaler, window_size, start_time_str):
    start_time = pd.to_datetime(start_time_str)

    df_site = df[df["SCATS Number"] == site_id].copy()
    df_site["DateTime"] = pd.to_datetime(df_site["DateTime"])
    df_site = df_site.sort_values("DateTime")

    time_window = df_site[df_site["DateTime"] < start_time].tail(window_size)

    if len(time_window) < window_size:
        raise ValueError(f"Not enough data before {start_time} for SCATS site {site_id}.")

    window_values = time_window["Volume"].values.reshape(-1, 1)
    scaled_input = scaler.transform(window_values)
    X_input = np.expand_dims(scaled_input, axis=0)

    pred_scaled = model.predict(X_input)
    print("Raw prediction:", pred_scaled)

    pred_scaled = np.clip(pred_scaled, 0, 1)
    predicted_volume = scaler.inverse_transform(pred_scaled)[0][0]

    return round(predicted_volume, 2)

def predict_all_volumes(df, model, scaler, window_size, start_time_str):
    predicted_volumes = {}
    scats_sites = df["SCATS Number"].unique()

    for site_id in scats_sites:
        try:
            volume = predict_volume(df, site_id, model, scaler, window_size, start_time_str)
            predicted_volumes[site_id] = volume
        except Exception as e:
            print(f"Skipping SCATS {site_id}: {e}")

    return predicted_volumes
