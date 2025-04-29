import pandas as pd
import numpy as np

# Load site_data (from previous step)
site_data = pd.read_pickle('Main/site_data.pkl')

# Parameters
WINDOW_SIZE = 4  # past 4 time steps to predict next one

X_all = []
y_all = []

for site_id, series in site_data.items():
    # Sort timestamps
    timestamps = sorted(series.keys())
    
    # Extract volume values in sorted order
    volumes = [series[t] for t in timestamps if not pd.isna(series[t])]
    
    # Skip short series
    if len(volumes) < WINDOW_SIZE + 1:
        continue

    # Sliding window
    for i in range(len(volumes) - WINDOW_SIZE):
        X = volumes[i:i+WINDOW_SIZE]
        y = volumes[i+WINDOW_SIZE]
        X_all.append(X)
        y_all.append(y)

print(f"Created {len(X_all)} sequences.")

# Convert to numpy arrays for ML
X_all = np.array(X_all)
y_all = np.array(y_all)

# Optionally save
np.save('Main/X_all.npy', X_all)
np.save('Main/y_all.npy', y_all)

print("Saved to 'Main/X_all.npy' and 'Main/y_all.npy'")
