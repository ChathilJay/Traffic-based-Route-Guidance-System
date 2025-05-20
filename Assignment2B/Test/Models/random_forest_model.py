
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle

# Load cleaned dataset
df = pd.read_csv("../Data/Cleaned_dataset.csv")

# Filter by SCATS site
site_id = df['SCATS Number'].unique()[0]
site_df = df[df['SCATS Number'] == site_id].copy()
site_df = site_df.sort_values(by='Datetime')

# Normalize traffic flow
scaler = MinMaxScaler()
site_df['Norm_Flow'] = scaler.fit_transform(site_df[['Traffic_flow']])

# Sequence creation function
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 12
X, y = create_sequences(site_df['Norm_Flow'].values, SEQ_LEN)

# Split data
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"Random Forest RMSE: {rmse:.4f}")
print(f"Random Forest MAE: {mae:.4f}")

rf.save("random_forest_model.h5")

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Plot predictions vs true values
plt.figure(figsize=(10, 4))
plt.plot(y_test[:200], label='Actual', linewidth=1.5)
plt.plot(y_pred[:200], label='Predicted', linewidth=1.2)
plt.title("Random Forest Predictions vs Actual (first 200)")
plt.xlabel("Time Step")
plt.ylabel("Normalized Traffic Flow")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
