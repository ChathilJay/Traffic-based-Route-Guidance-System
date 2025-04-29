import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam

# Load data
X = np.load('Main/X_all.npy')
y = np.load('Main/y_all.npy')

# --- RANDOM FOREST ---
print("\n🔧 Training Random Forest...")
X_rf, X_rf_test, y_rf, y_rf_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)
y_rf_pred = rf.predict(X_rf_test)
rf_rmse = mean_squared_error(y_rf_test, y_rf_pred, squared=False)
print(f"Random Forest RMSE: {rf_rmse:.2f}")

# --- LSTM ---
print("\n🔧 Training LSTM...")
X_lstm = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]
X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

lstm = Sequential([
    LSTM(64, input_shape=(X.shape[1], 1)),
    Dense(1)
])
lstm.compile(optimizer=Adam(0.001), loss='mse')
lstm.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=1)
y_lstm_pred = lstm.predict(X_lstm_test)
lstm_rmse = mean_squared_error(y_lstm_test, y_lstm_pred, squared=False)
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# --- GRU ---
print("\n🔧 Training GRU...")
gru = Sequential([
    GRU(64, input_shape=(X.shape[1], 1)),
    Dense(1)
])
gru.compile(optimizer=Adam(0.001), loss='mse')
gru.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=1)
y_gru_pred = gru.predict(X_lstm_test)
gru_rmse = mean_squared_error(y_lstm_test, y_gru_pred, squared=False)
print(f"GRU RMSE: {gru_rmse:.2f}")
