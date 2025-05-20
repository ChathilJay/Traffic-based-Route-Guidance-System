
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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

# Reshape for LSTM input
X = X.reshape((X.shape[0], X.shape[1], 1))
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

model.save("lstm_model.h5")

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
