# Long Short-Term Memory (LSTM)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import r2_score


# Generate a sinusoidal wave with variations
def generate_varied_wave_data(
    num_samples, timesteps, amplitude_variation_rate, frequency_variation_rate
):
    x = np.linspace(0, num_samples * timesteps, num_samples)
    amplitude_variation = np.sin(amplitude_variation_rate * x)  #
    frequency_variation = 1 + 0.5 * np.sin(frequency_variation_rate * x)
    wave_data = amplitude_variation * np.sin(frequency_variation * x)
    return wave_data


# Parameters
num_samples = 1000
timesteps = 0.1
seq_length = 50
# Generate training data
train_wave_data = generate_varied_wave_data(num_samples, timesteps, 0.1, 0.05)

# Generate validation data with different variations
validation_wave_data = generate_varied_wave_data(num_samples, timesteps, 0.15, 0.06)

# Plot the training and validation data
plt.plot(train_wave_data, label="Training Data")
plt.plot(validation_wave_data, label="Validation Data")
plt.title("Sinusoidal Waves with Different Variations")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# Create training data
X_train, y_train = create_sequences(train_wave_data, seq_length)

# Create validation data
X_val, y_val = create_sequences(validation_wave_data, seq_length)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

print(f"Training input shape: {X_train.shape}")
print(f"Training output shape: {y_train.shape}")
print(f"Validation input shape: {X_val.shape}")
print(f"Validation output shape: {y_val.shape}")

# The LSTM model
model = Sequential()
model.add(
    LSTM(64, activation="relu", return_sequences=True, input_shape=(seq_length, 1))
)
model.add(Dropout(0.3))
model.add(LSTM(64, activation="relu", return_sequences=True))
model.add(Dropout(0.3))

model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

model.summary()

# Train the model
history = model.fit(
    X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val)
)

# Predict the next values for training and validation data
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

# Compute the R² score for the training data predictions
r2_train = r2_score(y_train, train_predictions)
print(f"R² score for training data: {r2_train:.4f}")

# Compute the R² score for the validation data predictions
r2_val = r2_score(y_val, val_predictions)
print(f"R² score for validation data: {r2_val:.4f}")

# Save the relevant data to a CSV file
data_to_save = {
    "Sample": np.arange(len(y_train)),
    "Amplitude_Train": y_train,
    "Predicted_Amplitude_Train": train_predictions.flatten(),
    "Amplitude_Val": y_val,
    "Predicted_Amplitude_Val": val_predictions.flatten(),
}

df = pd.DataFrame(data_to_save)
df.to_csv("wave_data_predictions_LSTM.csv", index=False)

print("Data saved to wave_data_predictions.csv")

# Plot the predictions against the actual values for training data
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(y_train, label="True - Training")
plt.plot(train_predictions, label="Predicted - Training")
plt.legend()
plt.title("Training Data Prediction")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Plot the predictions against the actual values for validation data
plt.subplot(1, 2, 2)
plt.plot(y_val, label="True - Validation")
plt.plot(val_predictions, label="Predicted - Validation")
plt.legend()
plt.title("Validation Data Prediction")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
