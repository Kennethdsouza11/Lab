from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np


def generate_data(n_samples=1000, seq_len=5, input_dim=3):
    X = np.random.rand(n_samples, seq_len, input_dim)
    y = np.sum(X, axis=1)
    return X, y


X, y = generate_data()

model = Sequential([SimpleRNN(10, input_shape=(5, 3)), Dense(3)])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, batch_size=32)

test_X, test_y = generate_data(3)
print(f"True Sum:", test_y)
print(f"Predicted Sum:", test_y)
