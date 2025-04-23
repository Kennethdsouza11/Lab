# Implement the back propagation algorithm for training a recurrent network using temporal operation as a parameter into a layer feed forward network.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

# Dummy data generation
batch_size = 100
time_steps = 10
input_dim = 8

# Generate random sequence data
X = np.random.random((batch_size, time_steps, input_dim))
y = np.random.random((batch_size, 1))  # Regression output

# Define RNN model using high-level Keras APIs
model = Sequential(
    [
        SimpleRNN(32, activation="tanh", input_shape=(time_steps, input_dim)),
        Dense(1),  # Output layer
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

# Train the model using built-in training
model.fit(X, y, epochs=20, batch_size=16)
