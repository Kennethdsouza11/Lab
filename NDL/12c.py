import tensorflow as tf
import numpy as np

# Define an LSTM cell from Keras (this is the library-implemented cell)
lstm_cell = tf.keras.layers.LSTMCell(units=4)

# Sample input: (batch_size=1, input_dim=3)
x = tf.constant([[0.5, 0.3, 0.1]], dtype=tf.float32)

# Initial hidden state and cell state: (batch_size=1, units=4)
initial_state = [tf.zeros((1, 4)), tf.zeros((1, 4))]

# Single step through the LSTM cell (1 forward step)
output, new_states = lstm_cell(x, initial_state)

# Output is the hidden state (h_t), and new_states = [h_t, c_t]
print("LSTM Output (h_t):", output.numpy())
print("New Hidden State (h_t):", new_states[0].numpy())
print("New Cell State (c_t):", new_states[1].numpy())
