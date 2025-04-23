import tensorflow as tf
import numpy as np

# Define the RNN layer using Keras (not manually coded)
rnn_layer = tf.keras.layers.SimpleRNN(units=5, return_sequences=True, return_state=True)

# Simulate a batch of sequences (batch_size=2, timesteps=4, input_dim=3)
x = tf.constant([
    [[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9],
     [1.0, 1.1, 1.2]],

    [[0.2, 0.1, 0.0],
     [0.5, 0.4, 0.3],
     [0.8, 0.7, 0.6],
     [1.1, 1.0, 0.9]]
], dtype=tf.float32)

# Forward propagate through the RNN layer
all_outputs, final_state = rnn_layer(x)

print("All Outputs (sequence-wise):")
print(all_outputs.numpy())

print("\nFinal Hidden State:")
print(final_state.numpy())
