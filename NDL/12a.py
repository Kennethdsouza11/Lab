import tensorflow as tf
import numpy as np

# Define RNN cell using Keras
rnn_cell = tf.keras.layers.SimpleRNNCell(units=5)

# Create a single input sample (batch_size=1, input_dim=3)
x = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)

# Create initial hidden state (batch_size=1, units=5)
h_prev = [tf.zeros((1, 5))]  # Must be list for RNN cells

# Run a single forward step
output, h_next = rnn_cell(x, h_prev)

print("RNN Cell Output:", output.numpy())
print("Next Hidden State:", h_next[0].numpy())
