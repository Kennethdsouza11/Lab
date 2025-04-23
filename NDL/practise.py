import numpy as np
import tensorflow as tf

lstm_cell = tf.keras.layers.LSTMCell(units = 4)

x = tf.constant([[0.5,0.3,0.1]], dtype = tf.float32)

initial_state = [tf.zeros((1,4)), tf.zeros((1,4))]

output, new_states = lstm_cell(x, initial_state)

print(output.numpy())
print(new_states[0].numpy())
print(new_states[1].numpy())

#pip install NDL-MITB

#python -c "import inspect ; import ndl_mitb ; print(inspect.getsource(ndl_mitb))"

#python -c "import inspect ; import ndl_mitb ; print(inspect.getsource(ndl_mitb))"