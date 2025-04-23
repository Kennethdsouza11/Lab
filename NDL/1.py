# Implement the basic neuron and update the learning parameter using different activation functions. Show the graph. i. Binary Step ii. Linear iii. Sigmoid iv. Tanh v. ReLU vi. Leaky ReLU vii. Softmax


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

x = np.linspace(-10, 10, 100)


# 1. Binary Step
def binary_step(x):
    return np.where(x >= 0, 1, 0)


# 2. Linear
def linear(x):
    return x


# 3. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 4. Tanh
def tanh(x):
    return np.tanh(x)


# 5. ReLU
def relu(x):
    return np.maximum(0, x)


# 6. Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# 7. Softmax (applied to a vector)
softmax_vals = softmax(x)

# Plotting
activations = {
    "Binary Step": binary_step(x),
    "Linear": linear(x),
    "Sigmoid": sigmoid(x),
    "Tanh": tanh(x),
    "ReLU": relu(x),
    "Leaky ReLU": leaky_relu(x),
    "Softmax": softmax_vals,  # on whole input
}

plt.figure(figsize=(12, 10))
for i, (name, act) in enumerate(activations.items()):
    plt.subplot(4, 2, i + 1)
    plt.plot(x, act)
    plt.title(name)
    plt.grid(True)

plt.tight_layout()
plt.show()
