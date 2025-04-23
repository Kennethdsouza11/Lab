# Implement the neuron using Hebbian Learning algorithm and show the graph to differentiate the hypothesis between Hebb and Co-variance learning.

import numpy as np
import matplotlib.pyplot as plt

# OR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Hebbian Learning: Δw = x * y
hebb_weights = np.zeros(X.shape[1])
for xi, yi in zip(X, y):
    hebb_weights += xi * yi

# Covariance Learning: Δw = (x - mean_x) * (y - mean_y)
mean_x = np.mean(X, axis=0)
mean_y = np.mean(y)
cov_weights = np.zeros(X.shape[1])
for xi, yi in zip(X, y):
    cov_weights += (xi - mean_x) * (yi - mean_y)

# Plot comparison
labels = ["w1", "w2"]
x_axis = np.arange(len(labels))
width = 0.3

plt.bar(x_axis - width / 2, hebb_weights, width, label="Hebbian")
plt.bar(x_axis + width / 2, cov_weights, width, label="Covariance")

plt.xticks(x_axis, labels)
plt.ylabel("Weight Value")
plt.title("Hebbian vs Covariance Learning Weights")
plt.legend()
plt.grid(True)
plt.show()
