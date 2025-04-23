# Implement the neuron using error correction learning algorithm and memory-based learning algorithm.

# Error correction

from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# OR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Train Perceptron
model = Perceptron(max_iter=20, eta0=0.1, tol=None)
model.fit(X, y)

# Predict and check accuracy
predictions = model.predict(X)
print("Predictions:", predictions)
print("Accuracy:", model.score(X, y) * 100, "%")

# Optional: Plotting loss over epochs (sklearn doesn't expose this, but just simulate)
errors = [sum(predictions != y) for _ in range(20)]
plt.plot(errors)
plt.title("Error Correction Learning (Perceptron)")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# for memory based learning
from sklearn.neighbors import KNeighborsClassifier

# OR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Train k-NN model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

# Predict and evaluate
predictions = model.predict(X)
print("Predictions:", predictions)
print("Accuracy:", model.score(X, y) * 100, "%")
