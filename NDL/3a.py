from sklearn.linear_model import Perceptron
import numpy as np

# Logic gate inputs
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Define gate outputs
gates = {
    "AND":  np.array([0, 0, 0, 1]),
    "OR":   np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "NOR":  np.array([1, 0, 0, 0])
}

# Train and test SLP for each gate
for gate, y in gates.items():
    model = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3)
    model.fit(X, y)
    predictions = model.predict(X)
    print(f"\n{gate} Gate:")
    print("Predictions:", predictions)
    print("Accuracy:   ", model.score(X, y) * 100, "%")
