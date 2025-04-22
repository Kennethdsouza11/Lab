import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Try different regularization values (alpha in Ridge = 1 / lambda)
alphas = [0.001, 0.01, 0.1, 1.0, 10]
accuracies = []

for alpha in alphas:
    model = make_pipeline(
        RBFSampler(gamma=1.0, random_state=1),  # gamma: RBF kernel width
        RidgeClassifier(alpha=alpha)
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    accuracies.append(acc)

    print(f"\nAlpha (λ) = {alpha}")
    print("Predictions:", y_pred)
    print("Accuracy   :", acc * 100, "%")

# Plot: Accuracy vs Regularization (alpha)
plt.plot(alphas, accuracies, marker='o')
plt.xscale('log')
plt.xlabel("Regularization Strength (alpha)")
plt.ylabel("Accuracy")
plt.title("GRBF XOR Accuracy vs Regularization")
plt.grid(True)
plt.show()
