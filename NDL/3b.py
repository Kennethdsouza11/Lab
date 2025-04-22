import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# XOR gate data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

learning_rates = [0.001, 0.01, 0.1, 1.0]
losses = []
accuracies = []

for lr in learning_rates:
    model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, learning_rate_init=lr, random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    acc = model.score(X, y)
    loss = log_loss(y, y_prob)
    
    accuracies.append(acc)
    losses.append(loss)

    print(f"\nLearning Rate = {lr}")
    print("Predictions  :", y_pred)
    print("Accuracy     :", acc * 100, "%")
    print("Log Loss     :", loss)

# Plotting
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(learning_rates, losses, marker='o')
plt.title("Loss vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Log Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(learning_rates, accuracies, marker='o', color='green')
plt.title("Accuracy vs Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()
