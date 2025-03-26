from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

learning_rates = [0.001, 0.01, 0.1]
epoch_errors = {lr: [] for lr in learning_rates}


for lr in learning_rates:
    mlp = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        max_iter=1,
        learning_rate_init=lr,
        random_state=42,
        warm_start=True,
    )

    errors = []
    for epoch in range(100):
        mlp.fit(X, y)
        y_pred = mlp.predict_proba(X)[:, 1]
        error = np.mean((y_pred - y) ** 2)
        errors.append(error)

    epoch_errors[lr] = errors
    plt.plot(errors, label=f"Learning Rate : {lr}")

plt.title("Error Propogation with Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()

for lr in learning_rates:
    mlp = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        max_iter=100,
        learning_rate_init=lr,
        random_state=42,
    ).fit(X, y)
    for i, x in enumerate(X):
        pred = mlp.predict([x])[0]
        proba = mlp.predict_proba([x])[0][1]
        error = abs(pred - y[i])
        print(f"{x} {y[i]} {pred} {error:.2f}")
