from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

learning_rate = [0.001, 0.01, 0.1]
epoch_error = {lr: [] for lr in learning_rate}

plt.figure(figsize = (10,4))

for lr in learning_rate:
    mlp = MLPClassifier(
        activation = "logistic",
        hidden_layer_sizes = (2,),
        max_iter = 1,
        warm_start = True,
        random_state = 42,
        learning_rate_init = lr)
    
    errors = []
    for epoch in range(100):
        mlp.fit(X,y)
        y_pred = mlp.predict_proba(X)[:,1]
        error = np.mean((y_pred - y) ** 2)
        errors.append(error)
    epoch_error[lr] = errors
    plt.plot(errors, label = f"Learning Rate : {lr}")
    
plt.title("Error propogation")
plt.xlabel("Training Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.show()

print("\n Final Results")
for lr in learning_rate:
    mlp = MLPClassifier(hidden_layer_sizes = (2,),
                        activation = "logistic",
                        max_iter = 100,
                        random_state = 42,
                        learning_rate_init = lr).fit(X,y)
    print(f"\n Learning Rate : {lr}")
    print("Input Output Prediction Error")
    for i, x in enumerate(X):
        pred = mlp.predict([x])[0]
        prob = mlp.predict_proba([x])[0][1]
        error = abs(prob - y[i])
        print(f"{x} y{i} {pred} {error:.4f}")