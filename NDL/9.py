import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Load dataset
digits = load_digits()
X, y = digits.data, np.eye(10)[digits.target]  # One-hot encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# Initialize weights
np.random.seed(42)
W1, b1 = np.random.randn(64, 32) * 0.01, np.zeros((1, 32))
W2, b2 = np.random.randn(32, 10) * 0.01, np.zeros((1, 10))

# Optimizer parameters
lr, eps, beta1, beta2 = 0.01, 1e-8, 0.9, 0.999
G1, G2, S1, S2, V1, V2 = 0, 0, 0, 0, 0, 0

# Training
for epoch in range(500):
    # Forward pass
    Z1 = X_train @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Backpropagation
    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2 / len(X_train)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X_train)

    dZ1 = (dZ2 @ W2.T) * A1 * (1 - A1)  # This was missing in the previous version
    dW1 = X_train.T @ dZ1 / len(X_train)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X_train)

    # Choose an optimizer (Uncomment the one you want)
    
    # AdaGrad
    G1, G2 = G1 + dW1**2, G2 + dW2**2
    W1 -= lr * dW1 / (np.sqrt(G1) + eps)
    W2 -= lr * dW2 / (np.sqrt(G2) + eps)

    # RMSProp
    # S1, S2 = 0.9 * S1 + 0.1 * dW1**2, 0.9 * S2 + 0.1 * dW2**2
    # W1 -= lr * dW1 / (np.sqrt(S1) + eps)
    # W2 -= lr * dW2 / (np.sqrt(S2) + eps)

    # Adam
    # V1 = beta1 * V1 + (1 - beta1) * dW1
    # V2 = beta1 * V2 + (1 - beta1) * dW2
    # S1 = beta2 * S1 + (1 - beta2) * dW1**2
    # S2 = beta2 * S2 + (1 - beta2) * dW2**2
    # W1 -= lr * (V1 / (np.sqrt(S1) + eps))
    # W2 -= lr * (V2 / (np.sqrt(S2) + eps))

    b1 -= lr * db1
    b2 -= lr * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}")

# Evaluation
y_pred = np.argmax(softmax(sigmoid(X_test @ W1 + b1) @ W2 + b2), axis=1)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy:.4f}")
