import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def binary_step(x):
    return np.where(x >= 0, 1, 0)

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Generate input values
x = np.linspace(-10, 10, 1000)

# Create figure with subplots
plt.figure(figsize=(15, 10))

# Plot each activation function
activation_functions = [
    (binary_step, 'Binary Step'),
    (linear, 'Linear'),
    (sigmoid, 'Sigmoid'),
    (tanh, 'Tanh'),
    (relu, 'ReLU'),
    (leaky_relu, 'Leaky ReLU'),
    (softmax, 'Softmax')
]

for i, (func, name) in enumerate(activation_functions, 1):
    plt.subplot(3, 3, i)
    if name == 'Softmax':
        # For softmax, we need to handle it differently since it's a multi-class function
        y = func(np.array([x, x+1, x+2]))
        for j in range(3):
            plt.plot(x, y[j], label=f'Class {j+1}')
    else:
        y = func(x)
        plt.plot(x, y)
    
    plt.title(name)
    plt.grid(True)
    if name == 'Softmax':
        plt.legend()

plt.tight_layout()
plt.show()

# Example of a simple neuron with learning
class SimpleNeuron:
    def __init__(self, input_size, activation_function):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation = activation_function
    
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
    
    def update_weights(self, x, y_true, y_pred, learning_rate=0.01):
        error = y_true - y_pred
        self.weights += learning_rate * error * x
        self.bias += learning_rate * error

# Example usage
if __name__ == "__main__":
    # Create a simple neuron with sigmoid activation
    neuron = SimpleNeuron(input_size=2, activation_function=sigmoid)
    
    # Example training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR problem
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        for i in range(len(X)):
            y_pred = neuron.forward(X[i])
            neuron.update_weights(X[i], y[i], y_pred)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Weights: {neuron.weights}, Bias: {neuron.bias}") 