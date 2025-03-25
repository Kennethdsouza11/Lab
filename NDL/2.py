<<<<<<< HEAD:2.py
import numpy as np


class ErrorCorrectionNeuron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation(summation)

    def train(self, training_inputs, labels, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    neuron = ErrorCorrectionNeuron(input_size=2, learning_rate=0.1)
    neuron.train(X, y, epochs=100)

    for inputs in X:
        print(f"{inputs} -> {neuron.predict(inputs)}")
=======
import numpy as np

>>>>>>> 4c3baa3cafff9d46df87d6d8724b63212d4dcd21:NDL/2.py
