
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])

model = Perceptron(max_iter = 20, eta0 = 0.1, tol = None)
prediction = model.predict(X)
print("Predictions : ", prediction)
print(f"Accuracy : {model.score * 100}%")