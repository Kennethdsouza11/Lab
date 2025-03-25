from sklearn.linear_model import Perceptron
import numpy as np

gates = {
    "AND":np.array([0,0,0,1]),
    "OR":np.array([0,1,1,1]),
    "NAND":np.array([1,1,1,0]),
    "NOR":np.array([1,0,0,0])
}

X = np.array([[0,0],[0,1],[1,0],[1,1]])

for gate_name, label in gates.items():
    model = Perceptron(max_iter = 1000, random_state = 42)
    model.fit(X,label)
    
    print(f"\n{gate_name} Gate : ")
    for x in X:
        print(f"{x} -> {model.predict([x])[0]}")
