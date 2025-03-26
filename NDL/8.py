import numpy as np
from minisom import MiniSom
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=200, n_features=3, centers=4, random_state=42)
som = MiniSom(x=5, y=5, sigma=1.0, learning_rate=0.1, random_seed=42, input_len=3)

som.train_random(X, num_iteration=1000)
print("SOM weights shape:", som.get_weights().shape)
print("Quantization error:", som.quantization_error(X))
for i in range(3):
    winner = som.winner(X[i])
    print(f"Sample {i} maps to neuron position {winner}")
