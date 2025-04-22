import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Generate random 2D data
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples with 2 features

# Initialize the SOM
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=500)

# Plot the SOM distance map (U-Matrix)
plt.figure(figsize=(8, 6))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Transpose to match (x,y) orientation
plt.colorbar(label='Distance')
plt.title("SOM U-Matrix (Distance Map) with Competitive Learning")

# Overlay data points on the SOM map
for i, x in enumerate(X):
    winner = som.winner(x)
    plt.plot(winner[0]+0.5, winner[1]+0.5, 'o', markerfacecolor='None',
             markeredgecolor='black', markersize=8, markeredgewidth=1)

plt.grid(True)
plt.show()
