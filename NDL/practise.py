import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

np.random.seed(0)
X = np.random.rand(100,2)

som = MiniSom(x = 10, y = 10, input_len = 2, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration = 500)

plt.figure(figsize = (10,4))
plt.pcolor(som.distance_map().T, cmap = 'coolwarm')
plt.colorbar(label = 'Distance')
plt.title('SOM')

for i, x in enumerate(X):
    winner = som.winner(x)
    plt.plot(winner[0] + 0.5, winner[1]+0.5, 'o', markerfacecolor = 'None', markeredgecolor = 'black', markersize = 8, markeredgewidth = 1)
    
plt.grid(True)
plt.show()