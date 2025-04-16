import matplotlib.pyplot as plt
import skfuzzy as fuzzy
import numpy as np

data = np.array([
    [1,2],
    [2,1],
    [3,4],
    [5,7],
    [3,5],
    [8,9],
    [9,8],
    [10,10]
]).T

n_clusters = 2

coordinates, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
    data, c = n_clusters, m = 2, error = 0.005, maxiter = 1000, init = None
)

cluster_membership = np.argmax(u, axis = 0)

colors = ['r','b']
for i in range(n_clusters):
    plt.scatter(data[0, cluster_membership == i],
                data[1, cluster_membership == i],
                label = f'Cluster {i + 1}', c = colors[i])
    
plt.scatter(coordinates[:,0],coordinates[:,1], marker = 'x', s = 200, c = 'black', label = 'Centers')
plt.legend()
plt.title('Fuzzy C-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
    



