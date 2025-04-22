import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate random data (like before)
np.random.seed(0)
X = np.random.multivariate_normal([0, 0], [[3, 1], [1, 2]], 500)

# Apply PCA (n_components=1 to get the first principal component)
pca = PCA(n_components=1)
pca.fit(X)

# Get the principal component (weight direction)
w = pca.components_.flatten()

# Plot data and the learned principal component
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Input Data')
plt.quiver(0, 0, w[0], w[1], color='red', scale=3, label='Learned PC (PCA)')
plt.title("PCA for Principal Component Extraction")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
