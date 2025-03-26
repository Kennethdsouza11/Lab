import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

np.random.seed(42)
cov = np.array([[2, 1.5], [1.5, 1]])
X = np.random.multivariate_normal(cov=cov, size=200, mean=[0, 0])
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=1)
pca.fit(X)
sklearn_pc = pca.components_[0]

hebbian_pca = MLPRegressor(
    hidden_layer_sizes=(),
    activation="identity",
    solver="sgd",
    learning_rate_init=0.1,
    random_state=42,
    max_iter=100,
)

hebbian_pca.fit(X, X)
hebbian_weights = hebbian_pca.coefs_[0].flatten()
hebbian_weights /= np.linalg.norm(hebbian_weights)

plt.figure(figsize=(10, 4))
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Input Data")

plt.quiver(
    0,
    0,
    sklearn_pc[0],
    sklearn_pc[1],
    color="r",
    scale=5,
    label=f"PC1 : [{sklearn_pc[0]:.3f}, {sklearn_pc[1]:.3f}]",
)

plt.quiver(
    0,
    0,
    hebbian_weights[0],
    hebbian_weights[1],
    color="g",
    linestyle = '--',
    scale=5,
    label=f"PC1 : [{hebbian_weights[0]:.3f}, {hebbian_weights[1]:.3f}]",
)

plt.title("PCA vs Hebbian Learning")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()
