import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Define Gaussian RBF function
def gaussian_rbf(x, centers, sigma):
    return np.exp(-np.linalg.norm(x[:, None] - centers, axis=2)**2 / (2 * sigma**2))

# Train RBF network
def train_rbf(X, y, n_centers, sigma, alpha):
    # Step 1: Determine RBF centers using k-means clustering
    kmeans = KMeans(n_clusters=n_centers, random_state=42)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    
    # Step 2: Transform input data using RBFs
    rbf_outputs = gaussian_rbf(X, centers, sigma)
    
    # Step 3: Train output layer using Ridge regression (with regularization parameter alpha)
    ridge = Ridge(alpha=alpha)
    ridge.fit(rbf_outputs, y)
    
    return centers, ridge.coef_, ridge.intercept_, sigma

# Predict using trained RBF network
def predict_rbf(X, centers, weights, bias, sigma):
    rbf_outputs = gaussian_rbf(X, centers, sigma)
    return rbf_outputs @ weights + bias

# Experiment with different regularization parameters (alpha)
alphas = [0.01, 0.1, 1.0, 10.0]
errors = []

for alpha in alphas:
    # Train the RBF network
    centers, weights, bias, sigma = train_rbf(X, y, n_centers=2, sigma=1.0, alpha=alpha)
    
    # Predict on training data
    y_pred = predict_rbf(X, centers, weights, bias, sigma)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y, y_pred)
    errors.append(mse)

# Plot MSE vs Regularization Parameter
plt.figure(figsize=(8, 6))
plt.plot(alphas, errors, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter (alpha)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Effect of Regularization Parameter on MSE')
plt.grid(True)
plt.show()
