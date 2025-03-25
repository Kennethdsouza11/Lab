import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)
y = np.array([0,0,0,1], dtype = np.float32)

hebb_weights = tf.Variable(tf.random.normal([2]))
for _ in range(100):
    for xi,yi in zip(X,y):
        hebb_weights.assign_add(0.1 * xi * yi)
        
cov_weights = tf.Variable(tf.random.normal([2]))
mean_x = tf.reduce_mean(X,axis = 0)
mean_y = tf.reduce_mean(y)
for _ in range(100):
    for xi,yi in zip(X,y):
        cov_weights.assign_add(0.1 * (xi - mean_x) * (yi - mean_y))
        
plt.figure(figsize = (10,4))
plt.subplot(121)
plt.scatter(X[:,0], X[:,1], c = y)
plt.title(f"Hebbian\nWeights : {hebb_weights.numpy().round(2)}")
plt.subplot(122)
plt.scatter(X[:,0], X[:,1], c = y)
plt.title(f"Covariance\nWeights : {cov_weights.numpy().round(2)}")
plt.show()
