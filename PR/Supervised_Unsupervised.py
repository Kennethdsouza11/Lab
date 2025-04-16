from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = LogisticRegression(max_iter = 10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Supervised Accuracy : ", accuracy_score(y_test, y_pred))


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters = 10, random_state = 42)
kmeans.fit(X)
fig, axes = plt.subplots(2, 5, figsize = (10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8,8), cmap = 'gray')
    ax.set_title(f"Cluster : {kmeans.labels_[i]}")
    ax.axis('off')
    
plt.show()
