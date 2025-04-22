from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzzy
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_T = X_scaled.T

n_clusters = 3

cntrd, u, u0, d, jm, pc, fpc = fuzzy.cluster.cmeans(
    X_scaled_T, c = n_clusters, m = 2, error = 0.005, maxiter = 1000, init = None
)

print('Enter values for the following features  :')
user_input = []
for name in feature_names:
    val = float(input(f"{name} : "))
    user_input.append(val)

user_input_scaled = scaler.transform([user_input])

u_user, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    user_input_scaled.T, cntrd, m = 2, error = 0.005, maxiter = 1000
)

print('Fuzzy Classification Result : ')
for i, prob in enumerate(u_user[:,0]):
    print(f'{target_names[i]} : {prob * 100:.2f}%')
