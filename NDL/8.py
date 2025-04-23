from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use built-in library model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train (using fit)
model.fit(X_train_scaled, y_train)

# Test (using predict)
y_pred = model.predict(X_test_scaled)

# Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
