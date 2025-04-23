import autokeras as ak
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import classification_report, accuracy_score
#pip install autokeras
# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Run AutoKeras for each optimizer
optimizers = ['adam', 'rmsprop', 'adagrad']

for opt_name in optimizers:
    print(f"\n🔧 Training AutoKeras ImageClassifier with {opt_name.upper()} optimizer")

    # AutoKeras handles architecture search
    clf = ak.ImageClassifier(max_trials=1, overwrite=True)

    # Fit classifier
    clf.fit(x_train, y_train, epochs=5)

    # Export model and change optimizer
    model = clf.export_model()
    model.compile(optimizer=opt_name, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Re-train for fairness with the chosen optimizer
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)

    # Predict and evaluate
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"{opt_name.upper()} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))
