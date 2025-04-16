#alexnet optimization

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Define LeNet architecture
def create_lenet():
    model = keras.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        keras.layers.Conv2D(16, kernel_size=5, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(84, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Define AlexNet architecture (adjusted for MNIST)
def create_alexnet():
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1), padding='same'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        keras.layers.Conv2D(192, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        keras.layers.Conv2D(384, kernel_size=3, activation='relu', padding='same'),
        keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train and evaluate model
def train_and_evaluate(model, name):
    print(f"\nTraining {name}...")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    return model

# Function to visualize filters
def visualize_filters(model, layer_name):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and layer.name == layer_name:
            filters, _ = layer.get_weights()
            fig, axes = plt.subplots(1, min(6, filters.shape[-1]))
            for i, ax in enumerate(axes):
                ax.imshow(filters[:, :, 0, i], cmap='gray')
                ax.axis('off')
            plt.show()
            break

# Train models
lenet_model = train_and_evaluate(create_lenet(), "LeNet")
alexnet_model = train_and_evaluate(create_alexnet(), "AlexNet")

# Visualize first convolutional layer filters
visualize_filters(lenet_model, 'conv2d')
visualize_filters(alexnet_model, 'conv2d_1')