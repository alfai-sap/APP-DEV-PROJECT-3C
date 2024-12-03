# train_emnist_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import gzip

def load_emnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28, 1)

def load_emnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_emnist():
    x_train = load_emnist_images('emnist-balanced-train-images-idx3-ubyte.gz')
    y_train = load_emnist_labels('emnist-balanced-train-labels-idx1-ubyte.gz')
    x_test = load_emnist_images('emnist-balanced-test-images-idx3-ubyte.gz')
    y_test = load_emnist_labels('emnist-balanced-test-labels-idx1-ubyte.gz')
    return (x_train, y_train), (x_test, y_test)

def create_and_save_model():
    # Load EMNIST dataset
    (x_train, y_train), (x_test, y_test) = load_emnist()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(47, activation='softmax')  # 47 classes for EMNIST Balanced
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    # Save
    model.save('emnist_model.h5')

if __name__ == "__main__":
    create_and_save_model()