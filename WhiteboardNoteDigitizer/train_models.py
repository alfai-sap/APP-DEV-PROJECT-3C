# train_models.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from emnist import extract_training_samples, extract_test_samples

def create_digit_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Create model
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save('digit_model.h5')

def create_letter_model():
    # Load EMNIST dataset using emnist package
    x_train, y_train = extract_training_samples('letters')
    x_test, y_test = extract_test_samples('letters')
    
    # Preprocess data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # Adjust labels to be 0-based
    y_train = y_train - 1
    y_test = y_test - 1

    # Create model 
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(26, activation='softmax')  # 26 letters
    ])

    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save('letter_model.h5')

if __name__ == "__main__":
    create_digit_model()
    create_letter_model()