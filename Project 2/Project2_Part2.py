from __future__ import absolute_import, division, print_function, unicode_literals

#!/usr/bin/env python
# coding: utf-8

# Import libraries
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import util_mnist_reader as mnist_reader
import tensorflow as tf
import matplotlib.pyplot as plt

# Function that models the multi-layer perceptron
def MLP_Neural_Network(train_images, train_labels, validation_images, validation_labels, test_images, y_labels):
    #Normalization and resizing
    train_images = train_images.reshape(60000, 28, 28)
    test_images = test_images.reshape(5000, 28, 28)
    validation_images = validation_images.reshape(5000, 28, 28)
    validation_labels = validation_labels.reshape(5000, 1)
    train_labels = train_labels.reshape(60000, 1)
    test_labels = y_labels.reshape(5000, 1)
    validation_labels = to_categorical(validation_labels, 10)
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # plt.figure()
    # plt.imshow(train_images[1])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    train_images = train_images / 255.0
    validation_images = validation_images / 255.0
    test_images = test_images / 255.0

    # Build the MLP
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    n_epoch = 23
    train_loss = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), epochs=n_epoch)

    # Plot the training loss and the validation loss versus the number of epochs
    fig, ax = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss versus Epoch')
    ax.plot(range(n_epoch), train_loss.history['loss'], 'r', label='Training Dataset')
    ax.plot(range(n_epoch), train_loss.history['val_loss'], 'b', label='Validation Dataset')
    legend = ax.legend(loc='upper center', shadow=False, fontsize='large')
    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    predictions = model.predict(test_images)
    np_predictions = np.argmax(predictions, axis=1)
    
    # print confusion matrix and classification report
    print(confusion_matrix(np_predictions, y_labels))
    print(classification_report(np_predictions, y_labels))
