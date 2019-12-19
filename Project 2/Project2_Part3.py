from __future__ import absolute_import, division, print_function, unicode_literals

#!/usr/bin/env python
# coding: utf-8

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow import keras
import util_mnist_reader as mnist_reader
import keras.layers
import numpy as np
import tensorflow
import matplotlib.pyplot as plt


# Function that implements CNN
def CNN_Model(x_train, y_train, x_validation, y_validation, x_test, y_test):

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Data pre-processing
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    validation_images = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
    validation_labels = to_categorical(y_validation, 10)
    train_labels = to_categorical(y_train, 10)
    test_labels = to_categorical(y_test, 10)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    validation_images = validation_images.astype('float32') / 255

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.21))
    model.add(Flatten())
    model.add(Dense(144, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # fit model
    nepochs = 5
    history = model.fit(x_train, train_labels, epochs=nepochs, batch_size=36,
                        validation_data=(validation_images, validation_labels))

    # Plot the training loss and the validation loss versus the number of epochs
    fig, ax = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss versus Epoch')
    ax.plot(range(nepochs), history.history['loss'], 'r', label='Training Dataset')
    ax.plot(range(nepochs), history.history['val_loss'], 'b', label='Validation Dataset')
    legend = ax.legend(loc='upper center', shadow=False, fontsize='large')
    plt.show()

    # evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, test_labels, verbose=2)

    predictions = model.predict(x_test)
    np_predictions = np.argmax(predictions, axis=1)

    # print confusion matrix and classification report
    print(confusion_matrix(np_predictions, y_test))
    print(classification_report(np_predictions, y_test))
