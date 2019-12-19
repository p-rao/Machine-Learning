#!/usr/bin/env python
# coding: utf-8

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas
import numpy as np
import util_mnist_reader as mnist_reader
import matplotlib.pyplot as plt

# Function to calculate softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Function to calculate sigmoid
def sigmoid(z):
    return 1/(1 + np.exp(-1*z))

# Function to calculate loss
def compute_loss(Y, a):
    L_sum = np.sum(np.multiply(Y, np.log(a)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L

# Function that performs single hidden layer neural network
def Single_Hidden_Layer(train_images, train_labels, validation_images, validation_labels, test_images, test_labels):
    # Normalize the data
    train_images = train_images/255
    train_images = train_images.T
    test_images = test_images/255
    test_images = test_images.T
    validation_images = validation_images/255
    validation_images = validation_images.T

    digits = 10
    examples = train_labels.shape[0]
    train_labels = train_labels.reshape(1, examples)
    train_labels = np.eye(digits)[train_labels.astype('int32')]
    train_labels = train_labels.T.reshape(digits, examples)
    examples1 = validation_labels.shape[0]
    validation_labels = validation_labels.reshape(1, examples1)
    validation_labels = np.eye(digits)[validation_labels.astype('int32')]
    validation_labels = validation_labels.T.reshape(digits, examples1)

    lossarray = []
    lossarray_val = []

    n_x = train_images.shape[0]
    # Hyperparameters
    n_h = 64
    learning_rate = 1
    e = 1501
    m = 60000
    # Initialize the weights and the bias
    W_1 = np.random.randn(n_h, n_x)
    b_1 = np.zeros((n_h, 1))
    W_2 = np.random.randn(digits, n_h)
    b_2 = np.zeros((digits, 1))

    # Run for e number of epochs
    for i in range(e):
        # forward propagation
        # training
        Z_1 = np.dot(W_1, train_images) + b_1
        A_1 = sigmoid(Z_1)
        Z_2 = np.dot(W_2, A_1) + b_2
        A_2 = softmax(Z_2)
        cost = compute_loss(train_labels, A_2)
        lossarray.append(cost)

        # validation
        Z_1_val = np.dot(W_1, validation_images) + b_1
        A_1_val = sigmoid(Z_1_val)
        Z_2_val = np.dot(W_2, A_1_val) + b_2
        A_2_val = softmax(Z_2_val)
        cost_val = compute_loss(validation_labels, A_2_val)
        lossarray_val.append(cost_val)

        # backpropagation
        dZ2 = A_2-train_labels
        dW2 = (1/m) * np.matmul(dZ2, A_1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(W_2.T, dZ2)
        dZ1 = dA1 * sigmoid(Z_1) * (1 - sigmoid(Z_1))
        dW1 = (1/m) * np.matmul(dZ1, train_images.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # update the weights and bias
        W_2 = W_2 - learning_rate * dW2
        b_2 = b_2 - learning_rate * db2
        W_1 = W_1 - learning_rate * dW1
        b_1 = b_1 - learning_rate * db1
        if i % 100 == 0:
            print("Epoch : ", i, "cost : ", cost)
    print("Final Cost : ", cost)

    # predict the testing data
    Z_1 = np.matmul(W_1, test_images) + b_1
    A_1 = sigmoid(Z_1)
    Z_2 = np.matmul(W_2, A_1) + b_2
    A_2 = softmax(Z_2)
    predictions = np.argmax(A_2, axis=0)

    # print graph, confusion matrix and classification report
    fig, ax = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss versus Epoch')
    ax.plot(range(e), lossarray, 'r', label='Training Dataset')
    ax.plot(range(e), lossarray_val, 'b', label='Validation Dataset')
    legend = ax.legend(loc='upper center', shadow=False, fontsize='large')
    plt.show()
    print(confusion_matrix(predictions, test_labels))
    print(classification_report(predictions, test_labels))
