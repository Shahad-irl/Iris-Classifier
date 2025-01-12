# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:06:55 2025

@author: shahad-irl
"""

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#loading the iris dataset
iris = load_iris()
#features being the sepal & petal length
x = iris.data
#Labels
y = iris.target.reshape(-1, 1)


#One-hot encode the labels 
encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()

# Standardize the dataset
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Sigmoid activation function and its derivative
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Initialize weights for a simple three-layer network
np.random.seed(1)
input_size = x_train.shape[1]
hidden_size = 5  # Hidden layer with 5 neurons
output_size = y_train.shape[1]  # Number of classes (3 for Iris dataset)

# Randomly initialize weights
weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

# Training loop
epochs = 5000
learning_rate = 0.01

for epoch in range(epochs):
    # Forward propagation
    l0 = x_train
    l1 = sigmoid(np.dot(l0, weights_input_hidden))
    l2 = sigmoid(np.dot(l1, weights_hidden_output))

    # Compute error
    l2_error = y_train - l2

    # Backpropagation
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(weights_hidden_output.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # Update weights
    weights_hidden_output += learning_rate * l1.T.dot(l2_delta)
    weights_input_hidden += learning_rate * l0.T.dot(l1_delta)

    # Print error every 1000 epochs
    if (epoch % 1000) == 0:
        print(f"Error after {epoch} epochs: {np.mean(np.abs(l2_error))}")

# Testing
l0_test = x_test
l1_test = sigmoid(np.dot(l0_test, weights_input_hidden))
l2_test = sigmoid(np.dot(l1_test, weights_hidden_output))

# Convert predictions to class labels
predictions = np.argmax(l2_test, axis=1)
actual = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print(f"Test Accuracy: {accuracy:.2f}%")