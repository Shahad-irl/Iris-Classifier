
# Iris Dataset Neural Network Classifier

This repository contains a simple neural network implementation to classify the Iris dataset using Python and NumPy.

## Overview

The script demonstrates the implementation of a feedforward neural network from scratch to classify the Iris dataset. The key steps include:

1. Loading and preprocessing the Iris dataset.
2. One-hot encoding the target labels.
3. Normalizing the feature set.
4. Training a simple 3-layer neural network using forward and backpropagation.
5. Evaluating the model's accuracy on test data.

## Requirements

To run this script, you need to have the following libraries installed:

- **NumPy**: For numerical computations.
- **scikit-learn**: To load the Iris dataset and preprocess the data.

You can install these dependencies using pip:
```bash
pip install numpy scikit-learn
```

## Code Details

### Features

- **Dataset**: The Iris dataset consists of 150 samples of iris flowers, categorized into three species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*. Each sample has four features: sepal length, sepal width, petal length, and petal width.
- **Neural Network Structure**:
  - Input layer: 4 neurons (one for each feature).
  - Hidden layer: 5 neurons.
  - Output layer: 3 neurons (one for each class).

### Main Steps

1. **Load Dataset**:
   ```python
   from sklearn.datasets import load_iris
   iris = load_iris()
   x = iris.data
   y = iris.target.reshape(-1, 1)
   ```

2. **One-Hot Encoding**:
   ```python
   from sklearn.preprocessing import OneHotEncoder
   encoder = OneHotEncoder()
   y = encoder.fit_transform(y).toarray()
   ```

3. **Standardization**:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   x = scaler.fit_transform(x)
   ```

4. **Split Dataset**:
   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   ```

5. **Training**:
   - The weights are updated using gradient descent based on the sigmoid activation function and its derivative.
   - Training runs for 5000 epochs, with a learning rate of 0.01.
   - Error is printed every 1000 epochs.

6. **Testing and Accuracy Calculation**:
   - Predictions are made using the trained weights.
   - Accuracy is calculated by comparing predicted and actual labels.

### Output Example

During training, the error is printed every 1000 epochs. At the end of training, the test accuracy is displayed:
```
Error after 0 epochs: 0.482843
Error after 1000 epochs: 0.050871
Error after 2000 epochs: 0.034192
Error after 3000 epochs: 0.028287
Error after 4000 epochs: 0.024809
Test Accuracy: 96.67%
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/shahad-irl/iris-classifier.git
   cd iris-classifier
   ```

2. Run the script:
   ```bash
   python iris_classifier.py
   ```

## Author

**Shahad-irl**  
A computer engineering graduate passionate about artificial intelligence and machine learning.
