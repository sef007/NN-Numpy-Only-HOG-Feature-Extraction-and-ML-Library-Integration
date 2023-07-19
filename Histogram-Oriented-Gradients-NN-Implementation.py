import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Fetch and load the MNIST dataset
mnist = pd.read_csv('train.csv')

y = mnist['label']
X = mnist.iloc[:, 1:].values  # Convert to numpy array

# Perform HOG feature extraction
list_hog = []
print("Extracting Features\nHistogram of Oriented Gradients -")
for feature in tqdm(X):
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    list_hog.append(fd)
print("'HOG' Feature Extraction Complete\n")
hog_features = np.array(list_hog, dtype='float64')

# Scale the HOG features
preProcess = preprocessing.MaxAbsScaler().fit(hog_features)
hog_features_transformed = preProcess.transform(hog_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features_transformed, y, test_size=0.2, random_state=0)

# Neural Network Training
m_train, _ = X_train.shape


def init_params():
    """
    Initialise the parameters of the neural network

    Returns:
        tuple: Initialised weights and biases.
    """
    W1 = np.random.rand(20, hog_features_transformed.shape[1]) - 0.5
    b1 = np.random.rand(20, 1) - 0.5
    W2 = np.random.rand(10, 20) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    """
    applies the Rectified Linear Unit (ReLU) activation function element-wise to the input Z.

    Args:
        Z (numpy.ndarray): Input to the activation function

    Returns:
        numpy.ndarray: Result of applying the ReLU activation function to Z
    """
    return np.maximum(Z, 0)


def softmax(Z):
    """
    applies the softmax activation function to the input Z

    Args:
        Z (numpy.ndarray): Input to the activation function

    Returns:
        numpy.ndarray: Result of applying the softmax activation function to Z
    """
    max_Z = np.max(Z, axis=0)
    A = np.exp(Z - max_Z) / np.sum(np.exp(Z - max_Z), axis=0)
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    """
    performs forward propagation through the neural network

    Args:
        W1 (numpy.ndarray): Weights for the first layer.
        b1 (numpy.ndarray): Biases for the first layer.
        W2 (numpy.ndarray): Weights for the second layer.
        b2 (numpy.ndarray): Biases for the second layer.
        W3 (numpy.ndarray): Weights for the third layer.
        b3 (numpy.ndarray): Biases for the third layer.
        X (numpy.ndarray): Input data.

    Returns:
        tuple: Outputs and activations of each layer
    """
    Z1 = W1.dot(X.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def ReLU_deriv(Z):
    """
    computes the derivative of the ReLU activation function with respect to the input Z

    Args:
        z (numpy.ndarray): Input to the activation function

    Returns:
        numpy.ndarray: Derivative of the ReLU activation function with respect to Z
    """
    return (Z > 0).astype(int)


def one_hot(Y):
    """
    converts the true labels to one-hot encoded form

    Args:
        Y (numpy.ndarray): True labels.

    Returns:
        numpy.ndarray: One-hot encoded labels.
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    
    one_hot_Y = one_hot(Y)

    # Layer 1:
    dZ3 = A3 - one_hot_Y
    dW3 = (1 / m_train) * dZ3.dot(A2.T)
    db3 = (1 / m_train) * np.sum(dZ3, axis=1, keepdims=True)

    # Layer 2
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = (1 / m_train) * dZ2.dot(A1.T)
    db2 = (1 / m_train) * np.sum(dZ2, axis=1, keepdims=True)

    # Layer 3:
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m_train) * dZ1.dot(X)
    db1 = (1 / m_train) * np.sum(dZ1, axis=1, keepdims=True)

    # Gradient Clipping
    clip_value = 5  # Set the maximum value to clip the gradients
    for dparam in [dW1, db1, dW2, db2, dW3, db3]:
        np.clip(dparam, -clip_value, clip_value, out=dparam)

    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    """
        updates the parameters of the NN and implements alpha (the learning rate)
    """
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    """
    computes the predicted labels based on the output activations

    Args:
        A3 (numpy.ndarray): Output activations of the neural network

    Returns:
        numpy.ndarray: Predicted labels
    """
    return np.argmax(A3, axis=0)


def get_accuracy(predictions, Y):
    """
    computes the accuracy of the predictions compared to the true labels

    Args:
        predictions (numpy.ndarray): Predicted labels.
        Y (numpy.ndarray): True labels.

    Returns:
        float: Accuracy of the predictions
    """
    return np.mean(predictions == Y)


def gradient_descent(X, Y, alpha, iterations):
    """
    Performs gradient descent to train the neural network.

    Args:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): True labels.
        alpha (float): Learning rate.
        iterations (int): Number of iterations.

    Returns:
        tuple: Updated weights and biases.
    """
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i}, Accuracy: {accuracy}")

    return W1, b1, W2, b2, W3, b3


# Training The Model:
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 0.3, 500)

# Final Training Accuracy:
Z1_train, A1_train, Z2_train, A2_train, Z3_train, A3_train = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
train_predictions = get_predictions(A3_train)
train_accuracy = get_accuracy(train_predictions, y_train)
print("\nFinal Training Accuracy:", round(train_accuracy, 3) * 100, "%")

# Final Test Accuracy (Test For Overfitting):
Z1_test, A1_test, Z2_test, A2_test, Z3_test, A3_test = forward_prop(W1, b1, W2, b2, W3, b3, X_test)
test_predictions = get_predictions(A3_test)
test_accuracy = get_accuracy(test_predictions, y_test)
print("Resulting Test Accuracy:", round(test_accuracy, 3) * 100, "%")
