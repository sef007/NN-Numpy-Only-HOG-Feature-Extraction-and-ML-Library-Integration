import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, total_features, hidden_size, output_size):
        """
        initialises a neural network with the specified number of features, hidden size, and output size.

        Args:
            total_features (int): total number of features in the input data.
            hidden_size (int): Num of hidden layers
            output_size (int): Num of neurons in the output layer.
        """
        self.total_features = total_features
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(hidden_size, total_features) - 0.5
        self.biases_input_hidden = np.random.rand(hidden_size, 1) - 0.5
        self.weights_hidden_output = np.random.rand(output_size, hidden_size) - 0.5
        self.biases_hidden_output = np.random.rand(output_size, 1) - 0.5

    @staticmethod
    def ReLU(Z):
        """
        applies the Rectified Linear Unit (ReLU) activation function element-wise to the input Z.

        Args:
            Z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Result of applying the ReLU activation function to Z
        """
        return np.maximum(Z, 0)

    @staticmethod
    def softmax(Z):
        """
        applies the softmax activation function to the input Z

        Args:
            Z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Result of applying the softmax activation function to Z
        """
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0)

    @staticmethod
    def ReLU_deriv(Z):
        """
        computes the derivative of the ReLU activation function with respect to the input Z

        Args:
            z (numpy.ndarray): Input to the activation function

        Returns:
            numpy.ndarray: Derivative of the ReLU activation function with respect to Z
        """
        return Z > 0

    def forward_propagation(self, X):
        """
        performs forward propagation through the neural network

        args:
            X (numpy.ndarray): Input data

        returns:
            tuple: Tuple containing the intermediate results of forward propagation
                   (Z1, A1, Z2, A2)
        """
        Z1 = self.weights_input_hidden.dot(X) + self.biases_input_hidden
        A1 = self.ReLU(Z1)
        Z2 = self.weights_hidden_output.dot(A1) + self.biases_hidden_output
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        """
        Performs backward propagation to compute the gradients of the parameters.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.
            Z1 (numpy.ndarray):  output of forward propagation (hidden layer pre-activation)
            A1 (numpy.ndarray): output result of forward propagation (hidden layer after activation)
            Z2 (numpy.ndarray): result of forward propagation (output layer pre-activation)
            A2 (numpy.ndarray):  result of forward propagation (output layer after activation) final

        Returns:
            tuple: Tuple containing the gradients of the parameters:
                   (dW1, db1, dW2, db2)
        """
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / X.shape[1] * dZ2.dot(A1.T)
        db2 = 1 / X.shape[1] * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.weights_hidden_output.T.dot(dZ2) * self.ReLU_deriv(Z1)
        dW1 = 1 / X.shape[1] * dZ1.dot(X.T)
        db1 = 1 / X.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        """
        Updates the parameters of the NN and implements alpha (the learning rate)

        Args:
            dW1 (numpy.ndarray): gradients of weights between input and hidden layers
            db1 (numpy.ndarray): gradients of biases in the hidden layer
            dW2 (numpy.ndarray): gradients of weights between hidden and output layers
            db2 (numpy.ndarray): gradients of biases in the output layer
            learning_rate (float): learning rate for gradient descent (alpha)
        """
        self.weights_input_hidden -= learning_rate * dW1
        self.biases_input_hidden -= learning_rate * db1
        self.weights_hidden_output -= learning_rate * dW2
        self.biases_hidden_output -= learning_rate * db2

    @staticmethod
    def one_hot(Y):
        """
        Converts the true labels to one-hot encoded form.

        Args:
            Y (numpy.ndarray): True labels.

        Returns:
            numpy.ndarray: One-hot encoded labels.
        """
        num_classes = np.max(Y) + 1
        m = Y.shape[0]
        one_hot_Y = np.zeros((num_classes, m))
        one_hot_Y[Y, np.arange(m)] = 1
        return one_hot_Y

    @staticmethod
    def get_predictions(A2):
        """
        Computes the predicted labels based on the output activations.

        Args:
            A2 (numpy.ndarray): Output activations of the neural network.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        return np.argmax(A2, axis=0)

    @staticmethod
    def get_accuracy(predictions, Y):
        """
        Computes the accuracy of the predictions compared to the true labels.

        Args:
            predictions (numpy.ndarray): Predicted labels.
            Y (numpy.ndarray): True labels.

        Returns:
            float: Accuracy of the predictions.
        """
        return np.mean(predictions == Y)

    def train(self, X_train, Y_train, X_dev, Y_dev, learning_rate=0.3, iterations=500):
        """
        Trains the neural network using the specified training data.

        Args:
            X_train (numpy.ndarray): Input training data.
            Y_train (numpy.ndarray): True labels for training data.
            X_dev (numpy.ndarray): Input validation data.
            Y_dev (numpy.ndarray): True labels for validation data.
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.3.
            iterations (int, optional): Number of iterations for training. Defaults to 500.
        """
        m_train = X_train.shape[1]
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X_train)
            dW1, db1, dW2, db2 = self.backward_propagation(X_train, Y_train, Z1, A1, Z2, A2)
            self.update_params(dW1, db1, dW2, db2, learning_rate)
            if i % 10 == 0:
                print("Iteration:", i)
                train_predictions = self.get_predictions(A2)
                train_accuracy = self.get_accuracy(train_predictions, Y_train)
                print("Training Accuracy:", train_accuracy)
        print("\nFinal Training Accuracy:", self.evaluate(X_train, Y_train))
        print("Test Accuracy:", self.evaluate(X_dev, Y_dev))

    def evaluate(self, X, Y):
        """
        Evaluates the neural network on the given data and computes the accuracy.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): True labels.

        Returns:
            float: Accuracy of the neural network on the given data.
        """
        _, _, _, output_A2 = self.forward_propagation(X)
        predictions = self.get_predictions(output_A2)
        accuracy = self.get_accuracy(predictions, Y)
        return accuracy


# load and preprocess data
data = pd.read_csv('train.csv')
data = np.array(data)
np.random.shuffle(data)
# prep testing data
data_dev = data[0:1000].T
Y_dev = data_dev[0]  # gets the labels
X_dev = data_dev[1:] / 255.
# prep training data
data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:] / 255.

# TRain the Neural Network:
# 41,000 data instances in the training set, and each instance has 784 features.
total_features, total_data_ins = X_train.shape
hidden_size = 10
output_size = 10  # there are 10 classes (10 digits)
nn = NeuralNetwork(total_features, hidden_size, output_size)
nn.train(X_train, Y_train, X_dev, Y_dev, learning_rate=0.3, iterations=500)

# Final Training Accuracy
train_predictions = nn.get_predictions(nn.forward_propagation(X_train)[-1])
train_accuracy = nn.get_accuracy(train_predictions, Y_train)
print("Final Training Accuracy:", round(train_accuracy * 100, 2), "%")

# Final Test Accuracy (Test For Overfitting)
dev_predictions = nn.get_predictions(nn.forward_propagation(X_dev)[-1])
dev_accuracy = nn.get_accuracy(dev_predictions, Y_dev)
print("Resulting Test Accuracy:", round(dev_accuracy * 100, 2), "%")

# Approximately 90% accuracy on Test Data.

# References
# Implemented after completing Sentdex Neural Network Course
# Code adapted based on Samson Zhang's notebook and video demonstration.
# written: July 2023
