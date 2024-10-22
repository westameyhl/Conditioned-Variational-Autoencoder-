# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:30:02 2023

@author: westa
"""

import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the activation function
def active(x):
    # relu
    return np.maximum(0, x)

# Define the derivative
def active_derivative(x):
    # relu
    return np.where(x > 0, 1, 0)


# Define a simple feedforward neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, input_data):
        # Forward pass
        self.input_data = input_data
        self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = active(self.hidden_input)
        self.output = active(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output

    def backward(self, target, learning_rate):
        # Backpropagation
        error = target - self.output
        d_output = error * active_derivative(self.output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * active_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += self.input_data.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, input_data, target, learning_rate, epochs):
        for _ in range(epochs):
            output = self.forward(input_data)
            self.backward(target, learning_rate)

    def predict(self, input_data):
        return self.forward(input_data)

# Example usage
if __name__ == '__main__':
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 4
    output_size = 1
    learning_rate = 0.2

    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    print("Training for 10000 epochs...")
    neural_network.train(input_data, target, learning_rate, 10000)

    print("Predictions after training:")
    for i in range(len(input_data)):
        prediction = neural_network.predict(input_data[i])
        print(f"Input: {input_data[i]}, Prediction: {prediction}")