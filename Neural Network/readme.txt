Backup

import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import vertical_data


class layer:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons) # initializes a layer of given size with random values
        self.biases  = np.zeros((1, n_neurons)) # sets the biases of the layer to 0 by default
    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases # takes the dot product of the inputs with weights and adds it to biases

class forwardActivationFunction: #Same activation function for the entire layer, can be used to tune to each neuron but in this case it works on the whole layer?
    def ReLU(self, input):
        self.ReLUOutput = np.maximum(0, input)

    def linear(self, input):
        self.output = input

    def softMax(self, input):
        exp_values = np.exp(input - np.max(input, axis=0, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = prob

    def softMax_Batch(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.softMax_BatchOutput = prob

    def step(self, input):
        self.output = np.maximum(0, input)
        for i in range(len(self.output)):
            self.output[i] = np.choose(self.output > 0, [0, 1])

    def sigmoid(self, input): # does not work for batch (probably!)
        self.output = input
        for i in range(len(input)):
            self.output[i] = 1/(1 + pow(2.718, -input[i]))

"""
class Loss: #need to clip values for log
    def CCE(self, input, target_input):
        self.output = np.sum(-np.log(input) * target_input, axis=0, keepdims=True)

    def CCE_Batch(self, input, target_input):
        self.output = np.sum(-np.log(input) * target_input, axis=1, keepdims=True)

    def CCE_book(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]

        elif (len(y_true.shape) == 2):
            correct_confidences = np.sum(y_pred_clipped * y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        self.output =  negative_log_likelihoods
"""

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Accuracy:
    def acc(self, input, target_input):
        predictions = np.argmax(input, axis=0)
        self.output = np.mean(predictions==target_input)

    def acc_batch(self, input, target_input):
        predictions = np.argmax(input, axis=1)
        self.output = np.mean(predictions==target_input)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


X, y = vertical_data(samples=100, classes=3) # X has the x and y coordinates and Y is the number of each
#print("X: ", X) # input data to the network
#print("Y: ", y) # output data to train the network
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

input_layer = layer(2, 3) # 2 inputs 3 neurons
layer1 = layer(3, 3)

activation1 = forwardActivationFunction()
activation2 = forwardActivationFunction()

#loss_function = Loss()
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 999
best_inputLayer_weight = input_layer.weights.copy()
best_inputLayer_bias = input_layer.biases.copy()
best_Layer1_weight = layer1.weights.copy()
best_Layer1_bias = layer1.biases.copy()

input_layer.weights = 0.05 * np.random.randn(2, 3)
input_layer.biases = 0.05 * np.random.randn(1, 3)
layer1.weights = 0.05 * np.random.randn(3, 3)
layer1.biases = 0.05 * np.random.randn(1, 3)

for i in range(1000000):

    input_layer.forward(X)
    #print("1", input_layer.output, "0")
    activation1.ReLU(input_layer.output)
    layer1.forward(activation1.ReLUOutput)
    activation2.softMax_Batch(layer1.output)

    #loss_function.CCE_book(activation2.softMax_BatchOutput, y)
    loss = loss_function.calculate(activation2.softMax_BatchOutput, y)
    #print("Loss:", loss)

    predictions = np.argmax(activation2.softMax_BatchOutput, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print("Better set found: ", i, "Loss: ", loss, "Accuracy", accuracy)
        best_inputLayer_weight = input_layer.weights.copy()
        best_inputLayer_bias = input_layer.biases.copy()
        best_Layer1_weight = layer1.weights.copy()
        best_Layer1_bias = layer1.biases.copy()
        lowest_loss = loss

    input_layer.weights = best_inputLayer_weight + np.random.randn(2, 3)%10
    input_layer.biases = best_inputLayer_bias + np.random.randn(1, 3)%10
    layer1.weights = best_Layer1_weight + np.random.randn(3, 3)%10
    layer1.biases = best_Layer1_bias + np.random.randn(1, 3)%10

print("Lowest Loss: ", lowest_loss, file=open("output.txt", "a"))

print("w1: ", best_inputLayer_weight, file=open("output.txt", "a"))
print("b1: ", best_inputLayer_bias, file=open("output.txt", "a"))
print("w2: ", best_Layer1_weight, file=open("output.txt", "a"))
print("b2: ", best_Layer1_bias, file=open("output.txt", "a"))

print("END\n", file=open("output.txt", "a"))

print("Lowest Loss: ", lowest_loss)

print("w1: ", best_inputLayer_weight)
print("b1: ", best_inputLayer_bias)
print("w2: ", best_Layer1_weight)
print("b2: ", best_Layer1_bias)

l1 = layer(2, 3)
l2 = layer(3, 3)

a1 = forwardActivationFunction()
a2 = forwardActivationFunction()

l1.weights = best_inputLayer_weight
l1.biases = best_inputLayer_bias
l2.weights = best_Layer1_weight
l2.biases = best_Layer1_bias

l1.forward(X)
a1.ReLU(l1.output)
l2.forward(a1.ReLUOutput)
a2.softMax_Batch(l2.output)

op = a2.softMax_BatchOutput

plt.scatter(X[:, 0], X[:, 1], c=op, s=40, cmap='brg')
plt.show()

print("END")

