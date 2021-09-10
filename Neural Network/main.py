import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import vertical_data

class layer:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons) # initializes a layer of given size with random values
        self.biases  = np.zeros((1, n_neurons)) # sets the biases of the layer to 0 by default
    def forward(self, input):

        self.inputs = input
        self.output = np.dot(input, self.weights) + self.biases # takes the dot product of the inputs with weights and adds it to biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

class ActivationFunction: #Same activation function for the entire layer, can be used to tune to each neuron but in this case it works on the whole layer?
    def ReLU(self, input):
        self.inputs = input
        self.ReLUOutput = np.maximum(0, input)

    def ReLU_backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

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

    def backward_softMax(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):

            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

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

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)


    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Accuracy:
    def acc(self, input, target_input):
        predictions = np.argmax(input, axis=0)
        self.output = np.mean(predictions==target_input)

    def acc_batch(self, input, target_input):
        predictions = np.argmax(input, axis=1)
        self.output = np.mean(predictions==target_input)

class SGD_Optimizer:
    def __init__(self, learningRate=1.0):
        self.learning_rate = learningRate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class SGD_Momentum:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
                                 layer.weight_momentums + \
                                 (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
                               layer.bias_momentums + \
                               (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) +
                             self.epsilon)
        layer.biases += -self.current_learning_rate * \
                         bias_momentums_corrected / \
                         (np.sqrt(bias_cache_corrected) +
                             self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

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

X, y = spiral_data(100, 3)

dense1 = layer(2, 64)
dense2 = layer(64, 3)

activation1 = ActivationFunction()

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy() #?

accuracy = Accuracy()

training = False
if (training == True):
    sgd = SGD_Optimizer(1)
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

    for epoch in range(10001):

        #Forward Pass
        dense1.forward(X)
        activation1.ReLU(dense1.output)
        dense2.forward(activation1.ReLUOutput)

        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)

        accuracy.acc_batch(loss_activation.output, y)

        print("Epoch: ", epoch)
        #print(loss_activation.output[:5])
        print('Loss: ', loss)
        print("Accuracy: ", accuracy.output, "\n")


        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.ReLU_backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # sgd.update_params(dense1)
        # sgd.update_params(dense2)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

        X, y = spiral_data(100, 3) #Data refresh

    print("Loss: ", loss , file=open("output.txt", "a"))
    print("Accuracy: ", accuracy , file=open("output.txt", "a"))

    print("w1: ", dense1.weights, file=open("output.txt", "a"))
    print("b1: ", dense1.biases, file=open("output.txt", "a"))
    print("w2: ", dense2.weights, file=open("output.txt", "a"))
    print("b2: ", dense2.biases, file=open("output.txt", "a"))

else:
    dense1.weights = [[ -1.57595237,   2.95128169,   0.21637163,   0.1657238,    0.22763541,
   -7.65646695,   0.22306632,   3.83556337,   0.22158385,   5.58941069,
   -8.92643246,  -6.08413303,  -8.58318249,   0.19936783,   1.49717931,
    4.51786927,  -5.31586772,   0.20462804,  -4.58976883,   3.14857197,
    0.09446582,  -0.1165431 ,   4.41410028,  -8.47925013,   5.79318586,
    0.40457185,   0.06401995,   0.1532525 ,   2.96668147,  -4.23047334,
   -9.63387669, -13.11309413,  -1.57986673,   5.11074176,  -6.02281528,
   -0.04435747,   0.22694682, -12.91799168,   2.34441923,  -1.38128299,
    9.44259071,  -6.40096286,  -8.52245862,  10.13598578,   0.16745687,
   -1.57659045,   4.26145821,  -0.12012533,  -7.49153901,   0.15454769,
   -5.42234035,   1.15716443,  -0.17554996,  -4.89841467,   0.22558315,
   -0.21795542, -11.36113422,  -5.89079937,   2.65717203,   9.05893833,
    1.48400745, 12.80983223 , -0.01827072 ,  4.93695043],
 [ 12.11976517,  11.58915925,  -0.18299038,   0.13232309,   0.18828353,
    0.12591016,   0.04377872,  -9.8740213 ,   0.20613865,  -4.2334755,
    4.11303138,   0.87297083,   4.15691564,   0.06104006,  -6.30710007,
   -0.44181073,   7.63662608,   0.10794529,   4.07194996, -14.73278452,
   -0.12203185,   0.22309619,  -1.15675565,  -2.22415246, -10.82877561,
    0.11250924,  -0.10018341,  -0.07482339,  -6.16368453,   3.76376772,
    6.37533622,   4.77994591,   1.12678176, -11.87161767,   5.50721798,
    0.07609438,   0.16667815,   5.38454176,   5.83085882,   1.12360468,
    4.40595962,   3.82266775,   2.42633645,   8.57606009,  -0.06988527,
    1.12189443,  13.84455295,  0.05985586,  0.11862843,  -0.17619507,
    7.5656186,  12.26494657, -0.16651588,  -4.10465127,  -0.12650059,
   -0.1705855,  -4.88964496,   3.74101994,  -5.63117113,  10.05641559,
   -6.36080017,   2.11517854,  -0.16941355,  -6.53897629]]
    dense1.biases = [[ 0.3292159,  -0.47921437, -0.37074264, -0.39938633, -0.38689495,  0.67971414,
  -0.35553591, -5.65840801, -0.34906729, -0.27405583,  1.66740847, -3.44478876,
   1.50897782, -0.41539714,  3.81837031, -2.2652391,   0.64244106, -0.41667419,
   0.5525733,  2.76310495, -0.38437679, -0.40146323,  1.66635208, -5.52577804,
   4.74257421, -0.42175414, -0.40119917, -0.38246177,  3.04259012,  0.50899667,
  -2.97430554,  0.55244157,  1.53719815,  4.0713095,   4.75796877, -0.41912633,
  -0.37907614, -0.64790967,  2.58009666,  1.49494554, -4.62923603,  3.36130645,
  -3.28559876, -7.78321578, -0.42176464,  1.53393737,  2.72192004, -0.35948022,
   0.66402969, -0.42043231,  0.59612015,  3.93814778, -0.40889115, -1.22075656,
  -0.40753681, -0.40101498,  3.55738197,  3.28263242,  2.796565,    0.86499831,
   3.85128028,  1.37309816, -0.18878351, -1.9924873 ]]
    dense2.weights = [[ 1.57666774e+00, -1.52979224e+01,  4.51061273e+00],
 [-6.60414897e+00,  2.20453400e+01, -1.90804822e+00],
 [ 3.03336647e-01 ,-2.93158129e-01, -1.63536796e-01],
 [-4.28428969e-01, -8.28539060e-02,  1.82435879e-01],
 [-4.22339273e-01, -2.45950814e-02,  1.18958277e-01],
 [-4.57802183e+00,  3.30237630e+00,  2.64485485e+00],
 [-7.86399077e-03, -1.25613339e-01,  1.17348437e-01],
 [ 1.98734169e+01, -6.25429554e+01, -6.27060632e+00],
 [-4.12109818e-01, -5.83114867e-02,  1.88046847e-01],
 [-1.93247272e+00,  4.27151002e+00,  1.11412935e+00],
 [-5.33725583e+00,  1.86383582e+00,  3.92191582e+00],
 [-5.67525128e+01,  3.80468676e+00,  1.49876476e+01],
 [-5.13774165e+00,  2.67458191e+00,  3.00534813e+00],
 [-4.08294933e-01,  4.51030720e-02,  1.25242054e-01],
 [-3.24917698e+00,  7.69862394e+00,  1.17403368e+00],
 [ 7.97252963e+00, -8.79179399e+00, -2.45856661e+00],
 [ 4.36980647e+00,  8.08594319e-01, -5.60973211e+00],
 [ 2.89573267e-01, -2.93353578e-01, -1.66375359e-01],
 [ 2.50660029e+00, -6.52667681e-01, -3.29145048e+00],
 [-2.61443266e+00, -8.39032266e+00,  5.90728799e+00],
 [ 3.09379580e-02, -1.58925190e-01,  3.62200489e-01],
 [-1.13508751e-01, -1.46578687e-01,  2.09060736e-01],
 [-2.24792058e+00,  2.71283762e+00,  1.00775769e+00],
 [ 3.24451119e+01, -3.17921237e+01,  2.50977570e+00],
 [ 1.14399069e+00,  4.79754508e+00, -5.42080227e+00],
 [ 1.52227926e-01, -4.07195149e-01,  1.90942541e-01],
 [ 1.95042741e-01, -2.48823976e-01,  4.32567378e-01],
 [-1.71897547e-01, -2.37263057e-01,  3.46811175e-01],
 [-3.14588855e+00,  6.52487866e+00,  1.48569558e+00],
 [ 2.02337023e+00,-2.09264524e-01 ,-3.27175813e+00],
 [ 1.34034567e+01, -9.11151759e+00, -5.84578761e+00],
 [ 5.02258090e+00,  4.29822897e+00, -7.94964480e+00],
 [ 3.94156970e-01, -2.58204558e+00,  1.08232564e+00],
 [ 5.91213520e+00, -8.24624781e+00, -3.20587775e+00],
 [ 4.91630228e+00, -4.98111079e+00, -2.48240689e+00],
 [ 2.65400001e-01, -3.09061948e-01,  3.74419801e-01],
 [-3.51532221e-01 , 1.79748463e-01, -9.80820934e-02],
 [-1.09725770e+01 , 1.10184745e+00,  9.32135082e+00],
 [-2.52946911e+00,  1.39035303e+00,  3.10224260e+00],
 [ 2.35075519e-01, -2.42796081e+00,  1.11151236e+00],
 [ 3.84242880e+00, -2.96108531e+01,  7.25198925e+00],
 [ 3.43841867e+00, -4.03894500e+00, -1.05740949e+00],
 [-1.76931474e+01,  2.09132648e+01, -4.61862723e+00],
 [-1.42782856e+01,  3.03117258e+01, -2.47246979e+01],
 [ 9.05636734e-02, -5.05263589e-02, -3.21115327e-01],
 [ 3.44694734e-01, -2.54978797e+00,  1.08910279e+00],
 [-3.23370632e+00 , 9.05675418e+00, -2.65051744e+00],
 [-3.64894777e-01, -2.44700290e-01,  2.81415519e-01],
 [-4.65108464e+00,  2.79059663e+00,  2.77838506e+00],
 [ 2.42544687e-01,  1.28570832e-01, -3.12329676e-01],
 [ 4.68341737e+00 ,-2.35008543e-02, -5.29901584e+00],
 [-5.76053939e+00,  3.90175499e+00,  5.69921216e+00],
 [ 1.86987940e-01, -1.89364135e-01,  2.09143728e-01],
 [ 2.28777047e+00,  1.14878647e+01, -5.91057184e+00],
 [ 2.63417832e-01, -2.59092036e-01, -2.01640844e-01],
 [ 2.67728529e-01, -2.51637685e-01,  2.69197999e-01],
 [ 4.93640426e+00, -6.48680585e+00, -3.42313781e-01],
 [ 3.25762390e+00, -4.49523293e+00, -6.67132586e-01],
 [-2.62175478e+00,  6.24765779e+00,  1.25034039e+00],
 [ 5.49770516e+00, -1.20174359e+01, -4.10567500e-01],
 [-3.25498408e+00,  7.79262306e+00,  1.10670212e+00],
 [ 6.62860576e+00, -1.14817811e+01, -1.79055419e+00],
 [-1.21357690e-01, -4.18301037e-02,  3.93518197e-01],
 [-8.15527557e+00,  1.33965660e+01,  4.97928418e+00]]
    dense2.biases = [[-0.38054341, -1.33520703,  1.25093594]]

    dense1.forward(X)
    activation1.ReLU(dense1.output)
    dense2.forward(activation1.ReLUOutput)

    loss = loss_activation.forward(dense2.output, y)
    accuracy.acc_batch(loss_activation.output, y)

    print('Loss: ', loss)
    print("Accuracy: ", accuracy.output, "\n")

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

output = np.empty(300)
i = 0
while i < 300:
     output[i] = np.argmax(loss_activation.output[i])
     i += 1

plt.scatter(X[:, 0], X[:, 1], c=output, s=40, cmap='brg')
plt.show()