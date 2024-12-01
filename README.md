# Neural Network From Scratch

## Overview
This project implements a neural network from scratch using Python, demonstrating various layers, activation functions, and optimizers. It is designed to handle tasks such as image recognition and classification through a custom-built architecture.

## Features
- **Custom Neural Network Layers**: Includes fully connected layers with forward and backward propagation.
- **Activation Functions**: Implements ReLU, Softmax, and Sigmoid functions for non-linear transformations.
- **Loss Functions**: Supports Categorical Crossentropy for multi-class classification tasks.
- **Optimizers**: Features various optimization algorithms including SGD, Adam, RMSprop, and Adagrad.
- **Data Generation**: Capable of generating spiral data for testing the model.
- **Image Recognition**: Processes images to recognize characters using a trained model.

## Requirements
To run this project, ensure you have the following Python packages installed:
- `numpy`
- `matplotlib`
- `PIL` (Pillow)
- `opencv-python`
- `pandas`
- `tensorflow` (if using TensorFlow functionalities)

You can install the required packages using pip:
```bash
pip install numpy matplotlib pillow opencv-python pandas tensorflow
```

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/yourproject.git](https://github.com/RidheeshAmarthya/Neural_Network.git)
   cd Neural_Network
   ```
2. Install the required dependencies as mentioned above.

## Usage
To train the model on spiral data:
```python
python main.py
```

To run character recognition:
```python
python char_recognition.py
```

### Example Code Snippet
Here’s a brief example of how to define a layer and perform a forward pass:
```python
import numpy as np

class Layer:
    def __init__(self, n_input, n_neurons):
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases

# Usage
layer1 = Layer(2, 64)
layer1.forward(np.array([[1, 2], [3, 4]]))
print(layer1.output)
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or features you would like to add.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project is inspired by various machine learning resources and tutorials. Special thanks to the community for their contributions to open-source projects.
