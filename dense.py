# Dense Layer of the Network, Contains:
    # 1. Constructor (inputs_size, ):
        # 1. NumPy array: Initialization of weights of respective synaptic inputs of neurons
        # 2. NumPy array: Initialization of biases of respective outputs of neurons
        # 3. NumPy array: Storing the output of all neurons 'Zeros for now'
        # 3. Initialze an activation (Logistic Sigmoid)

    # 2. Method: Forward Signal (Inputs): 
        # 1. NumPy array: Calculation of respective activation potentials (linear combinations) of all neurons
        # 2. NumPy array: Adding respective biases to all activation potentials, assuming that input synaptic
        #                 signal for biases terms of all neurons is '1'
        # 3. NumPy array: Activating the vector from previous step, using 'logistic activation function'
        # 4. NumPy array: Storing the outputs in the outputs vector of this object
        # 5. NumPy array: Returning the output vector containing output of all neurons in the Dense Layer

    # 3. Method: Backward Signal (Not Sure Yet)

# Importing NumPy, will be using it for vector programming
import numpy as np
# Importing Activation.py
from activation import Activation
class Dense:
    # Definiation and Initialization of the Dense Layer
    def __init__(self, num_neurons, input_shape):
        print('Dense Layer: Outputs: {}, Inputs: {}'.format(num_neurons, input_shape))
        # Let's initialize the weights in interval [0,1) for respective synaptic inputs
        self.weights = np.random.uniform(low = 0, high = 1, size = input_shape)

        # Lets initialize the biases all with value '1' for every neurons' activation potential in self layer
        self.biases = np.ones(num_neurons)
        
        # Outputs of this layer
        self.outputs = np.zeros(num_neurons)

        # And finally the activation function, for non-linearity of outputs
        self.activation = Activation()

    # Forward Signal Definition
    def forward_signal(self, inputs):
        # Output Calculation: activate(input * weight + bias)
        outputs = self.activation.activate(inputs * self.weights + self.biases)
        return outputs
    
    # Backward Signal Definition
    def backward_signal(self,):
        # Let's wait for this, will do this later
        pass