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
        print('Adding Layer: input_shape: {}, number of neurons: {}, output_shape: {}'.format(input_shape, num_neurons, num_neurons))
        # Let's initialize the weights in interval [0,1) for respective synaptic inputs
        self.weights = np.random.uniform(low = 0, high = 1, size = input_shape)

        # Lets initialize the biases all with value '1' for every neuron in current layer
        self.biases = np.ones(num_neurons)
        
        # Lets initialize the activation_potentials all with value '0' for every neuron in current layer
        self.activation_potentials = np.zeros(num_neurons)

        # Outputs of this layer
        self.outputs = np.zeros(num_neurons)

        # Local Gradients of all the neurons in current layer
        self.local_gradients = np.zeros(num_neurons)
        
        # And finally the activation function, for non-linearity of outputs
        self.activation = Activation()

        # Inputs to this layer -> Outputs from previous layer
        self.previous_layers_outputs = []
        print('Added Layer ... ')
    def get_gradients(self):
        return self.local_gradients
    
    def get_weights(self):
        return self.weights

    # The activation potential vector calculator for a given layer
    def activation_potential(self, inputs):
        self.activation_potentials = np.dot(inputs, self.weights) + self.biases
        
    #Local Gradient  
    def local_gradient(self, error_at_end, layer, network, next_gradients, next_weights):
        if layer == network[-1]:
            # Output layer = error * derivative of activation function
            self.local_gradients = np.dot(error_at_end, self.activation.activate_derivative(self.activation_potentials))
        else:
            # Hidden layers = derivative of activation function * sum of all (derivative of activation
            # function of next layer * weights associated with this neuron going to all neurons in next layer)
            diff_of_activation_funct = self.activation.activate_derivative(self.activation_potentials)
            self.local_gradients = np.dot(diff_of_activation_funct, np.dot(next_gradients, next_weights))
        
    # Forward Signal Definition
    def forward_signal(self, inputs):
        # Calculate activation potentials for all neurons in this layer
        self.activation_potential(inputs)

        # Activate activation_potentials and save it in self.outputs
        self.outputs = self.activation.activate(self.activation_potentials)

        # Return the outputs of this layer, will need it in next layer
        return self.outputs
    
    # Backward Signal Definition
    def backward_signal(self, error_at_end, layer, network, learning_rate, next_gradients, next_weights, inputs):
        # Calculate local_gradients
        self.local_gradient(error_at_end, layer, network, next_gradients, next_weights)
       
        # Update weights
        self.weights = np.sum(self.weights, learning_rate * np.dot(self.local_gradients, inputs))
        # Update Biases
        self.biases = np.sum(self.biases, learning_rate * np.dot(self.local_gradients, 1))