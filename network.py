from dense import Dense
from activation import Activation
import numpy as np
# Definition of the multilayer perceptron network
    # Parameters:
        #  1. Integer Value: Number of features in training set
    # Placeholders:
        # 1. List to hold the layers of the network, each element is a Dense() layer
class MultilayerPerceptron:
    def __init__(self, input_shape):
        print("Creating Network, got {} as input shape".format(input_shape))
        self.network = []  # Initially the layer is empty
        
        # There are n_features in training data so n_features input nodes
        # First hidden layer has 12 neurons, and n_features inputs
        self.network.append(Dense(12, input_shape))
        # Secon hidden layer has 6 neurons, and 12 inputs
        self.network.append(Dense(6,12))
        # Output layer has 2 neurons, and 6 inputs
        self.network.append(Dense(2,6))

        #previous outputs to use as inputs to next layer in network
        self.previous_layers_outputs = []
        print('Done, Network Created...')
    # Train the model
    def train(self, X_train, desired_output, learning_rate = 0.1, epochs = 100):
        print('Training the Network: input_data: {}, learning_rate: {}, epochs: {}'.format(X_train.shape, learning_rate, epochs) )
        while epochs >= 1: # While epochs are remaining
            print('Epochs: ', epochs)
            self.previous_layers_outputs = []

            for current_sample in X_train:
                self.previous_layers_outputs.append(current_sample)

                for layer in self.network:
                    # Take latest output feed it to next layer as input and store its result as latest output
                    self.previous_layers_outputs.append(layer.forward_signal(self.previous_layers_outputs[-1]))    

                # Error E(n) at end of the forward pass    
                error_at_end = 10 #np.sum(desired_output, self.previous_layers_outputs[-1]) ** 2

                # The Backward pass loop
                for index, layer in enumerate(self.network):
                    print("Backpass Loop:")
                    print("Error: ", error_at_end)
                    print("Layer: ", layer)
                    print("Network: ", self.network)
                    print("Previous Local Gradients: ", self.network[index + 1].get_gradients())
                    print("Previous Weights: ", self.network[index + 1].get_weights())
                    print("Previous Inputs: ",self.previous_layers_outputs[index])
                    
                    layer.backward_signal(error_at_end, layer, self.network, self.network[index + 1].get_gradients(), 
                    self.network[index + 1].get_weights(), learning_rate, self.previous_layers_outputs[index])
        epochs -= 1