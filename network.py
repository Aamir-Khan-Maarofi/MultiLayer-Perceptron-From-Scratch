from dense import Dense
# Definition of the multilayer perceptron network
    # Parameters:
        #  1. Integer Value: Number of features in training set
    # Placeholders:
        # 1. List to hold the layers of the network, each element is a Dense() layer
class MultilayerPerceptron:
    def __init__(self, input_shape):
        self.network = []  # Initially the layer is empty

        # There are n_features in training data so n_features input nodes
        # First hidden layer has 12 neurons, and n_features inputs
        self.network.append(Dense(12, input_shape))
        # Secon hidden layer has 6 neurons, and 12 inputs
        self.network.append(Dense(6,12))
        # Output layer has 2 neurons, and 6 inputs
        self.network.append(Dense(2,6))
        print("Network Created...")

    #Local Gradient  
    def loacal_gradient():
        pass

    # Train the model
    def train(self, X_train, y_train, learning_rate = 0.1, epochs = 100):
        print('In MLP.train() method')
        print('Input Shape: ', X_train.shape)
        while epochs >= 1: # While epochs are remaining
            
            for current_sample in X_train: # for current trainng sample in X_train
                previous_layers_outputs = []
                previous_layers_outputs.append(current_sample) # Previous outputs is current training example
                
                for network in self.network:
                    # Take latest output feed it to next layer as input and store its result as latest output
                    previous_layers_outputs.append(network.forward_signal(previous_layers_outputs[-1]))

                error_at_output = (y_train - previous_layer_output) ** 2

                for network in self.network:
                    
                # Backpropagation Now
                
        pass