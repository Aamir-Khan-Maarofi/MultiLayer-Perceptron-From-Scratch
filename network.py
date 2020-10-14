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
    
    # Train the model
    def train(X_train, y_train, learning_rate = 0.1, epochs = 100):
        # While epochs are remaining
            # for current trainng sample in X_train
                # Previous outputs is current training example
                # For every layer: 
                    # find the respective output vector, store it will be used in next  layer
                # For the last output vector find error
                # Do backward prop - update weights - Need Local Gradient Calculation
        pass
        # Forward Propagation
        # Backward Propagation