# Importing make_classification to generate data with n_features and 2 target classes
from sklearn.datasets import make_classification
# Importing train_test_split to split data in training and test sets
from sklearn.model_selection import train_test_split
# Importing Network.py 
from network import MultilayerPerceptron

def generate_data():
    #Generaing Data with 500 training examples, 10 features, and 2 target classes 
    data = make_classification(n_samples = 500, n_features = 10, n_classes = 2)

    #Unpacking data from tuples to the inputs and targets variables
    inputs, targets = data

    #Normmalizing the data as with respect to max and min values in the inputs  
    inputs = inputs/ inputs.max()
    targets = targets / targets.max()
    print('Data Generated')
    #Spliting data into test and train classes
    return train_test_split(inputs,targets) #Leaving test_size and random_state default

def main():
    print('In main.. initiating')

    # Generating Data; generate_data() returns the training and test sets
    X_train, X_test, y_train, y_test = generate_data()

    # Creating Multilayer Perceptron Object with X_train.shape[1] input nodes
    network = MultilayerPerceptron(X_train.shape[1])
    
    print('All set, go ahead')
    # Training the model with X_train, y_train
    network.train(X_train, y_train, learning_rate = 0.7, epochs = 90)

    # Testing the model with X_test, y_test
    #network.test(X_test, y_test)


# Call to main method
if __name__ == '__main__':
    main()