import numpy as np
# This file contains the Activation class; which defines the activation function used by a layer of neurons
class Activation:
    # Definition of the activation function
    def __init__(self):
        pass
    # The value of sigmoid logistic activation function on activation potential
    def activate(self, activation_potential):
        #returns the scalled values of activation potentials using sigmoid logistic activation function
        return (1 / (1 - np.exp(activation_potential)))

    # The activation function to calculate value of first differential of sigmoid logistic activation function on activation potential
    def activate_derivative(self, activation_potential):
        return activation_potential * (1 - activation_potential)