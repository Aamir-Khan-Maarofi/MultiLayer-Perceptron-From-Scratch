# This file contains the Activation class; which defines the activation function used by a layer of neurons
class Activation:
    # Definition of the activation function
    def __init__(self):
        print("Activation Intialized")
        pass

    # The activation function used for activation of activation potential
    def activate(self, activation_potential):
        return activation_potential * (1 - activation_potential)
        