# MultiLayer-Perceptron-From-Scratch


## Structure of Modules, Classes and Methods used in Multi Layer Perceptron Implementation
There are 4 modules:
    1. activation.py
    2. dense.py
    3. network.py
    4. main.py

## 1. activation.py
    => Has one class Activaion, with
        => Constructor:
            => No implementation - Default
        => activate() method
            => parameters: self, activation_potential as ndarray
            => operation: sigmoid logistic activation on activation_potential  
            => returns: ndarray of nonlinear activation values of same shape as activation_potential
        => activate_derivitive() method
            => parameters: self, activation potential as ndarray
            => operation: first differential of sigmoid logistic activation calculated on activation_potential
            => returns: ndarray of nonlinear activation values of same shape as activation_potential
        
## 2. dense.py
    => Has one class Dense, with
        => Constructor:
            => parameters:
                => num_neurons: Number of Neurons in this Dense Layer
                => input_shape: Shape of inputs to this layer
            => operations:
                => Initialize weights as ndarray of size as input dimension
                => Initialize biases as ndarray of size number of neurons
                => Initialize activation_potentials as size of number of neurons
                => Initialize outputs as ndarray of size number of neurons
                => Initialize local_gradient as ndarray of size number of neurons
        => get_weights() method:
            => returns: weights of the layer
        => get_gradients() method:
            => returns: local_gradients of the layer
        => activation_potential() method:
            => prameters: inputs -> ndarray, input vector of the inputs to neurons in this Dense layer
            => operations: (inputs * weights) +  biases
            => return: activaion_potential -> ndarray, activation_potential of all neurons in this Dense layer  
        => local_gradient() method:
            => parameters: 
                => error_at_end -> ndarray, error at output neurons at end of predections, with respect to desired target
                => layer -> Dense Object, to determine whether layer is output of hidden
                => network -> list of Dense() objects to which the layer belongs
                => next_gradients -> ndarray, vector of local gradients of next layer to current
                => next_weights -> ndarray, vector of weights associated with outputs of this layer neurons that are feeded to next layer as input
            => operations: 
                => Whether the layer is output or hidden, it perform two oprations
                    => If the layer  is output layer: loc_grad is error_at_end times 'activate_derivative' of this layer
                    => Else: loc_grad is 'activate_derivative' of this layer times sum of all local gradients of next layer neurons times weights of next layer
            => returns: None
        => forward_signal() method:
            => parameters: inputs -> ndarray, vector outputs of prevoius layer
            => operations:
                => Calculate activation potential and save it to self.activation_potential ndarray
                => Activate the activation potential and add it to self.outputs ndarray
            => returns: outputs of  the current dense layer -> ndarray, will need it as input for next layer 
        => backward_signal() method:
            => parameters: 
                => error_at_end -> ndarray, error at output neurons at end of predictions, with respective to desired target, passed in turn to local_gradient()
                => layer -> Dense Object, passed in turn to local_gradient()
                => learning_rage -> float object, determine the step size of weight update
                => network -> list of Dense() objects to which the layer belongs, will be passed to local_gradient()
                => next_gradients -> ndarray, vector of local gradients of next layer to current
                => next_weights -> ndarray, vector of weights associated with outputs of this layer neurons that are feeded to next layer as input
            => operations:
                => Calculate Local Gradient using local_gradient(error_at_end, layer, network, next_gradients, next_weights) method
                => Update the layer weights on basis of w(n + 1) = w(n) + lr * local_grad * inputs_to_this_layer
                => Set this layer weights to updated weights
            => returns: None
## 3. network.py
    => Has one class Network, with
        => Constructor:
            => Initialize network -> list of layers
            => Append layers you want in the network, using Dense() class
            => Initialize previous_layers_output -> list of ndarrays, will use this to feed input to next layer
        => train() method:
            => prameters:
                => 
            => operations:
                => While epochs are still remaining, do
                    => Discard previous outputs
                    => Get a sample from training set
                    => On this sample, perform feedforward and save outputs of all layers
                    => Calculate error_at_end
                    => Perform feedbackward, update weights and biases
            => returns: None
            
## Has not updated this part yet will do it soon            
## 4. main.py
    => Has generate_data() method:
        => parameters:
        => operations:
        => returns:
    => Has main() method:
        => parameters:
        => operations:
        => returns:
