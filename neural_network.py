import numpy as np
import scipy.special # sigmoid function

# Neural network class
class NeuralNetwork():

    # Init the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in each layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Learning rate
        self.l_rate = learning_rate

        # Weight matrices between layers (input_hidden, hidden_output)
        # Random values between [-n; n], n = 1/sqrt(layer_inputs)
        self.wih = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Activation function
        self.activation_function = lambda x: scipy.special.expit(x)


    # Train the neural network
    def train(self, inputs_list, tragets_list):
        # Convert lists to 2d arrays
        inputs = np.array(inputs_list, ndim=2).T
        targets = np.array(tragets_list, ndim=2).T

         # Calculate inputs into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate outputs from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate inputs into output layer
        output_inputs = np.dot(self.who, hidden_outputs)
        # Calculate outputs from output layer
        output_outputs = self.activation_function(output_inputs)

        # Calculate output errors
        output_errors = targets - output_outputs
        # Calculate hidden layer errors: output_errors split by weights
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update hidden_output weights
        self.who += self.l_rate * np.dot((output_errors * output_outputs * (1.0 - output_outputs)), np.transpose(hidden_outputs))
        # Update input_hidden weights
        self.wih += self.l_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


    # 
    def feed_forward(self, inputs_list):
        # Convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate inputs into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate outputs from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate inputs into output layer
        output_inputs = np.dot(self.who, hidden_outputs)
        # Calculate outputs from output layer
        output_outputs = self.activation_function(output_inputs)

        return output_outputs


if __name__ == "__main__":
    n = NeuralNetwork(3, 3, 3, 0.3)
    print(n.feed_forward([1.0, 0.5, -1.5]))