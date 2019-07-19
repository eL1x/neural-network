import numpy as np
import scipy.special # sigmoid function
import scipy.ndimage
import sys

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
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(tragets_list, ndmin=2).T

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


    # Return outputs from neural network
    def feedforward(self, inputs_list):
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
    # Number of nodes in each layer
    input_nodes = 28*28 # Images are 28x28 pixels
    hidden_nodes = 200
    output_nodes = 10

    # Learning rate
    learning_rate = 0.1

    # Istance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    
    # Load the mnist training data
    with open("mnist_dataset/mnist_train.csv", "r") as training_data_file:
        training_data_records = training_data_file.readlines()

    # Train the neural network
    epochs = 5
    for e in range(epochs):
        print("# {0} epoch".format(e))
        for record in training_data_records:
            all_values = record.split(',')
            # Scale input to range 0.01 to 0.99
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            
            # Data augumentation - rotate by 10 and -10 degrees
            inputs_rotated_p = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
            inputs_rotated_m = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
            
            # Train on original and augumented data
            n.train(inputs, targets)
            n.train(inputs_rotated_p.reshape(784), targets)
            n.train(inputs_rotated_m.reshape(784), targets)

    # Load the mnist test data
    with open("mnist_dataset/mnist_test.csv", "r") as test_data_file:
        test_data_records = test_data_file.readlines()

    # Test the neural network
    scorecard = []

    for record in test_data_records:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        outputs = n.feedforward(inputs)
        label = np.argmax(outputs)
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = np.asfarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)

# TODO 
# Refactor of train function - use feedforward