from nn_03 import Value, draw_dot
import math
import random
import numpy as np
import matplotlib.pyplot as plt
class Neuron:

    def __init__(self, nin):
        # Create one weight for each input.
        # We initialize weights randomly so that different neurons
        # don't all start with the same values.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # Bias term for the neuron, also initialized randomly.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Compute the weighted sum of the inputs.
        #
        # zip(self.w, x) pairs each weight with its corresponding input:
        #   [(w1, x1), (w2, x2), ...]
        #
        # wi * xi computes the contribution of each input.
        #
        # sum(..., self.b) means:
        #   Start the sum with the bias value,
        #   then add all the weighted inputs.
        #
        # Equivalent to:
        # act = b + w1*x1 + w2*x2 + ...
        #
        # 'act' stands for activation (also called the pre-activation value),
        # i.e. the value before applying the activation function.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply the tanh activation function to introduce non-linearity.
        out = act.tanh()

        return out

    def parameters(self):
        # Return all learnable parameters of this neuron:
        # every weight and the bias.
        return self.w + [self.b]
# x = [2.0, 3.0]
# n = Neuron(2) #Create 1 neuron which expects 2 inputs
# print(n(x)) #calls call function o/p = Value(data=0.8723540545575671)
class Layer:

    def __init__(self, nin, nout):
        # Create 'nout' neurons.
        # Each neuron expects 'nin' inputs.
        # Example:
        # Layer(3, 4)
        #
        # Creates:
        #   Neuron(input_size=3)
        #   Neuron(input_size=3)
        #   Neuron(input_size=3)
        #   Neuron(input_size=3)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Pass the same input 'x' to every neuron in this layer.
        #
        # Each neuron computes:
        # output = tanh(w*x + b)
        #
        # The result is a list containing the output of every neuron.
        outs = [n(x) for n in self.neurons]

        # If there is only one neuron, return the single Value object
        # instead of returning a list containing one element.
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Collect parameters (weights and bias) from every neuron
        # into one flat list.
        return [p for neuron in self.neurons for p in neuron.parameters()]    
    
# x = [2.0, 3.0]
# n = Layer(2, 3) 
# print(n(x))  #[Value(data=-0.41956886899900714), Value(data=0.14639031428846722), Value(data=0.9995659050631185)]
class MLP:
    # MLP = Multi-Layer Perceptron
    #
    # A Multi-Layer Perceptron is simply a neural network made up of
    # multiple layers connected one after another.
    #
    # Data flows like:
    #
    # Input → Layer 1 → Layer 2 → ... → Final Layer → Output

    def __init__(self, nin, nouts):
        # nin   = Number of input features.
        #
        # nouts = List specifying the number of neurons in each layer.
        #
        # Example:
        # MLP(3, [4, 4, 1])
        #
        # Means:
        #   - Input has 3 values.
        #   - First layer has 4 neurons.
        #   - Second layer has 4 neurons.
        #   - Final layer has 1 neuron.

        # Create a list representing the size of every stage.
        #
        # Example:
        # nin = 3
        # nouts = [4, 4, 1]
        #
        # sz becomes:
        # [3, 4, 4, 1]
        #
        # This makes it easy to connect consecutive layers.
        sz = [nin] + nouts

        # Create all layers of the neural network.
        #
        # Using sz = [3, 4, 4, 1]:
        #
        # Layer(3,4)
        # Layer(4,4)
        # Layer(4,1)
        #
        # Each layer's output becomes the next layer's input.
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Pass the input through every layer one by one.
        #
        # Layer1 processes the original input.
        # Layer2 processes Layer1's output.
        # Layer3 processes Layer2's output.
        #
        # This process is called the Forward Pass.
        for layer in self.layers:
            x = layer(x)

        # Return the final prediction from the last layer.
        return x
    def parameters(self):
    # Collect all learnable parameters from the entire neural network.
    #
    # Each Layer contains multiple Neurons.
    # Each Neuron contains:
    #   - Weights (w)
    #   - Bias (b)
    #
    # We iterate through every layer, ask it for its parameters,
    # and combine everything into one flat list.
    #
    # This list is later used during training to:
    #   1. Reset gradients.
    #   2. Update weights and biases using gradient descent.
        params = []

        # Go through every layer in the neural network.
        for layer in self.layers:

            # Get all parameters (weights and biases) from that layer.
            for p in layer.parameters():
                params.append(p)

        return params
    
if __name__ == "__main__":

    # ==========================================
    # Example 1 : Single Neuron
    # ==========================================

    # x = [2.0, 3.0]
    # neuron = Neuron(2)
    # print(neuron(x))

    # ==========================================
    # Example 2 : Layer
    # ==========================================

    # x = [2.0, 3.0]
    # layer = Layer(2, 3)
    # print(layer(x))

    # ==========================================
    # Example 3 : Multi Layer Perceptron (MLP)
    # ==========================================

    x = [2.0, 3.0, -1.0]

    n = MLP(3, [4, 4, 1])

    # print(n(x))

    # ==========================================
    # Training Dataset
    # ==========================================

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    # Ground truth (expected outputs)
    ys = [1.0, -1.0, -1.0, 1.0]

    # Forward pass
    ypred = [n(x) for x in xs]

    print("Predictions")
    print(ypred)

    # Mean Squared Error (actually Sum Squared Error here)
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    print("\nLoss")
    print(loss)

    #dot = draw_dot(loss)
    #dot.render("computation_graph", view=True)

    # Train the neural network for 20 iterations.
    #
    # Each iteration is called an "epoch" in this example because we are
    # passing the entire training dataset through the network once.
    for k in range(20):

        # ============================================================
        # Step 1: Forward Pass
        # ============================================================
        #
        # Pass every training example through the neural network
        # to get its prediction.
        #
        # ypred will contain one prediction for every input in xs.
        ypred = [n(x) for x in xs]

        # Compute the total loss.
        #
        # Loss measures how far the predictions are from the
        # expected (ground truth) values.
        #
        # Smaller loss = better predictions.
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # ============================================================
        # Step 2: Backward Pass
        # ============================================================
        #
        # Gradients from the previous iteration are still stored
        # in every Value object, so clear them before computing
        # new gradients.
        for p in n.parameters():
            p.grad = 0.0

        # Compute gradients for every weight and bias in the network.
        #
        # This uses backpropagation + the chain rule.
        loss.backward()

        # ============================================================
        # Step 3: Gradient Descent (Parameter Update)
        # ============================================================
        #
        # Move every parameter slightly in the direction that
        # decreases the loss.
        #
        # 0.1 is the learning rate.
        #
        # A larger learning rate takes bigger steps.
        # A smaller learning rate takes smaller steps.
        for p in n.parameters():
            p.data += -0.1 * p.grad

        # Print the current loss after this iteration.
        print(k, loss.data)