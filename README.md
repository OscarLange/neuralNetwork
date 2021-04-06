# neuralNetwork
## Description

A simple neural network editor, that allows you to create a fully connected Neural Network(NN).
An optic Interface makes it easy to test various variables of a NN, and see what effect they have on the convergence of the Network.

You can choose any input dimension, biases, weights, number and dimension of hidden layers.
Currently the software only supports a ReLu and a Sigmoid activation function, and only 1 function per layer.

Each layer has to be fully connected to the last and the next (if present), so no CNN is possible as of now.
The output layer has to be 1 node, as at this time only the binary cross entropy loss function is implemented to estimate risk.

Everything is setup from the get go, so just pressing the button in the top left corner starts the training.

## Configuration

There are two main ways to interact with the neural network.

First you can change the input values of the NN in `input.csv`, please follow the established form.
Each row until the "END" line is one dimension of the input, and the comma separated values are the samples.
The row after the "END" is the outcome of these samples, and has to have the same length as all rows above.

The other interaction is the screen created by pygame, and there are a few options:
- Left click:
  - On empty space => creates a fresh node
  - On a node => settings of a node like the type, function, bias and connection to other nodes
  - On an edge => Type afterwards to set the weight of the edge
- Right click:
  - Deletes the object the cursor is on
- Buttons:
  - Top Left: Starts/Stops Training of the NN
  - Right of Top Left: Resets NN to original form

## Required Software
- python (3.8.5)
- numpy (1.20.1)
- pygame (2.0.1)

Have fun!