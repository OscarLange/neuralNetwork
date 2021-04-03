import numpy as np
from neuralnetwork import NN

X = np.matrix([[0,0,1,1,1],[0,1,0,1,0]])
Y = np.matrix([0,1,1,1,1])
W = []
W.append(np.matrix([[1,0],[0,1]]))
W.append(np.matrix([[1,0],[0,1]]))
W.append(np.matrix([1,-1]))
b = []
b.append(np.matrix([[0],[0]]))
b.append(np.matrix([[0],[0]]))
b.append(np.matrix([0]))
a = "sig"
l = "ce"
alpha = 1
iterations = 1000

nn1 = NN(X, W, b, Y, a, l, alpha)

while(iterations > 0):
    risk = nn1.epoch()
    print(risk)
    risk = nn1.epoch()
    print(risk)
    iterations -= 1