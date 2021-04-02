import numpy as np 

#Implements a Neural Network(NN) with one output node with a binary prediction
class NN:
    #initialize the NN:
    # X => input as matrix
    # W => weights as list of matrices
    # n => number of observations
    # b => bias as list of matrices
    # Y => label of input as matrix
    # h => number of hidden layers given by the number of weigt matrices - the output layer
    # a => activation function (sig, ...) as string
    # l => loss function (ce, ...) as string
    # alpha => learning rate
    def __init__(self, X, W, b, Y, a, l, alpha):
        self.X = X
        self.W = W
        self.b = b
        self.Y = Y
        self.h = len(W) - 1
        self.a = a
        self.l = l
        self.n = X.shape[1]
        self.alpha = alpha

    #negates binary variables and likelihoods
    def negate(self, y):
        return 1-y

    #sigmoid Function for 1 element
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #return derivative of outputlayer with given loss function 
    #and outputlayer activation function
    def dOutput(self,A):
        if(self.l == "ce"):
            if(self.a == "sig"):
                return self.dCeSig(A)

    #partial derivative of crossEntropyLoss regarding the sigmoid
    #times the derivative of the sigmoid regarding linear Comb Z
    def dCeSig(self, A):
        return np.subtract(A,self.Y)

    #return derivative of hidden lay with given activation function
    def dhidden(self,W,A, primer):
        if(self.a == "sig"):
            return self.dlClC(W,A, primer)

    #partial derivative of Z=W*A+b regarding A
    #times the derivative of A=sig(Z) regarding 
    def dlClC(self, W, A, primer):
        tmp = np.dot(W.transpose(), primer)
        return np.multiply(tmp, np.multiply(A, self.negate(A)))

    #partial derivative of Z=(A*W+b) regarding W
    def dLinCombW(self, A):
        return A.transpose()

    #partial derivative of Z=(A*W+b) regarding b
    def dLinCombB(self, A):
        return np.ones([A.shape[1],1])

    #calculate the cross-entropy loss of a prediction matrix A
    def crossEntropyLoss(self, A):
        # - (Y * ln(A) + (1-Y) * ln(1-A))
        return np.negative(np.add(np.multiply(self.Y, np.log(A)), np.multiply(self.negate(self.Y), np.log(self.negate(A)))))

    # calculate the risk based upon the given loss function
    def risk(self, A):
        if(self.l == "ce"):
            lossMatrix = self.crossEntropyLoss(A)
            return 1/lossMatrix.shape[1] * (np.sum(lossMatrix))

    # calculate one forward Pass in the neural network
    def forwardPass(self):
        #store the input as the Activation matrix, so you can reuse without changing Input 
        A = []
        currA = self.X
        #iterate over every layers activation function
        for index in range(self.h+1):
            #apply weights and biases W*A+b from the current activation
            currW = self.W[index]
            currB = self.b[index]
            Z = np.add(np.dot(currW,currA),currB)
            if(self.a == "sig"):
                #calculate activation function sig(Z)
                currA = self.sigmoid(Z)
                A.append(currA)
        return A

    #calculate on backward Pass in the neural network
    def backwardPass(self, A):
        #loop backwards through the activation results and propagate back iteratively
        #then in each iteration calculate weights
        dZ = np.subtract(A[1], self.Y)
        dW2 = np.multiply(np.dot(dZ,A[0].transpose()), (1/self.n))
        db2 = np.multiply(np.dot(dZ,np.ones([dZ.shape[1],1])), (1/self.n))
        dA1 = np.dot(self.W[1].transpose() ,dZ)
        dZ1 = np.multiply(dA1, np.multiply(A[0], self.negate(A[0])))
        dW1 = np.multiply(np.dot(dZ1,self.X.transpose()), (1/self.n))
        db1 = np.multiply(np.dot(dZ1,np.ones([dZ1.shape[1],1])), (1/self.n))
        self.W[0] = self.W[0] - self.alpha * dW1
        self.W[1] = self.W[1] - self.alpha * dW2
        self.b[0] = self.b[0] - self.alpha * db1
        self.b[1] = self.b[1] - self.alpha * db2

    #does one forward pass and one backward pass
    def epoch(self):
        A = self.forwardPass()
        risk = self.risk(A[len(A)-1])
        self.backwardPass(A)
        return risk
        