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

    #------------------------------------------------------ Heuristics and Activation --------------------------------------------------------------

    #negates binary variables and likelihoods
    def negate(self, y):
        return 1-y

    #sigmoid Function for 1 element
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #calculate the cross-entropy loss of a prediction matrix A
    def crossEntropyLoss(self, A):
        # - (Y * ln(A) + (1-Y) * ln(1-A))
        return np.negative(np.add(np.multiply(self.Y, np.log(A)), np.multiply(self.negate(self.Y), np.log(self.negate(A)))))

    # calculate the risk based upon the given loss function
    def risk(self, A):
        if(self.l == "ce"):
            lossMatrix = self.crossEntropyLoss(A)
            return 1/lossMatrix.shape[1] * (np.sum(lossMatrix))

    #------------------------------------------------------ Derivatives for Backpropagation --------------------------------------------------------

    #return derivative of outputlayer with given loss function 
    #and outputlayer activation function
    def dOutput(self,A):
        if(self.l == "ce"):
            if(self.a == "sig"):
                return self.dCeSig(A)

    #partial derivative of crossEntropyLoss regarding the sigmoid
    #times the derivative of the sigmoid regarding linear Comb Z
    def dCeSig(self, A):
        return np.subtract(A, self.Y)

    #return partial derivative of hidden layer activation function
    def dhidden(self, dA, A):
        if(self.a == "sig"):
            return self.dlClC(dA,A)

    #partial derivative of the sigmoid function regarding 
    # the linear combination Z=W*A+b 
    def dlClC(self, dA, A):
        return np.multiply(dA, np.multiply(A, self.negate(A)))

    #partial derivative of Z=(A*W+b) regarding W 
    #times the partial derivative of output regarding Z = dZ
    #divided by number of observations
    def dLinCombW(self, dZ, A):
        return np.multiply(np.dot(dZ,A.transpose()), (1/self.n))

    #partial derivative of Z=(A*W+b) regarding b
    #times the partial derivative of output regarding Z = dZ
    #divided by number of observations
    def dLinCombB(self, dZ):
        return np.multiply(np.dot(dZ,np.ones([dZ.shape[1],1])), (1/self.n))

    #partial derivative of Z=(A*W+b) regarding A
    #times the partial derivative of output regarding Z = dZ
    #divided by number of observations
    def dLinCombA(self, dZ, W):
        return np.dot(W.transpose() ,dZ)

    #update weights and biases
    def updateParams(self, dW, db):
        for index in range(self.h+1):
            self.W[index] = self.W[index] - self.alpha * dW[index]
            self.b[index] = self.b[index] - self.alpha * db[index]

    #------------------------------------------------------ Main Functions ------------------------------------------------------------------

    # calculate one forward Pass in the neural network
    def forwardPass(self):
        #store the input as the Activation matrix, so you can reuse without changing Input 
        A = []
        currA = self.X
        A.append(currA)
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
        dW = []
        db = []
        dZ = None
        dA = None
        for index in range(self.h, -1, -1):
            if(index == self.h):
                dZ = self.dOutput(A[index+1])
            else:
                dA = self.dLinCombA(dZ, self.W[index+1])
                dZ = self.dhidden(dA, A[index+1])    
            dW.append(self.dLinCombW(dZ, A[index]))
            db.append(self.dLinCombB(dZ))
        #reverse list as backpropagation is backwards
        dW.reverse()
        db.reverse()
        #update newly calculated weights and biases
        self.updateParams(dW, db)

    #does one forward pass and one backward pass trough all data
    def epoch(self):
        A = self.forwardPass()
        risk = self.risk(A[len(A)-1])
        self.backwardPass(A)
        return risk     