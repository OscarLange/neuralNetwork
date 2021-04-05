import numpy as np
class Node:
    def __init__(self, pos):
        self.pos = pos
        self.function = None
        self.bias = " "
        self.type = "hidden"

    def appendBias(self, strVal):
        if(strVal == "-"):
            if(self.bias == " "):
                self.bias = "-"
        elif(len(self.bias) <= 8):
            self.bias += strVal
    
    def shrinkBias(self):
        if(len(self.bias) > 1):
            self.bias = self.bias[:-1]
        else:
            self.bias = " "

class Edge:
    def __init__(self, startPos, endPos):
        self.startPos = startPosreturnInput()- self.endPos[1])/2
        if(self.startPos[0] > self.endPos[0]):
            middleX = self.startPos[0] - middleX
        else:
            middleX = self.endPos[0] - middleX
        if(self.startPos[1] > self.endPos[1]):
            middleY = self.startPos[1] - middleY
        else:
            middleY = self.endPos[1] - middleY
        self.middle = (middleX, middleY)

    def appendWeight(self, strVal):
        if(strVal == "-"):
            if(self.weight == " "):
                self.weight = "-"
        elif(len(self.weight) <= 8):
            self.weight += strVal
    
    def shrinkWeight(self):
        if(len(self.weight) > 1):
            self.weight = self.weight[:-1]
        else:
            self.weight = " "

    