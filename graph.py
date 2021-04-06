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

    def getBias(self):
        if(self.bias != " " and self.bias != "-"):
            return float(self.bias)
        else:
            return 0

class Edge:
    def __init__(self, startPos, endPos):
        self.startPos = startPos
        self.endPos = endPos
        self.rect = None
        self.weight = " "
        self.visited = False
        middleX = np.abs(self.startPos[0] - self.endPos[0])/4
        middleY = np.abs(self.startPos[1] - self.endPos[1])/4
        if(self.startPos[0] > self.endPos[0]):
            middleX = self.startPos[0] - middleX
            if(self.startPos[1] > self.endPos[1]):
                middleY = self.startPos[1] - middleY
            else:
                middleY = self.startPos[1] + middleY
        else:
            middleX = self.endPos[0] - middleX
            if(self.startPos[1] < self.endPos[1]):
                middleY = self.endPos[1] - middleY
            else:
                middleY = self.endPos[1] + middleY
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
    
    def getWeight(self):
        if(self.weight != " " and self.weight != "-"):
            return float(self.weight)
        else:
            return 1

#filter nodes for input nodes
def filterInput(circles):
    inputNodes = []
    for node in circles:
        if(node.type == "input"):
            inputNodes.append(node)
    return sorted(inputNodes, key=lambda node: node.pos[1])

#filter node by center pos
def filterNode(pos, circles):
    for node in circles:
        if node.pos == pos:
            return node

#filter Edge by center pos
def filterEdge(pos1, pos2, lines):
    for line in lines:
        if (line.startPos == pos1 and line.endPos == pos2) or (line.startPos == pos2 and line.endPos == pos1):
            return line

#filter connected nodes by connected edge
def filterConnections(node, lines, circles):
    connectedNodes = []
    for line in lines:
        if(not line.visited):
            if(node.pos == line.startPos):
                connectedNodes.append(filterNode(line.endPos, circles))
            elif (node.pos == line.endPos):
                connectedNodes.append(filterNode(line.endPos, circles))
    return sorted(connectedNodes, key=lambda node: node.pos[1])

