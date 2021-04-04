import numpy as np
class Node:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.function = None
        self.bias = None
        self.type = "hidden"

class Edge:
    def __init__(self, startPos, endPos):
        self.startPos = startPos
        self.endPos = endPos
        self.rect = None

    