import numpy as np
from neuralnetwork import NN
from graph import Node, Edge, filterInput, filterConnections, filterEdge, filterNode
from filereader import returnInput

import pygame
import time
import random

#start lib
pygame.init()
#load pictures
goImg = pygame.image.load('go.png')
stopImg = pygame.image.load('stop.png')

#define constants
GREEN   = ( 0, 255, 0)
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
RED = (255, 0, 0)
GREY = (192, 192, 192)
BLUE = ( 0, 0, 255)
width = 1600
height = 900
popup_width = 100
popup_height = 148
popup2_height = 75
popup3_height = 112
lineChartHeight = 450
lineChartWidth = 800
lineChartPos = (width-lineChartWidth, height-lineChartHeight)
granularity = 10
radius = 40
lineThickness = 4
margin = 10
LEFT = 1
RIGHT = 3
fps = 120
options1 = ["Connect", "Function", "Bias", "Type"]
options2 = ["None", "Sigmoid", "Relu"]
options3 = ["input", "hidden", "output"]

font_style = pygame.font.SysFont(None, 50)
popup_font = pygame.font.SysFont(None, 20)

#set display and title 
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Neural Network")

#set clock
clock = pygame.time.Clock()

#function to display messages
def message(msg,color):
    mesg = font_style.render(msg, True, color)
    text_width, text_height = font_style.size(msg)
    screen.blit(mesg, [(width - text_width)/2, (height - text_height)/2])

#euclidean distance
def euclideanDistance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

#function to calculate whether position is in circle
#and return circle center if that is the case
def insideCircle(pos, circles):
    for center in circles:
        euclideanDist = euclideanDistance(pos, center.pos)
        if(euclideanDist < radius):
            return center
    return None

#detect if a line was clicked
def lineclicked(pos, line):
    positions = []
    #check multiple collision points,
    #as the rect of line is too big
    positions.append(pos)
    if(np.abs(line.startPos[1] - line.endPos[1]) > margin):
        if(np.abs(line.startPos[0] - line.endPos[0]) > margin):
            positions.append((pos[0]-margin,pos[1]))
            positions.append((pos[0]+margin,pos[1]))
            positions.append((pos[0],pos[1]-margin))
            positions.append((pos[0],pos[1]+margin))
    for position in positions:
        if(not line.rect.collidepoint(position)):
            return False
    return True

#remove a line if rightclicked on it
def removeLines(pos, lines):
    newlines = set()
    for line in lines:
        if(not lineclicked(pos, line)):
            newlines.add(line)
    return newlines

#function to calculate where position overlapps with another circle
def overlappCircle(pos, circles):
    for center in circles:
        euclideanDist = euclideanDistance(pos, center.pos)
        if(euclideanDist <= radius*2):
            return True
    return False

#function to calculate wheter a position is inside a rectangle
def insideRectangle(pos, rectPos, size):
    return pos[0] >= rectPos[0] and pos[1] >= rectPos[1] and pos[0] <= rectPos[0]+size[0] and pos[1] <= rectPos[1]+size[1] 

#create a popup for when somebody selects a node
def circlePopup(currPos, options, height):
    popUp = pygame.Surface((popup_width,height))
    popUp.fill(GREY)
    pygame.draw.rect(popUp, BLACK, (0,0,popup_width,height), lineThickness)
    currheight = margin
    cutoff = []
    for option in options:
        msg = popup_font.render(option, True, BLACK)
        text_width, text_height = popup_font.size(option)
        popUp.blit(msg, ((popup_width - text_width)/2,currheight))
        currheight += (text_height + margin)
        pygame.draw.line(popUp, BLACK, (0, currheight), (popup_width, currheight), lineThickness)
        cutoff.append(currheight + currPos[1])
        currheight += (margin + lineThickness)
    return (cutoff, popUp)

#create a surface to draw the lineChart on
def lineChart(risk):
    lineChart = pygame.Surface((lineChartWidth,lineChartHeight))
    lineChart.fill(WHITE)
    pygame.draw.rect(lineChart, BLACK, (0,0,lineChartWidth,lineChartHeight), lineThickness)

    maxRisk = max(risk)
    riskGranularity = maxRisk/granularity
    yOffset = (lineChartHeight - (3*margin))/(granularity)
    y = margin
    currRisk = maxRisk
    text_width, text_height = popup_font.size(str(round(currRisk, 5)))
    for i in range(granularity+1):
        txt = str(round(currRisk, 5))
        msg = popup_font.render(txt, True, BLACK)
        lineChart.blit(msg, (margin,y-text_height/4))
        pygame.draw.line(lineChart, BLACK, (1.5*margin+text_width,y), (2.5*margin+text_width,y), lineThickness)
        y += yOffset
        currRisk -= riskGranularity
    xgraphOffset = 2*margin+text_width
    xOffset = (lineChartWidth - (xgraphOffset + margin))/(len(risk)-1)
    yOffset = (lineChartHeight-(2*margin))/maxRisk
    x = xgraphOffset
    for i in range(1,len(risk), +1):
        pos1 = (x, margin + ((maxRisk - risk[i-1])*yOffset))
        x += xOffset
        pos2 = (x, margin + ((maxRisk - risk[i])*yOffset))
        pygame.draw.line(lineChart, BLUE, pos1, pos2, lineThickness)

    pygame.draw.line(lineChart, BLACK, (xgraphOffset, margin), (xgraphOffset, lineChartHeight-margin), lineThickness)
    pygame.draw.line(lineChart, BLACK, (xgraphOffset, lineChartHeight-margin), (lineChartWidth - margin, lineChartHeight-margin), lineThickness)

    return lineChart


#function to convert keypress to nummeric value (including minus)
def keyToVal(eventKey):
    if eventKey == pygame.K_0:
        return "0"
    elif eventKey == pygame.K_1:
        return "1"
    elif eventKey == pygame.K_2:
        return "2"
    elif eventKey == pygame.K_3:
        return "3"
    elif eventKey == pygame.K_4:
        return "4"
    elif eventKey == pygame.K_5:
        return "5"
    elif eventKey == pygame.K_6:
        return "6"
    elif eventKey == pygame.K_7:
        return "7"
    elif eventKey == pygame.K_8:
        return "8"
    elif eventKey == pygame.K_9:
        return "9"
    elif eventKey == pygame.K_BACKSPACE:
        return "BS"
    elif eventKey == pygame.K_MINUS:
        return "-"
    else:
        return None

# the runmode where the nn is trained with the data
def runMode(circles, lines):
    running = True
    timer = 0

    xMatrix, yMatrix = returnInput()
    currNodes = filterInput(circles)
    if(xMatrix.shape[0] != len(currNodes)):
        message("Dim(Input Matrix) not equal to Dim(Input Nodes):"+ str(xMatrix.shape[0]) + "!=" + str(len(currNodes)),RED)
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == LEFT:
                            return
    weightList = []
    biasList = []
    activationList = []
    index = 0
    #while loop for layers
    while currNodes[0].type != "output":
        nxtNodes = filterConnections(currNodes[0], lines, circles)
        weightList.append(np.empty((len(nxtNodes),len(currNodes))))
        activationList.append(nxtNodes[0].function)
        for i in range(len(nxtNodes)):
            for j in range(len(currNodes)):
                line = filterEdge(nxtNodes[i].pos, currNodes[j].pos, lines)
                weightList[index][i][j] = line.getWeight()
                line.visited = True
        biasList.append(np.empty((len(nxtNodes),1))) 
        for i in range(len(nxtNodes)):
            biasList[index][i][0] = nxtNodes[i].getBias()
        currNodes = nxtNodes
        index +=1

    for line in lines:
        line.visited = False

    nn1 = NN(xMatrix, weightList, biasList, yMatrix, activationList, "ce", 1)
    risk = []
    risk.append(nn1.epoch())

    while(running):#user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        circle = insideCircle(pos, circles)
                        if(pos[0] < stopImg.get_width() and  pos[1] < stopImg.get_height()):
                            running = False
        #reset screen 
        screen.fill(WHITE)
        #calculate one epoch of nn
        risk.append(nn1.epoch())
        #draw buttons top left
        screen.blit(stopImg, (0,0))
        #draw every line for every edge
        for line in lines:
            line.rect = pygame.draw.line(screen, BLACK, line.startPos, line.endPos, lineThickness)
            msg = popup_font.render(line.weight, True, RED)
            text_width, text_height = popup_font.size(line.weight)
            screen.blit(msg, (line.middle[0]-text_width/2,line.middle[1]))
        #draw circle for every Node in set
        for circle in circles:
            pygame.draw.circle(screen, WHITE, circle.pos, radius)
            if(circle.type == "input"):
                pygame.draw.circle(screen, GREEN, circle.pos, radius, lineThickness)
            else:
                if(circle.type == "hidden"):
                    pygame.draw.circle(screen, BLACK, circle.pos, radius, lineThickness)
                else:
                    pygame.draw.circle(screen, RED, circle.pos, radius, lineThickness)
                if(circle.function != None):
                    msg = popup_font.render(circle.function, True, BLACK)
                    text_width, text_height = popup_font.size(circle.function)
                    screen.blit(msg, (circle.pos[0]-text_width/2,circle.pos[1]-text_height/2))
                msg = popup_font.render(circle.bias, True, BLUE)
                text_width, text_height = popup_font.size(circle.bias)
                screen.blit(msg, (circle.pos[0]-text_width/2,circle.pos[1]+text_height))
        #draw line chart

        screen.blit(lineChart(risk), lineChartPos)
        #reset window
        pygame.display.flip()
        #refresh time
        timer += (clock.tick(fps)/1000)

#already draw a basic nn
def quickStart():
    circles = set()
    lines = set()
    xOffset = (width - radius*8)/5
    yOffset = (height - radius*6)/4
    x = xOffset + radius
    y = yOffset + radius
    oldPos = []
    for i in range(3):
        oldPos.append((x,y))
        tmpNode = Node((x,y))
        tmpNode.type = "input"
        circles.add(tmpNode)
        y += yOffset + (2*radius)
    x2 = x + xOffset + (2*radius)
    y2 = yOffset + radius
    tpmWeights = [[1,1,0],[1,0,1],[0,1,1]]
    newPos = []
    for i in range(3):
        newPos.append((x2,y2))
        tmpNode = Node((x2,y2))
        tmpNode.function = "Relu"
        tmpNode.bias = "0"
        circles.add(tmpNode)
        for j in range(len(oldPos)):
            tmpLine = Edge(oldPos[j],(x2,y2))
            tmpLine.weight = str(tpmWeights[i][j])
            lines.add(tmpLine)
        y2 += yOffset + (2*radius)
    yOffset = (height - radius*4)/3
    x = x2 + xOffset + (2*radius)
    y = yOffset + radius
    tpmWeights = [[1,0,1],[0,1,1]]
    oldPos = []
    for i in range(2):
        oldPos.append((x,y))
        tmpNode = Node((x,y))
        tmpNode.function = "Sigmoid"
        tmpNode.bias = "0"
        circles.add(tmpNode)
        for j in range(len(newPos)):
            tmpLine = Edge(newPos[j],(x,y))
            tmpLine.weight = str(tpmWeights[i][j])
            lines.add(tmpLine)
        y += yOffset + (2*radius)
    x += xOffset + (2*radius)
    y = height/2
    tmpNode = Node((x,y))
    tmpNode.function = "Sigmoid"
    tmpNode.type = "output"
    tmpNode.bias = "0"
    circles.add(tmpNode)
    tpmWeights = [[1,-1]]
    for j in range(len(oldPos)):
        tmpLine = Edge(oldPos[j],(x,y))
        tmpLine.weight = str(tpmWeights[0][j])
        lines.add(tmpLine)
    return (circles,lines)

#main loop
def drawMode():
    active = True
    circles,lines = quickStart()
    mode = "Standard"
    lineStart = (0,0)
    lineEnd = (0,0)
    popUp = None
    popUpPos = (0,0)
    cutoff = []
    timer = 0
    currLine = None
    # huge main loop with different modes, has to be split up later
    while active:
        #user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if(mode == "Standard"):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        circle = insideCircle(pos, circles)
                        if(pos[0] < goImg.get_width() and  pos[1] < goImg.get_height()):
                            runMode(circles,lines)
                        elif(circle != None):
                            cutoff, popUp = circlePopup(circle.pos, options1, popup_height)
                            popUpPos = circle.pos
                            mode = "Circle"
                        else:
                            for line in lines:
                                if(lineclicked(pos, line)):
                                    mode = "Line"
                                    currLine = line
                                    timer = 0
                            if mode != "Line":
                                if(not overlappCircle(pos, circles)):
                                    circles.add(Node(pos))
                                else:
                                    message("Too close to other node!",RED)
                                    pygame.display.flip()
                                    pygame.time.wait(1500)
                    if event.button == RIGHT:
                        pos = pygame.mouse.get_pos()
                        circle = insideCircle(pos, circles)
                        if(circle != None):
                            circles.remove(circle)
                            toRemove = []
                            for line in lines:
                                if(line.startPos == circle.pos or line.endPos == circle.pos):
                                    toRemove.append(line)
                            for line in toRemove:
                                lines.remove(line)
                        lines = removeLines(pos, lines)
            elif(mode == "Connect" and timer >= 0.2):
                if event.type==pygame.MOUSEMOTION:
                    lineEnd = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    circle = insideCircle(pos, circles)
                    if (event.button == LEFT and circle != None and circle.pos != lineStart):
                        lines.add(Edge(lineStart, circle.pos))
                        mode = "Standard"
                    else:
                        mode = "Standard"
                else:
                    mode = "Standard"
            elif(mode == "Circle"):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mode = "Standard"
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        tmp = popUpPos
                        if(insideRectangle(pos, tmp, ((popup_width,popup_height)))):
                            if(pos[1] < cutoff[0]):
                                mode = options1[0]
                                lineStart = popUpPos
                                lineEnd = pos
                                timer = 0
                            elif(pos[1] < cutoff[1]):
                                mode = options1[1]
                                cutoff, popUp = circlePopup(popUpPos, options2, popup3_height)
                                timer = 0
                            elif(pos[1] < cutoff[2]):
                                mode = options1[2]
                                timer = 0
                            elif(pos[1] < cutoff[3]):
                                mode = options1[3]
                                cutoff, popUp = circlePopup(popUpPos, options3, popup3_height)
                                timer = 0            
            elif(mode == "Function" and timer >= 0.2):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        tmp = popUpPos
                        if(insideRectangle(pos, tmp, ((popup_width,popup3_height)))):
                            circle = insideCircle(popUpPos, circles)
                            if(pos[1] < cutoff[0]):
                                circle.function = None
                            elif(pos[1] < cutoff[1]):
                                circle.function = "Sigmoid"
                            elif(pos[1] < cutoff[2]):
                                circle.function = "Relu"
                    mode = "Standard"
            elif(mode == "Bias" and timer >= 0.2):
                if event.type == pygame.KEYDOWN:
                    circle = insideCircle(popUpPos, circles)
                    keypress = keyToVal(event.key)
                    if keypress != None:
                        if keypress == "BS":
                            circle.shrinkBias()
                        else:
                            circle.appendBias(keypress)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mode = "Standard"
            elif(mode == "Type" and timer >= 0.2):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        tmp = popUpPos
                        if(insideRectangle(pos, tmp, ((popup_width,popup3_height)))):
                            circle = insideCircle(popUpPos, circles)
                            if(pos[1] < cutoff[0]):
                                circle.type = "input"
                            elif(pos[1] < cutoff[1]):
                                circle.type = "hidden"
                            elif(pos[1] < cutoff[2]):
                                circle.type = "output"
                    mode = "Standard"
            elif(mode == "Line" and timer >= 0.2):
                if event.type == pygame.KEYDOWN:
                    keypress = keyToVal(event.key)
                    if keypress != None:
                        if keypress == "BS":
                            currLine.shrinkWeight()
                        else:
                            currLine.appendWeight(keypress)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mode = "Standard"
        #reset screen 
        screen.fill(WHITE)
        #draw buttons top left
        screen.blit(goImg, (0,0))
        #draw every line for every edge
        for line in lines:
            line.rect = pygame.draw.line(screen, BLACK, line.startPos, line.endPos, lineThickness)
            msg = popup_font.render(line.weight, True, RED)
            text_width, text_height = popup_font.size(line.weight)
            screen.blit(msg, (line.middle[0]-text_width/2,line.middle[1]))
        #draw circle for every Node in set
        for circle in circles:
            pygame.draw.circle(screen, WHITE, circle.pos, radius)
            if(circle.type == "input"):
                pygame.draw.circle(screen, GREEN, circle.pos, radius, lineThickness)
            else:
                if(circle.type == "hidden"):
                    pygame.draw.circle(screen, BLACK, circle.pos, radius, lineThickness)
                else:
                    pygame.draw.circle(screen, RED, circle.pos, radius, lineThickness)
                if(circle.function != None):
                    msg = popup_font.render(circle.function, True, BLACK)
                    text_width, text_height = popup_font.size(circle.function)
                    screen.blit(msg, (circle.pos[0]-text_width/2,circle.pos[1]-text_height/2))
                msg = popup_font.render(circle.bias, True, BLUE)
                text_width, text_height = popup_font.size(circle.bias)
                screen.blit(msg, (circle.pos[0]-text_width/2,circle.pos[1]+text_height))
        #if the mode is Connect mode draw a line
        if(mode == "Connect"):
            pygame.draw.line(screen, BLACK, lineStart, lineEnd, lineThickness)
        elif(mode == "Circle" or mode == "Function" or mode == "Type"):
            screen.blit(popUp, popUpPos)
        
        msg = popup_font.render(str(pygame.mouse.get_pos()), True, BLACK)
        text_width, text_height = popup_font.size(str(pygame.mouse.get_pos()))
        screen.blit(msg, (width-text_width,height-text_height))
        #reset window
        pygame.display.flip()
        #refresh time
        timer += (clock.tick(fps)/1000)

#start game
drawMode()