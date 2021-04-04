import numpy as np
from neuralnetwork import NN
from graph import Node, Edge

import pygame
import time
import random

#start lib
pygame.init()

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
radius = 40
lineThickness = 4
margin = 10
LEFT = 1
RIGHT = 3
fps = 120
options1 = ["Connect", "Function", "Bias", "Type"]
options2 = ["None", "Sigmoid"]
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
    opened = True 
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


#main loop
def game_loop():
    id = 0
    active = True
    circles = set()
    lines = set()
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
                        if(circle != None):
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
                                    circles.add(Node(id, pos))
                                    id += 1
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
                                cutoff, popUp = circlePopup(popUpPos, options2, popup2_height)
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
                        if(insideRectangle(pos, tmp, ((popup_width,popup2_height)))):
                            circle = insideCircle(popUpPos, circles)
                            if(pos[1] < cutoff[0]):
                                circle.function = None
                            elif(pos[1] < cutoff[1]):
                                circle.function = "sigmoid"
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
        #reset window
        pygame.display.flip()
        #refresh time
        timer += (clock.tick(fps)/1000)

#start game
game_loop()