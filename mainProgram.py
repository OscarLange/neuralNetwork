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
width = 1600
height = 900
popup_width = 100
popup_height = 148
radius = 40
lineThickness = 4
margin = 10
LEFT = 1
RIGHT = 3
fps = 120

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
def circlePopup(currPos):
    opened = True 
    popUp = pygame.Surface((popup_width,popup_height))
    popUp.fill(GREY)
    pygame.draw.rect(popUp, BLACK, (0,0,popup_width,popup_height), lineThickness)
    currheight = margin
    options = ["Connect", "Function", "Bias", "Type"]
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
                            cutoff, popUp = circlePopup(circle.pos)
                            popUpPos = circle.pos
                            mode = "Circle"
                        elif(not overlappCircle(pos, circles)):
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
                    if event.button == LEFT:
                        pos = pygame.mouse.get_pos()
                        tmp = popUpPos
                        if(insideRectangle(pos, tmp, ((popup_width,popup_height)))):
                            if(pos[1] < cutoff[0]):
                                mode = "Connect"
                                lineStart = popUpPos
                                lineEnd = pos
                                timer = 0
                        else:
                            mode = "Standard"
                    else:
                        mode = "Standard"
        #reset screen 
        screen.fill(WHITE)
        #if the mode is Connect mode draw a line
        if(mode == "Connect"):
            pygame.draw.line(screen, BLACK, lineStart, lineEnd, lineThickness)
        elif(mode == "Circle"):
            screen.blit(popUp, popUpPos)
        #draw circle for every Node in set
        for circle in circles:
            pygame.draw.circle(screen, BLACK, circle.pos, radius, lineThickness)
        #draw every line for every edge
        for line in lines:
            pygame.draw.line(screen, BLACK, line.startPos, line.endPos, lineThickness)
        #reset window
        pygame.display.flip()
        #refresh time
        timer += (clock.tick(fps)/1000)


#start game
game_loop()