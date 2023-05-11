#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pygame


class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Cumulative reward')      
        #self.ax.set_ylim([-200, 0])
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,y,label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y,label=label)
        else:
            self.ax.plot(y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

class ComparisonPlot:

    def __init__(self,title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('Parameter (exploration)')
        self.ax.set_ylabel('Average reward') 
        self.ax.set_xscale('log')
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self,x,y,label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x 
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

# Makes a plot of the action with the maximum value for each stat
def plotarrows(s: str, path: str) -> None:
    WIDTH = 768
    HEIGHT = 768
    TILEWIDTH = WIDTH//12
    TILEHEIGHT = HEIGHT//12
    surface = pygame.Surface((WIDTH,HEIGHT))
    surface.fill((255,255,255))
    arrow = pygame.transform.scale(pygame.image.load("arrow.png"), (TILEWIDTH,TILEHEIGHT))
    # Draw grid
    for i, item in enumerate(s):
        # If one of the starting positions (not represented in s, therefore this if)
        if i == 26 or i == 110:
            pygame.draw.rect(surface, (0,0,255), ((i%12)*TILEWIDTH,(i//12)*TILEHEIGHT,TILEWIDTH,TILEHEIGHT))
        if item in "UDLR":
            arr = arrow.copy()
            if item == "U": arr = pygame.transform.rotate(arr, 90)
            elif item == "D": arr = pygame.transform.rotate(arr, 270)
            elif item == "L": arr = pygame.transform.rotate(arr, 180) 
            surface.blit(arr,((i%12)*TILEWIDTH,(i//12)*TILEHEIGHT))
        else:
            if item == "G": pygame.draw.rect(surface, (0,255,0), ((i%12)*TILEWIDTH,(i//12)*TILEHEIGHT,TILEWIDTH,TILEHEIGHT))
            elif item == "C": pygame.draw.rect(surface, (255,0,0), ((i%12)*TILEWIDTH,(i//12)*TILEHEIGHT,TILEWIDTH,TILEHEIGHT))
    # Draw lines on the surface
    for i in range(12):
        pygame.draw.line(surface, (0,0,0), (TILEWIDTH*i, 0), (TILEWIDTH*i, HEIGHT))
        pygame.draw.line(surface, (0,0,0), (0,TILEHEIGHT*i), (WIDTH,TILEHEIGHT*i))
    pygame.draw.line(surface, (0,0,0), (WIDTH-1, 0), (WIDTH-1, HEIGHT))
    pygame.draw.line(surface, (0,0,0), (0,HEIGHT-1), (WIDTH,HEIGHT-1))  
    pygame.image.save(surface,path)

# Renders the (optimal) greedy trail of an trained agent 
def render_policy(env, agent):
    s = ""
    for r in range(env.r):
        for c in range(env.c):
            if env.s[r][c] == "X":
                a = np.argmax(agent.Q[r * env.c + c])
                if a == 0:
                    s += "U"
                elif a == 1:
                    s += "D"
                elif a == 2:
                    s += "L"
                elif a == 3:
                    s += "R"
            else:
                s += env.s[r][c]
    return s

if __name__ == '__main__':
    # Test Learning curve plot
    x = np.arange(100)
    y = 0.01*x + np.random.rand(100) - 0.4 # generate some learning curve y
    LCTest = LearningCurvePlot(title="Test Learning Curve")
    LCTest.add_curve(y,label='method 1')
    LCTest.add_curve(smooth(y,window=35),label='method 1 smoothed')
    LCTest.save(name='learning_curve_test.png')

    # Test Performance plot
    PerfTest = ComparisonPlot(title="Test Comparison")
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 1')
    PerfTest.add_curve(np.arange(5),np.random.rand(5),label='method 2')
    PerfTest.save(name='comparison_test.png') 