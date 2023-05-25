#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from Agents import *

class TomAndJerryEnvironment:

    class Cat:
        def __init__(self, coord:tuple):
            """
            coord: coordinate
            """    
            self.coord = np.array(coord)
            self.maxtime = 16
            self.time = np.random.randint(0,self.maxtime)

            self.speed = np.random.choice([-1,1],size=1)
            
            self.path = [ # gets overwritten anyways in setPath
                [(0,),(1,11),(2,10)],
                [(5,15),(4,12),(3,9)],
                [(6,14),(7,13),(8,)]
            ]

        def setPath(self,path):
            self.path = path
            self.maxtime = max([max([max(i) for i in x]) for x in path]) + 1
            self.time = np.random.randint(0,self.maxtime)
        
        def step(self):
            """
            Adjust the timestep, and thus the position when called
            """
            self.time = (self.time+self.speed+self.maxtime)%self.maxtime

        def pos(self):
            """
            Returns the delta position of the mid point of the cat
            """
            # find the current time in path
            for i in range(len(self.path)):
                for j in range(len(self.path[0])):
                    if self.time in self.path[i][j]:
                        return np.array([j,i])
            raise Exception("Cat made invalid move") # shouldn't occur

    def __init__(self, render_mode = None):        
        self.env = [
            '0000',
            '0000',
            '0000',
            '0000'
        ]

        self.rewards = {
            'cat': -10,
            'step': -1,
            'goal': 30,
        }

        self.width = 4
        self.height = 4

        self.n_states = self.height * self.width
        self.n_actions = 8
                
        self.cats = []
        self.info = {}
        self.goal = np.array((3,3))
        self.done = False
        self.state = self.reset()
        
        if render_mode == "human":
            self._render_setup()

    def stateToPos(self,s):
        """
        Converts a state s (int) to a grid position (tuple)
        """
        return np.array((s%self.width,s//self.width))
    
    def posToState(self,pos):
        """
        Converts a grid position (tuple) to a state s (int)
        """
        return pos[0] + pos[1]*self.width
    
    def reset(self, seed=None):
        """
        Resets the environment and randomizes the cats cycle pos
        """
        np.random.seed(seed)

        self.done = False
        self.info = {'pathLength':0}

        self.cats = [
            self.Cat((1,1))
        ]
        self.cats[0].setPath([
            [(0,),(1,)],
            [(3,),(2,)],
        ])
        
        s = self.posToState((0,0))
        self.state = s
        return s

    def shallow_reset(self):
        self.done = False

        self.cats = [
            self.Cat((1,1))
        ]
        self.cats[0].setPath([
            [(0,),(1,)],
            [(3,),(2,)],
        ])
        
        s = self.posToState((0,0))
        self.state = s
        return s

    def state_size(self):
        return self.n_states
    
    def action_size(self):
        return self.n_actions
    
    def _get_action_definitions(self):
        """
        Returns the step definitions by action.
        """
        return {0:np.array((0,-1)),1:np.array((-1,0)),2:np.array((0,1)),3:np.array((1,0)), 
                4:np.array((1,1)), 5:np.array((1,-1)), 6:np.array((-1,1)), 7:np.array((-1,-1))}
       
    def step(self,a):
        ''' Forward the environment based on action a 
        Returns the next state, the obtained reward, and a boolean whether the environment terminated '''
        
        r = 0
        
        # Move the agent
        s_next = self.stateToPos(self.state) + self._get_action_definitions()[a] 
        # bound within grid
        s_next[0] = min(max(0,s_next[0]),self.width-1)
        s_next[1] = min(max(0,s_next[1]),self.height-1)

        # Generate info: if clear via upwards or downwards, denote in info
        self.info['pathLength'] += 1

        # if cat is hit
        for cat in self.cats:
            coord = cat.coord
            oldpos = cat.pos()
            
            cat.step()

            delta = cat.pos()
            pos = coord+delta
            
            if np.all(s_next == pos):
                r += self.rewards['cat']
                s_next = self.shallow_reset()
                return s_next, r, self.done, self.info
            elif np.all(oldpos == s_next) and np.all(self.stateToPos(self.state) == delta):
                r += self.rewards['cat']
                s_next = self.shallow_reset()
                return s_next, r, self.done, self.info

        # check if not wall is hit
        if self.env[s_next[1]][s_next[0]] != '1':
            self.state = self.posToState(s_next)

        # Check reward and termination
        if np.all(s_next == self.goal):
            self.done = True
            r += self.rewards['goal']
        else:
            self.done = False
            r += self.rewards['step']
        
        return self.state, r, self.done, self.info

    def __str__(self):
        """
        Returns the map when `print(env)` is called
        """
        curMap = [[x for x in row] for row in self.env]
            
        for cat in self.cats:
            coord = cat.coord
            delta = cat.pos()
            pos = coord+delta

            curMap[pos[1]][pos[0]] = 'C'
        
        # agent position
        state = self.stateToPos(self.state)
        curMap[state[1]][state[0]] = 'M'

        # goal position
        curMap[self.goal[1]][self.goal[0]] = 'G'

        return '\n'.join([''.join(x) for x in curMap])+"\n"

    def __bool__(self):
        """
        Returns whether the environment has terminated by `if env == [True/False]`
        """
        return self.done

    def _render_setup(self):
        """
        Sets up the interactive plot of the environment.
        """
        plt.ion()
        self.fig, self.axs = plt.subplots(1,1)
        self.cmap = mpl.colors.ListedColormap(['white','black','green','orange','blue'])
        self.bounds= np.array(range(6))
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

    def render(self):
        """
        The render function that is called to render the current environment
        """
        curMap = self.__str__().rstrip().split('\n')
        newMap = []
        for row in curMap:
            row = row.replace('M','4').replace('C','3').replace('G','2').replace('T','5')
            row = np.array([int(x) for x in row])
            newMap.append(row)
        newMap = np.array(newMap)
        time.sleep(0.1)

        self.axs.cla()

        # tell imshow about color map so that only set colors are used
        img = self.axs.imshow(newMap,interpolation='nearest',
                            cmap = self.cmap,norm=self.norm)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
                


def test():
    # Hyperparameters
    n_test_steps = 25
    step_pause = 0.5
    
    # Initialize environment and Q-array
    env = TomAndJerryEnvironment(render_mode=None)
    s = env.reset()
    Q_sa = np.zeros((env.n_states,env.n_actions)) # Q-value array of flat zeros

    agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.01)
    agent.load()
    done = False

    heatmap = [0]*env.state_size()

    # Test
    for i in range(1000):
        while not done:
            #a = int(input())
            heatmap[s] += 1

            a = agent.select_action(s)   
            s_next,r,done,info = env.step(a) # execute action in the environment
            if done:
                s = env.reset()
            else:
                s = s_next

    heatmap = np.array(heatmap)
    heatmap = np.reshape(heatmap, (4,4))
    print(heatmap)

    plt.imshow(heatmap, cmap='autumn')
    plt.show()
    
if __name__ == '__main__':
    test()
