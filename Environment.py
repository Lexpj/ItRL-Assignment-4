#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class TomAndJerryEnvironment:

    class Cat:
        def __init__(self, mid:tuple):
            """
            mid: coordinate
            """    
            self.mid = np.array(mid)
            self.time = np.random.randint(0,16) # Random number between 0 and 15

            self.mirrorX = np.random.randint(0,1)
            self.mirrorY = np.random.randint(0,1)
            self.speed = np.random.choice([-1,1],size=1)
            
            self.eating = 0
            
            self.path = [
                [(0,),(1,11),(2,10)],
                [(5,15),(4,12),(3,9)],
                [(6,14),(7,13),(8,)]
            ]
            self.transformPath()
            

        def transformPath(self):
            """
            Adjust the path by mirroring the self.path 2D array
            Mirroring can happen in the X-axis and in the Y-axis
            Along with this, the speed (-1 or 1) is the last randomizable factor,
            allowing up to 2^3 = 8 variations of the path
            """
            if self.mirrorX:
                self.path = self.path[::-1]
            if self.mirrorY:
                self.path = [x[::-1] for x in self.path]    

        def step(self,treats):
            """
            Adjust the timestep, and thus the position when called
            """
            # Get position
            pos = self.mid + self.pos()
            toBeDeleted = []
            
            for treat in treats:
                if np.all(pos == treat):
                    toBeDeleted.append(treat)
                    self.eating += 2
            
            for treat in toBeDeleted:
                treats.remove(treat)
            
            if not self.eating:
                self.time = (self.time+self.speed+16)%16
            else:
                self.eating -= 1
            
            return self.eating > 0


        def pos(self):
            """
            Returns the delta position of the mid point of the cat
            """
            # find the current time in path
            for i in range(3):
                for j in range(3):
                    if self.time in self.path[i][j]:
                        return np.array([j-1,i-1])
            raise Exception("Cat made invalid move") # shouldn't occur

    def __init__(self, render_mode = None):        
        self.env = [
            '0000000',
            '0000011',
            '0000000',
            '0001000',
            '0001000',
            '0001001',
            '0001000',
            '0000000'
        ]

        self.rewards = {
            'cat': -10,
            'step': -1,
            'goal': 30,
            'eating': 50
        }

        self.width = 7
        self.height = 8

        self.n_states = self.height * self.width
        self.n_actions = 5
                
        self.cats = []
        self.goal = np.array((5,6))
        self.done = False
        self.state = self.reset()
        self.nr_treats = 1
        self.treats = []
        
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
        self.nr_treats = 1
        self.treats = []
        
        self.cats = [
            self.Cat((1,5)),
            self.Cat((5,3)),
            self.Cat((5,6))
        ]

        s = self.posToState((0,1))
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
        return {0:np.array((0,-1)),1:np.array((-1,0)),2:np.array((0,1)),3:np.array((1,0)),4:np.array((0,0))}
       
    def step(self,a):
        ''' Forward the environment based on action a 
        Returns the next state, the obtained reward, and a boolean whether the environment terminated '''
        
        r = 0
        # If treats available, lay one down
        if a == 4 and self.nr_treats > 0:
            self.nr_treats -= 1
            self.treats.append(self.stateToPos(self.state))
        
        # info
        info = {'treatsLeft':self.treats}

        # Move the agent
        s_next = self.stateToPos(self.state) + self._get_action_definitions()[a] 
        # bound within grid
        s_next[0] = min(max(0,s_next[0]),self.width-1)
        s_next[1] = min(max(0,s_next[1]),self.height-1)

        # if cat is hit
        for cat in self.cats:
            mid = cat.mid
            eating = cat.step(self.treats)
            if eating: r += self.rewards['eating'] 

            delta = cat.pos()
            pos = mid+delta
            if np.all(s_next == pos):
                self.done = True
                r += self.rewards['cat']
                return s_next, r, self.done, info

        # check if wall is hit
        if self.env[s_next[1]][s_next[0]] == '1':
            pass # no update
        else:
            self.state = self.posToState(s_next)

        # Check reward and termination
        if np.all(s_next == self.goal):
            self.done = True
            r += self.rewards['goal']
        else:
            self.done = False
            r += self.rewards['step']
        
        return self.state, r, self.done, info

    def __str__(self):
        """
        Returns the map when `print(env)` is called
        """
        curMap = [[x for x in row] for row in self.env]

        for treat in self.treats:
            curMap[treat[1]][treat[0]] = "T"
            
        for cat in self.cats:
            mid = cat.mid
            delta = cat.pos()
            pos = mid+delta

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
        self.cmap = mpl.colors.ListedColormap(['white','black','green','orange','yellow','blue'])
        self.bounds= np.array(range(7))
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
    env = TomAndJerryEnvironment(render_mode="human")
    env.reset()
    Q_sa = np.zeros((env.n_states,env.n_actions)) # Q-value array of flat zeros

    done = False

    # Test
    while not done:
        #a = int(input())
        a = np.random.randint(5) # sample random action    
        s_next,r,done,info = env.step(a) # execute action in the environment
        #env.render(Q_sa=Q_sa,plot_optimal_policy=False,step_pause=step_pause) # display the environment
        env.render()
    print(env)

if __name__ == '__main__':
    test()

