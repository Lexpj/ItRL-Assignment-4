#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class TomAndJerryEnv:

    class Cat:
        def __init__(self, mid:tuple):
            """
            mid: coordinate
            """    
            self.mid = mid
            self.time = np.random.randint(0,17) # Random number between 0 and 16
            self.rotation = np.random.randint(0,4)
            self.mirror = np.random.randint(0,1)

        def move(self):
            """
            Returns the state respect to the current env grid
            """
            pass
            return 
        

    def __init__(self):
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




        self.height = 7
        self.width = 10
        self.shape = (self.width, self.height)
        self.n_states = self.height * self.width
        self.n_actions = 4
        self.winds = (0,0,0,1,1,1,2,2,1,0)
        
        self.action_effects = {
                0: (0, 1),  # up
                1: (1, 0),   # right
                2: (0, -1),   # down
                3: (-1, 0),  # left
                }
        self.fig = None
        self.Q_labels = None
        self.arrows = None
        
        self.reset()

    def state_to_location(self,state):
        ''' bring a state index to an (x,y) location of the agent '''
        return np.unravel_index(state,self.shape)
    
    def location_to_state(self,location):
        ''' bring an (x,y) location of the agent to a state index '''
        return np.ravel_multi_index(location,self.shape)

    def reset(self):
        ''' set the agent back to the start location '''
        self.location = np.array([0,3])
        s = self.location_to_state(self.location)
        return s
        
    def step(self,a):
        ''' Forward the environment based on action a 
        Returns the next state, the obtained reward, and a boolean whether the environment terminated '''
        # Move the agent
        self.location += self.action_effects[a] # effect of action
        self.location = np.clip(self.location,(0,0),np.array(self.shape)-1) # bound within grid
        self.location[1] += self.winds[self.location[0]] # effect of wind
        self.location = np.clip(self.location,(0,0),np.array(self.shape)-1) # bound within grid
        s_next = self.location_to_state(self.location)    
        
        # Check reward and termination
        if np.all(self.location == (7,3)):
            done = True
            r = 20
        else:
            done = False
            r = -1
        
        return s_next, r, done 

    def render(self,Q_sa=None,plot_optimal_policy=False,step_pause=0.001):
        ''' Plot the environment 
        if Q_sa is provided, it will also plot the Q(s,a) values for each action in each state
        if plot_optimal_policy=True, it will additionally add an arrow in each state to indicate the greedy action '''
        # Initialize figure
        if self.fig == None:
            self._initialize_plot()
            
        # Add Q-values to plot
        if Q_sa is not None:
            # Initialize labels
            if self.Q_labels is None:
                self._initialize_Q_labels()
            # Set correct values of labels
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q_labels[state][action].set_text(np.round(Q_sa[state,action],1))

        # Add arrows of optimal policy
        if plot_optimal_policy and Q_sa is not None:
            self._plot_arrows(Q_sa)
            
        # Update agent location
        self.agent_circle.center = self.location+0.5
            
        # Draw figure
        plt.pause(step_pause)

    def _initialize_plot(self):
        self.fig,self.ax = plt.subplots()#figsize=(self.width, self.height+1)) # Start a new figure
        self.ax.set_xlim([0,self.width])
        self.ax.set_ylim([0,self.height]) 
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)

        for x in range(self.width):
            for y in range(self.height):
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0, facecolor='k',alpha=self.winds[x]/4))       
                self.ax.add_patch(Rectangle((x, y),1,1, linewidth=0.5, edgecolor='k', fill=False))     

        self.ax.axvline(0,0,self.height,linewidth=5,c='k')
        self.ax.axvline(self.width,0,self.height,linewidth=5,c='k')
        self.ax.axhline(0,0,self.width,linewidth=5,c='k')
        self.ax.axhline(self.height,0,self.width,linewidth=5,c='k')


        # Indicate start and goal state
        self.ax.add_patch(Rectangle((0.0, 3.0),1.0,1,0, linewidth=0, facecolor='r',alpha=0.2))
        self.ax.add_patch(Rectangle((7.0, 3.0),1.0,1,0, linewidth=0, facecolor='g',alpha=0.2))
        self.ax.text(0.05,3.75, 'S', fontsize=20, c='r')
        self.ax.text(7.05,3.75, 'G', fontsize=20, c='g')


        # Add agent
        self.agent_circle = Circle(self.location+0.5,0.3)
        self.ax.add_patch(self.agent_circle)
        
    def _initialize_Q_labels(self):
        self.Q_labels = []
        for state in range(self.n_states):
            state_location = self.state_to_location(state)
            self.Q_labels.append([])
            for action in range(self.n_actions):
                plot_location = np.array(state_location) + 0.42 + 0.35 * np.array(self.action_effects[action])
                next_label = self.ax.text(plot_location[0],plot_location[1]+0.03,0.0,fontsize=8)
                self.Q_labels[state].append(next_label)

    def _plot_arrows(self,Q_sa):
        if self.arrows is not None: 
            for arrow in self.arrows:
                arrow.remove() # Clear all previous arrows
        self.arrows=[]
        for state in range(self.n_states):
            plot_location = np.array(self.state_to_location(state)) + 0.5
            max_actions = full_argmax(Q_sa[state])
            for max_action in max_actions:
                new_arrow = arrow = Arrow(plot_location[0],plot_location[1],self.action_effects[max_action][0]*0.2,
                                          self.action_effects[max_action][1]*0.2, width=0.05,color='k')
                ax_arrow = self.ax.add_patch(new_arrow)
                self.arrows.append(ax_arrow)
                
def full_argmax(x):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max '''
    return np.where(x == np.max(x))[0]


def test():
    # Hyperparameters
    n_test_steps = 25
    step_pause = 0.5
    
    # Initialize environment and Q-array
    env = WindyGridworld()
    env.reset()
    Q_sa = np.zeros((env.n_states,env.n_actions)) # Q-value array of flat zeros

    # Test
    for t in range(n_test_steps):
        a = np.random.randint(4) # sample random action    
        s_next,r,done = env.step(a) # execute action in the environment
        env.render(Q_sa=Q_sa,plot_optimal_policy=False,step_pause=step_pause) # display the environment
    
if __name__ == '__main__':
    test()
