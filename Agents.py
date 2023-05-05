import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        e_greedy = np.zeros(self.n_actions)
        e_greedy.fill(self.epsilon / (self.n_actions - 1))
        e_greedy[np.argmax(self.Q[state])] = 1 - self.epsilon
        return np.random.choice(self.n_actions, p = e_greedy)
        
    def update(self, state, action, reward, stateprime, alpha = 0.1, gamma = 1):
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma * np.max(self.Q[stateprime]) - self.Q[state][action])

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        e_greedy = np.zeros(self.n_actions)
        e_greedy.fill(self.epsilon / (self.n_actions - 1))
        e_greedy[np.argmax(self.Q[state])] = 1 - self.epsilon
        return np.random.choice(self.n_actions, p = e_greedy)
        
    def update(self, state, action, reward, stateprime, actionprime, alpha = 0.1, gamma = 1):
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma * self.Q[stateprime][actionprime] - self.Q[state][action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        e_greedy = np.zeros(self.n_actions)
        e_greedy.fill(self.epsilon / (self.n_actions - 1))
        e_greedy[np.argmax(self.Q[state])] = 1 - self.epsilon
        return np.random.choice(self.n_actions, p = e_greedy)
        
    def update(self, state, action, reward, stateprime, alpha = 0.1, gamma = 1):
        prob = np.zeros(self.n_actions)
        prob.fill(self.epsilon / (self.n_actions - 1))
        prob[np.argmax(self.Q[state])] = 1 - self.epsilon
        self.Q[state][action] = self.Q[state][action] + alpha * (reward + gamma * sum([prob[a] * self.Q[stateprime][a] 
                                                                                     for a in range(self.n_actions)]) - self.Q[state][action])
