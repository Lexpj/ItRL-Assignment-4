from Environment import *
from Agents import *
import sys
import matplotlib.pyplot as plt
import numpy as np
from Helper import LearningCurvePlot, ComparisonPlot, smooth, plotarrows, render_policy


def run_repetitions(agent_type, n_episodes = 10000, n_rep = 100, alpha = 0.1, gamma = 1.0, epsilon = 0.1):
    
    res = np.zeros((n_rep, n_episodes))
    env = TomAndJerryEnvironment()
    
    if agent_type == "Q_Learning":

        for rep in range(n_rep):
            agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            for episode in range(n_episodes):
                #The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Choose A from S using egreedy derived from Q 
                    action = agent.select_action(state)
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Q(S,A) = Q(S,A) + alpha(R + gamma*max_a(Q(S',A)) - Q(S,A))
                    agent.update(state, action, reward, stateprime, alpha = alpha, gamma = gamma)
                    # S = S'
                    state = stateprime
                # Reset environment
                env.reset()  
        

    elif agent_type == "SARSA":
        
        for rep in range(n_rep):
            agent = SARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            for episode in range(n_episodes):
                #The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Choose A from S using egreedy derived from Q 
                action = agent.select_action(state)
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Choose A' from S' using egreedy derived from Q
                    actionprime = agent.select_action(stateprime)
                    # Q(S,A) = Q(S,A) + alpha(R + gamma * Q(S',A') - Q(S,A))
                    agent.update(state, action, reward, stateprime, actionprime, alpha = alpha, gamma = gamma)
                    # S = S'
                    state = stateprime
                    # A = A'
                    action = actionprime
                # Reset environment
                env.reset()      
    
    
    elif agent_type == "Expected_SARSA":

        for rep in range(n_rep):
            agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon)
            for episode in range(n_episodes):
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Choose A from S using egreedy derived from Q 
                    action = agent.select_action(state)
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Q(S,A) = Q(S,A) + alpha(R + gamma*sum_{A'}(p(A'|S')*Q(S',A')) - Q(S,A))
                    agent.update(state,action,reward,stateprime,alpha=alpha,gamma=gamma)
                    # S = S'
                    state = stateprime
                # Reset environment
                env.reset()        
           
    return agent, env, res

# Visualizing the action with the maximum value for each state
def action_max_value_plot(agent_type, windy = False, n_rep = 1, n_episodes = 10000):
    if windy:
        agent, env, _ = run_repetitions(agent_type = agent_type, windy = windy, n_rep = n_rep, n_episodes = n_episodes)
        s = render_policy(env, agent)
        plotarrows(s, path = agent_type + '_windy_1.png')
    else:
        agent, env, _ = run_repetitions(agent_type = agent_type, windy = windy, n_rep = n_rep, n_episodes = n_episodes)
        s = render_policy(env, agent)
        plotarrows(s, path = agent_type + '_1.png')

# Plotting the averaged learning curves
def averaged_learning_curve_plot(agent_type, n_rep = 100, n_episodes = 1000):
    smoothing_window = 29
    
    _, _, result = run_repetitions(agent_type = agent_type, n_rep = n_rep, n_episodes = n_episodes)
    averaged_rewards = np.empty(n_episodes)
    for i in range(n_episodes):
        averaged_rewards[i] = result[:, i].mean()

    learning_curve = LearningCurvePlot()
    learning_curve.add_curve(smooth(y = averaged_rewards, window = smoothing_window), label = 'alpha = 0.1')
    learning_curve.save(name = agent_type + '_2.png')
    
# Plotting the final performance for different alpha values
def varying_alpha_experiment(agent_type, alpha = [0.01, 0.1, 0.5, 0.9], n_rep = 100, n_episodes = 1000):
    smoothing_window = 29
    
    learning_curve = LearningCurvePlot()
    
    for i in range(len(alpha)):
        _, _, result = run_repetitions(agent_type = agent_type, n_rep = n_rep, n_episodes = n_episodes, alpha = alpha[i])
        averaged_rewards = np.empty(n_episodes) 
        for k in range(n_episodes):
            averaged_rewards[k] = result[:, k].mean()
        learning_curve.add_curve(smooth(y = averaged_rewards, window = smoothing_window), label = 'alpha = ' + str(alpha[i]))

    learning_curve.save(name = agent_type + '_3.png')  


def main():
    # Q-learning experiments
    action_max_value_plot(agent_type = 'Q_Learning')
    averaged_learning_curve_plot(agent_type = 'Q_Learning')
    varying_alpha_experiment(agent_type = 'Q_Learning')
    
    # SARSA experiments
    action_max_value_plot(agent_type = 'SARSA')
    averaged_learning_curve_plot(agent_type = 'SARSA')
    varying_alpha_experiment(agent_type = 'SARSA')
    
    # Stormy weather experiment
    action_max_value_plot(agent_type = 'Q_Learning', windy = True)
    action_max_value_plot(agent_type = 'SARSA', windy = True)
    
    #Expected-SARSA experiments
    action_max_value_plot(agent_type = 'Expected_SARSA')
    varying_alpha_experiment(agent_type = 'Expected_SARSA')
    # Optimal alpha: 0.1
    averaged_learning_curve_plot(agent_type = 'Expected_SARSA')


if __name__ == '__main__':
    main()