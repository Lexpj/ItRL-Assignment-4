# ItRL: Assignment 4
# Leiden University
# Maksim Terentev and Lex Jansssens
# Last changes 21-05-2023

import sys
import numpy as np

from Environment import *
from Agents import *
from Helper import LearningCurvePlot, ComparisonPlot, smooth
import matplotlib.pyplot as plt
from matplotlib import cm

## Runs an experiment n_episodes times for n_repetitions for the agent_type ##
def run_repetitions(agent_type, ratio, n_episodes = 1000, n_repetitions = 100, alpha = 0.1, gamma = 1.0, epsilon = 0.01):
    # Initialize the cumulative rewards array and the environment
    cumulative_rewards = np.zeros((n_repetitions, n_episodes))
    cumulative_pathlength = np.zeros((n_repetitions, n_episodes))
    env = TomAndJerryEnvironment(render_mode = None)
    env.rewards['cat'] = env.rewards['step']*ratio

    # Q-learning agent
    if agent_type == "Q-learning":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}    ")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Loop until the goal state is achieved
                while not env.done:
                    # Choose action a using the agent's policy
                    a = agent.select_action(s)
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, info = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    cumulative_pathlength[repetition][episode] += info['pathLength']
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, alpha = alpha, gamma = gamma)
                    s = s_prime
                    
            agent.save()

    # SARSA agent
    elif agent_type == "SARSA":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = SARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}    ")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Choose action a using the agent's policy
                a = agent.select_action(s)
                # Loop until the goal state is achieved
                while not env.done:
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, info = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    cumulative_pathlength[repetition][episode] += info['pathLength']
                    s_prime = env.state
                    # Choose a' from s' using the agent's policy
                    a_prime = agent.select_action(s_prime)
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, a_prime, alpha = alpha, gamma = gamma)
                    s = s_prime
                    a = a_prime

            agent.save()

    # Expected SARSA agent                
    elif agent_type == "Expected SARSA":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = ExpectedSARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}    ")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Loop until the goal state is achieved
                while not env.done:
                    # Choose the action using the agent's policy
                    a = agent.select_action(s)
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, info = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    cumulative_pathlength[repetition][episode] += info['pathLength']
                    s_prime = env.state
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, alpha = alpha, gamma = gamma)
                    s = s_prime  

            agent.save()
    
    return (cumulative_rewards.mean(axis = 0), cumulative_pathlength.mean(axis = 0))

## Heatmap
def heatmap(agent_type):  
    """
    Note that it will make a heatmap based on random agents present in some folder
    """  
    # Initialize environment and Q-array
    env = TomAndJerryEnvironment(render_mode=None)
    s = env.reset()
    
    agents = {
        'Q-learning': QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.01),
        'SARSA': SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.01),
        'Expected SARSA': ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.01)
    }


    agent = agents[agent_type]
    agent.load()
    done = False

    heatmap = [0]*env.state_size()

    # Run 1000 repetitions with randomly picked agents.
    for i in range(10000):
        agent.load()
        s = env.reset()
        done = False
        while not done:
            heatmap[s] += 1

            a = agent.select_action(s)   
            s_next,r,done,info = env.step(a) # execute action in the environment

            s = s_next

    heatmap = np.array(heatmap)
    # Normalize
    heatmap = (heatmap - min(heatmap))/(max(heatmap)-min(heatmap))
    # Get rid of the starting cell steps
    heatmap[0] = 0
    # Reshape into 4x4
    heatmap = np.reshape(heatmap, (4,4))

    plt.imshow(heatmap, cmap='autumn')
    plt.show()

## 3D plot of episodes and ratio's as a function to yield pathlength and rewards
def make3d(agent_type):

    # Get the ranges to make data
    X = np.arange(100,1000,1)
    Y = np.arange(1,16,2)
    X, Y = np.meshgrid(X, Y)

    # Run the experiment
    reward, path = [],[]
    for i in range(1, 16, 2):
        r, p = run_repetitions(agent_type,i)
        reward.append(smooth(r[100:],29))
        path.append(smooth(p[100:],29))
    

    # Plot the surface for path length
    Z = np.array(path)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Ratio")
    ax.set_zlabel("Path length")
    ax.invert_xaxis()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


    # Plot the surface for rewards
    Z = np.array(reward)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Ratio")
    ax.set_zlabel("Reward")
    ax.invert_xaxis()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


## Plotting the averaged cumulative reward for alpha = 0.1 ##
def averaged_learning_curve_plot(agent_type, n_repetitions = 100, n_episodes = 1000):
    cumulative_rewards = run_repetitions(agent_type = agent_type, n_repetitions = n_repetitions, n_episodes = n_episodes)
    avg_cumulative_rewards = cumulative_rewards.mean(axis = 0)

    smoothing_window = 29
    plot = LearningCurvePlot()
    plot.add_curve(smooth(y = avg_cumulative_rewards, window = smoothing_window), label = 'alpha = 0.1')
    plot.save(name = agent_type + '_avg_rewards.png')
    
## Plotting the averaged cumulative reward for various alpha values ##
def varying_alpha_plot(agent_type, n_repetitions = 100, n_episodes = 1000, alpha = [0.01, 0.1, 0.5, 0.9]):
    smoothing_window = 29
    plot = LearningCurvePlot()
    for i in range(len(alpha)):
        cumulative_rewards = run_repetitions(agent_type = agent_type, n_repetitions = n_repetitions, n_episodes = n_episodes, alpha = alpha[i])
        avg_cumulative_rewards = cumulative_rewards.mean(axis = 0)
        plot.add_curve(smooth(y = avg_cumulative_rewards, window = smoothing_window), label = 'alpha = ' + str(alpha[i]))
    plot.save(name = agent_type + '_various_alpha.png')      
    
## Plotting the averaged cumulative rewards for all three agents ##
def optimal_agents_plot(n_repetitions = 10, n_episodes = 500, optimal_alpha = [0.1, 0.1, 0.1]):
    cum_rewards_QLearning = run_repetitions(agent_type = "Q-learning", n_repetitions = n_repetitions, n_episodes = n_episodes, epsilon = optimal_alpha[0])
    avg_cumulative_rewards_QLearning = cum_rewards_QLearning.mean(axis = 0)
    cum_rewards_SARSA = run_repetitions(agent_type = "SARSA", n_repetitions = n_repetitions, n_episodes = n_episodes, epsilon = optimal_alpha[1])
    avg_cumulative_rewards_SARSA = cum_rewards_SARSA.mean(axis = 0)
    cum_rewards_ExpectedSARSA = run_repetitions(agent_type = "Expected SARSA", n_repetitions = n_repetitions, n_episodes = n_episodes, epsilon = optimal_alpha[2])
    avg_cumulative_rewards_ExpectedSARSA = cum_rewards_ExpectedSARSA.mean(axis = 0)
    
    smoothing_window = 29
    plot = LearningCurvePlot() # Plot the learning curves
    plot.add_curve(smooth(y = avg_cumulative_rewards_QLearning, window = smoothing_window), label = 'Q-learning')
    plot.add_curve(smooth(y = avg_cumulative_rewards_SARSA, window = smoothing_window), label = 'SARSA')
    plot.add_curve(smooth(y = avg_cumulative_rewards_ExpectedSARSA, window = smoothing_window), label = 'Expected SARSA')
    plot.save(name = 'OptimalAgents.png')
    

def main():
    # Averaged learning plots
    # averaged_learning_curve_plot(agent_type = 'Q-learning', n_repetitions = 1, n_episodes = 100)
    #averaged_learning_curve_plot(agent_type = 'SARSA', n_repetitions = 1, n_episodes = 1000)
    #averaged_learning_curve_plot(agent_type = 'Expected SARSA', n_episodes = 1000)
    
    # Varying alpha values plots
    #varying_alpha_plot(agent_type = 'Q-learning', n_episodes = 5000)
    #varying_alpha_plot(agent_type = 'SARSA')
    #varying_alpha_plot(agent_type = 'Expected SARSA')
    
    optimal_agents_plot(n_episodes = 1000)


if __name__ == '__main__':
    main()