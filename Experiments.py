# ItRL: Assignment 4
# Leiden University
# Maksim Terentev and Lex Jansssens
# Last changes 21-05-2023

import sys
import numpy as np

from Environment import *
from Agents import *
from Helper import LearningCurvePlot, ComparisonPlot, smooth

## Runs an experiment n_episodes times for n_repetitions for the agent_type ##
def run_repetitions(agent_type, n_episodes = 1000, n_repetitions = 100, alpha = 0.1, gamma = 1.0, epsilon = 0.01):
    # Initialize the cumulative rewards array and the environment
    cumulative_rewards = np.zeros((n_repetitions, n_episodes))
    env = TomAndJerryEnvironment(render_mode = None)

    # Q-learning agent
    if agent_type == "Q-learning":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Loop until the goal state is achieved
                while not env.done:
                    # Choose action a using the agent's policy
                    a = agent.select_action(s)
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, _ = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, alpha = alpha, gamma = gamma)
                    s = s_prime
                    #env.render()
    # SARSA agent
    elif agent_type == "SARSA":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = SARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Choose action a using the agent's policy
                a = agent.select_action(s)
                # Loop until the goal state is achieved
                while not env.done:
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, _ = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    s_prime = env.state
                    # Choose a' from s' using the agent's policy
                    a_prime = agent.select_action(s_prime)
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, a_prime, alpha = alpha, gamma = gamma)
                    s = s_prime
                    a = a_prime
                    #env.render()          
    # Expected SARSA agent                
    elif agent_type == "Expected SARSA":
        # Conduct the experiment n_repetitions
        for repetition in range(n_repetitions):
            # Initialize the agent
            agent = ExpectedSARSAAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            # Run the agent for n_episodes
            for episode in range(n_episodes):
                # The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20) + "."*(20-(int(((episode+1+(repetition*n_episodes))/(n_episodes*n_repetitions)) * 20))) +f"] {episode+1+(repetition*n_episodes)}/{(n_episodes*n_repetitions)}")
                sys.stdout.flush()
                # Initialize state s
                s = env.reset()
                # Loop until the goal state is achieved
                while not env.done:
                    # Choose the action using the agent's policy
                    a = agent.select_action(s)
                    # Take action a, observe state s' and reward r
                    s_prime, r, _, _ = env.step(a)
                    # Save the reward
                    cumulative_rewards[repetition][episode] += r
                    s_prime = env.state
                    # Update the Q-table
                    agent.update(s, a, r, s_prime, alpha = alpha, gamma = gamma)
                    s = s_prime        

    agent.save()
    return cumulative_rewards


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