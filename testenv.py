import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import sys
from Environment import *
from Agents import *
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def run_repetitions(agent_type, ratio, n_episodes = 1000, n_repetitions = 10, alpha = 0.1, gamma = 1.0, epsilon = 0.01):
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


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(100,1000,1)
Y = np.arange(1,12,2)
X, Y = np.meshgrid(X, Y)

# Z = np.array([smooth(run_repetitions('Q-learning',i)[1][100:],29) for i in range(1,12,2)])
reward, path = [],[]
for i in range(2, 3, 2):
    r, p = smooth(run_repetitions('Q-learning',i),29)
    reward.append(r[100:])
    path.append(p[100:])
exit()
Z = np.array(reward)
print(Z)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# ax.set_zlim(5,20)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


