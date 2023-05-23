from Environment import *
from Agents import *
import sys
import matplotlib.pyplot as plt
import numpy as np
from Helper import LearningCurvePlot, ComparisonPlot, smooth, plotarrows, render_policy


def run_repetitions(agent_type, n_episodes = 10000, n_rep = 100, alpha = 0.1, gamma = 1.0, epsilon = 0.3):
    
    res = np.zeros((n_rep, n_episodes))
    env = TomAndJerryEnvironmentTunnel(render_mode=None)
    
    if agent_type == "Q_Learning":

        for rep in range(n_rep):
            agent = QLearningAgent(n_actions = env.action_size(), n_states = env.state_size(), epsilon = epsilon)
            for episode in range(n_episodes):
                #The progress bar
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.reset()
                # Loop for each step of episode, until S is terminal
                while not env.done:
                    # Choose A from S using egreedy derived from Q 
                    action = agent.select_action(state)
                    # Take action A, observe R, S'
                    stateprime, reward, done, _ = env.step(action)
                    
                    res[rep][episode] += reward

                    # Q(S,A) = Q(S,A) + alpha(R + gamma*max_a(Q(S',A)) - Q(S,A))
                    agent.update(state, action, reward, stateprime, alpha = alpha, gamma = gamma)
                    # S = S'
                    state = stateprime
           
    return agent, res


def experimenttest():
    smoothing_window = 29
    
    learning_curve = LearningCurvePlot()

    agent, res = run_repetitions(agent_type = "Q_Learning", n_rep = 10, n_episodes = 10000)
    averaged_rewards = res.mean(axis=0)

    learning_curve.add_curve(smooth(y = averaged_rewards, window = smoothing_window))
    learning_curve.save(name = "Q_Learning.png")

    return agent  


def main():
    # Q-learning experiments
    agent = experimenttest()

    # Initialize environment and Q-array
    env = TomAndJerryEnvironmentTunnel(render_mode="human")
    s = env.reset()

    done = False

    # Test
    while True:
        #a = int(input())
        a = agent.select_action(s) # sample random action    

        s_next,r,done,info = env.step(a) # execute action in the environment

        #env.render(Q_sa=Q_sa,plot_optimal_policy=False,step_pause=step_pause) # display the environment
        env.render()
        s = s_next

        if done:
            s = env.reset()



    # averaged_learning_curve_plot(agent_type = 'Q_Learning')
    # varying_alpha_experiment(agent_type = 'Q_Learning')
    
    # # SARSA experiments
    # action_max_value_plot(agent_type = 'SARSA')
    # averaged_learning_curve_plot(agent_type = 'SARSA')
    # varying_alpha_experiment(agent_type = 'SARSA')
    
    # # Stormy weather experiment
    # action_max_value_plot(agent_type = 'Q_Learning', windy = True)
    # action_max_value_plot(agent_type = 'SARSA', windy = True)
    
    # #Expected-SARSA experiments
    # action_max_value_plot(agent_type = 'Expected_SARSA')
    # varying_alpha_experiment(agent_type = 'Expected_SARSA')
    # # Optimal alpha: 0.1
    # averaged_learning_curve_plot(agent_type = 'Expected_SARSA')


if __name__ == '__main__':
    main()