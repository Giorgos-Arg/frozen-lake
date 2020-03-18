from __future__ import division
import gym
from gym.envs.registration import register
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import clear_output
import sys

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8', 'is_slippery': False},
)


def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class Agent:
    def __init__(self, env, environment):
        self.stateCnt = env.observation_space.n
        self.actionCnt = env.action_space.n  # left:0; down:1; right:2; up:3
        if(environment == "-s"):
            self.learning_rate = 0.1
        else:
            self.learning_rate = 1
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.9
        self.Q = self._initialiseModel()

    def _initialiseModel(self):
        return np.zeros((self.stateCnt, self.actionCnt))

    # Returns a vector with the value of each action in state s.
    def predict_value(self, s):
        return self.Q[s, :]

    #  Updates the current estimate of the value of the state-action pair <s,a> using Q-learning.
    def update_value_qlearning(self, s, a, r, s_next, terminalStateNotReached):
        if terminalStateNotReached:
            self.Q[s, a] = self.Q[s, a] + self.learning_rate * (
                r + self.gamma * np.max(self.Q[s_next, :]) - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + \
                self.learning_rate * (r - self.Q[s, a])

    #  Updates the current estimate of the value of the state-action pair <s,a> using Sarsa.
    def update_value_sarsa(self, s, a, r, s_next, a_next, terminalStateNotReached):
        if terminalStateNotReached:
            self.Q[s, a] = self.Q[s, a] + self.learning_rate * \
                (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + \
                self.learning_rate * (r - self.Q[s, a])

    # Returns the action to execute in state s, implementing an ε-greedy policy.
    def choose_action(self, s):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.epsilon:
            action = np.argmax(self.predict_value(s))
        else:
            action = env.action_space.sample()
        return action


class World:
    def __init__(self, env, nbOfTrainingEpisodes):
        self.env = env
        print('Environment has %d states and %d actions.' %
              (self.env.observation_space.n, self.env.action_space.n))
        self.stateCnt = env.observation_space.n
        self.actionCnt = env.action_space.n
        self.maxStepsPerEpisode = 200
        self.q_Sinit_progress = np.zeros(
            (nbOfTrainingEpisodes, self.actionCnt))

    #  Runs an episode, learning with the Q-learning algorithm.
    def run_episode_qlearning(self, agent, episodeCount):
        state = self.env.reset()  # "reset" environment to start state
        # fill q_Sinit_progress table
        temp = agent.predict_value(state)
        for j in range(self.actionCnt):
            self.q_Sinit_progress[episodeCount, j] = temp[j]
        r_total = 0
        episodeStepsCnt = 0
        success = False
        for i in range(self.maxStepsPerEpisode):
            episodeStepsCnt += 1
            action = agent.choose_action(state)
            new_state, reward, terminalStateReached, info = env.step(action)
            # Update Q-table
            agent.update_value_qlearning(
                state, action, reward, new_state, not terminalStateReached)
            state = new_state
            r_total += reward
            if terminalStateReached:
                if reward == 1:
                    success = True
                break
        # We only want the number of steps for successful episodes throughout training.
        if not success:
            episodeStepsCnt = 0
        return r_total, episodeStepsCnt

    #  Runs an episode, learning with the Sarsa algorithm.
    def run_episode_sarsa(self, agent, episodeCount):
        state = self.env.reset()  # "reset" environment to start state
        action = agent.choose_action(state)
        # fill q_Sinit_progress table
        temp = agent.predict_value(state)
        for j in range(self.actionCnt):
            self.q_Sinit_progress[episodeCount, j] = temp[j]
        r_total = 0
        episodeStepsCnt = 0
        success = False
        for i in range(self.maxStepsPerEpisode):
            episodeStepsCnt += 1
            next_state, reward, terminalStateReached, info = env.step(action)
            next_action = agent.choose_action(next_state)
            # Update Q-table
            agent.update_value_sarsa(
                state, action, reward, next_state, next_action, not terminalStateReached)
            state = next_state
            action = next_action
            r_total += reward
            if terminalStateReached:
                if reward == 1:
                    success = True
                break
        # we only want the number of steps for successful episodes throughout training.
        if not success:
            episodeStepsCnt = 0
        return r_total, episodeStepsCnt

    # runs an episode executing the currently optimal policy.
    def run_evaluation_episode(self, agent):
        state = env.reset()
        agent.epsilon = 0
        terminalStateReached = False
        print(
            "\n========================\n Evaluation Episode \n========================\n")
        time.sleep(1)
        for i in range(self.maxStepsPerEpisode):
            env.render()
            action = agent.choose_action(state)
            next_state, reward, terminalStateReached, info = env.step(action)
            if terminalStateReached:
                env.render()
                if reward == 1:
                    print(
                        "\n========================\n Success:  goal reached! \n========================\n")
                else:
                    print(
                        "\n========================\n Failure: agent fell in a hole! \n========================\n")
                break
            state = next_state
        if not terminalStateReached:
            env.render()
            print(
                "\n========================\n Failure: Maximum number of steps! \n========================\n")


if __name__ == '__main__':
    # Parse command line arguments
    if(len(sys.argv) == 4):
        environment = sys.argv[1]
        algorithm = sys.argv[2]
        nbOfTrainingEpisodes = int(sys.argv[3])
    if(len(sys.argv) != 4 or (algorithm != "-s" and algorithm != "-q") or (environment != "-s" and environment != "-d")):
        print("\npy frozen-lake.py <environment> <algorithm> <number_of_training_episodes>\nClass can be:\n -s "
              "for Sarsa\n -q for Qlearning\nEnvironment can be:\n -s for stochastic environment\n -d for deterministic"
              " environment")
        sys.exit(0)
    if(environment == "-s"):
        env = gym.make('FrozenLake8x8-v0')
        print("\nUsing stochastic environment.")
    else:
        env = gym.make('FrozenLakeNotSlippery-v0')
        print("\nUsing deterministic environment.")
    if(algorithm == "-s"):
        print("Using Sarsa algorithm.")
    else:
        print("Using Qlearning algorithm.")
    print('Training for ', nbOfTrainingEpisodes, ' episodes...')
    world = World(env, nbOfTrainingEpisodes)
    agent = Agent(env, environment)
    r_total_progress = []
    episodeStepsCnt_progress = []
    for i in range(nbOfTrainingEpisodes):
        if(algorithm == "-s"):
            r_total, episodeStepsCnt = world.run_episode_sarsa(agent, i)
        else:
            r_total, episodeStepsCnt = world.run_episode_qlearning(agent, i)
        r_total_progress.append(r_total)
        episodeStepsCnt_progress.append(episodeStepsCnt)
    print('Training done!!!')
    # Run evaluation episode
    world.run_evaluation_episode(agent)
    # Plot the evolution of the number of steps per successful episode throughout training. A successful episode is
    # an episode where the agent reached the goal.
    plt.plot(episodeStepsCnt_progress)
    plt.title("Number of steps per successful episode")
    plt.show()
    # Plot the evolution of the total collected rewards per episode throughout training (use the running_mean
    # function to smooth the plot)
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.title("Rewards collected per episode")
    plt.show()
