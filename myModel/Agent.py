import os
import numpy as np

from util.util import *

class GreedyPolicyAgent(object):
    """
    An agent that uses a greedy policy
    (i.e. chooses the highest reward action each step)
    """

    def __init__(self, action_shape, policy_function):
        self.policy = policy_function
        self.possible_action_directions = generate_possible_action_directions(action_shape)

    def act(self, state):
        prev_action = state[-1,COL_IN_ACTION]

        state = state.copy().squeeze()
        state = np.expand_dims(state, state.ndim - 1)
        state = np.dot(np.ones((len(self.possible_action_directions), 1)), state)

        predict_fn = lambda x: self.policy.predict(state, x)

        min_action = prev_action.copy()
        min_action[:2] = 0
        min_action[2:] *= 0.67

        max_action = prev_action.copy()
        max_action[:2] = 200
        max_action[2:] *= 1.5

        best_action, best_reward = step_wise_minimize(
            prev_action, self.possible_action_directions,
            predict_fn, min_action, max_action
        )

        return best_action, best_reward

    def batch_act(self, states):
        best_actions = []
        best_rewards = []

        for state in states:
            prev_action = state[COL_IN_ACTION]

            state = state.copy().squeeze()
            state = np.expand_dims(state, state.ndim - 1)
            state = np.dot(np.ones((len(self.possible_action_directions), 1)), state)

            predict_fn = lambda x: self.policy.predict(state, x)

            min_action = prev_action.copy()
            min_action[:2] = 0
            min_action[2:] *= 0.67

            max_action = prev_action.copy()
            max_action[:2] = 200
            max_action[2:] *= 1.5

            best_action, best_reward = step_wise_minimize(
                prev_action, self.possible_action_directions,
                predict_fn, min_action, max_action
            )

            best_actions.append(best_action)
            best_rewards.append(best_reward)

        best_actions = np.array(best_actions)
        best_rewards = np.array(best_rewards)

        return best_actions, best_rewards
