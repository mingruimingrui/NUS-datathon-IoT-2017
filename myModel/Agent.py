"""
Reference from
https://github.com/Rochester-NRT/RocAlphaGo/blob/develop/AlphaGo/ai.py
"""

import os
import numpy as np

from util.util import *

class GreedyPolicyAgent(object):
    """
    An agent that uses a greedy policy
    (i.e. chooses the highest reward action each step)
    """

    def __init__(self, policy_function):
        self.policy = policy_function

    def act(self, state):
        prev_action = state[-1]
        sensible_actions = generate_sensible_actions(state)

        
