import os
import sys

# append the base directory to sys
file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(dir_path)

from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from myGym.Env import Env
from myGym.Memory import Memory
from myModel.Agent import GreedyPolicyAgent
from myModel.Policy import SimplePolicy, ConvPolicy

from util.util import *
from util.const import *

N_EPISODE = 200
LR = 0.005 # 0.001
GAMMA = 0.9
BATCH_SIZE = 32
BATCH_PER_EP = 4

JSON_FILE = 'myModel/model/convModel.json'
WEIGHT_FILE = 'myModel/model/convWeights.h5'

def main():
    print('Starting training.py\n')

    hist = {'loss':[], 'val_loss':[], 'ep_n':0}

    memory = Memory(max_size=10000)
    policy = SimplePolicy(learning_rate=LR)
    policy.save_model(JSON_FILE, WEIGHT_FILE)
    policy.compile()

    while memory.get_size() < 10000:
        date_folder_name = np.random.choice(TRAIN_FOLDERS)
        temp_memory = []

        env = Env(date_folder_name, verbose=None)
        cur_state = env.cur_state
        cur_action = env.get_cur_action()

        done = False

        while not done:
            next_state, reward, _, cw_hcr, pc, done = env.step()

            if not done:
                next_action = env.get_cur_action
            else:
                next_action = None

            if (reward is not None) & (pc > MIN_PC_CUTOFF):
                temp_memory.append((cur_state, cur_action, reward, cw_hcr, pc, done))
            else:
                temp_memory.append((cur_state, cur_action, np.nan, np.nan, np.nan, done))

            cur_state = next_state
            cur_action = next_action

        for i in range(len(temp_memory) - 5):
            parital_rewards = list(map(lambda x: x[2], temp_memory[i:(i+5)]))
            partial_cwhcr = list()
            if np.all(~np.isnan(parital_rewards)):
                memory.append((cur_state, cur_action, reward, cw_hcr, pc, done))
