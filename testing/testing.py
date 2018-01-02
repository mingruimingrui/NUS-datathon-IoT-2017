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
from myModel.Policy import SimplePolicy

from util.util import *
from util.const import *

JSON_FILE = 'myModel/model/model.json'
WEIGHT_FILE = 'myModel/model/weight.h5'


def main():
    date_folder_name = np.random.choice(TEST_FOLDERS)

    env = Env(date_folder_name)
    policy = SimplePolicy(json_file=JSON_FILE)
    agent = GreedyPolicyAgent(env.action_shape, policy)
    cur_state = env.cur_state

    done = False
    count = 0
    actual = []
    suggested = []

    for _ in range(200):
    # while not done:
        count += 1
        cur_action = env.get_cur_action()
        next_state, reward, _, _, pc, done = env.step()

        best_cur_action, best_cur_reward = agent.act(cur_state)
        # best_next_action, best_next_reward = agent.act(next_state)

        # action_diff = best_cur_action - cur_action
        # diff.append(action_diff)
        actual.append(cur_action)
        suggested.append(best_cur_action)

        sys.stdout.write('\r'+ str(count))
        sys.stdout.flush()

        cur_state = next_state

        # if not done:
        #     next_action = env.get_cur_action()
        #
        #     cur_pred_reward = policy.predict(cur_state, cur_action)[0,0]
        #     next_pred_reward = policy.predict(next_state, next_action)[0,0]



            # print(cur_pred_reward, next_pred_reward)
            # print(best_cur_reward, best_next_reward)
            # print(best_cur_reward - best_next_reward * 0.9)

    # diff = np.array(diff).reshape(len(diff), -1)
    actual = np.array(actual).reshape(len(actual), -1)
    suggested = np.array(suggested).reshape(len(suggested), -1)
    x = np.arange(len(actual))

    for i in range(24):
        # plt.hist(diff[:,i])
        plt.figure()
        plt.plot(x, actual[:,i], label='actual')
        plt.plot(x, suggested[:,i], label='suggested')
        plt.legend()
        plt.title(CONFIGURABLE_COLS[i//4])
        plt.show()


if __name__ == '__main__':
    main()
