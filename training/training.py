import os
import sys

# append the base directory to sys
file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(os.path.dirname(file_path))
sys.path.append(dir_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from myGym.Env import Env
from myGym.Memory import Memory
# from myModel.Agent import GreedyPolicyAgent
# from myModel.Policy import ConvPolicy

from util.util import *
from util.const import *

def main():
    hist = {'chhc':[], 'cwhc':[], 'pc':[], 'COP': []}

    choice = np.random.choice(len(DATE_FOLDERS))
    choice_date_folder = DATE_FOLDERS[choice]
    env = Env(choice_date_folder)

    cur_state = env.cur_state
    next_action = env.get_next_action()
    done = False

    while not done:
        env.get_next_action()
        next_state, reward, chhc, cwhc, pc, done = env.step()
        if pc > MIN_PC_CUTOFF:
            hist['chhc'].append(chhc)
            hist['cwhc'].append(cwhc)
            hist['pc'].append(pc)
            hist['COP'].append(reward)

    x = range(len(hist['chhc']))
    plt.plot(x, hist['chhc'])
    plt.figure()
    plt.plot(x, hist['cwhc'])
    plt.figure()
    plt.plot(x, hist['pc'])
    plt.figure()
    plt.plot(x, hist['COP'])
    plt.figure()
    plt.scatter(
        np.array(hist['chhc']) + np.array(hist['cwhc']),
        hist['pc'],
        c = hist['COP']
    )
    plt.show()


if __name__ == '__main__':
    main()
