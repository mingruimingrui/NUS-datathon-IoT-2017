"""
Used one of my old projects https://github.com/mingruimingrui/myEnv/blob/master/Env.py
and made some edits
currently this does not work like how a proper env should
there is no taking of actions, since all actions are pre defined
"""

import os
import types

import numpy as np
import pandas as pd

from util.util import *
from util.const import *

def isListLike(x):
    return isinstance(x, np.ndarray) | isinstance(x, list) | isinstance(x, tuple) | isinstance(x, pd.Series)

def isAscending(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))

class Env:
    """
    Home made env, works like OpenAI gym but works with multi dimensional arrays

    date_folder: string
        the folder location where the csv are stored
    start, end: int (optional)
        Set start and end time, inclusive.
    step_size: int
        With every step, how many timesteps to move forward to next state
        Minimum 1
        Default(1)
    lookback: int
        Signifies the number of timesteps to look back when returning state
        Minimum 1
        Default(5)
    """

    def __init__(self, date_folder, start=None, end=None,
        step_size=1, lookback=1, verbose=True):

        timestamps, columns, data = get_data_from_folder(date_folder)

        assert isListLike(timestamps), 'timestamps must be list-like'
        assert isAscending(timestamps), 'timestamps must be ascending'
        assert isinstance(data, np.ndarray), 'data must be np.ndarray object'
        assert len(timestamps) == len(data), 'there must be just as many timestamps as data entries'

        if start is not None:
            assert isinstance(start, int), 'start must be int'
            data = data[timestamps >= start]
            timestamps = timestamps[timestamps >= start]

        if end is not None:
            assert isinstance(end, int), 'end must be int'
            data = data[timestamps <= end]
            timestamps = timestamps[timestamps <= end]

        assert isinstance(step_size, int), 'step_size must be an int'
        assert step_size >= 1, 'step_size must be 1 or greater'

        assert isinstance(lookback, int), 'lookback must be an int'
        assert lookback >= 1, 'lookback must be 1 or greater'

        assert len(timestamps) >= lookback + step_size, 'not enough timesteps'

        self.name = date_folder
        self.timestamps = timestamps[np.arange(len(timestamps)) % step_size == 0]
        self.data = data[0::step_size]
        # self.data = data[np.arange(len(data)) % step_size == 0]
        self.step_size = step_size
        self.lookback = lookback

        self.action_shape = (-1,6,4)

        self.cur_data_index = 0
        self.cur_time = timestamps[self.cur_data_index + self.lookback - 1]
        self.cur_state = self.data[self.cur_data_index:(self.cur_data_index + self.lookback)]
        self.has_next = True

        if verbose is not None:
            print('New Env initiated')
            print('Data date:', date_folder)
            # print('Timestamps from', timestamps[0], 'to', timestamps[len(timestamps)-1])
            print('Step size:', step_size)
            print('Lookback:', lookback)
            print()


    def get_cur_action(self):
        """
        get next action in data

        returns action
        """

        assert self.has_next, 'no more steps left in env, please reset'

        next_action = self.data.copy()[(self.cur_data_index + self.lookback)]
        next_action = next_action[COL_IN_ACTION]
        next_action = np.expand_dims(next_action, 0)

        return next_action


    def step(self):
        """
        no action!!!

        returns next_state, reward, ch_heat_capacity_rate, cw_heat_capacity_rate, power_comsumption, done
        """

        assert self.has_next, 'no more steps left in env, please reset'

        # generate new state
        next_state = self.data.copy()[
            (self.cur_data_index + 1):
            (self.cur_data_index + 1 + self.lookback)
        ]

        # calculate reward
        # if chiller is above 15 degree then we assume that the chiller is off
        chillers_on = (
            (next_state[-1,COLS_TO_USE.index('p_ch1Watt')] > 0) &
            (next_state[-1,COLS_TO_USE.index('p_ch2Watt')] > 0) &
            (next_state[-1,COLS_TO_USE.index('p_ch3Watt')] > 0)
        )

        if np.sum(chillers_on) > 0:
            # calculate only rewards of chillers which are on
            ch_heat_capacity_rate = 1000 * next_state[-1,COLS_TO_USE.index('c_flowRate')]
            ch_heat_capacity_rate *= 4.19 * (
                next_state[-1,COLS_TO_USE.index('t_value2')] -
                next_state[-1,COLS_TO_USE.index('t_value1')]
            )
            ch_heat_capacity_rate /= 3600
            ch_heat_capacity_rate = np.sum(ch_heat_capacity_rate[chillers_on])

            cw_heat_capacity_rate = 1000 * next_state[-1,COLS_TO_USE.index('e_flowRate')]
            cw_heat_capacity_rate *= 4.19 * (
                next_state[-1,COLS_TO_USE.index('t_value3')] -
                next_state[-1,COLS_TO_USE.index('t_value4')]
            )
            cw_heat_capacity_rate /= 3600
            cw_heat_capacity_rate = np.sum(cw_heat_capacity_rate[chillers_on])

            power_comsumption = (next_state[-1, COLS_TO_USE.index('p_ch1Watt')] +
                                 next_state[-1, COLS_TO_USE.index('p_ch2Watt')] +
                                 next_state[-1, COLS_TO_USE.index('p_ch3Watt')])
            power_comsumption = np.sum(power_comsumption[chillers_on])

            # reward is COP for now which is a terrible optimization objective
            reward = cw_heat_capacity_rate / power_comsumption

            if (ch_heat_capacity_rate < 0) | (cw_heat_capacity_rate < 0) | (power_comsumption < 0):
                ch_heat_capacity_rate = None
                cw_heat_capacity_rate = None
                power_comsumption = None
                reward = None

        else:
            # if the chiller is off, we don't return rewards
            ch_heat_capacity_rate = None
            cw_heat_capacity_rate = None
            power_comsumption = None
            reward = None


        # check if done
        done = self.cur_data_index + self.lookback + 1 >= len(self.data)

        # update state
        self.cur_data_index = self.cur_data_index + 1
        self.cur_time = self.timestamps[self.cur_data_index + self.lookback - 1]
        self.cur_state = next_state

        if done:
            self.has_next = False

        return next_state, reward, ch_heat_capacity_rate, cw_heat_capacity_rate, power_comsumption, done


    def reset():
        """
        resets your env with same init values
        """

        self.cur_data_index = 0
        self.cur_time = timestamps[self.cur_data_index + self.lookback - 1]
        self.cur_state = self.data[self.cur_data_index:(self.cur_data_index + self.lookback)]
        self.has_next = True

        print('Env reset')
