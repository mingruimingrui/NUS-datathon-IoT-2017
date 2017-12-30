import os

import numpy as np
import pandas as pd

from util.const import *

def get_data_from_folder(date_folder):
    file_names= list(map(lambda x: 'data/' + date_folder + '/' + x, os.listdir('data/' + date_folder)))
    dfs = [pd.read_csv(file_name) for file_name in file_names]

    timestamps = dfs[0].ts
    columns = COLS_TO_USE
    data = []

    for df in dfs:
        df = df.copy()[COLS_TO_USE]
        data.append(df.values.astype('float32'))

    data = np.array(data).transpose((1,2,0))

    return timestamps, columns, data

def normalize_data(data):
    data = data.copy()

    data[:,COLS_TO_USE.index('c_flowRate')] = data[:,COLS_TO_USE.index('c_flowRate')].clip(0,200) / 200
    # data[:,COLS_TO_USE.index('c_flowSpeed')] = data[:,COLS_TO_USE.index('c_flowSpeed')].clip(0,2) / 2

    data[:,COLS_TO_USE.index('e_flowRate')] = data[:,COLS_TO_USE.index('e_flowRate')].clip(0,250) / 250
    # data[:,COLS_TO_USE.index('e_flowSpeed')] = data[:,COLS_TO_USE.index('e_flowSpeed')].clip(0,3.5) / 3.5

    data[:,COLS_TO_USE.index('p_ch1Watt')] = data[:,COLS_TO_USE.index('p_ch1Watt')].clip(0,8e4) / 8e4
    data[:,COLS_TO_USE.index('p_ch2Watt')] = data[:,COLS_TO_USE.index('p_ch2Watt')].clip(0,8e4) / 8e4
    data[:,COLS_TO_USE.index('p_ch3Watt')] = data[:,COLS_TO_USE.index('p_ch3Watt')].clip(0,8e4) / 8e4

    data[:,COLS_TO_USE.index('t_value1')] = (data[:,COLS_TO_USE.index('t_value1')].clip(5,26) - 5) / 21
    data[:,COLS_TO_USE.index('t_value2')] = (data[:,COLS_TO_USE.index('t_value2')].clip(6,31) - 6) / 25
    data[:,COLS_TO_USE.index('t_value3')] = (data[:,COLS_TO_USE.index('t_value3')].clip(22,41) - 22) / 19
    data[:,COLS_TO_USE.index('t_value4')] = (data[:,COLS_TO_USE.index('t_value4')].clip(20,36) - 20) / 16

    return data

def normalize_action(action):
    action = action.copy()

    action[CONFIGURABLE_COLS.index('c_flowRate')] = action[CONFIGURABLE_COLS.index('c_flowRate')].clip(0,200) / 200

    action[CONFIGURABLE_COLS.index('e_flowRate')] = action[CONFIGURABLE_COLS.index('e_flowRate')].clip(0,250) / 250

    action[CONFIGURABLE_COLS.index('t_value1')] = (action[CONFIGURABLE_COLS.index('t_value1')].clip(5,26) - 5) / 21
    action[CONFIGURABLE_COLS.index('t_value2')] = (action[CONFIGURABLE_COLS.index('t_value2')].clip(6,31) - 6) / 25
    action[CONFIGURABLE_COLS.index('t_value3')] = (action[CONFIGURABLE_COLS.index('t_value3')].clip(22,41) - 22) / 19
    action[CONFIGURABLE_COLS.index('t_value4')] = (action[CONFIGURABLE_COLS.index('t_value4')].clip(20,36) - 20) / 16

    return action

def generate_sensible_actions(state):
    prev_action = state[-1]
    prev_action
