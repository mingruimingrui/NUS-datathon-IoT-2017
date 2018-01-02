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

    action[:, CONFIGURABLE_COLS.index('c_flowRate')] = action[:, CONFIGURABLE_COLS.index('c_flowRate')].clip(0,200) / 200

    action[:, CONFIGURABLE_COLS.index('e_flowRate')] = action[:, CONFIGURABLE_COLS.index('e_flowRate')].clip(0,250) / 250

    action[:, CONFIGURABLE_COLS.index('t_value1')] = (action[:, CONFIGURABLE_COLS.index('t_value1')].clip(5,26) - 5) / 21
    action[:, CONFIGURABLE_COLS.index('t_value2')] = (action[:, CONFIGURABLE_COLS.index('t_value2')].clip(6,31) - 6) / 25
    action[:, CONFIGURABLE_COLS.index('t_value3')] = (action[:, CONFIGURABLE_COLS.index('t_value3')].clip(22,41) - 22) / 19
    action[:, CONFIGURABLE_COLS.index('t_value4')] = (action[:, CONFIGURABLE_COLS.index('t_value4')].clip(20,36) - 20) / 16

    return action


def generate_possible_action_directions(action_shape):
    possible_action_directions = [np.zeros(action_shape[1:])]
    for i in range(action_shape[2]):
        action_direction = np.zeros(action_shape[1:])
        action_direction[0:2,i] += 200
        possible_action_directions.append(action_direction)

        action_direction = np.zeros(action_shape[1:])
        action_direction[0:2,i] -= 200
        possible_action_directions.append(action_direction)

    for i in np.arange(2, action_shape[1]):
        for j in range(action_shape[2]):
            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] += 0.5
            possible_action_directions.append(action_direction)

            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] -= 0.5
            possible_action_directions.append(action_direction)

            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] += 1.5
            possible_action_directions.append(action_direction)

            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] -= 1.5
            possible_action_directions.append(action_direction)

            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] += 5
            possible_action_directions.append(action_direction)

            action_direction = np.zeros(action_shape[1:])
            action_direction[i,j] -= 5
            possible_action_directions.append(action_direction)

    possible_action_directions = np.array(possible_action_directions)

    return possible_action_directions


def step_wise_minimize(init_state, action_directions, fn, min_state, max_state, max_iter=32):
    best_state = init_state.copy()
    best_reward = -1e15
    hashtable = {}
    done = False
    count = 0

    while not done:
        count += 1

        best_state = np.squeeze(best_state.copy())
        best_state = np.expand_dims(best_state, best_state.ndim - 1)

        new_states = action_directions + np.dot(np.ones((len(action_directions),1)), best_state)
        new_states -= min_state
        new_states[new_states < 0] = 0
        new_states += min_state - max_state
        new_states[new_states > 0] = 0
        new_states += max_state

        # new_states[new_states - min_state < 0] = min_state[new_states - min_state < 0]
        # new_states[new_states - max_state > 0] = max_state[new_states - max_state > 0]

        rewards = np.zeros(len(new_states))

        hashes = np.array([x.tostring() for x in new_states])
        in_hashtable = np.array([(x in hashtable.keys()) for x in hashes])

        # in hashtable
        rewards[in_hashtable] = [hashtable[x] for x in hashes[in_hashtable]]

        # needs to be computed
        rewards[~in_hashtable] = fn(new_states[~in_hashtable])

        best_index = np.argmax(rewards)
        new_best_state = new_states[best_index]
        new_best_reward = rewards[best_index]

        if (new_best_reward > best_reward) & (count < max_iter):
            best_state = new_best_state
            best_reward = new_best_reward
        else:
            done = True

    best_state = np.squeeze(best_state)
    best_state = np.expand_dims(best_state, 0)

    return best_state, best_reward
