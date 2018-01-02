import json

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Flatten
from keras.layers import Dense, Conv1D
from keras.layers import BatchNormalization
from keras.optimizers import Adam

from keras.models import model_from_json

from util.util import *

def load_model(json_file):
    with open(json_file, 'r') as f:
        model_specs = json.load(f)

    model = model_from_json(model_specs['model'])
    model.load_weights(model_specs['weights_file'])

    return model


class SimplePolicy(object):
    def __init__(self, learning_rate=0.001, state_dim=(-1,9,4), action_dim=(1,6,4),
        hidden_size=32, json_file=None):
        # if json_file_present, then we load from json

        if json_file is None:
            input_size = (state_dim[1] + action_dim[1]) * state_dim[2]

            self.model = Sequential()

            self.model.add(Dense(hidden_size, activation='relu',
                                 input_dim=input_size))
            self.model.add(Dense(hidden_size, activation='relu'))
            self.model.add(Dense(2, activation='linear'))
        else:
            self.model = load_model(json_file)

        self.optimizer = Adam(lr=learning_rate)

    def save_model(self, json_file, weights_file):
        model_specs = {
            'model': self.model.to_json(),
            'weights_file': weights_file
        }

        with open(json_file, 'w') as f:
            json.dump(model_specs, f)

        self.model.save_weights(weights_file)

    def compile(self, loss='mse'):
        self.model.compile(loss=loss, optimizer=self.optimizer)

    def predict(self, state, action):
        inputs = np.concatenate([
            normalize_data(state), normalize_action(action)
        ], axis=1)
        inputs = inputs.reshape(len(state), -1)

        reward = self.model.predict_on_batch(inputs)

        return reward

    def train(self, state, action, reward, validation_batch=None):
        inputs = np.concatenate([
            normalize_data(state), normalize_action(action)
        ], axis=1)

        inputs = inputs.reshape(len(state), -1)

        # return self.model.train_on_batch(inputs, reward)
        if validation_batch is None:
            return self.model.fit(inputs, reward, epochs=1, verbose=0)
        else:
            state_v, action_v, reward_v = validation_batch
            inputs_v = np.concatenate([
                normalize_data(state_v), normalize_action(action_v)
            ], axis=1)
            inputs_v = inputs_v.reshape(len(validation_batch[0]), -1)

            return self.model.fit(inputs, reward,
                                  validation_data=(inputs_v, reward_v),
                                  epochs=1, verbose=0)

class ConvPolicy(object):
    def __init__(self, learning_rate=0.001, state_dim=(-1,9,4),
        action_dim=(1,6,4), json_file=None):
        # if json_file_present, then we load from json

        if json_file is None:
            inputs = Input(shape=(state_dim[1] + action_dim[1], state_dim[2]))

            x = Conv1D(16, 1, strides=1, activation='relu')(inputs)
            x = Conv1D(16, 15, strides=1, activation='relu')(x)
            x = Flatten()(x)

            x = BatchNormalization()(x)
            x = Dense(16, activation='relu')(x)
            outputs = Dense(2, activation='linear')(x)
        else:
            self.model = load_model(json_file)

        self.optimizer = Adam(lr=learning_rate)

    def save_model(self, json_file, weights_file):
        model_specs = {
            'model': self.model.to_json(),
            'weights_file': weights_file
        }

        with open(json_file, 'w') as f:
            json.dump(model_specs, f)

        self.model.save_weights(weights_file)

    def compile(self, loss='mse'):
        self.model.compile(loss=loss, optimizer=self.optimizer)

    def predict(self, state, action):
        inputs = np.concatenate([
            normalize_data(state), normalize_action(action)
        ], axis=1)
        inputs = inputs.reshape(len(state), -1)

        reward = self.model.predict_on_batch(inputs)

        return reward

    def train(self, state, action, reward, validation_batch=None):
        inputs = np.concatenate([
            normalize_data(state), normalize_action(action)
        ], axis=1)

        inputs = inputs.reshape(len(state), -1)

        # return self.model.train_on_batch(inputs, reward)
        if validation_batch is None:
            return self.model.fit(inputs, reward, epochs=1, verbose=0)
        else:
            state_v, action_v, reward_v = validation_batch
            inputs_v = np.concatenate([
                normalize_data(state_v), normalize_action(action_v)
            ], axis=1)
            inputs_v = inputs_v.reshape(len(validation_batch[0]), -1)

            return self.model.fit(inputs, reward,
                                  validation_data=(inputs_v, reward_v),
                                  epochs=1, verbose=0)
