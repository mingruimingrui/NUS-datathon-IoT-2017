from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

class SimplePolicy:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10):
        self.model = Sequential()

        self.model.add(Dense(hidden_size, activation='relu',
                             input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.optimizer = Adam(lr=learning_rate)

    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)

    def save_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def compile(self, loss='mse'):
        self.model.compile(loss=loss, optimizer=self.optimizer)
