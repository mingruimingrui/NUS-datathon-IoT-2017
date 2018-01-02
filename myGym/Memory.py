import numpy as np
from collections import deque

class Memory(object):
    def __init__(self, max_size=3000):
        self.buffer = deque(maxlen=max_size)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)

        return [self.buffer[i] for i in idx]

    def get_size(self):
        return len(self.buffer)
