from collections import deque# Ordered collection with ends
import numpy as np           # Handle matrices

class Memory():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen = max_size)
        self.size = 0

    def add(self, experience):
        if self.size < self.max_size:
            self.size += 1
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = self.size
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]

    def size(self):
        return self.size
