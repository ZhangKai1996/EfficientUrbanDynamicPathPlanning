import numpy as np


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.size = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(args)
        else:
            self.buffer[self.pos] = args
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        n = len(self.buffer)
        if n <= batch_size: return self.buffer[:]          # 或 random.sample，n 小无所谓
        idx = np.random.randint(0, n, size=batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
