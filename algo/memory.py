import random
import numpy as np
from collections import deque


class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ScalableReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = {}
        self.counter = 0

    def push(self, *args, key=None):
        if key is None: key = 0

        if key not in self.buffer.keys():
            self.buffer[key] = deque(maxlen=self.capacity)
        self.buffer[key].append(args)
        self.counter += 1

    def sample(self, batch_size, num_keys=None):
        keys = list(self.buffer.keys())
        if num_keys is not None:
            keys = np.random.choice(keys, num_keys)

        batch_dict = {}
        for key in keys:
            value = self.buffer[key]
            if len(value) >= batch_size:
                batch_dict[key] = random.sample(value, batch_size)
        return batch_dict

    def sample_(self, batch_size):
        batch = []
        for key, value in self.buffer.items():
            num = min(len(value), batch_size)
            batch += random.sample(self.buffer[key], num)
        return batch

    def __len__(self):
        return self.counter


class RolloutBuffer:
    def __init__(self):
        self.states, self.dyns, self.masks = [], [], []
        self.actions, self.log_probs, self.values = [], [], []
        self.rewards, self.dones = [], []
        self.next_states, self.next_dyns, self.next_masks = [], [], []

    def store(self, state, dyn, mask,
              action, log_prob,
              reward, done, value,
              next_state=None, next_dyn=None, next_mask=None):
        self.states.append(state)
        self.dyns.append(dyn)
        self.masks.append(mask)

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        self.rewards.append(reward)
        self.dones.append(done)

        if next_state is not None: self.next_states.append(next_state)
        if next_dyn is not None: self.next_dyns.append(next_dyn)
        if next_mask is not None: self.next_masks.append(next_mask)

    def clear(self):
        self.states, self.dyns, self.masks = [], [], []
        self.actions, self.log_probs, self.values = [], [], []
        self.rewards, self.dones = [], []
        self.next_states, self.next_dyns, self.next_masks = [], [], []


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # 控制“偏重程度”，0=均匀采样

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(args)
        self.priorities.append(max_priority)  # 新样本设为最高优先级（保证能被学）

    def sample(self, batch_size, beta=0.4):
        assert len(self.buffer) >= batch_size

        # 计算概率分布
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样 indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化处理

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + 1e-6  # 避免 0 priority

    def __len__(self):
        return len(self.buffer)