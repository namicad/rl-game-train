from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 2000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, candidate: np.ndarray, action_idx: int, reward: float) -> None:
        self.buffer.append((state.copy(), candidate.copy(), int(action_idx), float(reward)))

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
