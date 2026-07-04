from __future__ import annotations

import numpy as np


CANDIDATE_POOL = np.array(
    [
        [0.9, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.7, 0.2, 0.6, 0.4, 0.5],
        [0.4, 0.5, 0.5, 0.3, 0.6, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.2, 0.3, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.7],
        [0.7, 0.6, 0.4, 0.5, 0.5, 0.6, 0.5, 0.9, 0.7, 0.8, 0.6, 0.4],
        [0.8, 0.5, 0.3, 0.2, 0.4, 0.7, 0.4, 0.5, 0.3, 0.4, 0.6, 0.7],
        [0.3, 0.7, 0.6, 0.8, 0.8, 0.2, 0.5, 0.2, 0.4, 0.3, 0.5, 0.4],
        [0.6, 0.4, 0.5, 0.4, 0.7, 0.6, 0.3, 0.8, 0.2, 0.5, 0.7, 0.6],
        [0.1, 0.2, 0.9, 0.6, 0.2, 0.5, 0.8, 0.1, 0.7, 0.8, 0.3, 0.2],
        [0.95, 0.6, 0.1, 0.9, 0.2, 0.7, 0.3, 0.8, 0.1, 0.7, 0.3, 0.6],
        [0.25, 0.8, 0.7, 0.2, 0.85, 0.3, 0.6, 0.2, 0.45, 0.4, 0.7, 0.5],
        [0.55, 0.4, 0.6, 0.4, 0.65, 0.75, 0.4, 0.7, 0.25, 0.6, 0.8, 0.7],
        [0.15, 0.3, 0.9, 0.6, 0.15, 0.4, 0.8, 0.1, 0.8, 0.9, 0.2, 0.3],
        [0.82, 0.6, 0.4, 0.3, 0.55, 0.7, 0.45, 0.6, 0.35, 0.5, 0.6, 0.8],
        [0.38, 0.2, 0.7, 0.4, 0.75, 0.4, 0.6, 0.3, 0.6, 0.7, 0.5, 0.4],
        [0.65, 0.75, 0.3, 0.6, 0.35, 0.6, 0.35, 0.8, 0.4, 0.6, 0.7, 0.55],
        [0.18, 0.4, 0.8, 0.5, 0.8, 0.3, 0.7, 0.2, 0.75, 0.85, 0.3, 0.25],
    ],
    dtype=np.float64,
)


def _reward(state: np.ndarray, candidate: np.ndarray) -> float:
    gold, army, control = state[1], state[2], state[3]
    aggression = candidate[0]
    economy = candidate[4]
    defense = candidate[8]
    return (
        0.001 * gold
        + 0.18 * army
        + 0.24 * control
        + 0.40 * aggression
        + 0.28 * economy
        + 0.18 * defense
        - 0.20 * candidate[3]
        - 0.10 * abs(candidate[1] - 0.5)
    )


def _next_state(state: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    next_state = state.copy()
    next_state[1] += 25.0 + 20.0 * candidate[4] - 8.0 * candidate[8]
    next_state[2] += 0.9 + 0.35 * candidate[0] + 0.25 * candidate[1]
    next_state[3] += 0.15 + 0.22 * candidate[2] + 0.12 * candidate[3]
    next_state[0] += 1.0
    return next_state


def generate_selfplay_episodes(
    num_episodes: int = 220,
    turns_per_episode: int = 12,
    seed: int = 7,
    states: list[np.ndarray] | None = None,
    candidate_pool: np.ndarray | None = None,
):
    rng = np.random.default_rng(seed)
    episodes = []
    pool = candidate_pool if candidate_pool is not None else CANDIDATE_POOL
    if states is None:
        states = [np.array([1.0, 500.0, 3.0, 0.0], dtype=np.float64)]

    for _ in range(num_episodes):
        state = states[int(rng.integers(0, len(states)))].copy()
        for _ in range(turns_per_episode):
            candidate_idx = int(rng.integers(0, len(pool)))
            candidate = pool[candidate_idx]
            reward = _reward(state, candidate)
            label = int(candidate_idx % 4)
            episodes.append((state.copy(), candidate, label, reward))
            state = _next_state(state, candidate)
    return episodes
