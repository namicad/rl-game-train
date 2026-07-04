from __future__ import annotations

import numpy as np

from .selfplay import generate_selfplay_episodes


def build_training_data(
    num_episodes: int = 80,
    turns_per_episode: int = 8,
    seed: int = 7,
    states: list[np.ndarray] | None = None,
    candidate_pool: np.ndarray | None = None,
):
    episodes = generate_selfplay_episodes(
        num_episodes=num_episodes,
        turns_per_episode=turns_per_episode,
        seed=seed,
        states=states,
        candidate_pool=candidate_pool,
    )
    states = []
    candidates = []
    actions = []
    rewards = []
    for state, candidate, action_idx, reward in episodes:
        states.append(state)
        candidates.append(candidate)
        actions.append(action_idx)
        rewards.append(reward)

    feature_dim = len(states[0]) + len(candidates[0])
    features = np.zeros((len(states), feature_dim), dtype=np.float64)
    for i, (state, candidate) in enumerate(zip(states, candidates)):
        features[i] = np.concatenate([state, candidate], axis=0)

    return features, np.array(actions, dtype=np.int64), np.array(rewards, dtype=np.float64)
