from __future__ import annotations

import numpy as np


def extract_state_features(state: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[3]),
        ],
        dtype=np.float64,
    )


def extract_candidate_features(candidate: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            float(candidate[3]),
            float(candidate[4]),
            float(candidate[5]),
            float(candidate[6]),
            float(candidate[7]),
            float(candidate[8]),
            float(candidate[9]),
            float(candidate[10]),
            float(candidate[11]),
        ],
        dtype=np.float64,
    )


def extract_interaction_features(state: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(state[1]) * 0.001,
            float(state[2]) * 0.01,
            float(state[3]) * 0.01,
            float(candidate[0]) - float(candidate[4]),
            float(candidate[1]) - float(candidate[5]),
            float(candidate[2]) - float(candidate[6]),
        ],
        dtype=np.float64,
    )


def extract_features(state: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    state_features = extract_state_features(state)
    candidate_features = extract_candidate_features(candidate)
    interaction_features = extract_interaction_features(state, candidate)
    return np.concatenate([state_features, candidate_features, interaction_features], axis=0)
