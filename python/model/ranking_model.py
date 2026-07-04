from __future__ import annotations

import numpy as np
from pathlib import Path


class LinearPolicy:
    def __init__(self, feature_dim: int, action_dim: int) -> None:
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.weight = np.zeros((feature_dim, action_dim), dtype=np.float64)
        self.bias = np.zeros(action_dim, dtype=np.float64)
        self.baseline = 0.0
        self.weight_decay = 1e-4

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features)
        if norm < 1e-8:
            return features
        return features / norm

    def _features(self, state: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        raw = np.concatenate([state, candidate], axis=0)
        expanded = np.concatenate([
            raw,
            raw * raw,
            np.abs(raw),
            np.sign(raw),
        ], axis=0)
        return expanded

    def logits(self, features: np.ndarray) -> np.ndarray:
        normalized = self._normalize(features)
        return normalized @ self.weight + self.bias

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return self._softmax(self.logits(features))

    def choose_action(self, state: np.ndarray, candidate_pool: np.ndarray, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        scores = []
        for candidate in candidate_pool:
            features = self._features(state, candidate)
            scores.append(self.logits(features))
        scores = np.vstack(scores)
        probs = self._softmax(scores.mean(axis=1))
        action_idx = int(rng.choice(len(probs), p=probs))
        return action_idx, probs

    def update(self, state: np.ndarray, candidate: np.ndarray, action_idx: int, reward: float, lr: float = 0.001) -> None:
        features = self._features(state, candidate)
        normalized = self._normalize(features)
        probs = self.probabilities(features)
        grad_logits = probs.copy()
        grad_logits[action_idx] -= 1.0
        grad_w = np.outer(normalized, grad_logits)
        grad_b = grad_logits
        advantage = (reward - self.baseline) / (1.0 + abs(reward))
        self.weight -= lr * advantage * grad_w + lr * self.weight_decay * self.weight
        self.bias -= lr * advantage * grad_b
        self.baseline = 0.95 * self.baseline + 0.05 * reward

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self.weight, bias=self.bias, baseline=np.array([self.baseline]))

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(path)
        model = cls(int(data["weight"].shape[0]), int(data["weight"].shape[1]))
        model.weight = data["weight"]
        model.bias = data["bias"]
        model.baseline = float(data["baseline"][0])
        return model
