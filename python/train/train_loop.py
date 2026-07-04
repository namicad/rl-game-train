from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from python.data.replay_buffer import ReplayBuffer
from python.feature.feature_extractor import extract_features
from python.model.ranking_model import LinearPolicy
from tools.export_weights import export_weights


class TrainingLoop:
    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_buffer = ReplayBuffer(capacity=4000)
        self.policy = LinearPolicy(feature_dim=4 * (4 + 12), action_dim=4)
        self.log_path = self.output_dir / "training_log.csv"
        self._init_log()

    def _init_log(self) -> None:
        if not self.log_path.exists():
            with self.log_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "reward", "loss", "match_rate"])

    def _dense_reward(self, state: np.ndarray, candidate: np.ndarray, next_state: np.ndarray) -> float:
        reward = 0.0
        reward += 0.02 * (next_state[1] - state[1])
        reward += 0.03 * (next_state[2] - state[2])
        reward += 0.04 * (next_state[3] - state[3])
        reward += 0.8 * float(candidate[0])
        reward += 0.4 * float(candidate[4])
        reward += 0.3 * float(candidate[8])
        return reward

    def record(self, state: np.ndarray, candidate: np.ndarray, action_idx: int, reward: float) -> None:
        self.replay_buffer.push(state, candidate, action_idx, reward)

    def train_step(self, state: np.ndarray, candidate: np.ndarray, action_idx: int, reward: float, lr: float = 0.001) -> float:
        features = extract_features(state, candidate)
        self.policy.update(state, candidate, int(action_idx), float(reward), lr=lr)
        self.record(state, candidate, action_idx, reward)
        return float(reward)

    def train_from_replay(self, steps: int = 128, lr: float = 0.001) -> tuple[float, float]:
        if len(self.replay_buffer) == 0:
            return 0.0, 0.0
        batch = self.replay_buffer.sample(min(steps, len(self.replay_buffer)))
        losses = []
        for state, candidate, action_idx, reward in batch:
            features = extract_features(state, candidate)
            self.policy.update(state, candidate, int(action_idx), float(reward), lr=lr)
            losses.append(abs(float(reward)))
        return float(np.mean(losses)), float(np.mean(losses))

    def evaluate(self, states, candidate_pool, eval_rounds: int = 40) -> float:
        wins = 0
        for _ in range(eval_rounds):
            state = states[random.randint(0, len(states) - 1)].copy()
            chosen_idx, _ = self.policy.choose_action(state, candidate_pool)
            heuristic_best = max(candidate_pool, key=lambda cand: float(cand[0]) + float(cand[4]) + float(cand[8]))
            heuristic_idx = int(np.where(np.all(candidate_pool == heuristic_best, axis=1))[0][0])
            wins += int(chosen_idx == heuristic_idx)
        return wins / max(1, eval_rounds)

    def save(self) -> None:
        self.policy.save(self.output_dir / "policy_weights.npz")
        export_weights(self.policy, self.output_dir / "weights.txt", self.output_dir / "weights.cpp")

    def log(self, step: int, reward: float, loss: float, match_rate: float) -> None:
        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([step, reward, loss, match_rate])
