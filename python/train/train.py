from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from python.data.dataset_builder import build_training_data
from python.model.ranking_model import LinearPolicy
from python.train.loss import cross_entropy_loss
from python.train.result_summary import ResultSummary
from python.train.train_loop import TrainingLoop
from tools.export_weights import export_weights
from tools.update_sample_code import update_sample_code


def _candidate_score(state: np.ndarray, candidate: np.ndarray) -> float:
    return (
        0.32 * candidate[0]
        + 0.16 * candidate[1]
        + 0.12 * candidate[2]
        + 0.10 * candidate[3]
        + 0.20 * candidate[4]
        + 0.12 * candidate[5]
        + 0.10 * candidate[6]
        + 0.16 * candidate[8]
        + 0.08 * candidate[9]
        + 0.04 * candidate[10]
        + 0.002 * state[1]
        + 0.003 * state[2]
        + 0.004 * state[3]
    )


def _build_competition_samples(states, candidate_pool, num_rounds=180):
    if states is None:
        states = [np.array([1.0, 500.0, 3.0, 0.0], dtype=np.float64)]
    if candidate_pool is None:
        candidate_pool = np.array([
            [0.9, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.7, 0.2, 0.6, 0.4, 0.5],
            [0.2, 0.3, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.7],
        ], dtype=np.float64)

    samples = []
    for _ in range(num_rounds):
        state = states[np.random.randint(0, len(states))].copy()
        for idx_a, candidate_a in enumerate(candidate_pool):
            for idx_b, candidate_b in enumerate(candidate_pool):
                if idx_a == idx_b:
                    continue
                score_a = _candidate_score(state, candidate_a)
                score_b = _candidate_score(state, candidate_b)
                reward_a = 1.0 if score_a >= score_b else -1.0
                reward_b = -reward_a
                samples.append((state, candidate_a, idx_a % 4, reward_a))
                samples.append((state, candidate_b, idx_b % 4, reward_b))
    return samples


def _evaluate_policy(policy, states, candidate_pool, eval_rounds=60):
    if states is None:
        states = [np.array([1.0, 500.0, 3.0, 0.0], dtype=np.float64)]
    if candidate_pool is None:
        candidate_pool = np.array([
            [0.9, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.7, 0.2, 0.6, 0.4, 0.5],
            [0.2, 0.3, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.7],
        ], dtype=np.float64)

    wins = 0
    for _ in range(eval_rounds):
        state = states[np.random.randint(0, len(states))].copy()
        chosen_idx, _ = policy.choose_action(state, candidate_pool)
        chosen = candidate_pool[chosen_idx]
        heuristic_best = max(candidate_pool, key=lambda cand: _candidate_score(state, cand))
        heuristic_idx = int(np.where(np.all(candidate_pool == heuristic_best, axis=1))[0][0])
        wins += int(chosen_idx == heuristic_idx)
    return wins / max(1, eval_rounds)


def main(
    states: list[np.ndarray] | None = None,
    candidate_pool: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    epochs: int = 600,
    log_every: int = 50,
) -> None:
    features, actions, rewards = build_training_data(
        num_episodes=280,
        turns_per_episode=16,
        seed=7,
        states=states,
        candidate_pool=candidate_pool,
    )
    competition_samples = _build_competition_samples(states, candidate_pool)
    input_dim = features.shape[1] * 4
    policy = LinearPolicy(feature_dim=input_dim, action_dim=4)

    rewards = np.clip((rewards - rewards.mean()) / (rewards.std() + 1e-8), -2.0, 2.0)

    out_dir = Path(output_dir) if output_dir is not None else ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    loop = TrainingLoop(output_dir=out_dir)
    summary = ResultSummary(output_dir=out_dir)
    summary.write(status="training", epochs_completed=0, final_loss=None, eval_match_rate=None)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        lr = 0.0007 / (1.0 + 0.0018 * epoch)

        for feature, action_idx, reward in zip(features, actions, rewards):
            state = feature[:4]
            candidate = feature[4:]
            loop.train_step(state, candidate, int(action_idx), float(reward), lr=lr)
            logits = loop.policy.logits(loop.policy._features(state, candidate))
            epoch_loss += cross_entropy_loss(np.array([logits]), np.array([int(action_idx)]))

        for state, candidate, action_idx, reward in competition_samples:
            loop.train_step(state, candidate, int(action_idx), float(reward), lr=lr * 0.5)
            epoch_loss += abs(float(reward)) * 0.01

        average_loss = epoch_loss / max(1, len(actions) + len(competition_samples))
        losses.append(average_loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"epoch={epoch + 1}/{epochs} loss={average_loss:.4f}")
            loop.log(epoch + 1, float(np.mean(rewards)), float(average_loss), 0.0)
            summary.write(
                status="training",
                epochs_completed=epoch + 1,
                final_loss=float(average_loss),
                eval_match_rate=None,
                weights_file=str(out_dir / "weights.txt"),
                training_log=str(out_dir / "training_log.csv"),
            )

    loop.save()

    eval_score = _evaluate_policy(loop.policy, states, candidate_pool)
    loop.log(epochs, float(np.mean(rewards)), float(losses[-1]), float(eval_score))

    final_payload = {
        "status": "completed",
        "epochs": epochs,
        "epochs_completed": epochs,
        "final_loss": float(losses[-1]),
        "eval_match_rate": float(eval_score),
        "weights_file": str(out_dir / "weights.txt"),
        "training_log": str(out_dir / "training_log.csv"),
    }
    summary.write(**final_payload)
    summary.promote_best_run(
        current_payload=final_payload,
        current_weights_path=out_dir / "weights.txt",
        current_cpp_path=out_dir / "weights.cpp",
        current_policy_path=out_dir / "policy_weights.npz",
    )

    print(f"trained_policy={len(losses)}_epochs")
    print(f"final_loss={losses[-1]:.4f}")
    print(f"eval_match_rate={eval_score:.4f}")
    print(f"weights_file={out_dir / 'weights.txt'}")
    print(f"summary_file={out_dir / 'result_summary.json'}")

    update_sample_code()
    print("promoted best run and updated sample-code")


if __name__ == "__main__":
    main()
