from __future__ import annotations

import numpy as np

from python.train.train import main


def run_multiple_rounds(rounds: int = 3, epochs_per_round: int = 200) -> None:
    states = [np.array([1.0, 500.0, 3.0, 0.0], dtype=np.float64)]
    candidate_pool = np.array([
        [0.9, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.7, 0.2, 0.6, 0.4, 0.5],
        [0.2, 0.3, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.7],
    ], dtype=np.float64)
    for round_idx in range(1, rounds + 1):
        print(f"=== training round {round_idx}/{rounds} ===")
        main(states=states, candidate_pool=candidate_pool, output_dir="models", epochs=epochs_per_round, log_every=25)


if __name__ == "__main__":
    run_multiple_rounds()
