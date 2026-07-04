from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from python.train.result_summary import ResultSummary
from python.train.train import main


def test_training_pipeline_runs() -> None:
    states = [np.array([1.0, 500.0, 3.0, 0.0], dtype=np.float64)]
    candidate_pool = np.array([
        [0.9, 0.8, 0.2, 0.7, 0.3, 0.8, 0.2, 0.7, 0.2, 0.6, 0.4, 0.5],
        [0.2, 0.3, 0.8, 0.2, 0.9, 0.2, 0.7, 0.3, 0.6, 0.4, 0.8, 0.7],
    ], dtype=np.float64)
    main(states=states, candidate_pool=candidate_pool, output_dir="models")


def test_best_summary_is_promoted(tmp_path: Path) -> None:
    summary = ResultSummary(output_dir=tmp_path)

    first_payload = {"status": "completed", "epochs_completed": 10, "final_loss": 1.2, "eval_match_rate": 0.4}
    first_weights = tmp_path / "weights_a.txt"
    first_weights.write_text("first", encoding="utf-8")
    summary.promote_best_run(
        current_payload=first_payload,
        current_weights_path=first_weights,
        current_cpp_path=tmp_path / "weights_a.cpp",
        current_policy_path=tmp_path / "policy_a.npz",
    )

    second_payload = {"status": "completed", "epochs_completed": 20, "final_loss": 0.8, "eval_match_rate": 0.9}
    second_weights = tmp_path / "weights_b.txt"
    second_weights.write_text("second", encoding="utf-8")
    summary.promote_best_run(
        current_payload=second_payload,
        current_weights_path=second_weights,
        current_cpp_path=tmp_path / "weights_b.cpp",
        current_policy_path=tmp_path / "policy_b.npz",
    )

    best_summary = json.loads((tmp_path / "best_result_summary.json").read_text(encoding="utf-8"))
    assert best_summary["eval_match_rate"] == 0.9
    assert (tmp_path / "weights.txt").read_text(encoding="utf-8") == "second"
    assert (tmp_path / "best_weights.txt").read_text(encoding="utf-8") == "second"


if __name__ == "__main__":
    test_training_pipeline_runs()
