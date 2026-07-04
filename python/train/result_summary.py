from __future__ import annotations

import json
from pathlib import Path


class ResultSummary:
    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else Path("models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "result_summary.json"
        self.latest_path = self.output_dir / "latest_result.txt"
        self.best_summary_path = self.output_dir / "best_result_summary.json"
        self.best_weights_path = self.output_dir / "best_weights.txt"
        self.best_cpp_path = self.output_dir / "best_weights.cpp"
        self.best_policy_path = self.output_dir / "best_policy_weights.npz"
        self.promoted_weights_path = self.output_dir / "weights.txt"
        self.promoted_cpp_path = self.output_dir / "weights.cpp"
        self.promoted_policy_path = self.output_dir / "policy_weights.npz"

    def write(self, **payload) -> None:
        existing = {}
        if self.summary_path.exists():
            try:
                existing = json.loads(self.summary_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(payload)
        self.summary_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = []
        for key, value in existing.items():
            lines.append(f"{key}={value}")
        self.latest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _is_better(self, candidate: dict, current_best: dict | None) -> bool:
        if not current_best:
            return True
        candidate_eval = float(candidate.get("eval_match_rate", -1.0))
        current_eval = float(current_best.get("eval_match_rate", -1.0))
        if candidate_eval != current_eval:
            return candidate_eval > current_eval
        candidate_loss = float(candidate.get("final_loss", float("inf")))
        current_loss = float(current_best.get("final_loss", float("inf")))
        if candidate_loss != current_loss:
            return candidate_loss < current_loss
        candidate_epochs = int(candidate.get("epochs_completed", 0))
        current_epochs = int(current_best.get("epochs_completed", 0))
        return candidate_epochs > current_epochs

    def promote_best_run(
        self,
        current_payload: dict,
        current_weights_path: str | Path,
        current_cpp_path: str | Path | None = None,
        current_policy_path: str | Path | None = None,
    ) -> dict:
        current_payload = dict(current_payload)
        current_best = {}
        if self.best_summary_path.exists():
            try:
                current_best = json.loads(self.best_summary_path.read_text(encoding="utf-8"))
            except Exception:
                current_best = {}

        if not self._is_better(current_payload, current_best):
            return current_best

        self.best_summary_path.write_text(json.dumps(current_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        weights_path = Path(current_weights_path)
        if weights_path.exists():
            self.best_weights_path.write_bytes(weights_path.read_bytes())
            self.promoted_weights_path.write_bytes(weights_path.read_bytes())

        if current_cpp_path is not None:
            cpp_path = Path(current_cpp_path)
            if cpp_path.exists():
                self.best_cpp_path.write_bytes(cpp_path.read_bytes())
                self.promoted_cpp_path.write_bytes(cpp_path.read_bytes())

        if current_policy_path is not None:
            policy_path = Path(current_policy_path)
            if policy_path.exists():
                self.best_policy_path.write_bytes(policy_path.read_bytes())
                self.promoted_policy_path.write_bytes(policy_path.read_bytes())

        return current_payload

    def read(self) -> dict:
        if not self.summary_path.exists():
            return {}
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
