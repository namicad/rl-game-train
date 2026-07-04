from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "models" / "result_summary.json"
BEST_SUMMARY_PATH = ROOT / "models" / "best_result_summary.json"
SAMPLE_CODE_PATH = ROOT / "cpp" / "sample-code.cpp"


def update_sample_code() -> None:
    summary_path = BEST_SUMMARY_PATH if BEST_SUMMARY_PATH.exists() else SUMMARY_PATH
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    final_loss = float(summary.get("final_loss", 0.0))
    eval_match_rate = float(summary.get("eval_match_rate", 0.0))
    epochs = int(summary.get("epochs", 0))

    text = SAMPLE_CODE_PATH.read_text(encoding="utf-8")

    marker = "// AUTO-RL-UPDATE:START"
    end_marker = "// AUTO-RL-UPDATE:END"
    block = f"""{marker}
static const char* kLatestRlSummary = "epochs={epochs};final_loss={final_loss:.6f};eval_match_rate={eval_match_rate:.6f}";
{end_marker}
"""

    if marker in text and end_marker in text:
        pattern = re.compile(rf"{re.escape(marker)}.*?{re.escape(end_marker)}", re.S)
        text = pattern.sub(block, text, count=1)
    else:
        insert_after = "//////////////////////////////////\n//// WRITE YOUR STRATEGY HERE ////\n//////////////////////////////////\n"
        text = text.replace(insert_after, insert_after + "\n" + block)

    SAMPLE_CODE_PATH.write_text(text, encoding="utf-8")

    print(f"updated sample-code with summary: epochs={epochs}, final_loss={final_loss:.6f}, eval_match_rate={eval_match_rate:.6f}")


if __name__ == "__main__":
    update_sample_code()
