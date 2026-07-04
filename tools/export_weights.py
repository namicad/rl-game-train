from __future__ import annotations

from pathlib import Path
import numpy as np


def export_weights(policy, txt_path: str | Path, cpp_path: str | Path) -> None:
    txt_path = Path(txt_path)
    cpp_path = Path(cpp_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    cpp_path.parent.mkdir(parents=True, exist_ok=True)

    flat = np.concatenate([policy.weight.reshape(-1), policy.bias.reshape(-1)])
    np.savetxt(txt_path, flat.reshape(1, -1), fmt="%.8f")

    cpp_body = "#include <vector>\n\nstd::vector<float> load_exported_weights() {\n"
    cpp_body += "    return {"
    for index, value in enumerate(flat):
        suffix = "," if index < len(flat) - 1 else ""
        cpp_body += f"{value:.8f}{suffix}"
    cpp_body += "};\n}\n"
    cpp_path.write_text(cpp_body, encoding="utf-8")

    return None
