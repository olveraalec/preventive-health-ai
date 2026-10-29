"""Run unchanged Streamlit pages on synthetic data in the Explorer-only environment."""

import argparse
import subprocess
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worktree", type=Path)
    args = parser.parse_args()
    root = args.worktree.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == "c28e2ad51f2b9304f99637f6f720e41f2be92bdf"
    for name in ("predictions.csv", "eval_predictions.csv"):
        assert not (root / "data" / name).exists(), "No-CSV synthetic demo required"
    pages = [
        ("dashboard/app.py", "ICU Retrospective Research Dashboard", 1),
        ("dashboard/pages/1_Adaptive_Decisions.py", "Adaptive AI decisions", 1),
        ("dashboard/pages/2_Model_Evaluation.py", "Model evaluation and reliability", 2),
    ]
    for file, title, charts in pages:
        app = AppTest.from_file(str(root / file)).run(timeout=45)
        assert not app.exception, f"{file}: {[e.message for e in app.exception]}"
        assert title in [item.value for item in app.title]
        assert any("SIMULATED DATA" in item.value for item in app.warning)
        assert len(app.get("plotly_chart")) == charts
        if "Model_Evaluation" in file:
            assert len(app.dataframe) == 2
            assert any("not experimental results" in item.value for item in app.warning)
        print(f"PASS {file}: synthetic warning, expected charts, no runtime exception")


if __name__ == "__main__":
    main()
