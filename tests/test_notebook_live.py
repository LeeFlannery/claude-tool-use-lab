"""Run the whole solution notebook headless against the real API.

Skipped unless ANTHROPIC_API_KEY is set. Costs a few cents per run.

    ANTHROPIC_API_KEY=... uv run pytest tests/test_notebook_live.py
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

SOLUTION = Path(__file__).resolve().parent.parent / "lab" / "claude-tool-use-lab-solution.ipynb"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_solution_notebook_runs_top_to_bottom(tmp_path):
    nb = nbformat.read(SOLUTION, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    outputs = "".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
    )
    for n in range(1, 6):
        assert f"Exercise {n} passed" in outputs
