"""Retired compatibility wrapper; label shapes are validated in preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._deprecated_notebook_mutator import run


if __name__ == "__main__":
    raise SystemExit(run("fix_label_dim.py"))
