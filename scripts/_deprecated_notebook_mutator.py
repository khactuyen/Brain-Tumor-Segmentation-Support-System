"""Shared entry point for retired one-off notebook rewrite scripts."""

from __future__ import annotations

from scripts.build_v2_notebook import validate_notebook


def run(script_name: str) -> int:
    print(
        f"{script_name} is retired. SIC_Capstone_v2.ipynb is now the canonical "
        "source and will not be rewritten by one-off scripts."
    )
    notebook = validate_notebook()
    print(f"Canonical notebook is valid: {len(notebook['cells'])} cells")
    return 0
