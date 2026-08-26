"""Validate or copy the canonical SIC_Capstone_v2 notebook.

The checked-in notebook is the single source of truth. Older versions of this
script duplicated hundreds of cell definitions and could overwrite reviewed
work with stale training code and sample metrics.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "SIC_Capstone_v2.ipynb"
REQUIRED_SNIPPETS = (
    "PREPROCESS_VERSION =",
    "cache_is_current",
    "load_validated_cache",
    "reusable_cases",
    "cases_to_build",
    "conversion skipped",
    "NUM_SAMPLES = 1",
    "test_cases = train_test_split",
    "val_loader_full",
    "test_loader",
    "compute_case_metrics",
    "include_surface=True",
    "checkpoint[\"model_state\"]",
    "save_checkpoint_atomic",
    'map_location="cpu"',
    "nib.Nifti1Image",
    "ResearchSegmentationReport",
    "not for clinical diagnosis or treatment",
)
FORBIDDEN_SNIPPETS = (
    "Quantitative volume estimates generated for radiotherapy planning",
    "export_clinical_pdf(v_cid, {'dice_wt': 0.912",
)


def _python_source_for_validation(source: str) -> str:
    """Replace line-oriented IPython commands while preserving line numbers."""
    lines = source.splitlines(keepends=True)
    first_code_line = next((line.lstrip() for line in lines if line.strip()), "")
    if first_code_line.startswith("%%"):
        return ""

    validated_lines = []
    command_indent = ""
    in_ipython_command = False
    for line in lines:
        stripped = line.lstrip()
        starts_ipython_command = stripped.startswith(("!", "%"))
        if starts_ipython_command:
            command_indent = line[: len(line) - len(stripped)]
            in_ipython_command = True
        if in_ipython_command:
            newline = "\n" if line.endswith("\n") else ""
            validated_lines.append(f"{command_indent}pass  # IPython command{newline}")
            in_ipython_command = line.rstrip().endswith("\\")
        else:
            validated_lines.append(line)
    return "".join(validated_lines)


def validate_notebook(path: Path = NOTEBOOK_PATH) -> dict:
    """Validate structure and the safety-critical V2 pipeline contract."""
    if not path.is_file():
        raise FileNotFoundError(f"Notebook not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    if notebook.get("nbformat") != 4:
        raise ValueError(f"Unsupported notebook format: {notebook.get('nbformat')}")
    minor_version = notebook.get("nbformat_minor")
    if not isinstance(minor_version, int) or minor_version < 0:
        raise ValueError(f"Invalid notebook minor version: {minor_version}")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Notebook has no cells")

    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in code]
    forbidden = [snippet for snippet in FORBIDDEN_SNIPPETS if snippet in code]
    if missing:
        raise ValueError(f"Notebook is missing required pipeline contracts: {missing}")
    if forbidden:
        raise ValueError(f"Notebook contains obsolete/unsafe content: {forbidden}")

    cell_ids = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"Cell {index} is not an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise ValueError(f"Cell {index} has invalid type: {cell_type}")
        source_lines = cell.get("source")
        if not isinstance(source_lines, list) or not all(
            isinstance(line, str) for line in source_lines
        ):
            raise ValueError(f"Cell {index} source must be a list of strings")
        if minor_version >= 5:
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError(f"Cell {index} is missing an nbformat 4.5+ id")
            cell_ids.append(cell_id)
        if cell_type != "code":
            continue
        source = "".join(source_lines)
        try:
            ast.parse(_python_source_for_validation(source))
        except SyntaxError as exc:
            raise ValueError(
                f"Code cell {index} has invalid Python syntax at line "
                f"{exc.lineno}: {exc.msg}"
            ) from exc
        if cell.get("execution_count") is not None:
            raise ValueError(f"Code cell {index} contains a stale execution count")
        if cell.get("outputs"):
            raise ValueError(f"Code cell {index} contains stale outputs")

    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Notebook contains duplicate cell ids")

    return notebook


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the canonical V2 notebook or copy it to another path."
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        help="Optional destination for a byte-for-byte copy of the validated notebook.",
    )
    args = parser.parse_args(argv)

    notebook = validate_notebook()
    print(f"Validated {NOTEBOOK_PATH.name}: {len(notebook['cells'])} cells")
    if args.copy_to:
        destination = args.copy_to.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NOTEBOOK_PATH, destination)
        print(f"Copied canonical notebook to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
