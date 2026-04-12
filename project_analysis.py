"""Command-line helper that prints the project's markdown analysis report."""

from __future__ import annotations

from pathlib import Path

from mixx import DEFAULT_SOURCE_DATA_PATH, load_dataset
from mixx.project_analysis import render_analysis_report


def _resolve_dataset_path() -> Path:
    """Prefer the runtime CSV when present, otherwise fall back to the seed dataset."""
    runtime_path = Path("data/runtime/restaurant_data.csv")
    if runtime_path.exists():
        return runtime_path
    return Path(DEFAULT_SOURCE_DATA_PATH)


def main() -> None:
    """Print a full markdown project-analysis report generated from Python code."""
    df = load_dataset(_resolve_dataset_path())
    print(render_analysis_report(df))


if __name__ == "__main__":
    main()
