"""Dataset loading and persistence helpers.

These functions hide CSV-specific details from the rest of the project so the
app and modeling code can work with clean pandas DataFrames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "dish_name",
    "category",
    "price_level",
    "temperature_c",
    "is_holiday",
    "cooked_qty",
    "sold_qty",
]

CSV_COLUMNS = [
    "date",
    "weekday",
    "dish_name",
    "category",
    "price_level",
    "temperature_c",
    "is_holiday",
    "cooked_qty",
    "sold_qty",
    "wasted_qty",
]


def ensure_working_dataset(source_path: str | Path, working_path: str | Path) -> Path:
    """Create the writable runtime CSV the first time the app is launched."""
    source = Path(source_path)
    working = Path(working_path)
    working.parent.mkdir(parents=True, exist_ok=True)

    # The repository keeps a seed CSV under version control, while the dashboard
    # writes updates to a separate runtime file so the original data stays intact.
    if not working.exists():
        if not source.exists():
            raise FileNotFoundError(f"Seed dataset not found at {source}")
        shutil.copyfile(source, working)

    return working


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the dataset and normalize the columns expected by the project."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.copy()
    # Convert the date column once so the rest of the project can use datetime logic directly.
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()

    numeric_columns = ["temperature_c", "is_holiday", "cooked_qty", "sold_qty"]
    if "wasted_qty" in df.columns:
        numeric_columns.append("wasted_qty")

    # Fail fast on bad numeric data instead of silently coercing invalid rows.
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Recompute waste from cooked and sold quantities so derived values always stay consistent.
    df["wasted_qty"] = (df["cooked_qty"] - df["sold_qty"]).clip(lower=0)
    return df.sort_values(["date", "dish_name"]).reset_index(drop=True)


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a clean, CSV-friendly copy of the current dataset to disk."""
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = df.copy()
    # Rebuild presentation fields just before writing so the saved CSV stays normalized.
    serializable["date"] = pd.to_datetime(serializable["date"])
    serializable["weekday"] = serializable["date"].dt.day_name()
    serializable["wasted_qty"] = (
        serializable["cooked_qty"] - serializable["sold_qty"]
    ).clip(lower=0)

    serializable["date"] = serializable["date"].dt.strftime("%Y-%m-%d")
    serializable = serializable.sort_values(["date", "dish_name"]).reset_index(drop=True)
    serializable[CSV_COLUMNS].to_csv(dataset_path, index=False)
    return dataset_path


def build_daily_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the editable dish grid shown in the dashboard's daily update tab."""
    latest_rows = (
        df.sort_values(["dish_name", "date"])
        .groupby("dish_name", as_index=False)
        .tail(1)
        .sort_values("dish_name")
        .reset_index(drop=True)
    )

    # Use each dish's latest known metadata and quantities as the default form values.
    template = latest_rows[
        ["dish_name", "category", "price_level", "sold_qty", "cooked_qty"]
    ].copy()
    template["sold_qty"] = template["sold_qty"].astype(int)
    template["cooked_qty"] = template["cooked_qty"].astype(int)
    return template


def build_waste_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily cooked, sold, and wasted quantities for trend reporting."""
    summary = (
        df.groupby("date", as_index=False)[["cooked_qty", "sold_qty", "wasted_qty"]]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    # Waste percentage answers the business question more directly than raw waste units alone.
    summary["waste_pct"] = (
        summary["wasted_qty"] / summary["cooked_qty"].where(summary["cooked_qty"] > 0)
    ).fillna(0.0) * 100.0
    return summary
