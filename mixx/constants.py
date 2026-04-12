from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_DATA_PATH = PROJECT_ROOT / "mixx_synthetic_restaurant_data.csv"
DEFAULT_WORKING_DATA_PATH = PROJECT_ROOT / "data" / "runtime" / "restaurant_data.csv"

DEFAULT_LATITUDE = 60.1699
DEFAULT_LONGITUDE = 24.9384
DEFAULT_TIMEZONE = "Europe/Helsinki"

FEATURE_COLUMNS = ["lag_1", "lag_7", "rolling_mean_7", "temperature_c", "is_holiday"]
MIN_HISTORY_ROWS = 14
DEFAULT_SPLIT_DATE = "2025-11-01"
