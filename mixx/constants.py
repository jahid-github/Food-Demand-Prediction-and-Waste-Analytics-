"""Shared project constants.

Keeping these values in one place makes the app, scripts, and modeling layer
use the same defaults for file paths, locations, and feature settings.
"""

from __future__ import annotations

from pathlib import Path

# Project root lets the code build absolute paths no matter where it is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data is the read-only seed CSV committed to the repository.
DEFAULT_SOURCE_DATA_PATH = PROJECT_ROOT / "mixx_synthetic_restaurant_data.csv"
# Working data is the writable runtime copy that the dashboard updates over time.
DEFAULT_WORKING_DATA_PATH = PROJECT_ROOT / "data" / "runtime" / "restaurant_data.csv"

# Default coordinates point to Helsinki because the sample project uses Finland holidays.
DEFAULT_LATITUDE = 60.1699
DEFAULT_LONGITUDE = 24.9384
DEFAULT_TIMEZONE = "Europe/Helsinki"

# These are the model inputs reused by both benchmarking and live forecasting.
FEATURE_COLUMNS = ["lag_1", "lag_7", "rolling_mean_7", "temperature_c", "is_holiday"]
# Below this history size, the project falls back to a simple average instead of fitting a model.
MIN_HISTORY_ROWS = 14
# Static train/test cutoff used for reproducible benchmark comparisons.
DEFAULT_SPLIT_DATE = "2025-11-01"
