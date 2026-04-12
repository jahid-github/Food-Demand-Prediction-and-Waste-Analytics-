"""Public package surface for the MIXX project.

The app and helper scripts import from `mixx` directly, so this file re-exports
the functions that make up the project's supported API.
"""

# Re-export path defaults used by the dashboard and helper scripts.
from .constants import DEFAULT_SOURCE_DATA_PATH, DEFAULT_WORKING_DATA_PATH

# Re-export data access helpers.
from .data import (
    build_daily_input_frame,
    build_waste_summary,
    ensure_working_dataset,
    load_dataset,
    save_dataset,
)

# Re-export modeling and forecast helpers.
from .modeling import (
    add_daily_data_and_predict,
    benchmark_regression_models,
    build_holiday_calendar,
    predict_next_day_all_dishes_smart,
    predict_next_day_all_dishes_with_forecast,
)

# Re-export narrative analysis helpers used by the README companion scripts.
from .project_analysis import (
    build_experiment_trail,
    build_exploratory_findings,
    build_data_profile,
    choose_final_model,
    describe_analysis_process,
    describe_feature_choices,
    render_analysis_report,
)

# Re-export the weather integration used by the dashboard sidebar.
from .weather import get_tomorrow_temperature

__all__ = [
    "DEFAULT_SOURCE_DATA_PATH",
    "DEFAULT_WORKING_DATA_PATH",
    "add_daily_data_and_predict",
    "benchmark_regression_models",
    "build_daily_input_frame",
    "build_holiday_calendar",
    "build_waste_summary",
    "build_data_profile",
    "build_experiment_trail",
    "build_exploratory_findings",
    "choose_final_model",
    "describe_analysis_process",
    "describe_feature_choices",
    "ensure_working_dataset",
    "get_tomorrow_temperature",
    "load_dataset",
    "predict_next_day_all_dishes_smart",
    "predict_next_day_all_dishes_with_forecast",
    "render_analysis_report",
    "save_dataset",
]
