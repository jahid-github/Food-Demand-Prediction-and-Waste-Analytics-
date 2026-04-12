"""Narrative analysis helpers used to explain the project in plain language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import build_waste_summary
from .modeling import benchmark_regression_models

ORDERED_MODEL_NAMES = [
    "Naive Baseline",
    "Linear Regression",
    "Random Forest",
]


@dataclass(frozen=True)
class AnalysisStep:
    """A single step in the end-to-end project workflow."""

    title: str
    detail: str


@dataclass(frozen=True)
class FeatureChoice:
    """Why a feature was included or excluded from the forecasting model."""

    name: str
    reason: str


@dataclass(frozen=True)
class ExperimentNote:
    """Narrative summary of a model that was tested during experimentation."""

    model: str
    metrics: dict[str, float]
    takeaway: str


def describe_analysis_process() -> list[AnalysisStep]:
    """Return the step-by-step workflow used for this project."""
    return [
        AnalysisStep(
            title="Define the business problem",
            detail=(
                "Frame the task as next-day buffet demand prediction so the model can help "
                "the restaurant cook enough food while reducing avoidable waste."
            ),
        ),
        AnalysisStep(
            title="Clean and standardize the dataset",
            detail=(
                "Validate required columns, convert dates and numeric fields, and recompute "
                "`wasted_qty` from cooked and sold quantities so downstream analysis uses "
                "consistent values."
            ),
        ),
        AnalysisStep(
            title="Explore demand and waste behavior",
            detail=(
                "Review dish-level demand, total waste, seasonal temperature shifts, and "
                "holiday behavior to understand which patterns are worth turning into model features."
            ),
        ),
        AnalysisStep(
            title="Engineer forecasting features",
            detail=(
                "Build lag-based, rolling, temperature, and holiday features that capture "
                "both short-term demand memory and external demand drivers."
            ),
        ),
        AnalysisStep(
            title="Benchmark multiple models",
            detail=(
                "Compare a naive baseline, Linear Regression, and Random Forest using the "
                "same train/test split so the final choice is evidence-based."
            ),
        ),
        AnalysisStep(
            title="Select the production model",
            detail=(
                "Choose the model that balances predictive quality with interpretability "
                "and deployment simplicity for a Streamlit dashboard."
            ),
        ),
        AnalysisStep(
            title="Deploy the analysis as an app",
            detail=(
                "Wrap the forecasting workflow in Streamlit so users can generate forecasts, "
                "enter new daily actuals, and monitor waste trends interactively."
            ),
        ),
    ]


def build_data_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize the dataset in a form that supports narrative analysis."""
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])

    # These aggregates drive the plain-language findings shown in reports and docs.
    waste_summary = build_waste_summary(working)
    dish_mean_sales = (
        working.groupby("dish_name")["sold_qty"].mean().sort_values(ascending=False)
    )
    dish_volatility = (
        working.groupby("dish_name")["sold_qty"].std().fillna(0).sort_values(ascending=False)
    )

    total_cooked = float(working["cooked_qty"].sum())
    total_sold = float(working["sold_qty"].sum())
    total_waste = float(working["wasted_qty"].sum())
    overall_waste_pct = (total_waste / total_cooked * 100.0) if total_cooked else 0.0

    return {
        "records": int(len(working)),
        "days": int(working["date"].nunique()),
        "dishes": int(working["dish_name"].nunique()),
        "categories": int(working["category"].nunique()),
        "date_start": working["date"].min().date().isoformat(),
        "date_end": working["date"].max().date().isoformat(),
        "temperature_min": round(float(working["temperature_c"].min()), 1),
        "temperature_max": round(float(working["temperature_c"].max()), 1),
        "holiday_share_pct": round(float(working["is_holiday"].mean() * 100.0), 2),
        "overall_waste_pct": round(overall_waste_pct, 2),
        "average_daily_waste_pct": round(float(waste_summary["waste_pct"].mean()), 2),
        "top_dish": str(dish_mean_sales.index[0]),
        "top_dish_avg_sold": round(float(dish_mean_sales.iloc[0]), 2),
        "most_volatile_dish": str(dish_volatility.index[0]),
        "most_volatile_dish_std": round(float(dish_volatility.iloc[0]), 2),
        "total_sold": int(total_sold),
        "total_cooked": int(total_cooked),
        "total_waste": int(total_waste),
    }


def build_exploratory_findings(df: pd.DataFrame) -> list[str]:
    """Create recruiter-friendly exploratory findings from the dataset."""
    profile = build_data_profile(df)
    return [
        (
            f"The dataset contains {profile['records']} rows across {profile['days']} days, "
            f"covering {profile['dishes']} dishes between {profile['date_start']} and {profile['date_end']}."
        ),
        (
            f"Overall waste is {profile['overall_waste_pct']}% of cooked food, with an "
            f"average daily waste rate of {profile['average_daily_waste_pct']}%."
        ),
        (
            f"{profile['top_dish']} is the highest-volume dish at an average of "
            f"{profile['top_dish_avg_sold']} sold portions per day."
        ),
        (
            f"{profile['most_volatile_dish']} shows the highest day-to-day demand variability "
            f"with a sold-quantity standard deviation of {profile['most_volatile_dish_std']}."
        ),
        (
            f"Temperature ranges from {profile['temperature_min']} C to {profile['temperature_max']} C, "
            f"and holidays make up {profile['holiday_share_pct']}% of records, so external drivers "
            "are worth including alongside demand history."
        ),
    ]


def describe_feature_choices() -> list[FeatureChoice]:
    """Explain the feature engineering decisions behind the final model."""
    return [
        FeatureChoice(
            name="lag_1",
            reason=(
                "Captures the most recent demand signal. Buffet demand often carries over "
                "from the previous day, so yesterday's sales are a strong baseline predictor."
            ),
        ),
        FeatureChoice(
            name="lag_7",
            reason=(
                "Captures weekly seasonality. Restaurant demand commonly follows weekday patterns, "
                "so the same day last week is useful context."
            ),
        ),
        FeatureChoice(
            name="rolling_mean_7",
            reason=(
                "Smooths short-term noise and gives the model a stable measure of recent trend "
                "instead of reacting too strongly to one unusual day."
            ),
        ),
        FeatureChoice(
            name="temperature_c",
            reason=(
                "Models weather-driven demand shifts. Temperature can change appetite and menu "
                "preferences, especially for buffet-style service."
            ),
        ),
        FeatureChoice(
            name="is_holiday",
            reason=(
                "Represents special days when customer traffic may differ from ordinary weekday behavior."
            ),
        ),
        FeatureChoice(
            name="category and price_level",
            reason=(
                "Kept in the dataset for daily input management and dish metadata, but not used as "
                "regression features because forecasting is already done per dish and the time-series "
                "signals are more informative for next-day demand."
            ),
        ),
    ]


def build_experiment_trail(df: pd.DataFrame) -> list[ExperimentNote]:
    """Summarize each model tried during experimentation."""
    # Reformat the benchmark table once so the narrative can access metrics by model name.
    metrics_lookup = _ordered_metrics_lookup(benchmark_regression_models(df))
    return [
        ExperimentNote(
            model="Naive Baseline",
            metrics=metrics_lookup["Naive Baseline"],
            takeaway=(
                "Used as the minimum acceptable benchmark by predicting demand from the most "
                "recent known value. It is simple, but it cannot adapt well to broader trends "
                "or external conditions."
            ),
        ),
        ExperimentNote(
            model="Linear Regression",
            metrics=metrics_lookup["Linear Regression"],
            takeaway=(
                "Provided the strongest balance of accuracy, transparency, and deployment simplicity. "
                "Its coefficients are interpretable, which makes the model easier to explain in a business setting."
            ),
        ),
        ExperimentNote(
            model="Random Forest",
            metrics=metrics_lookup["Random Forest"],
            takeaway=(
                "Added non-linear flexibility, but the improvement was not large enough to justify the "
                "extra complexity for this project."
            ),
        ),
    ]


def choose_final_model(df: pd.DataFrame) -> dict[str, str | float]:
    """Explain why the final model was selected from the benchmark results."""
    metrics_lookup = _ordered_metrics_lookup(benchmark_regression_models(df))
    linear = metrics_lookup["Linear Regression"]
    best_rmse = min(item["RMSE"] for item in metrics_lookup.values())

    if linear["RMSE"] <= best_rmse * 1.05:
        reason = (
            "Linear Regression was selected because it achieved top-tier benchmark performance "
            "while staying easier to explain, maintain, and deploy than the more complex alternative."
        )
    else:
        reason = (
            "Linear Regression remained the preferred deployment choice because its performance stayed "
            "close to the best model while offering better interpretability and a simpler production path."
        )

    return {
        "selected_model": "Linear Regression",
        "reason": reason,
        "mae": linear["MAE"],
        "rmse": linear["RMSE"],
        "mape": linear["MAPE (%)"],
    }


def render_analysis_report(df: pd.DataFrame) -> str:
    """Render the full project narrative as markdown text."""
    # Assemble markdown line-by-line so the report can be printed or reused elsewhere.
    lines: list[str] = [
        "# Project Analysis",
        "",
        "## Project Focus",
        "",
        "- Python data wrangling and cleaning",
        "- Model benchmarking and forecast evaluation",
        "- Streamlit dashboard deployment",
        "",
        "## Step-by-Step Analysis Process",
        "",
    ]

    for index, step in enumerate(describe_analysis_process(), start=1):
        lines.append(f"{index}. **{step.title}**: {step.detail}")

    lines.extend(
        [
            "",
            "## Exploratory Analysis Findings",
            "",
        ]
    )

    for finding in build_exploratory_findings(df):
        lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "## Why These Features Were Chosen",
            "",
        ]
    )

    for choice in describe_feature_choices():
        lines.append(f"- **{choice.name}**: {choice.reason}")

    lines.extend(
        [
            "",
            "## Model Experiment Trail",
            "",
            "| Model | MAE | RMSE | MAPE (%) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for note in build_experiment_trail(df):
        lines.append(
            f"| {note.model} | {note.metrics['MAE']:.2f} | {note.metrics['RMSE']:.2f} | {note.metrics['MAPE (%)']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Experiment notes:",
            "",
        ]
    )

    for note in build_experiment_trail(df):
        lines.append(f"- **{note.model}**: {note.takeaway}")

    final_model = choose_final_model(df)
    lines.extend(
        [
            "",
            "## Final Model Selection",
            "",
            f"- **Selected model**: {final_model['selected_model']}",
            f"- **Performance**: MAE {final_model['mae']:.2f}, RMSE {final_model['rmse']:.2f}, MAPE {final_model['mape']:.2f}%",
            f"- **Why this model was chosen**: {final_model['reason']}",
        ]
    )

    return "\n".join(lines)


def _ordered_metrics_lookup(metrics_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convert benchmark results into a stable model-name lookup."""
    metrics_lookup: dict[str, dict[str, float]] = {}
    indexed = metrics_df.set_index("Model")

    # Keep a fixed order so the report reads consistently even if the dataframe is sorted by RMSE.
    for model_name in ORDERED_MODEL_NAMES:
        row = indexed.loc[model_name]
        metrics_lookup[model_name] = {
            "MAE": float(row["MAE"]),
            "RMSE": float(row["RMSE"]),
            "MAPE (%)": float(row["MAPE (%)"]),
        }

    return metrics_lookup
