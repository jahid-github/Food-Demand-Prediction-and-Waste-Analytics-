from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def ensure_dashboard_table(
    project_id: str,
    dataset_id: str = "mixx",
    table_id: str = "dashboard_predictions",
    location: str = "EU",
) -> str:
    bigquery = _import_bigquery()
    client = bigquery.Client(project=project_id)

    dataset_ref = bigquery.Dataset(f"{project_id}.{dataset_id}")
    dataset_ref.location = location
    client.create_dataset(dataset_ref, exists_ok=True)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    schema = [
        bigquery.SchemaField("run_ts", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("scenario", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("forecast_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("dish_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("predicted_next_day_demand", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("model_used", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("temp_c", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("is_holiday", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("sold_qty", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("cooked_qty", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("waste_qty", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("waste_pct", "FLOAT", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table, exists_ok=True)
    return table_ref


def push_dashboard_to_bigquery(
    dashboard_df: pd.DataFrame,
    scenario: str,
    forecast_date: str,
    project_id: str,
    dataset_id: str = "mixx",
    table_id: str = "dashboard_predictions",
    temp_c: float | None = None,
    is_holiday: int | None = None,
    sold_today: Mapping[str, int] | None = None,
    cooked_today: Mapping[str, int] | None = None,
    replace_scenario: bool = True,
) -> pd.DataFrame:
    bigquery = _import_bigquery()
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    df_out = _normalize_dashboard_columns(dashboard_df).copy()
    required_columns = {"dish_name", "predicted_next_day_demand"}
    missing_columns = required_columns.difference(df_out.columns)
    if missing_columns:
        raise ValueError(f"Dashboard data is missing required columns: {missing_columns}")

    if "model_used" not in df_out.columns:
        df_out["model_used"] = None

    df_out["scenario"] = str(scenario)
    df_out["forecast_date"] = pd.to_datetime(forecast_date).date()
    df_out["run_ts"] = pd.Timestamp.utcnow()
    df_out["temp_c"] = None if temp_c is None else float(temp_c)
    df_out["is_holiday"] = None if is_holiday is None else int(is_holiday)

    sold_today = sold_today or {}
    cooked_today = cooked_today or {}
    df_out["sold_qty"] = df_out["dish_name"].map(sold_today)
    df_out["cooked_qty"] = df_out["dish_name"].map(cooked_today)

    cooked_series = pd.to_numeric(df_out["cooked_qty"], errors="coerce")
    sold_series = pd.to_numeric(df_out["sold_qty"], errors="coerce")
    waste_qty = (cooked_series - sold_series).clip(lower=0)
    df_out["waste_qty"] = waste_qty
    df_out["waste_pct"] = (
        waste_qty / cooked_series.where(cooked_series > 0)
    ).fillna(0.0) * 100.0

    ordered_columns = [
        "run_ts",
        "scenario",
        "forecast_date",
        "dish_name",
        "predicted_next_day_demand",
        "model_used",
        "temp_c",
        "is_holiday",
        "sold_qty",
        "cooked_qty",
        "waste_qty",
        "waste_pct",
    ]
    df_out = df_out[ordered_columns]

    if replace_scenario:
        delete_sql = f"""
            DELETE FROM `{table_ref}`
            WHERE scenario = @scenario
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("scenario", "STRING", str(scenario))
            ]
        )
        client.query(delete_sql, job_config=job_config).result()

    load_job = client.load_table_from_dataframe(df_out, table_ref)
    load_job.result()
    return df_out


def _normalize_dashboard_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Dish Name": "dish_name",
        "Predicted Next Day Demand": "predicted_next_day_demand",
        "Model Used": "model_used",
        "Forecast Date": "forecast_date",
        "Temperature (C)": "temp_c",
        "Holiday": "is_holiday",
    }
    return df.rename(columns=rename_map)


def _import_bigquery() -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery support requires google-cloud-bigquery to be installed."
        ) from exc
    return bigquery
