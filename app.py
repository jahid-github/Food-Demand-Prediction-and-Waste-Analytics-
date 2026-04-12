from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Import the reusable project functions from the local `mixx` package.
from mixx import (
    DEFAULT_SOURCE_DATA_PATH,
    DEFAULT_WORKING_DATA_PATH,
    add_daily_data_and_predict,
    benchmark_regression_models,
    build_daily_input_frame,
    build_holiday_calendar,
    build_waste_summary,
    ensure_dashboard_table,
    ensure_working_dataset,
    get_tomorrow_temperature,
    load_dataset,
    predict_next_day_all_dishes_smart,
    predict_next_day_all_dishes_with_forecast,
    push_dashboard_to_bigquery,
    save_dataset,
)
from mixx.constants import DEFAULT_LATITUDE, DEFAULT_LONGITUDE

# Configure the page before rendering any Streamlit elements.
st.set_page_config(page_title="MIXX Dashboard", layout="wide")


def main() -> None:
    """Render the full dashboard and connect UI actions to the project logic."""
    # Apply the custom visual styling used by the landing section and metrics.
    _inject_styles()

    # Resolve where the app should read the seed dataset from and where it should
    # store the writable runtime CSV used for daily updates.
    source_path = Path(os.getenv("MIXX_SOURCE_DATA_PATH", DEFAULT_SOURCE_DATA_PATH))
    working_path = ensure_working_dataset(
        source_path=source_path,
        working_path=Path(os.getenv("MIXX_DATA_PATH", DEFAULT_WORKING_DATA_PATH)),
    )

    # Load the active dataset and precompute values that are reused across tabs.
    df = load_dataset(working_path)
    holiday_calendar = build_holiday_calendar(df)
    waste_summary = build_waste_summary(df)
    latest_date = df["date"].max().date()
    forecast_date = latest_date + pd.Timedelta(days=1)
    latest_temp = float(df.sort_values("date")["temperature_c"].iloc[-1])

    # Store the forecast temperature in session state so it survives Streamlit reruns
    # caused by widget interactions.
    if "forecast_temp_c" not in st.session_state:
        st.session_state["forecast_temp_c"] = latest_temp

    # Hero section that explains what this dashboard is and how it is intended to run.
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Food Demand Prediction and Waste Analytics</div>
            <h1>MIXX Streamlit Dashboard</h1>
            <p>
                Local CSV is the default runtime so the project stays portable on GitHub,
                in Docker, and for anyone cloning the repo. BigQuery stays available as an
                optional export path for the original Colab -> BigQuery -> Looker Studio flow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # Show which CSV file is acting as the app's live data source.
        st.subheader("Runtime")
        st.caption("Default mode is local CSV for GitHub and Docker portability.")
        st.code(str(working_path), language="text")
        st.caption(
            "Mount `data/runtime` as a Docker volume if you want writes to survive container restarts."
        )

        # Let the user override location inputs for the weather forecast request.
        latitude = st.number_input(
            "Latitude",
            value=float(os.getenv("MIXX_LATITUDE", DEFAULT_LATITUDE)),
            format="%.4f",
        )
        longitude = st.number_input(
            "Longitude",
            value=float(os.getenv("MIXX_LONGITUDE", DEFAULT_LONGITUDE)),
            format="%.4f",
        )
        if st.button("Fetch tomorrow weather", use_container_width=True):
            try:
                # Save the fetched value into session state so the forecast tab uses it.
                st.session_state["forecast_temp_c"] = get_tomorrow_temperature(
                    lat=latitude,
                    lon=longitude,
                )
                st.success(
                    f"Tomorrow max temperature: {st.session_state['forecast_temp_c']:.1f} C"
                )
            except Exception as exc:  # pragma: no cover - network dependent
                st.error(f"Weather fetch failed: {exc}")

        st.subheader("Architecture")
        st.markdown(
            "- `Local CSV -> Streamlit` is the default app mode.\n"
            "- `Colab -> BigQuery -> Looker Studio` stays available via optional export."
        )

    # Top-level summary metrics for the currently loaded dataset.
    metric_columns = st.columns(4)
    metric_columns[0].metric("Records", f"{len(df):,}")
    metric_columns[1].metric("Dishes", str(df["dish_name"].nunique()))
    metric_columns[2].metric("Latest Date", latest_date.isoformat())
    metric_columns[3].metric(
        "Latest Waste %",
        f"{waste_summary['waste_pct'].iloc[-1]:.2f}",
    )

    forecast_tab, update_tab, data_tab = st.tabs(
        ["Forecast", "Daily Update", "Data and Models"]
    )

    with forecast_tab:
        # This tab is for generating next-day predictions and downloading/exporting them.
        st.subheader("Next-Day Forecast")
        left_col, right_col = st.columns([1, 2], gap="large")

        with left_col:
            # The forecast date cannot be earlier than the day after the latest data row.
            selected_forecast_date = st.date_input(
                "Forecast date",
                value=forecast_date,
                min_value=forecast_date,
            )
            forecast_temp_c = st.number_input(
                "Temperature for forecast (C)",
                value=float(st.session_state["forecast_temp_c"]),
                step=0.5,
            )
            display_mode = st.radio(
                "Forecast mode",
                options=["Weather-aware", "Baseline"],
                horizontal=True,
            )

            # Build both forecast versions first, then show the one chosen in the radio.
            baseline_dashboard = predict_next_day_all_dishes_smart(
                df, forecast_date=selected_forecast_date
            )
            weather_dashboard = predict_next_day_all_dishes_with_forecast(
                df,
                temp_tomorrow_c=forecast_temp_c,
                date_tomorrow=selected_forecast_date,
                holiday_calendar=holiday_calendar,
            )
            active_dashboard = (
                weather_dashboard if display_mode == "Weather-aware" else baseline_dashboard
            )

            # Allow the current forecast table to be downloaded as a CSV file.
            st.download_button(
                label="Download forecast CSV",
                data=_to_csv_bytes(active_dashboard),
                file_name=f"mixx_forecast_{selected_forecast_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Optional export for users who still want to feed the BigQuery dashboard path.
            _render_bigquery_export(
                active_dashboard=active_dashboard,
                selected_forecast_date=selected_forecast_date,
                forecast_temp_c=forecast_temp_c,
                is_holiday=int(pd.to_datetime(selected_forecast_date) in holiday_calendar),
            )

        with right_col:
            # Show the selected forecast as both a table and a quick visual comparison.
            st.dataframe(
                _format_forecast_for_display(active_dashboard),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(
                active_dashboard.set_index("dish_name")["predicted_next_day_demand"],
                use_container_width=True,
            )

    with update_tab:
        # This tab captures actual cooked/sold values and writes them back to the runtime CSV.
        st.subheader("Write Daily Actuals to Local CSV")
        st.caption(
            f"Updates are written to `{working_path}`. In Docker, mount `data/runtime` to persist them."
        )

        # Default the entry form to the next day after the latest known dataset date.
        entry_date = st.date_input("Actuals date", value=forecast_date, key="actuals_date")
        observed_temp_c = st.number_input(
            "Observed temperature for that date (C)",
            value=latest_temp,
            step=0.5,
            key="observed_temp_c",
        )

        # Start the editor with the latest known metadata and quantities for each dish.
        entry_template = build_daily_input_frame(df)
        edited_entries = st.data_editor(
            entry_template,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "dish_name": st.column_config.TextColumn("dish_name", required=True),
                "category": st.column_config.TextColumn("category", required=True),
                "price_level": st.column_config.TextColumn("price_level", required=True),
                "sold_qty": st.column_config.NumberColumn(
                    "sold_qty", min_value=0, step=1, required=True
                ),
                "cooked_qty": st.column_config.NumberColumn(
                    "cooked_qty", min_value=0, step=1, required=True
                ),
            },
        )

        if st.button("Save daily data", type="primary"):
            try:
                # Validate and normalize the edited grid before turning it into dictionaries
                # expected by the modeling layer.
                cleaned_entries = _clean_daily_entries(edited_entries)
                sold_dict = dict(
                    zip(cleaned_entries["dish_name"], cleaned_entries["sold_qty"])
                )
                cooked_dict = dict(
                    zip(cleaned_entries["dish_name"], cleaned_entries["cooked_qty"])
                )
                new_dish_meta = {
                    row["dish_name"]: {
                        "category": row["category"],
                        "price_level": row["price_level"],
                    }
                    for _, row in cleaned_entries.iterrows()
                }

                # Append the new daily rows and immediately refresh tomorrow's prediction.
                updated_df, updated_dashboard, tomorrow = add_daily_data_and_predict(
                    df=df,
                    date_today=entry_date,
                    temp_today_c=observed_temp_c,
                    sold_dict=sold_dict,
                    cooked_dict=cooked_dict,
                    new_dish_meta=new_dish_meta,
                )
                # Persist the updated dataset back to the working CSV file.
                save_dataset(updated_df, working_path)

                st.success(
                    f"Saved {len(cleaned_entries)} dish records to {working_path} and refreshed the baseline forecast for {tomorrow}."
                )
                st.dataframe(
                    _format_forecast_for_display(updated_dashboard),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.error(str(exc))

    with data_tab:
        # This tab is a reference area for model quality, raw rows, and waste trend history.
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("Model Benchmark")
            try:
                metrics_df = benchmark_regression_models(df)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Benchmark could not be generated: {exc}")

            st.subheader("Current Dataset Snapshot")
            st.dataframe(
                df.sort_values(["date", "dish_name"], ascending=[False, True]).head(20),
                use_container_width=True,
                hide_index=True,
            )

        with right_col:
            st.subheader("Waste Trend")
            st.line_chart(
                waste_summary.set_index("date")["waste_pct"],
                use_container_width=True,
            )

            st.subheader("Download Runtime CSV")
            st.download_button(
                label="Download current dataset",
                data=working_path.read_bytes(),
                file_name=working_path.name,
                mime="text/csv",
                use_container_width=True,
            )


def _clean_daily_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Validate the editable daily input grid before saving it."""
    cleaned = entries.copy()

    # Trim whitespace so accidental spaces do not create duplicate or empty dish names.
    cleaned["dish_name"] = cleaned["dish_name"].astype(str).str.strip()
    cleaned["category"] = cleaned["category"].astype(str).str.strip()
    cleaned["price_level"] = cleaned["price_level"].astype(str).str.strip()

    # Remove incomplete rows that the user may have left blank in the editor.
    cleaned = cleaned[
        cleaned["dish_name"].ne("")
        & cleaned["category"].ne("")
        & cleaned["price_level"].ne("")
    ].copy()

    # Protect the downstream save/prediction functions from invalid input.
    if cleaned.empty:
        raise ValueError("At least one valid dish row is required.")
    if cleaned["dish_name"].duplicated().any():
        duplicated = cleaned.loc[cleaned["dish_name"].duplicated(), "dish_name"].tolist()
        raise ValueError(f"Duplicate dish names are not allowed: {duplicated}")

    # Convert quantity columns to integers and reject any negative values.
    cleaned["sold_qty"] = pd.to_numeric(cleaned["sold_qty"], errors="raise").astype(int)
    cleaned["cooked_qty"] = pd.to_numeric(cleaned["cooked_qty"], errors="raise").astype(int)
    if (cleaned["sold_qty"] < 0).any() or (cleaned["cooked_qty"] < 0).any():
        raise ValueError("Sold and cooked quantities must be non-negative.")

    return cleaned.sort_values("dish_name").reset_index(drop=True)


def _format_forecast_for_display(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Rename internal column names into labels that are easier to read in the UI."""
    display_df = forecast_df.rename(
        columns={
            "dish_name": "Dish Name",
            "predicted_next_day_demand": "Predicted Next Day Demand",
            "model_used": "Model Used",
            "forecast_date": "Forecast Date",
            "temp_c": "Temperature (C)",
            "is_holiday": "Holiday",
        }
    ).copy()
    display_df["Predicted Next Day Demand"] = display_df[
        "Predicted Next Day Demand"
    ].round(2)
    return display_df


def _render_bigquery_export(
    active_dashboard: pd.DataFrame,
    selected_forecast_date: object,
    forecast_temp_c: float,
    is_holiday: int,
) -> None:
    """Render the optional export panel for the original BigQuery workflow."""
    with st.expander("Optional BigQuery export"):
        st.caption(
            "This keeps the original Colab -> BigQuery -> Looker Studio path available."
        )

        # Read export settings from environment variables so the feature stays optional.
        project_id = os.getenv("MIXX_GCP_PROJECT", "").strip()
        dataset_id = os.getenv("MIXX_GCP_DATASET", "mixx").strip()
        table_id = os.getenv("MIXX_GCP_TABLE", "dashboard_predictions").strip()
        scenario = st.text_input("Scenario", value="streamlit_forecast")

        if not project_id:
            st.info("Set `MIXX_GCP_PROJECT` and Google credentials to enable this export.")
            return

        if st.button("Export to BigQuery", use_container_width=True):
            try:
                # Ensure the destination dataset/table exists before loading rows into it.
                ensure_dashboard_table(
                    project_id=project_id,
                    dataset_id=dataset_id,
                    table_id=table_id,
                )
                # Upload the currently selected forecast as one named scenario.
                push_dashboard_to_bigquery(
                    dashboard_df=active_dashboard,
                    scenario=scenario,
                    forecast_date=str(selected_forecast_date),
                    project_id=project_id,
                    dataset_id=dataset_id,
                    table_id=table_id,
                    temp_c=forecast_temp_c,
                    is_holiday=is_holiday,
                )
                st.success(
                    f"Exported forecast to {project_id}.{dataset_id}.{table_id} for scenario `{scenario}`."
                )
            except Exception as exc:
                st.error(f"BigQuery export failed: {exc}")


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a dataframe into bytes so Streamlit can serve it as a file download."""
    return df.to_csv(index=False).encode("utf-8")


def _inject_styles() -> None:
    """Inject custom CSS used to give the dashboard its branded look."""
    st.markdown(
        """
        <style>
            /* Style the header card at the top of the app. */
            .hero {
                background:
                    radial-gradient(circle at top left, rgba(43, 111, 82, 0.16), transparent 38%),
                    linear-gradient(135deg, #fff7ec 0%, #f7f0de 55%, #eef5ea 100%);
                border: 1px solid rgba(23, 49, 38, 0.08);
                border-radius: 24px;
                padding: 2rem 2.25rem;
                margin-bottom: 1.25rem;
            }
            /* Main dashboard title inside the hero card. */
            .hero h1 {
                color: #173126;
                margin: 0.2rem 0 0.6rem 0;
                font-size: 2.35rem;
            }
            /* Supporting description text inside the hero card. */
            .hero p {
                color: #315545;
                font-size: 1rem;
                margin: 0;
                max-width: 52rem;
            }
            /* Small uppercase label above the hero title. */
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.78rem;
                color: #7d5a2d;
                font-weight: 700;
            }
            /* Override Streamlit metric values so they match the app palette. */
            [data-testid="stMetricValue"] {
                color: #173126;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Run the app only when this file is executed directly by Streamlit/Python.
    main()
