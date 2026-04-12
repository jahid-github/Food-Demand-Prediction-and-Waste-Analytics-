"""Streamlit entrypoint for the MIXX forecasting dashboard.

This file is intentionally UI-focused. The data cleaning, modeling, and report
generation live in the `mixx` package so the app stays thin and easier to read.
"""

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
    ensure_working_dataset,
    get_tomorrow_temperature,
    load_dataset,
    predict_next_day_all_dishes_smart,
    predict_next_day_all_dishes_with_forecast,
    save_dataset,
)
from mixx.constants import DEFAULT_LATITUDE, DEFAULT_LONGITUDE

# Configure the page before rendering any Streamlit elements.
st.set_page_config(page_title="MIXX Forecasting Dashboard", layout="wide")


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
                An end-to-end Python project focused on data wrangling, feature engineering,
                model evaluation, and interactive forecasting for buffet restaurant operations.
                The final dashboard turns the analysis into a recruiter-friendly Streamlit app.
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

        st.subheader("Project Focus")
        st.markdown(
            "- `Python data wrangling and cleaning`\n"
            "- `Model benchmarking and forecast evaluation`\n"
            "- `Streamlit dashboard deployment`"
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
        # This tab is for generating next-day predictions and downloading them.
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


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a dataframe into bytes so Streamlit can serve it as a file download."""
    return df.to_csv(index=False).encode("utf-8")


def _inject_styles() -> None:
    """Inject custom CSS used to give the dashboard its branded look."""
    st.markdown(
        """
        <style>
            :root {
                --mixx-bg: #f7f3ec;
                --mixx-surface: #fbf7f1;
                --mixx-surface-strong: #efe6d7;
                --mixx-border: rgba(134, 109, 74, 0.18);
                --mixx-shadow: 0 18px 34px rgba(63, 52, 38, 0.08);
                --mixx-shadow-soft: 0 10px 22px rgba(63, 52, 38, 0.05);
                --mixx-text: #1b302b;
                --mixx-muted: #49645b;
                --mixx-accent: #315f56;
                --mixx-accent-soft: #dfeae4;
                --mixx-highlight: #b48352;
            }
            /* Give the app a soft layered background instead of a flat ash tone. */
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(180, 131, 82, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(49, 95, 86, 0.12), transparent 30%),
                    linear-gradient(180deg, #faf7f2 0%, var(--mixx-bg) 100%);
            }
            /* Keep the main content roomy and centered. */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            /* Give the sidebar a slightly raised panel feel. */
            [data-testid="stSidebar"] {
                background:
                    linear-gradient(180deg, rgba(255, 250, 243, 0.78) 0%, rgba(237, 228, 213, 0.96) 100%);
                box-shadow: inset -1px 0 0 rgba(134, 109, 74, 0.10);
            }
            /* Style the header card at the top of the app. */
            .hero {
                background:
                    radial-gradient(circle at top left, rgba(49, 95, 86, 0.18), transparent 36%),
                    radial-gradient(circle at bottom right, rgba(180, 131, 82, 0.16), transparent 32%),
                    linear-gradient(145deg, rgba(255, 251, 245, 0.96) 0%, rgba(243, 236, 224, 0.98) 100%);
                border: 1px solid rgba(134, 109, 74, 0.14);
                border-radius: 28px;
                box-shadow: var(--mixx-shadow), inset 0 1px 0 rgba(255, 255, 255, 0.7);
                padding: 2.15rem 2.35rem;
                margin-bottom: 1.4rem;
            }
            /* Main dashboard title inside the hero card. */
            .hero h1 {
                color: var(--mixx-text);
                margin: 0.2rem 0 0.6rem 0;
                font-size: 2.35rem;
                letter-spacing: -0.02em;
            }
            /* Supporting description text inside the hero card. */
            .hero p {
                color: var(--mixx-muted);
                font-size: 1rem;
                margin: 0;
                max-width: 52rem;
                line-height: 1.65;
            }
            /* Small uppercase label above the hero title. */
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.78rem;
                color: var(--mixx-highlight);
                font-weight: 700;
            }
            /* Add a soft card surface to key dashboard widgets for a simple 3D effect. */
            [data-testid="stMetric"],
            [data-testid="stDataFrame"],
            [data-testid="stVegaLiteChart"],
            [data-testid="stFileUploader"],
            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            div[data-baseweb="textarea"] > div,
            [data-testid="stDateInputField"] {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.84) 0%, rgba(249, 244, 235, 0.98) 100%);
                border: 1px solid var(--mixx-border);
                border-radius: 22px;
                box-shadow: var(--mixx-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.72);
            }
            /* Improve default metric card spacing and lift. */
            [data-testid="stMetric"] {
                padding: 1rem 1.1rem;
            }
            /* Override Streamlit metric values so they match the app palette. */
            [data-testid="stMetricValue"] {
                color: var(--mixx-text);
                letter-spacing: -0.02em;
            }
            /* Keep metric labels readable but quieter than the values. */
            [data-testid="stMetricLabel"] {
                color: var(--mixx-muted);
            }
            /* Style tabs as raised chips so section switching feels clearer. */
            div[data-baseweb="tab-list"] {
                gap: 0.6rem;
            }
            button[data-baseweb="tab"] {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.8) 0%, rgba(241, 232, 220, 0.95) 100%);
                border: 1px solid var(--mixx-border);
                border-radius: 18px 18px 0 0;
                box-shadow: var(--mixx-shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.74);
                color: var(--mixx-muted);
                font-weight: 600;
                padding: 0.7rem 1rem;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(223, 234, 228, 0.98) 100%);
                border-color: rgba(49, 95, 86, 0.22);
                color: var(--mixx-text);
            }
            /* Give the active tab panel the same card treatment as the rest of the dashboard. */
            .stTabs [data-baseweb="tab-panel"] {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.78) 0%, rgba(248, 243, 235, 0.94) 100%);
                border: 1px solid var(--mixx-border);
                border-radius: 0 22px 22px 22px;
                box-shadow: var(--mixx-shadow);
                padding: 1.2rem 1.15rem 1rem 1.15rem;
            }
            /* Make action buttons consistent with the theme and slightly raised. */
            .stButton > button,
            .stDownloadButton > button {
                background: linear-gradient(180deg, #3d7368 0%, var(--mixx-accent) 100%);
                border: 1px solid rgba(31, 65, 58, 0.18);
                border-radius: 999px;
                box-shadow: 0 14px 24px rgba(49, 95, 86, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.20);
                color: #f8f4ee;
                font-weight: 600;
                min-height: 2.9rem;
            }
            /* Add a gentle hover lift without making the UI feel busy. */
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: rgba(31, 65, 58, 0.24);
                transform: translateY(-1px);
            }
            /* Keep headings dark and easy to scan against the light palette. */
            h2, h3, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
                color: var(--mixx-text);
            }
            /* Make code/path displays look like small inset panels. */
            pre {
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.86) 0%, rgba(244, 238, 229, 0.98) 100%);
                border: 1px solid var(--mixx-border);
                border-radius: 18px;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.68);
            }
            /* Keep data editors and tables integrated with the same visual system. */
            [data-testid="stDataFrame"] {
                padding: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Run the app only when this file is executed directly by Streamlit/Python.
    main()
