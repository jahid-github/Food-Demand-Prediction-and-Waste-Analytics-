from __future__ import annotations

from typing import Any, Mapping

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .constants import DEFAULT_SPLIT_DATE, FEATURE_COLUMNS, MIN_HISTORY_ROWS


def build_holiday_calendar(df: pd.DataFrame) -> Any:
    years = set(pd.to_datetime(df["date"]).dt.year.astype(int).tolist())
    if years:
        years.add(max(years) + 1)
    else:
        current_year = pd.Timestamp.today().year
        years.update({current_year, current_year + 1})
    return holidays.Finland(years=sorted(years))


def prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    working = working.sort_values(["dish_name", "date"]).reset_index(drop=True)

    working["target_next_day"] = working.groupby("dish_name")["sold_qty"].shift(-1)
    working["lag_1"] = working.groupby("dish_name")["sold_qty"].shift(1)
    working["lag_7"] = working.groupby("dish_name")["sold_qty"].shift(7)
    working["rolling_mean_7"] = (
        working.groupby("dish_name")["sold_qty"].shift(1).rolling(7).mean()
    )

    model_frame = working.dropna(subset=["target_next_day", *FEATURE_COLUMNS]).reset_index(
        drop=True
    )
    return working, model_frame


def benchmark_regression_models(
    df: pd.DataFrame, split_date: str = DEFAULT_SPLIT_DATE
) -> pd.DataFrame:
    _, model_frame = prepare_model_frame(df)
    if model_frame.empty:
        raise ValueError("Model frame is empty. Check the input dataset.")

    split_ts = pd.to_datetime(split_date)
    train = model_frame[model_frame["date"] < split_ts]
    test = model_frame[model_frame["date"] >= split_ts]

    if train.empty or test.empty:
        raise ValueError(
            "The configured split date produced an empty train or test set."
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_next_day"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target_next_day"]

    results = []

    baseline_pred = X_test["lag_1"]
    results.append(_evaluate_model("Naive Baseline", y_test, baseline_pred))

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    results.append(_evaluate_model("Linear Regression", y_test, linear_pred))

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    random_forest.fit(X_train, y_train)
    random_forest_pred = random_forest.predict(X_test)
    results.append(_evaluate_model("Random Forest", y_test, random_forest_pred))

    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)


def predict_next_day_all_dishes_smart(
    df: pd.DataFrame,
    forecast_date: str | pd.Timestamp | None = None,
    minimum_history_rows: int = MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    forecast_ts = (
        pd.to_datetime(forecast_date)
        if forecast_date is not None
        else working["date"].max() + pd.Timedelta(days=1)
    )

    predictions = []
    for dish_name in working["dish_name"].sort_values().unique():
        dish_frame = (
            working[working["dish_name"] == dish_name]
            .sort_values("date")
            .reset_index(drop=True)
        )

        predictions.append(
            _predict_single_dish(
                dish_frame,
                forecast_date=forecast_ts,
                forecast_temp_c=None,
                forecast_is_holiday=None,
                minimum_history_rows=minimum_history_rows,
            )
        )

    return (
        pd.DataFrame(predictions)
        .sort_values("predicted_next_day_demand", ascending=False)
        .reset_index(drop=True)
    )


def predict_next_day_all_dishes_with_forecast(
    df: pd.DataFrame,
    temp_tomorrow_c: float,
    date_tomorrow: str | pd.Timestamp,
    holiday_calendar: Any | None = None,
    minimum_history_rows: int = MIN_HISTORY_ROWS,
) -> pd.DataFrame:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    forecast_ts = pd.to_datetime(date_tomorrow)
    holiday_calendar = holiday_calendar or build_holiday_calendar(working)
    is_holiday_tomorrow = int(forecast_ts in holiday_calendar)

    predictions = []
    for dish_name in working["dish_name"].sort_values().unique():
        dish_frame = (
            working[working["dish_name"] == dish_name]
            .sort_values("date")
            .reset_index(drop=True)
        )
        predictions.append(
            _predict_single_dish(
                dish_frame,
                forecast_date=forecast_ts,
                forecast_temp_c=float(temp_tomorrow_c),
                forecast_is_holiday=is_holiday_tomorrow,
                minimum_history_rows=minimum_history_rows,
            )
        )

    return (
        pd.DataFrame(predictions)
        .sort_values("predicted_next_day_demand", ascending=False)
        .reset_index(drop=True)
    )


def add_daily_data_and_predict(
    df: pd.DataFrame,
    date_today: str | pd.Timestamp,
    temp_today_c: float,
    sold_dict: Mapping[str, int],
    cooked_dict: Mapping[str, int],
    new_dish_meta: Mapping[str, Mapping[str, str]] | None = None,
    forecast_temp_tomorrow: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    updated_df = append_daily_records(
        df=df,
        date_today=date_today,
        temp_today_c=temp_today_c,
        sold_dict=sold_dict,
        cooked_dict=cooked_dict,
        new_dish_meta=new_dish_meta,
    )

    tomorrow = (pd.to_datetime(date_today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if forecast_temp_tomorrow is None:
        dashboard = predict_next_day_all_dishes_smart(updated_df, forecast_date=tomorrow)
    else:
        dashboard = predict_next_day_all_dishes_with_forecast(
            updated_df,
            temp_tomorrow_c=forecast_temp_tomorrow,
            date_tomorrow=tomorrow,
            holiday_calendar=build_holiday_calendar(updated_df),
        )

    return updated_df, dashboard, tomorrow


def append_daily_records(
    df: pd.DataFrame,
    date_today: str | pd.Timestamp,
    temp_today_c: float,
    sold_dict: Mapping[str, int],
    cooked_dict: Mapping[str, int],
    new_dish_meta: Mapping[str, Mapping[str, str]] | None = None,
    replace_existing: bool = True,
) -> pd.DataFrame:
    if not sold_dict:
        raise ValueError("At least one dish entry is required.")
    if set(sold_dict) != set(cooked_dict):
        raise ValueError("sold_dict and cooked_dict must contain the same dish names.")

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"])
    entry_date = pd.to_datetime(date_today)
    holiday_calendar = build_holiday_calendar(working)
    new_dish_meta = dict(new_dish_meta or {})

    if replace_existing:
        existing_mask = (working["date"] == entry_date) & (
            working["dish_name"].isin(sold_dict.keys())
        )
        working = working.loc[~existing_mask].copy()

    new_rows = []
    for dish_name in sorted(sold_dict):
        sold_qty = int(sold_dict[dish_name])
        cooked_qty = int(cooked_dict[dish_name])
        if sold_qty < 0 or cooked_qty < 0:
            raise ValueError("Sold and cooked quantities must be non-negative.")

        if dish_name in working["dish_name"].values:
            latest_meta = (
                working[working["dish_name"] == dish_name]
                .sort_values("date")
                .iloc[-1]
            )
            category = str(latest_meta["category"])
            price_level = str(latest_meta["price_level"])
        else:
            category = str(new_dish_meta.get(dish_name, {}).get("category", "Main"))
            price_level = str(
                new_dish_meta.get(dish_name, {}).get("price_level", "Medium")
            )

        new_rows.append(
            {
                "date": entry_date,
                "weekday": entry_date.day_name(),
                "dish_name": dish_name,
                "category": category,
                "price_level": price_level,
                "temperature_c": float(temp_today_c),
                "is_holiday": int(entry_date in holiday_calendar),
                "cooked_qty": cooked_qty,
                "sold_qty": sold_qty,
                "wasted_qty": max(cooked_qty - sold_qty, 0),
            }
        )

    updated = pd.concat([working, pd.DataFrame(new_rows)], ignore_index=True)
    return updated.sort_values(["date", "dish_name"]).reset_index(drop=True)


def _predict_single_dish(
    dish_frame: pd.DataFrame,
    forecast_date: pd.Timestamp,
    forecast_temp_c: float | None,
    forecast_is_holiday: int | None,
    minimum_history_rows: int,
) -> dict[str, Any]:
    last_row = dish_frame.iloc[-1]
    base_temp = float(last_row["temperature_c"])
    base_is_holiday = int(last_row["is_holiday"])

    if len(dish_frame) < minimum_history_rows:
        return {
            "dish_name": str(last_row["dish_name"]),
            "predicted_next_day_demand": round(float(dish_frame["sold_qty"].mean()), 2),
            "model_used": "Fallback Average",
            "forecast_date": forecast_date.date().isoformat(),
            "temp_c": float(forecast_temp_c if forecast_temp_c is not None else base_temp),
            "is_holiday": int(
                forecast_is_holiday if forecast_is_holiday is not None else base_is_holiday
            ),
        }

    prepared_frame = dish_frame.copy()
    prepared_frame["target_next_day"] = prepared_frame["sold_qty"].shift(-1)
    prepared_frame["lag_1"] = prepared_frame["sold_qty"].shift(1)
    prepared_frame["lag_7"] = prepared_frame["sold_qty"].shift(7)
    prepared_frame["rolling_mean_7"] = (
        prepared_frame["sold_qty"].shift(1).rolling(7).mean()
    )
    model_frame = prepared_frame.dropna(
        subset=["target_next_day", *FEATURE_COLUMNS]
    ).reset_index(drop=True)

    if len(model_frame) < 2:
        return {
            "dish_name": str(last_row["dish_name"]),
            "predicted_next_day_demand": round(float(dish_frame["sold_qty"].mean()), 2),
            "model_used": "Fallback Average",
            "forecast_date": forecast_date.date().isoformat(),
            "temp_c": float(forecast_temp_c if forecast_temp_c is not None else base_temp),
            "is_holiday": int(
                forecast_is_holiday if forecast_is_holiday is not None else base_is_holiday
            ),
        }

    X = model_frame[FEATURE_COLUMNS]
    y = model_frame["target_next_day"]
    model = LinearRegression()
    model.fit(X[:-1], y[:-1])

    X_next = model_frame.iloc[[-1]][FEATURE_COLUMNS].copy()
    if forecast_temp_c is not None:
        X_next["temperature_c"] = float(forecast_temp_c)
    if forecast_is_holiday is not None:
        X_next["is_holiday"] = int(forecast_is_holiday)

    prediction = max(float(model.predict(X_next)[0]), 0.0)

    return {
        "dish_name": str(last_row["dish_name"]),
        "predicted_next_day_demand": round(prediction, 2),
        "model_used": "Linear Regression",
        "forecast_date": forecast_date.date().isoformat(),
        "temp_c": float(
            forecast_temp_c if forecast_temp_c is not None else X_next["temperature_c"].iloc[0]
        ),
        "is_holiday": int(
            forecast_is_holiday if forecast_is_holiday is not None else X_next["is_holiday"].iloc[0]
        ),
    }


def _evaluate_model(name: str, y_true: pd.Series, y_pred: Any) -> dict[str, float | str]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "Model": name,
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE (%)": round(float(mape), 2),
    }
