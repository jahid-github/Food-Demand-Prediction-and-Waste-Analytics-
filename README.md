# MIXX - Food Demand Prediction and Waste Analytics

Machine learning based food demand prediction and waste analytics for buffet restaurants.

This repository now supports two deployment paths:

1. `Local CSV -> Streamlit`
2. `Google Colab -> BigQuery -> Looker Studio`

The default app runtime is local CSV because it is the easiest way to keep the project portable on GitHub, reusable by other people, and deployable with Docker without requiring Google Cloud credentials.

## What the project does

- Predicts next-day dish demand
- Tracks cooked, sold, and wasted quantities
- Benchmarks regression models
- Supports dish-level daily input updates
- Keeps BigQuery export available for the original dashboard flow

## Final model

The notebook compares three approaches:

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Linear Regression | 10.20 | 13.06 | 9.36 |
| Random Forest | 10.29 | 13.60 | 9.22 |
| Naive Baseline | 15.69 | 19.32 | 14.69 |

Linear Regression remains the selected default model because it is simpler to explain, cheaper to run, and matches the original notebook decision.

## Current architectures

### Option A: Local first Streamlit app

Default for this repo:

```text
CSV -> Python modules -> Streamlit dashboard
```

Why this is the default:

- Works directly from GitHub
- Easy to run locally
- Easy to package with Docker
- No cloud credentials required
- Other people can clone and use it immediately

Persistence behavior:

- The app reads from the seed dataset `mixx_synthetic_restaurant_data.csv`
- On first run it creates a writable runtime copy at `data/runtime/restaurant_data.csv`
- Daily updates are written to the runtime CSV
- In Docker, mount `data/runtime` as a volume if you want writes to survive container restarts

### Option B: Original cloud dashboard flow

Still supported:

```text
Google Colab -> BigQuery -> Looker Studio
```

Use this if you want:

- Shared cloud storage
- Looker Studio dashboards
- A managed analytics layer
- Scenario history inside BigQuery

The Streamlit app includes an optional BigQuery export panel. If Google Cloud credentials and environment variables are present, the current forecast can be pushed into the same `dashboard_predictions` table pattern used by the notebook.

## Project structure

```text
app.py
mixx/
  constants.py
  data.py
  modeling.py
  weather.py
  bigquery_export.py
mixx_synthetic_restaurant_data.csv
data/runtime/
requirements.txt
Dockerfile
MIXX_Final_Model.ipynb
```

## Python modules extracted from the notebook

The notebook logic was moved into reusable `.py` files:

- `mixx/data.py`
  - load and save CSV data
  - build the daily input table
  - summarize waste
- `mixx/modeling.py`
  - create lag features
  - benchmark models
  - generate baseline forecasts
  - generate weather-aware forecasts
  - append daily actuals and refresh forecasts
- `mixx/weather.py`
  - fetch tomorrow temperature from Open-Meteo
- `mixx/bigquery_export.py`
  - create the BigQuery table
  - export Streamlit forecasts to BigQuery

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

## Run with Docker

Build the image:

```bash
docker build -t mixx-dashboard .
```

Run the container:

```bash
docker run --rm -p 8501:8501 -v "${PWD}/data/runtime:/app/data/runtime" mixx-dashboard
```

Open:

```text
http://localhost:8501
```

## Environment variables

Optional runtime variables:

- `MIXX_DATA_PATH`
  - Custom path for the writable runtime CSV
- `MIXX_SOURCE_DATA_PATH`
  - Custom path for the seed CSV
- `MIXX_LATITUDE`
  - Latitude for weather lookup
- `MIXX_LONGITUDE`
  - Longitude for weather lookup

Optional BigQuery variables:

- `MIXX_GCP_PROJECT`
- `MIXX_GCP_DATASET`
- `MIXX_GCP_TABLE`
- `GOOGLE_APPLICATION_CREDENTIALS`

If `MIXX_GCP_PROJECT` is not set, the app stays in local CSV mode and BigQuery export remains disabled.

## Streamlit app features

- Weather-aware forecast mode
- Baseline forecast mode
- Daily actuals entry with dynamic dish rows
- Local CSV persistence
- Runtime CSV download
- Waste trend view
- Model benchmark table
- Optional BigQuery export

## Notes on deployment choice

For this repository, the best default deployment choice is:

```text
Local CSV + Streamlit + Docker
```

Reason:

- It keeps the project simple for GitHub users
- It avoids forcing every user to set up BigQuery
- It is easy to demonstrate and grade
- It can still grow into a cloud architecture later

If you later want a public production workflow, you can keep the Streamlit app as the front end and move persistence to BigQuery, a database, or object storage without changing the forecasting logic too much.

## Original notebook

The notebook is still included as:

- `MIXX_Final_Model.ipynb`

It remains useful as the project report, experiment log, and Colab-first reference implementation.
