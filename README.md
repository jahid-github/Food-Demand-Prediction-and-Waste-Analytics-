# MIXX - Food Demand Prediction and Waste Analytics

An end-to-end Python machine learning project for buffet restaurant operations. The project focuses on data wrangling, feature engineering, demand forecasting, waste analysis, model comparison, and deployment through a Streamlit dashboard.

## Project Summary

This project answers a practical operations question:

How much of each dish should a restaurant prepare for the next day to reduce waste without missing demand?

The workflow is built around four stages:

1. Clean and wrangle restaurant sales data in Python
2. Engineer time-based, holiday, and weather-related features
3. Compare forecasting models and select the most practical one
4. Deploy the final analysis in an interactive Streamlit dashboard

## What This Project Demonstrates

- Data cleaning and transformation with Pandas
- Feature engineering for forecasting problems
- Regression model benchmarking with scikit-learn
- Practical model selection based on both performance and explainability
- Streamlit app development for interactive analytics delivery
- Waste tracking using cooked, sold, and wasted quantities

## Selected Model

The notebook compares three approaches:

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| Linear Regression | 10.20 | 13.06 | 9.36 |
| Random Forest | 10.29 | 13.60 | 9.22 |
| Naive Baseline | 15.69 | 19.32 | 14.69 |

Linear Regression remains the selected model for the dashboard because it gives strong performance while staying simpler to explain, maintain, and present in a business setting.

## Workflow

### 1. Data Wrangling and Preparation

The source dataset contains daily dish-level records such as:

- dish name
- category
- price level
- temperature
- holiday indicator
- cooked quantity
- sold quantity

The Python pipeline cleans the dataset, standardizes types, computes `wasted_qty`, and prepares the data for both modeling and dashboard use.

### 2. Feature Engineering and Modeling

The forecasting pipeline creates:

- `lag_1`
- `lag_7`
- `rolling_mean_7`
- `temperature_c`
- `is_holiday`

These features are used to benchmark:

- Naive Baseline
- Linear Regression
- Random Forest Regressor

### 3. Streamlit Dashboard Deployment

The final dashboard turns the analysis into an interactive application where users can:

- view next-day demand forecasts
- compare baseline and weather-aware forecasts
- enter new daily actuals
- monitor waste trends
- review benchmark metrics
- download the active forecast or runtime dataset

## Dashboard Features

- Forecast tab for next-day dish demand prediction
- Daily Update tab for entering cooked and sold quantities
- Data and Models tab for benchmark results and waste trends
- Weather-aware forecasting using Open-Meteo
- Local CSV persistence through a runtime dataset

## Project Structure

```text
app.py
mixx/
  __init__.py
  constants.py
  data.py
  modeling.py
  weather.py
mixx_synthetic_restaurant_data.csv
data/runtime/
requirements.txt
Dockerfile
MIXX_Final_Model.ipynb
```

## Python Modules

- `mixx/data.py`
  - dataset loading and saving
  - daily input table generation
  - waste summary calculation
- `mixx/modeling.py`
  - feature creation
  - model benchmarking
  - next-day forecasting
  - daily actual append-and-refresh workflow
- `mixx/weather.py`
  - tomorrow temperature lookup for weather-aware forecasting
- `app.py`
  - Streamlit user interface and interaction flow

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
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

## Streamlit Community Cloud

To deploy on Streamlit Community Cloud:

1. Push this repository to GitHub
2. Create a new app in Streamlit Community Cloud
3. Select the repository branch
4. Set the main file path to `app.py`

## Runtime Data Behavior

- The app reads the seed dataset from `mixx_synthetic_restaurant_data.csv`
- On first run it creates `data/runtime/restaurant_data.csv`
- New daily updates are written to the runtime CSV
- This keeps the original source dataset unchanged

## Notebook

The original analysis notebook is included as:

- `MIXX_Final_Model.ipynb`

It serves as the experiment log and model development reference behind the deployed dashboard.
