# MIXX - Food Demand Prediction and Waste Analytics

## Live Dashboard

Use the deployed Streamlit app here:

https://mixx-dashboard.streamlit.app/

## Project Focus

- Python data wrangling and cleaning
- Model benchmarking and forecast evaluation
- Streamlit dashboard deployment

## Overview

This project predicts next-day buffet dish demand and tracks food waste using Python and Streamlit. The work is organized so that the reusable production logic lives in Python modules, while the notebook provides a readable step-by-step analysis companion.

## What We Did In This Project

1. Cleaned and standardized restaurant demand data
2. Explored dish demand, waste, temperature, and holiday patterns
3. Engineered forecasting features from historical sales and external signals
4. Benchmarked multiple regression models
5. Selected the final model based on both performance and interpretability
6. Deployed the workflow in a Streamlit dashboard

## Where The Analysis Lives Now

- [MIXX_Final_Model.ipynb](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/MIXX_Final_Model.ipynb:1)
  Clean analysis notebook for exploratory analysis, feature engineering review, model comparison, and project walkthrough.
- [mixx/data.py](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/mixx/data.py:1)
  Handles data loading, cleaning, runtime CSV persistence, and waste summaries.
- [mixx/modeling.py](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/mixx/modeling.py:1)
  Contains the forecasting pipeline, feature creation, benchmarking logic, and prediction workflow.
- [mixx/project_analysis.py](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/mixx/project_analysis.py:1)
  Contains the step-by-step analysis process, exploratory findings, feature rationale, experiment trail, and final model-selection narrative.
- [project_analysis.py](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/project_analysis.py:1)
  Prints a full markdown analysis report generated directly from the Python modules.
- [app.py](C:/Users/jahid/OneDrive/Documents/GitHub/Food-Demand-Prediction-and-Waste-Analytics-/app.py:1)
  Contains the Streamlit dashboard only.

## Feature Engineering

The final forecasting workflow uses:

- `lag_1` to capture the most recent demand signal
- `lag_7` to capture weekly seasonality
- `rolling_mean_7` to smooth short-term noise
- `temperature_c` to capture weather-driven demand shifts
- `is_holiday` to represent special-day traffic changes

`category` and `price_level` remain part of the dataset for data entry and dish context, but the final regression uses dish-level time-series features because they are more useful for next-day forecasting.

## Model Experimentation

The benchmark compares:

- Naive Baseline
- Linear Regression
- Random Forest

The project keeps the benchmark logic in Python so the same comparison can be reproduced consistently in both the notebook and the Streamlit app.

## Final Model Choice

Linear Regression is the final deployment model because it offers strong benchmark performance while staying easier to explain, maintain, and deploy in a recruiter-friendly Streamlit application.

## Project Structure

```text
app.py
MIXX_Final_Model.ipynb
project_analysis.py
mixx/
  __init__.py
  constants.py
  data.py
  modeling.py
  project_analysis.py
  weather.py
mixx_synthetic_restaurant_data.csv
data/runtime/
requirements.txt
Dockerfile
```

## Run The Dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Run The Python Analysis Report

To print the full step-by-step analysis narrative from Python code:

```bash
python project_analysis.py
```

## Dataset Behavior

- The source dataset is `mixx_synthetic_restaurant_data.csv`
- The app creates `data/runtime/restaurant_data.csv` as the writable runtime copy
- Daily updates are written to the runtime CSV so the source file stays unchanged
