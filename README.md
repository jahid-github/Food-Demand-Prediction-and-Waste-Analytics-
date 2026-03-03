
# MIXX — Food Demand Prediction & Waste Analytics

> Machine Learning–Based Food Demand Prediction and Waste Analytics for Buffet Restaurants

> A Regression & Cloud-Based Decision Support System

🌐 **Live Dashboard:** [https://mixx.meherix.com/](https://mixx.meherix.com/)
📄 **Project Report (PDF):** [Download Full Thesis Report](Food%20Demand%20Prediction%20and%20Waste%20Analytics.pdf) 


## Project Overview

Food waste in buffet restaurants is a major operational and environmental challenge.

Overproduction → Waste & Cost
Underproduction → Customer dissatisfaction

This project develops **MIXX**, a machine learning–based decision support system that:

* Predicts next-day dish demand
* Calculates food waste quantity & percentage
* Compares regression models
* Visualizes results in a live cloud dashboard

The focus of the project is **understanding machine learning concepts**, not just writing Python code.

---

## Problem Statement

How can buffet restaurants:

1. Accurately predict next-day food demand?
2. Reduce overproduction and food waste?
3. Use data-driven decision-making instead of intuition?

The problem is formulated as a **supervised regression task**.

---

## Machine Learning Approach

### Target Variable

`y = next-day demand`

### Input Features (X)

* Temperature
* Holiday indicator (0/1)
* Historical demand (lag feature)
* Dish encoding

---

## Model Comparison

| Model             | MAE   | RMSE  | MAPE (%) |
| ----------------- | ----- | ----- | -------- |
| Linear Regression | 10.20 | 13.06 | 9.36     |
| Random Forest     | 10.29 | 13.60 | 9.22     |
| Naive Baseline    | 15.69 | 19.32 | 14.69    |

### Final Model: Linear Regression

Although Random Forest slightly improved accuracy, Linear Regression was selected because:

* High interpretability
* Lower computational complexity
* Easier explanation to stakeholders
* Better alignment with learning objectives

This demonstrates understanding of **model trade-offs**, not just performance metrics.

---

## Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **MAPE (Mean Absolute Percentage Error)**

This ensures proper regression model evaluation.

---

## Food Waste Analytics

Waste is calculated using:

```
waste_qty = max(cooked_qty − sold_qty, 0)
waste_pct = (waste_qty / cooked_qty) × 100
```

This enables:

* Identification of overproduced dishes
* Monitoring of waste trends
* Sustainability tracking
* Operational optimization

---

## System Architecture

```
Google Colab → BigQuery → Looker Studio
```

### Google Colab

* Data preprocessing
* Feature engineering
* Model training
* Prediction generation
* Waste calculation

### BigQuery

* Cloud data warehouse
* Scenario tracking
* Historical storage

### Looker Studio

* Interactive dashboard
* Scenario filtering
* Waste + prediction comparison
* Real-time visualization

---

## Demonstration Scenarios

### Demo 1 – Baseline

* Historical data only
* Pure ML prediction

### Demo 2 – Daily Input

* Staff inputs cooked & sold quantities
* Waste calculated automatically
* Model updated

### Demo 3 – New Dish

* New dish without history
* Fallback strategy applied
* Demonstrates robustness

---

## Live Dashboard

👉 View the working system here:

**[https://mixx.meherix.com/](https://mixx.meherix.com/)**

The dashboard includes:

* Scenario selector
* Prediction vs Waste comparison
* Dish-level analytics
* Real-time BigQuery data updates


## Future Roadmap

This project can evolve into a full production-ready MIXX platform:

### 1️⃣ Full Web Software

* Dedicated MIXX SaaS dashboard
* Authentication & user roles
* Multi-restaurant support

### 2️⃣ POS Integration

* Direct data sync from restaurant systems
* Automated daily updates

### 3️⃣ Advanced ML Models

* XGBoost
* LSTM (time-series)
* AutoML pipelines

### 4️⃣ Real-Time Model Retraining

* Automated retraining pipeline
* Performance monitoring

### 5️⃣ Computer Vision Waste Detection

* Smart bin camera system
* Object detection models
* Automatic waste classification
* Real sustainability impact reporting

---

## Key Learning Outcomes

This project demonstrates understanding of:

* Supervised learning
* Regression modeling
* Feature engineering
* Model comparison
* Evaluation metrics
* Cloud architecture integration
* ML explainability
* Business-oriented AI


# Why This Project Stands Out

* Combines ML + Cloud + Sustainability
* Demonstrates model reasoning (not just coding)
* Includes model comparison
* Shows real-world architecture
* Live working dashboard
* Clear business impact

                                                      **THE END**
