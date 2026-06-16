#  Global Live AQI Forecaster

## Overview

Global Live AQI Forecaster is an AI-powered air quality forecasting platform that combines historical environmental data, live AQI feeds, weather information, and advanced machine learning models to predict future air quality levels.

The application automatically downloads historical weather and AQI data, trains predictive models, integrates real-time AQI measurements, and generates a 24-hour air quality forecast through an interactive Streamlit dashboard.

---

## Key Features

* Real-time AQI monitoring using live API data
* Historical weather and AQI data collection
* 24-hour AQI forecasting
* Multi-city air quality analysis
* Interactive Streamlit dashboard
* Advanced feature engineering
* Time-series forecasting
* Forecast visualization with Plotly
* Comparative machine learning model selection

---

## Supported Cities

* Delhi
* Dehradun
* Bangalore
* Srinagar
* London
* Chicago

---

## Predictive Models

### Machine Learning

* Random Forest Regressor
* XGBoost Regressor
* Stacking Ensemble

### Deep Learning

* Physics-Informed CNN-Attention Network
* LSTM-Based Forecasting
* Temporal Fusion Transformer (TFT-Lite)

---

## Data Sources

### Historical Data

* Open-Meteo Weather Archive API
* Open-Meteo Air Quality API

### Live Data

* World Air Quality Index (WAQI) API

---

## Workflow

1. Select a city and prediction model.
2. Download historical weather and AQI data.
3. Generate rolling statistical and time-series features.
4. Train the selected machine learning or deep learning model.
5. Fetch live AQI and weather conditions.
6. Integrate live observations into the forecasting pipeline.
7. Generate a 24-hour AQI forecast.
8. Visualize predictions through tables and interactive charts.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Matplotlib
* Scikit-Learn
* XGBoost
* TensorFlow / Keras
* Open-Meteo API
* WAQI API

---

## Running the Project

1. Download the project files.
2. Install dependencies:

pip install -r requirements.txt

3. Configure the WAQI API token.

4. Run:

streamlit run app.py

## Future Enhancements

* Additional city support
* Extended forecasting horizons
* Model performance comparison dashboard
* Cloud deployment
* Mobile-friendly interface
* Advanced transformer architectures

---

## Author

Devanshu Kumar

B.Tech Computer Science and Engineering

---

## Disclaimer

This project is intended for educational, research, and analytical purposes. Forecasts are generated using publicly available environmental data and machine learning models and should not be used for critical decision-making.
