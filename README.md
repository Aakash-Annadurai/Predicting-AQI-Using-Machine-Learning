# Predicting Air Quality Index (AQI)

A modern, vibrant, and interactive web application to analyze historical air quality trends and predict the Air Quality Index (AQI) based on various pollutant levels. This project was developed by analyzing air quality data across various cities and training a Random Forest Machine Learning model for prediction.

## Overview

This repository contains:
- `Project.ipynb`: An exploratory data analysis (EDA) notebook showing trends and training baseline models (Linear Regression, Random Forest, Decision Tree, Logistic Regression, SVM).
- `city_day.csv`: The dataset containing daily pollutant metrics across various cities.
- `train_model.py`: A Python script that preprocesses the dataset and trains a Random Forest Regressor, saving the model, scaler, and encoder to the `models/` directory for faster app loading.
- `app.py`: A Streamlit web application that provides interactive Data Insights using Plotly and a sleek user interface to predict the AQI for custom inputs.
- `models/`: Contains the pre-trained `.joblib` files required for predictions.

## Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/) with custom CSS for a glassmorphism dark mode aesthetic.
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (Random Forest Regressor).
- **Data Manipulation**: Pandas, NumPy.
- **Visualizations**: Plotly Express.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Aakash-Annadurai/Predicting-AQI-Using-Machine-Learning.git
cd Predicting-AQI-Using-Machine-Learning
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Install the required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Generate the Models (If not already present)
To generate or update the `rf_model.joblib`, `scaler.joblib`, and `encoder.joblib` files, run the training script:
```bash
python train_model.py
```
This script reads `city_day.csv`, trains a `RandomForestRegressor`, and saves the required files in the `models/` directory.

### 4. Run the Streamlit Application
Launch the web application locally:
```bash
streamlit run app.py
```
The application will be accessible at `http://localhost:8501`.

## Features
*   **Data Insights**: Interactive dashboard to view AQI distribution, yearly trends, and a time-series plot comparing different cities.
*   **Predict AQI**: Input specific pollutant levels (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene) along with the City and Date to get a real-time prediction of the AQI, color-coded by severity category (Good to Severe).

## Disclaimer
As noted in the exploratory analysis (`Project.ipynb`), predicting AQI directly and solely from these pollutants using standard regression models carries a margin of error. The Random Forest model provides an estimation intended for demonstration and research purposes.
