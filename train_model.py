import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv("city_day.csv")

    # Drop any potential NaN values just in case
    df = df.dropna()

    # Preprocessing
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day

    print("Encoding features...")
    encoder = LabelEncoder()
    df['City_encoded'] = encoder.fit_transform(df['City'])

    # Prepare features and target
    target = "AQI"
    
    # Selecting the exact features as in the notebook
    feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                    'Benzene', 'Toluene', 'Xylene', 'Year', 'Month', 'Day', 'City_encoded']

    X = df[feature_cols]
    y = df[target]

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=45)

    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    print("\nRandom Forest Performance:")
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_rf))
    print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    print("R2 Score:", r2_score(y_test, y_pred_rf))

    # Save models
    print("\nSaving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(encoder, 'models/encoder.joblib')
    print("Models saved successfully in 'models/' directory.")

if __name__ == "__main__":
    main()
