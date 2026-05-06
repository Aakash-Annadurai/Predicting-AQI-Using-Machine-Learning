import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Design ---
st.markdown("""
    <style>
    /* Dark Theme & Glassmorphism */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Inter', sans-serif;
    }
    .css-1d391kg, .css-1v3fvcr { 
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stMetric, .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #0f2027;
        font-weight: bold;
        border: none;
        border-radius: 30px;
        padding: 10px 24px;
        transition: transform 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        color: #0f2027;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.6);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #E0E0E0;
        text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
    }
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0));
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Models and Data ---
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        encoder = joblib.load('models/encoder.joblib')
        return rf_model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'train_model.py' has been run.")
        return None, None, None

@st.cache_data
def load_data():
    if os.path.exists("city_day.csv"):
        df = pd.read_csv("city_day.csv")
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Year'] = df['Datetime'].dt.year
        return df.dropna()
    return None

rf_model, scaler, encoder = load_models()
data = load_data()

# --- Utility Functions ---
def get_aqi_bucket(aqi):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Satisfactory", "#ffff00"
    elif aqi <= 200:
        return "Moderate", "#ff7e00"
    elif aqi <= 300:
        return "Poor", "#ff0000"
    elif aqi <= 400:
        return "Very Poor", "#8f3f97"
    else:
        return "Severe", "#7e0023"

# --- Sidebar Navigation ---
st.sidebar.title("🌍 AQI Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Data Insights", "🎯 Predict AQI"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**About**\n\n"
    "This application uses a Machine Learning model (Random Forest) to predict the Air Quality Index (AQI) based on various pollutant levels."
)

# --- Page: Data Insights ---
if page == "📊 Data Insights":
    st.title("📊 Air Quality Data Insights")
    st.markdown("Explore historical air quality trends across various cities.")
    
    if data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("AQI Distribution")
            fig1 = px.histogram(data, x="AQI", nbins=50, 
                                color_discrete_sequence=['#00C9FF'],
                                template="plotly_dark")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.subheader("Yearly Average AQI")
            yearly_aqi = data.groupby('Year')['AQI'].mean().reset_index()
            fig2 = px.line(yearly_aqi, x='Year', y='AQI', markers=True,
                           line_shape='spline', color_discrete_sequence=['#92FE9D'],
                           template="plotly_dark")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        st.subheader("City-wise AQI Trends")
        selected_city = st.selectbox("Select a City", sorted(data['City'].unique()))
        city_data = data[data['City'] == selected_city]
        
        if not city_data.empty:
            fig3 = px.scatter(city_data, x="Datetime", y="AQI", 
                              color="AQI_Bucket",
                              color_discrete_map={
                                  "Good": "#00e400", "Satisfactory": "#ffff00", 
                                  "Moderate": "#ff7e00", "Poor": "#ff0000", 
                                  "Very Poor": "#8f3f97", "Severe": "#7e0023"
                              },
                              template="plotly_dark")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Dataset not found. Please ensure 'city_day.csv' is available.")

# --- Page: Predict AQI ---
elif page == "🎯 Predict AQI":
    st.title("🎯 Predict Air Quality Index")
    st.markdown("Enter the pollutant values below to estimate the AQI.")
    
    if rf_model is None or scaler is None or encoder is None:
        st.warning("Model files not found. Please wait while models are being trained, or run `train_model.py` first.")
    else:
        with st.form("prediction_form"):
            st.markdown("### 📍 Location & Date")
            col1, col2 = st.columns(2)
            with col1:
                city = st.selectbox("City", encoder.classes_)
                date = st.date_input("Date")
            
            st.markdown("### 🌫️ Pollutant Levels (µg/m³)")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
                no = st.number_input("NO", min_value=0.0, value=20.0)
                co = st.number_input("CO", min_value=0.0, value=1.0)
            with c2:
                pm10 = st.number_input("PM10", min_value=0.0, value=100.0)
                no2 = st.number_input("NO2", min_value=0.0, value=30.0)
                so2 = st.number_input("SO2", min_value=0.0, value=15.0)
            with c3:
                nox = st.number_input("NOx", min_value=0.0, value=40.0)
                o3 = st.number_input("O3", min_value=0.0, value=30.0)
                xylene = st.number_input("Xylene", min_value=0.0, value=2.0)
            with c4:
                nh3 = st.number_input("NH3", min_value=0.0, value=20.0)
                benzene = st.number_input("Benzene", min_value=0.0, value=2.0)
                toluene = st.number_input("Toluene", min_value=0.0, value=5.0)
                
            submit_btn = st.form_submit_button("Predict AQI 🚀")
            
        if submit_btn:
            try:
                # Prepare input
                city_encoded = encoder.transform([city])[0]
                year = date.year
                month = date.month
                day = date.day
                
                input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, year, month, day, city_encoded]],
                                          columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month', 'Day', 'City_encoded'])
                
                # Scale and predict
                input_scaled = scaler.transform(input_data)
                prediction = rf_model.predict(input_scaled)[0]
                
                # Bucket and color
                bucket, color = get_aqi_bucket(prediction)
                
                # Display Results
                st.markdown("---")
                st.markdown("## Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {color};">
                            <h2 style="margin: 0; color: #E0E0E0;">Predicted AQI</h2>
                            <h1 style="margin: 0; font-size: 4rem; color: {color};">{int(prediction)}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {color};">
                            <h2 style="margin: 0; color: #E0E0E0;">Air Quality Category</h2>
                            <h1 style="margin: 0; font-size: 3rem; color: {color};">{bucket}</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.info(f"**Disclaimer:** This prediction is based on a Random Forest model. As indicated in the exploratory analysis, predicting AQI purely from these pollutants presents high variance, so consider this an estimation rather than a definitive reading.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
