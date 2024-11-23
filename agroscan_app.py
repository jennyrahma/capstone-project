import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model dan scaler
model = load_model('agroscan_model.h5')  # Ganti dengan path model Anda
scaler = MinMaxScaler()
scaler.fit(pd.read_csv('indonesia_environment_data.csv')[['Temperature', 'Humidity', 'Soil pH', 'Rainfall', 'Light Intensity']])

# Judul aplikasi
st.title("AgroScan - Health Prediction")

# Input data lingkungan
st.header("Input Environmental Data")
temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=40.0, value=30.0)
humidity = st.number_input("Humidity (%)", min_value=50.0, max_value=100.0, value=80.0)
soil_ph = st.number_input("Soil pH", min_value=4.0, max_value=8.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=200.0, value=100.0)
light_intensity = st.number_input("Light Intensity (lux)", min_value=1000.0, max_value=5000.0, value=3000.0)

# Upload citra daun
st.header("Upload Leaf Image")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# Prediksi jika semua input tersedia
if st.button("Predict"):
    if uploaded_file is not None:
        # Preprocess data lingkungan
        env_data = np.array([[temperature, humidity, soil_ph, rainfall, light_intensity]])
        env_data = scaler.transform(env_data)

        # Preprocess citra daun
        img = load_img(uploaded_file, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predicted_health = model.predict([env_data, img_array])
        st.success(f"Predicted Health Score: {predicted_health[0][0]:.2f}")
    else:
        st.error("Please upload a leaf image.")
