import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("CarsPrice.csv")

df = load_data()

# Load Target Encoding Pickle File
@st.cache_resource
def load_encodings():
    with open("car_encoder.pkl", "rb") as f:
        return pickle.load(f)

encoding = load_encodings()

# Load Scaler
@st.cache_resource
def load_scaler():
    with open("car_scaler.pkl", "rb") as f:
        return pickle.load(f)

scaler = load_scaler()

# Load ML Model
@st.cache_resource
def load_model():
    with open("cars_price.pkl", "rb") as f:
        return joblib.load(f)

lin_model = load_model()

# 🎨 Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🔍 Predict the price of your car instantly!</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for inputs
st.markdown("### 📋 Enter Car Details")

# Numeric Inputs
year = st.number_input("📅 Manufacture Year", min_value=1990, max_value=2021)
km_driven = st.number_input("🛣️ Distance Covered (KM)")
mileage = st.number_input("⛽ Mileage (KM/L)", min_value=3, max_value=20)
engine = st.number_input("⚙️ Engine (CC)", min_value=0, max_value=10000)
power = st.number_input("🐎 Horsepower (HP)", min_value=0, max_value=1000)
age = st.number_input("🕰️ Car's Age (Years)", min_value=0, max_value=20)

# Categorical Inputs
make = st.selectbox("🏢 Company Name", df["make"].unique())
encoded_make = encoding["make"].get(make, np.nan)

model = st.selectbox("🚘 Car Model", sorted(df["model"].unique()))
encoded_model = encoding["model"].get(model, np.nan)

Individual = st.selectbox("👤 Individual", ['Yes', 'No'])
Individual = 1 if Individual == "Yes" else 0

TrustmarkDealer = st.selectbox("✅ Trustmark Dealer?", ['Yes', 'No'])
TrustmarkDealer = 1 if TrustmarkDealer == "Yes" else 0

fuel = st.selectbox("⛽ Fuel Type", ['Diesel', 'Electric', 'LPG', 'Petrol'])
Diesel = 1 if fuel == "Diesel" else 0
Electric = 1 if fuel == "Electric" else 0
LPG = 1 if fuel == "LPG" else 0
Petrol = 1 if fuel == "Petrol" else 0

Manual = st.selectbox("🔄 Transmission", ['Manual', 'Automatic'])
Manual = 1 if Manual == 'Manual' else 0

gear = st.selectbox("⚙️ Gears", ['5', '>5'])
five = 1 if gear == "5" else 0
greater_five = 1 if gear == ">5" else 0

st.markdown("---")
st.markdown("### 🔎 Your Selected Car Details:")

# Display user inputs in two columns
col1, col2 = st.columns(2)

with col1:
    st.write(f"📅 **Year:** {year}")
    st.write(f"🛣️ **KM Driven:** {km_driven}")
    st.write(f"⛽ **Mileage:** {mileage} KM/L")
    st.write(f"⚙️ **Engine:** {engine} CC")
    st.write(f"🐎 **Horsepower:** {power} HP")

with col2:
    st.write(f"🏢 **Company:** {make}")
    st.write(f"🚘 **Model:** {model}")
    st.write(f"👤 **First Owner:** {'Yes' if Individual else 'No'}")
    st.write(f"✅ **Trustmark Dealer:** {'Yes' if TrustmarkDealer else 'No'}")
    st.write(f"🔄 **Transmission:** {'Manual' if Manual else 'Automatic'}")

st.markdown("---")

# Create DataFrame for Model Input
input_data = pd.DataFrame([[
    year, km_driven, mileage, engine, power, age, encoded_make, encoded_model,
    Individual, TrustmarkDealer, Diesel, Electric, LPG, Petrol, Manual, five, greater_five]])

# Apply Scaling
input_data_scaled = scaler.transform(input_data)

# Predict Price Button
if st.button("🎯 Predict Car Price"):
    prediction = lin_model.predict(input_data_scaled)[0]

    st.markdown(f"<h2 style='text-align: center; color: #28a745;'>💰 Estimated Car Price: ₹{round(prediction, 2)} lakhs</h2>", unsafe_allow_html=True)