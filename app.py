import streamlit as st
import pandas as pd
import joblib

# Page config
st.image("https://cdn-icons-png.flaticon.com/512/1041/1041885.png", width=80)
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# Load model
model = joblib.load("model/model.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.image("https://cdn-icons-png.flaticon.com/512/1041/1041885.png", width=80)

st.markdown("<p style='text-align: center;'>Get an estimated laptop price based on specifications</p>", unsafe_allow_html=True)

st.divider()

# ----------- INPUT SECTION -----------

col1, col2 = st.columns(2)

with col1:
    Company = st.selectbox("Brand", ["Apple", "HP", "Dell", "Lenovo", "Asus", "Acer"])
    TypeName = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible"])
    Inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6)
    Ram = st.slider("RAM (GB)", 4, 64, 8)
    Memory = st.slider("Storage (GB)", 128, 2048, 512)

with col2:
    ScreenResolution = st.selectbox("Resolution", ["1920x1080", "1366x768", "2560x1600"])
    Cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD"])
    Gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
    OpSys = st.selectbox("Operating System", ["Windows", "macOS", "No OS"])
    Weight = st.slider("Weight (kg)", 1.0, 3.0, 1.5)

st.divider()

# ----------- PREDICTION -----------

if st.button("💰 Predict Price", use_container_width=True):

    data = pd.DataFrame([{
        "Company": Company,
        "TypeName": TypeName,
        "Inches": Inches,
        "ScreenResolution": ScreenResolution,
        "Ram": Ram,
        "Memory": Memory,
        "OpSys": OpSys,
        "Weight": Weight,
        "Cpu_brand": Cpu_brand,
        "Gpu_brand": Gpu_brand
    }])

    prediction = model.predict(data)

    price_euro = prediction[0]
    price_inr = price_euro * 90

    st.success(f"💰 Estimated Price: ₹ {int(price_inr)}")

    # Extra info
    st.info(f"Approx € {round(price_euro,2)}")
