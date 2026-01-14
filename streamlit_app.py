# ==============================
# Safe Config for Streamlit Cloud
# ==============================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==============================
# Imports
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Tesla Stock Prediction | LSTM",
    page_icon="📈",
    layout="wide"
)

# ==============================
# Load Model & Scaler
# ==============================
@st.cache_resource
def load_model_safe():
    return load_model("tesla_lstm_model.h5")

@st.cache_resource
def load_scaler_safe():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model_safe()
scaler = load_scaler_safe()

# ==============================
# HEADER (Hero Section)
# ==============================
st.markdown(
    """
    <h1 style='text-align: center;'>📈 Tesla Stock Price Prediction</h1>
    <p style='text-align: center; font-size:18px;'>
    AI-powered stock forecasting using <b>LSTM Neural Networks</b>
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==============================
# Upload Dataset
# ==============================
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload Tesla stock CSV file (must contain Date & Close columns)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload a dataset to begin analysis and prediction.")
    st.stop()

# ==============================
# Load & Fix Dataset
# ==============================
df = pd.read_csv(uploaded_file)

if "Date" not in df.columns:
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)

# ==============================
# KPI METRICS (Top Company Style)
# ==============================
latest_price = df["Close"].iloc[-1]
price_change = latest_price - df["Close"].iloc[-2]

col1, col2, col3 = st.columns(3)
col1.metric("📌 Latest Close Price", f"${latest_price:.2f}")
col2.metric("📈 Daily Change", f"${price_change:.2f}")
col3.metric("🧠 Model Type", "LSTM")

st.divider()

# ==============================
# Tabs Layout
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📉 Analysis", "🔮 Prediction", "ℹ️ About"]
)

# ==============================
# TAB 1: OVERVIEW
# ==============================
with tab1:
    st.subheader("Tesla Closing Price Trend")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], df["Close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# ==============================
# TAB 2: ANALYSIS
# ==============================
with tab2:
    st.subheader("Moving Average Analysis")

    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Close"], label="Close")
    ax.plot(df["MA50"], label="50-Day MA")
    ax.plot(df["MA200"], label="200-Day MA")
    ax.legend()
    st.pyplot(fig)

# ==============================
# TAB 3: PREDICTION
# ==============================
with tab3:
    st.subheader("Future Stock Price Prediction")

    days = st.selectbox(
        "Select prediction horizon",
        [1, 5, 10]
    )

    if st.button("🚀 Predict Price"):
        close_prices = df[["Close"]].values
        scaled_data = scaler.transform(close_prices)

        time_steps = 60
        sequence = scaled_data[-time_steps:]

        predictions = []

        for _ in range(days):
            pred = model.predict(sequence.reshape(1, time_steps, 1), verbose=0)
            predictions.append(pred[0][0])
            sequence = np.append(sequence[1:], pred[0][0])

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        st.success("Prediction Completed")

        for i, value in enumerate(predictions.flatten(), 1):
            st.write(f"📅 Day {i}: **${value:.2f}**")

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df["Close"].tail(100).values, label="Historical")
        ax.plot(
            range(100, 100 + days),
            predictions.flatten(),
            marker="o",
            label="Forecast"
        )
        ax.legend()
        st.pyplot(fig)

# ==============================
# TAB 4: ABOUT
# ==============================
with tab4:
    st.markdown(
        """
        ### 📌 Project Overview
        - Predicts Tesla stock prices using **LSTM (RNN)**
        - Trained on historical closing prices
        - Scaled using MinMaxScaler
        - Deployed on **Streamlit Cloud**

        ### 🧠 Why LSTM?
        LSTM models capture long-term dependencies in time-series data,
        making them ideal for stock price forecasting.

        ### 👨‍💻 Built for:
        - Internship projects
        - Resume & portfolio
        - Interview demonstrations
        """
    )

st.divider()
st.caption("🚀 Built with Streamlit & TensorFlow | Tesla Stock Prediction")
