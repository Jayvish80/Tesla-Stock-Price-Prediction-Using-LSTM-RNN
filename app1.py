# ==============================
# Streamlit + TensorFlow Safe Config
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
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ==============================
# Load Model & Scaler
# ==============================
@st.cache_resource
def load_lstm_model():
    return load_model("tesla_lstm_model.h5")

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
scaler = load_scaler()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Tesla Stock Price Prediction", layout="centered")

st.title("📈 Tesla Stock Price Prediction")
st.write("Predict Tesla stock **1, 5, or 10 days ahead** using an LSTM model.")

# ==============================
# Upload Dataset
# ==============================
uploaded_file = st.file_uploader("Upload Tesla Stock CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data preprocessing
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    close_prices = df[["Close"]].values
    scaled_data = scaler.transform(close_prices)

    # Last 60 days
    time_steps = 60
    last_sequence = scaled_data[-time_steps:]

    # ==============================
    # Prediction Selection
    # ==============================
    days = st.selectbox("Select number of days to predict", [1, 5, 10])

    if st.button("Predict"):
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days):
            pred = model.predict(current_sequence.reshape(1, time_steps, 1), verbose=0)
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred[0, 0])

        # Inverse scaling
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        # ==============================
        # Display Results
        # ==============================
        st.subheader("🔮 Predicted Closing Prices")
        for i, price in enumerate(predictions.flatten(), 1):
            st.write(f"Day {i}: **${price:.2f}**")

        # ==============================
        # Visualization
        # ==============================
        st.subheader("📊 Forecast Visualization")
        plt.figure(figsize=(8, 4))
        plt.plot(df["Close"].tail(100).values, label="Historical Price")
        plt.plot(
            range(100, 100 + days),
            predictions.flatten(),
            marker="o",
            label="Predicted Price",
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Price")
        st.pyplot(plt)

else:
    st.info("👆 Upload a Tesla stock CSV file to start prediction.")
