import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load model & scaler
model = load_model("tesla_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("📈 Tesla Stock Price Prediction (LSTM)")
st.write("Predict 1, 5, or 10 days Tesla stock closing price")

# Upload CSV
uploaded_file = st.file_uploader("Upload Tesla Stock CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    data = df[['Close']]
    scaled_data = scaler.transform(data)

    time_steps = 60
    last_sequence = scaled_data[-time_steps:]

    days = st.selectbox("Select prediction days", [1, 5, 10])

    if st.button("Predict"):
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days):
            pred = model.predict(current_sequence.reshape(1, time_steps, 1))
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred[0, 0])

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1,1)
        )

        st.subheader("🔮 Predicted Prices")
        st.write(predictions.flatten())

        # Plot
        st.subheader("📊 Forecast Visualization")
        plt.figure(figsize=(8,4))
        plt.plot(data.tail(100).values, label="Historical Price")
        plt.plot(range(100, 100+days), predictions, marker='o', label="Prediction")
        plt.legend()
        st.pyplot(plt)



