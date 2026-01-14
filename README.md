# 📈 Tesla Stock Price Prediction Using LSTM-RNN

This project focuses on predicting **Tesla (TSLA) stock prices** using a **Long Short-Term Memory (LSTM) Recurrent Neural Network**, a deep learning model well-suited for **time-series forecasting**.  
The trained model is deployed as an **interactive Streamlit web application** for real-time prediction and visualization.

---

## 🚀 Project Highlights

- Time-series forecasting using **LSTM-RNN**
- Historical Tesla stock price analysis
- Data normalization using **MinMaxScaler**
- Model evaluation with actual vs predicted prices
- Interactive **Streamlit web app**
- Pre-trained model and scaler included

---

## 🧠 Technologies Used

- Python  
- TensorFlow / Keras  
- Pandas & NumPy  
- Matplotlib  
- Scikit-learn  
- Streamlit  

---

## 📊 Dataset

- **Source:** Historical Tesla (TSLA) stock price data  
- **File:** `TSLA.csv`  
- **Features used:** Open, High, Low, Close prices  
- Data is preprocessed and scaled before training the LSTM model.

---

## 🏗️ Project Structure

```text
Tesla-Stock-Price-Prediction-Using-LSTM-RNN/
│
├── TSLA.csv
├── Tesla_Stock_Price_Prediction.ipynb
├── tesla_lstm_model.h5
├── scaler.pkl
├── streamlit_app.py
├── app1.py
├── requirements.txt
└── README.md
```
## 📉 Model Overview

- Model Type: LSTM (Recurrent Neural Network)
- Purpose: Learn temporal patterns in Tesla stock prices
- Output: Predict future stock price trends based on past data
- The trained model is saved as tesla_lstm_model.h5

## 🖥️ Web Application
The Streamlit app allows users to:
- Visualize historical Tesla stock prices
- View actual vs predicted values
- Interact with the prediction model through a simple UI

## 🔗 Live App:
https://tesla-stock-price-prediction-using-lstm-rnn-gnbusjjppgnfjjfcfv.streamlit.app/

## 📌 Results
The model captures trends in Tesla stock prices and provides reasonable future price predictions based on historical data. 
This project demonstrates the application of deep learning in financial time-series forecasting.

## 🔮 Future Enhancements
- Add technical indicators (RSI, MACD, Moving Averages)
- Improve accuracy with GRU or Bidirectional LSTM
- Enable multi-stock prediction
- Add real-time stock data using APIs

## 👤 Author

Jayvish
GitHub: https://github.com/Jayvish80

