import streamlit as st
from src.forecast.xgboost_forecast import run_forecast
from src.visualizations.plot_forecast import plot_forecast

st.title("Stock Price Forecast + RL Trading")

ticker = st.selectbox("Select ticker", ["AAPL", "AMZN", "META", "GOOG", "NFLX"])

if st.button("Run Forecast"):
    df_forecast = run_forecast(ticker)
    st.write(f"Next day forecast: {df_forecast['future_pred']:.2f}")
    plot_forecast(df_forecast['dates'], df_forecast['actual'], df_forecast['predicted'], ticker)
