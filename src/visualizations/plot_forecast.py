import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_forecast(dates, actual, predicted, ticker):
    df_plot = pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predicted
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Actual'],
                             mode='lines+markers', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Predicted'],
                             mode='lines+markers', name='Predicted', line=dict(color='orange', dash='dot')))
    fig.update_layout(
        title=f"{ticker} Stock Forecast - Actual vs Predicted",
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white'
    )
    fig.show()
