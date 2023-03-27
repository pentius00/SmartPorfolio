import streamlit as st
import os
import numpy as np
import requests
import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame
import datetime as dt
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from pandas_datareader import data as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from finquant.portfolio import build_portfolio
from finquant.efficient_frontier import EfficientFrontier
from finquant.moving_average import compute_ma, ema
import finta as TA
import datetime
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data import CryptoDataStream
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import matplotlib.pyplot as plt
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest

# Set Alpaca API credentials
alpaca_api_key = os.getenv("Alpaca_Paper_Key_ID")
alpaca_secret_key = os.getenv("Alpaca_Paper_Secret_Key")
load_dotenv("key.env")
alpaca = REST(alpaca_api_key, alpaca_secret_key, "https://paper-api.alpaca.markets")

# Create Streamlit app

st.set_page_config(page_title='My App', layout='wide')
st.title('My App')

# Create empty list for stock tickers
names = []
new_name = ''

# Start a loop that will run until the user enters 'exit'.
while new_name != 'exit':
    # Ask the user for a stock ticker
    new_name = st.text_input("Please add stock ticker, or enter 'exit': ")

    # Add the new name to our list
    if new_name != 'exit':
        names.append(new_name)

year = int(st.text_input('Enter a year for start date: '))   #year for start date
month = int(st.text_input('Enter a month for start date: ')) #month for start date
day = int(st.text_input('Enter a day for start date: '))     #day for start date

# use datetime for start_date and end_date
start_date = datetime.datetime(year, month, day)
end_date = datetime.datetime.now()

# Use build_protfolio from finquant with yahoo finance as data api otherwise it will use quandl
pf = build_portfolio(names=names,start_date=start_date, end_date=end_date, data_api="yfinance")

    # Show cumulative returns plot
cumulative_returns = pf.calc_cumulative_return()
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(cumulative_returns.index, cumulative_returns.values)
plt.title("Portfolio Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
st.pyplot(fig)

# Calculate and plot efficient frontier
# Compute expected returns and sample covariance
mu = expected_returns.mean_historical_return(pf.data)
S = risk_models.sample_cov(pf.data)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
plotting.plot_efficient_frontier(ef, show_assets=True)

# Show portfolio weights
st.subheader("Portfolio Weights")
for key in cleaned_weights:
    st.write(f"{key}: {cleaned_weights[key]*100:.2f}%")
