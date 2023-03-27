
import os
import numpy as np
import requests
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
import datetime as dt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pandas_datareader import data as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
from finquant.monte_carlo import MonteCarloOpt
from finquant.portfolio import build_portfolio
from finquant.efficient_frontier import EfficientFrontier as EfficientFrontier2
from finquant.moving_average import compute_ma, ema
from plotly.subplots import make_subplots
import plotly.tools as tls
import streamlit as st
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
from dotenv import load_dotenv
import os
load_dotenv("key.env")
######################### Set up Alpaca credentials
alpaca_api_key = os.getenv("Alpaca_Paper_Key_ID")
alpaca_secret_key = os.getenv("Alpaca_Paper_Secret_Key")

alpaca = REST(alpaca_api_key, alpaca_secret_key, "https://paper-api.alpaca.markets")
trading_client = TradingClient(alpaca_api_key, alpaca_secret_key)
account = trading_client.get_account()

######################################## Define Streamlit app

# Add a title and sidebar input fields for stock tickers and start date
st.title('Portfolio Optimization')
names = st.sidebar.text_input("Stock Tickers (separated by comma)", value='AAPL,MSFT')
start_date = st.sidebar.date_input("Start Date", value=dt.datetime(2020, 1, 1))

# Split names into a list
names = [name.strip() for name in names.split(',')]

# Convert start date to datetime
end_date = dt.datetime.now()

# Use build_protfolio from finquant with yahoo finance as data api otherwise it will use quandl
pf = build_portfolio(names=names,start_date=start_date, end_date=end_date, data_api="yfinance")

##################################################### Plot cumulative returns
st.subheader('Plot cumulative returns')
fig = px.line(pf.comp_cumulative_returns(), x=pf.comp_cumulative_returns().index, y=pf.comp_cumulative_returns().columns)
fig.add_hline(y=0, line_dash="dot", line_color="black")
st.plotly_chart(fig)

######################################perform Monte Carlo optimization and plot the results
if st.button("Monte Carlo Optimization"):
    st.title('Monte Carlo Optimization')
    num_trials = st.slider("Enter the number of trials:", 100, 10000, 5000, step=100)
    opt_w, opt_res = pf.mc_optimisation(num_trials=num_trials)
    st.write("Optimized Weights:", opt_w)
    st.write("Optimization Results:", opt_res)
    fig2 = go.Figure()
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Weights Distribution", "Portfolio Returns"))
    fig2.add_trace(go.Histogram(x=opt_w, nbinsx=25, name="Weights"), row=1, col=1)
    fig2.add_trace(go.Histogram(x=opt_res, nbinsx=25, name="Returns"), row=1, col=2)
    fig2.update_layout(title="Monte Carlo Optimization", xaxis_title="Value", yaxis_title="Frequency")
    st.plotly_chart(fig2)
    fig2 = pf.mc_plot_results()
  
    
    
############################################
# Efficient Frontier optimization and plots

if st.button("Efficient Frontier"):
    st.title('Efficient Frontier Optimization')
    #ef = EfficientFrontier(pf.comp_mean_returns(), pf.comp_cov())
    #ig, ax = plt.subplots(figsize=(8,6))
    #opy = ef
    #f_max_sharpe = ef
    #f_max_sharpe = ef.max_sharpe(risk_free_rate=0.02)

    #lotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    # Find the tangency portfolio

    #ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    #ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    # rets = pf.dot(ef.expected_returns)
    # stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    # sharpes = rets / stds
    # ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output

    #ax.set_title("Efficient Frontier with random portfolios")
    #ax.legend()
    #plt.tight_layout()
    #plt.show()


    #st.plotly_chart(fig)

############################################################   
    
    
if st.button("Optimal Portfolios"):
    ef = EfficientFrontier(pf.comp_mean_returns(), pf.comp_cov())
    ef.plot_optimal_portfolios()
        # Maximum Sharpe Ratio optimization
    st.header('Maximum Sharpe Ratio Optimization')
    max_sharpe = ef.maximum_sharpe_ratio()
    st.write("Maximum Sharpe Ratio Portfolio Weights:")
    st.write(max_sharpe)

    st.pyplot()

#########################################    
    
if st.button("Stock Plots"):
    fig = pf.plot_stocks()
    st.pyplot()
##########################################
if st.button("Optimized Portfolio Results"):
    st.header('Minimum Volatility Optimization')
    ef = EfficientFrontier2(pf.comp_mean_returns(freq=1), pf.comp_cov())
    st.write("Minimum Volatility Portfolio:", ef.minimum_volatility())
    (expected_return, volatility, sharpe) = ef.properties(verbose=True)
    st.write("Expected Return:", expected_return)
    st.write("Volatility:", volatility)
    st.write("Sharpe Ratio:", sharpe)




########################################
# Find the portfolio with minimum volatility for a target return of 0.25
ef2 = EfficientFrontier(pf.comp_mean_returns(), pf.comp_cov())
target_return = 0.25
raw_weights = ef2.max_sharpe()
cleaned_weights = ef2.clean_weights()
weights = ef.efficient_return(target_return,save_weights=True)
ef2.portfolio_performance(verbose=True)
# Plot efficient frontier

############################################





# Place market order with Alpaca
st.title('Place Market Order')
st.write("Place a market order for your portfolio using Alpaca")
trading_client = TradingClient(alpaca)
order_request = MarketOrderRequest(
    symbols=cleaned_weights.keys(),
    quantities=list(cleaned_weights.values()),
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,
)
response = trading_client.place_order(order_request)
st.write("Order response:", response)
