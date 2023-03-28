import os
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
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
from alpaca.trading.client import TradingClient
import matplotlib.pyplot as plt
import yfinance as yf
from pypfopt import plotting
from alpaca_trade_api.rest import REST
################### ###### Set up Alpaca credentials
alpaca_api_key = os.getenv("Alpaca_Paper_Key_ID")
alpaca_secret_key = os.getenv("Alpaca_Paper_Secret_Key")

alpaca = REST(alpaca_api_key, alpaca_secret_key, "https://paper-api.alpaca.markets")
orders = alpaca.list_orders(status='open')
trading_client = TradingClient(alpaca_api_key, alpaca_secret_key)
alpaca_positions = alpaca.list_positions()
account = trading_client.get_account()

######################################## Define Streamlit app

# Add a title and sidebar input fields for stock tickers and start date
st.title('Portfolio Optimization')
names = st.sidebar.text_input("Stock Tickers (separated by comma)", value='AAPL,MSFT')

start_date = st.sidebar.date_input("Start Date", value=dt.datetime(2020, 1, 1))

# Split names into a list
names = [name.strip() for name in names.split(',')]
portfolio_symbols = names.copy()
# Convert start date to datetime
end_date = dt.datetime.now()

# Use build_protfolio from finquant with yahoo finance
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
    ef = EfficientFrontier2(pf.comp_mean_returns(), pf.comp_cov())
    fig, ax = plt.subplots(figsize=(8,6))
    ef_max_sharpe = ef.maximum_sharpe_ratio(ef, save_weights=True)

    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    #Find the tangency portfolio

    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    rets = pf.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output

    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.show()


    #st.plotly_chart(fig)

############################################################   
    
    
if st.button("Optimal Portfolios"):
    ef = EfficientFrontier2(pf.comp_mean_returns(), pf.comp_cov())
    ef.plot_optimal_portfolios()
        # Maximum Sharpe Ratio optimization
    st.header('Maximum Sharpe Ratio Optimization')
    max_sharpe = ef.maximum_sharpe_ratio()
    st.write("Maximum Sharpe Ratio Portfolio Weights:")
    st.write(max_sharpe)

    st.pyplot()

#########################################    
    
if st.button("Stock Plots"):
    fig5 = pf.plot_stocks()
    st.pyplot(fig5)
##########################################
if st.button("Optimized Portfolio Results"):
    st.header('Minimum Volatility Optimization')
    ef = EfficientFrontier2(pf.comp_mean_returns(freq=1), pf.comp_cov())
    st.write("Minimum Volatility Portfolio:", ef.minimum_volatility())
    (expected_return, volatility, sharpe) = ef.properties(verbose=True)
    st.write("Expected Return:", expected_return)
    st.write("Volatility:", volatility)
    st.write("Sharpe Ratio:", sharpe)




 #############################################       


########################################
if st.button("Place Market Order"):
    st.title('Market Order Placement')
    st.write("Place a market order for your portfolio using Alpaca")
    # Check if all portfolio symbols are eligible for fractional trading
    
    all_symbols_eligible_for_fractionals = all([[alpaca.get_asset(symbol).fractionable for symbol in portfolio_symbols]])
    if all_symbols_eligible_for_fractionals:
        
        # Get open orders and positions
        orders = alpaca.list_orders(status='open')
        positions = alpaca.list_positions()
        # Determine which positions to close
        symbols_to_close = [position.symbol for position in positions if position.symbol not in symbols]
        for symbol in symbols_to_close:
            alpaca.close_position(symbol)
            st.write(f"Position for {symbol} has been closed.")
        # Determine target allocations and rebalance equity
        equity = alpaca.get_account().equity
        if isinstance(equity, list):
            rebalance_equity = float(equity[0]) * 0.95  # 5% cash reserve
        else:
            rebalance_equity = float(equity) * 0.95  # 5% cash reserv
        target_allocations = optimize_portfolio(pf, rebalance_equity)
        # Determine symbols to sell and buy
        latest_allocations = {position.symbol: float(position.market_value) for position in positions}
        symbols_to_sell, symbols_to_buy = alpaca_symbols_to_sell_and_buy(target_allocations, latest_allocations)
        # Place sell orders
        for symbol, amount in symbols_to_sell.items():
            if amount > 1:
                alpaca.submit_order(
                    symbol=symbol,
                    qty=int(amount/pf[symbol].latest_price),
                    side="sell",
                    type="market",
                    time_in_force="day"
                )
                st.write(f"Sell order for {symbol} has been placed.")
        # Place buy orders
        for symbol, amount in symbols_to_buy.items():
            if amount > 1:
                alpaca.submit_order(
                    symbol=symbol,
                    notional=amount,
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
                st.write(f"Buy order for {symbol} has been placed.")
    else:
        st.write("All portfolio symbols must be eligible for fractional trading.")
#######################################################################################################
def _all_symbols_eligible_for_fractionals(symbols):
    """
    Returns True if all symbols are eligible for fractional trading on Alpaca.
    """
    eligible = True
    for symbol in symbols:
        asset = alpaca.get_asset(symbol)
        if not asset.tradable or not asset.fractionable:
            eligible = False
            print(f"{symbol} is not eligible for fractional trading on Alpaca.")
    return eligible
def _alpaca_latest_positions():
    """
    Returns the latest positions from Alpaca.
    """
    positions = {}
    for position in alpaca.list_positions():
        symbol = position.symbol
        qty = int(float(position.qty))
        positions[symbol] = qty
    return positions

def _portfolio_symbols_equity_allocations(target_allocations, total_equity):
    """
    Returns the equity allocation for each symbol based on the target allocation and total equity.
    """
    equity_allocations = {}
    for symbol, target_allocation in target_allocations.items():
        equity_allocations[symbol] = total_equity * target_allocation
    return equity_allocations

def _alpaca_symbols_to_sell_and_buy(portfolio_symbols_equity_allocations, latest_alpaca_positions_allocations):

    positions_to_sell, positions_to_buy = {}, {}

    # Loop Through Latest Desired Allocation
    for ticker, desired_allocation in portfolio_symbols_equity_allocations.items():

        current_allocation = latest_alpaca_positions_allocations.get(ticker, 0)
        allocation_to_adjust = (desired_allocation - current_allocation) * 100 / alpaca_account_equity
        if allocation_to_adjust > 0:
            positions_to_buy[ticker] = allocation_to_adjust
        else:
            positions_to_sell[ticker] = allocation_to_adjust * -1
    
    return positions_to_sell, positions_to_buy

def symbols_to_close(alpaca_positions, portfolio_symbols):
    alpaca_symbols_to_close = [x for x in alpaca_positions if x not in portfolio_symbols]
    return alpaca_symbols_to_close
alpaca_positions = _alpaca_latest_positions()
portfolio_symbols = names.copy()
alpaca_symbols_to_close = symbols_to_close(alpaca_positions, portfolio_symbols)

def _rebalance_equity(cash_weight):
    alpaca_account_equity = float(alpaca.get_account().equity)
    rebalance_equity = alpaca_account_equity - (alpaca_account_equity * cash_weight)
    return rebalance_equity


balance = alpaca.get_account().equity
print(balance)

rebalance_equity = _rebalance_equity(.05)
print(rebalance_equity)


def alpaca_close_positions(symbols_to_close):
    for symbol in symbols_to_close:
        # Check if position is open
        position = alpaca.get_position(symbol)
        if position.qty == "0":
            print(f"{symbol} position is already closed.")
            continue
        
        # Determine number of shares to close
        shares_to_close = int(position.qty)
        
        # Close position
        order = alpaca.submit_order(
            symbol=symbol,
            qty=shares_to_close,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        
        # Wait for order to fill
        while True:
            order_status = alpaca.get_order(order.id).status
            if order_status == "filled":
                print(f"{symbol} position closed.")
                break
            time.sleep(1) 
def target_allocation_maximum_sharpe(pf):
    # Compute expected returns and sample covariance

    # Optimize for maximum Sharpe ratio
    mu = expected_returns.mean_historical_returns(df)
    ef2 = EfficientFrontier(, pf.comp_cov())
    ef_max_sharpe = ef2.maximum_sharpe_ratio(save_weights=True)
    weights = ef.clean_weights()

    # Convert dictionary of weights to target allocations
    target_allocations = {}
    for ticker, weight in weights.items():
        target_allocations[ticker] = weight

    return target_allocations

def _portfolio_symbols_equity_allocations(target_allocations, rebalance_equity):
    if callable(target_allocations):
        target_allocations = target_allocations()

    portfolio_symbols_equity_allocations = {}
    for ticker, weight in target_allocations.items():
        portfolio_symbols_equity_allocations[ticker] = round((weight * rebalance_equity), 2)
    
    return portfolio_symbols_equity_allocations



def _alpaca_latest_positions_allocations():
    
    latest_alpaca_positions_allocations = {position.symbol: float(position.market_value) for position in alpaca.list_positions()}
    return latest_alpaca_positions_allocations



def _rebalance_equity(cash_weight):
    
    alpaca_account_equity = float(alpaca.get_account().equity)
    rebalance_equity = alpaca_account_equity - (alpaca_account_equity * cash_weight)
    
    return rebalance_equity

balance = alpaca.get_account().equity
print(balance)

rebalance_equity = _rebalance_equity(.05)
print(rebalance_equity)



def handle_buy_orders(positions_to_buy):
    
    for symbol, amount in positions_to_buy.items():
        alpaca.submit_order(symbol, amount, "buy")

def handle_sell_orders(positions_to_sell):
    
    for symbol, amount in positions_to_sell.items():
        alpaca.submit_order(symbol, amount, "sell")

                
def alpaca_rebalance(target_allocations, cash_weight=.05):

    portfolio_symbols = list(target_allocations.keys())

    # Check to Make Sure All Symbols are Eligible for Fractional Trading on Alpaca and Market is Open
    all_symbols_eligible_for_fractionals = _all_symbols_eligible_for_fractionals(portfolio_symbols)

    # Ensures All Symbols are Fractionable and the Market is Open
    if all_symbols_eligible_for_fractionals and alpaca.get_clock().is_open:

        # Grab Current Alpaca Holdings
        alpaca_latest_positions = _alpaca_latest_positions()

        # Construct a List of Equities to Close Based on Current Alpaca Holdings and Current Desired Holdings
        print("Closing Positions...")
        print(20*"~~")
        alpaca_symbols_to_close = _alpaca_symbols_to_close(alpaca_latest_positions, portfolio_symbols)
        
        # Close Any Alpaca Positions if Neccessary
        if alpaca_symbols_to_close:
            alpaca_close_positions(alpaca_symbols_to_close)

        # Calculate Rebalance Weight Taking Cash Weight % into Account
        print("Preparing Rebalance Equity...")
        print(20*"~~")
        rebalance_equity = _rebalance_equity(cash_weight)
        
        # Calculate Latest Desired Positions Equity Allocation
        print("Calculating Latest Desired Positions Equity Allocation...")
        print(20*"~~")
        portfolio_symbols_equity_allocation = _portfolio_symbols_equity_allocations(target_allocations, rebalance_equity)

        # Grab Latest Alpaca Positions and Their Respective Allocations
        print("Grabbing Latest Alpaca Positions...")
        print(20*"~~")
        alpaca_latest_positions_allocations = _alpaca_latest_positions_allocations()
        
        # Determine Which Symbols Need to be Bought and Which Symbols Need to be Sold to Rebalance
        print("Determining Which Symbols Need to be Bought and Sold...")
        print(20*"~~")
        positions_to_sell, positions_to_buy = _alpaca_symbols_to_sell_and_buy(portfolio_symbols_equity_allocation, alpaca_latest_positions_allocations)
        
        # Handle Any Sell Orders
        print("Handling Sell Orders...")
        print(20*"~~")
        handle_sell_orders(positions_to_sell)
        
        # Handle Any Buy Orders
        print("Handling Buy Orders...")
        print(20*"~~")
        handle_buy_orders(positions_to_buy)

        print("Rebalance Complete!")
def _alpaca_latest_positions_allocations():
    """
    Returns the latest positions and their respective allocations from Alpaca.
    """
    total_equity = float(alpaca.get_account().equity)
    positions = _alpaca_latest_positions()
    allocations = _portfolio_symbols_equity_allocations(positions, total_equity)
    return allocations
ef2 = EfficientFrontier(pf.comp_mean_returns(), pf.comp_cov())
target_allocations = target_allocation_maximum_sharpe(ef2)
portfolio_symbols = list(target_allocations.keys())


# Check to Make Sure All Symbols are Eligible for Fractional Trading on Alpaca and Market is Open
all_symbols_eligible_for_fractionals = _all_symbols_eligible_for_fractionals(portfolio_symbols)

# Ensures All Symbols are Fractionable and the Market is Open
if all_symbols_eligible_for_fractionals and alpaca.get_clock().is_open:

    # Grab Current Alpaca Holdings
    alpaca_latest_positions = _alpaca_latest_positions()

    # Construct a List of Equities to Close Based on Current Alpaca Holdings and Current Desired Holdings
    print("Closing Positions...")
    print(20*"~~")
    alpaca_symbols_to_close = _alpaca_symbols_to_close(alpaca_latest_positions, portfolio_symbols)
    
    # Close Any Alpaca Positions if Neccessary
    if alpaca_symbols_to_close:
        alpaca_close_positions(alpaca_symbols_to_close)

    # Calculate Rebalance Weight Taking Cash Weight % into Account
    print("Preparing Rebalance Equity...")
    print(20*"~~")
    total_equity = float(alpaca.get_account().equity)
    rebalance_equity = total_equity * (1 - cash_weight)
    
    # Calculate Latest Desired Positions Equity Allocation
    print("Calculating Latest Desired Positions Equity Allocation...")
    print(20*"~~")
    portfolio_symbols_equity_allocation = _portfolio_symbols_equity_allocations(target_allocations, rebalance_equity)

    # Grab Latest Alpaca Positions and Their Respective Allocations
    print("Grabbing Latest Alpaca Positions...")
    print(20*"~~")
    alpaca_latest_positions_allocations = _alpaca_latest_positions_allocations()
    
    # Determine Which Symbols Need to be Bought and Which Symbols Need to be Sold to Rebalance
    print("Determining Which Symbols Need to be Bought and Sold...")
    print(20*"~~")
    positions_to_sell, positions_to_buy = _alpaca_symbols_to_sell_and_buy(portfolio_symbols_equity_allocation, alpaca_latest)        
def _alpaca_latest_positions():
    
    alpaca_positions = alpaca.list_positions()
    alpaca_latest_positions = [position.symbol for position in alpaca_positions]

    return alpaca_latest_positions

alpaca_latest_positions_allocations = _alpaca_latest_positions_allocations()
alpaca_latest_positions_allocations
print(portfolio_symbols_equity_allocation)
alpaca_latest_positions = _alpaca_latest_positions()
#alpaca_latest_positions
print(alpaca_latest_positions_allocations)

def alpaca_order(symbol, amount, side):
    
    if amount > 1:
        
        alpaca_order_info = alpaca.submit_order(symbol=symbol, notional=amount, side=side, type="market", time_in_force="day")
        alpaca_client_order_id = alpaca_order_info.client_order_id
        
        order_pending = True
        while order_pending:
            latest_alpaca_order_info = alpaca.get_order_by_client_order_id(alpaca_client_order_id)
            alpaca_latest_status = latest_alpaca_order_info.status
            sleep(2.5)
            if alpaca_latest_status == "filled":
                order_pending = False

