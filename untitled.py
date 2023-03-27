


!pip install streamlit
!pip install plotly
!pip install pandas
!pip install numpy
!pip install alpaca-trade-api
!pip install pandas-datareader
!pip install pypfopt
!pip install finquant
!pip install finta
Once the necessary libraries are installed, we can create a new Python file and add the following code:

python
Copy code
# app.py

import os
import streamlit as st
import numpy as np
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