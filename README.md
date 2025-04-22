# black_mesa_stock_market_dashboard
Black Mesa Softwar3: Stock Market Dashboard created by Kevin Martinez 2025

Black Mesa Softwar3 presents Stock Market Dashboard

This software is a real-time stock market dashboard.
It was designed to help investors, analyst and financial advisors monitor markets, set alerts, and stay informed - all in once place.

Features:


Macro Economics
  U.S Economic Indicators (Unemployment, Gas Prices, Retail sales, Car sales) - via FRED
  Major financial secotors overview


Micro Economics
  -Enter a ticker and recieve:
    Live Stock Charts (Closing Prices with Volume overlays)
    Technical Indicators (RSI, Moving Averages, P/E Ratio, Beta)
    News Integration : Latest company headlines via Finhub API
    Analyst Consensus : Visual breakdown of Buy, Hold, and Sell ratings

Custom Alerts
  -Users can create alerts for a ticker regarding:
    -Price Treshold (above, below)
    -RSI and P/E Alerts
    -Volume Spike Detection
    -Moving Average crossover
  -Email + SMS Notifications of Alerts

  Tech Stack:
  Python, Streamlit
  yFinance, Plotly, Finhub API
  SQLLite3
  Twilo API
  FRED API
