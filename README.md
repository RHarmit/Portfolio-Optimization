# Portfolio-Optimization

Overview
This Python project optimizes stock portfolio allocations by maximizing the Sharpe Ratio to achieve the best risk-adjusted returns. It uses real historical data from Yahoo Finance and showcases the efficient frontier for various portfolio combinations.

Features
Fetches historical stock price data for user-selected or predefined stocks (e.g., TSLA, META, SPY, MSFT).
Calculates key portfolio metrics: expected returns, volatility, and Sharpe Ratio.
Identifies the optimal portfolio using scipy.optimize.
Visualizes the efficient frontier, highlighting the portfolio with the highest Sharpe Ratio.
Technologies Used
Python Libraries:
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib: For data visualization.
yfinance: For fetching real-time stock data.
scipy: For portfolio optimization.
