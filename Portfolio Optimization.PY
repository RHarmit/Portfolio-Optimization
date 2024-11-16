import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Fetch data excluding AAPL and including META
tickers = ['MSFT', 'TSLA', 'META', 'SPY']
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio optimization
def portfolio_metrics(weights):
    weights = np.array(weights)
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility

# Objective function to minimize (negative Sharpe ratio)
def negative_sharpe(weights):
    return -portfolio_metrics(weights)[2]

# Constraints: Weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(len(tickers))]
initial_weights = [1/len(tickers)] * len(tickers)

# Optimization
optimized = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Visualize efficient frontier
results = []
for i in range(1000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    results.append(portfolio_metrics(weights))
results = np.array(results)

plt.scatter(results[:, 1], results[:, 0], c=results[:, 0] / results[:, 1], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(portfolio_metrics(optimal_weights)[1], portfolio_metrics(optimal_weights)[0], color='red', marker='*', label='Optimal Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

# Output optimal weights
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")