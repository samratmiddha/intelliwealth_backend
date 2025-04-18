
import os
import io
from scipy.optimize import minimize, Bounds
from numpy.linalg import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.parser import parse
import math

# Get the directory where this file lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to the CSV files
PCA_PREDICTED_PATH = os.path.join(THIS_DIR, 'PCA_Predicted_Prices1.csv')
PCA_ACTUAL_PATH = os.path.join(THIS_DIR, 'PCA_Actual_Prices1.csv')

PCA_Predicted_Prices = pd.read_csv(PCA_PREDICTED_PATH)
PCA_Predicted_Prices['Date'] = pd.to_datetime(PCA_Predicted_Prices['Date'])
PCA_Predicted_Prices = PCA_Predicted_Prices.set_index('Date')

PCA_Actual_Prices = pd.read_csv(PCA_ACTUAL_PATH)
PCA_Actual_Prices['Date'] = pd.to_datetime(PCA_Actual_Prices['Date'])
PCA_Actual_Prices = PCA_Actual_Prices.set_index('Date')
PCA_Predicted_Returns = PCA_Predicted_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]
PCA_Actual_Returns = PCA_Actual_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]

def mean_returns(df, length):
    return df.sum(axis=0) / length

def monthdelta(date, delta):
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m:
        m = 12
    d = min(date.day, [31, 29 if y % 4 == 0 and not y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    new_date = date.replace(day=d, month=m, year=y)
    return parse(new_date.strftime('%Y-%m-%d'))

def windowGenerator(dataframe, lookback, horizon, step, cummulative=False):
    if cummulative:
        c = lookback
        step = horizon
    initial = min(dataframe.index)
    windows = []
    horizons = []
    while initial <= monthdelta(max(dataframe.index), -lookback):
        windowStart = initial
        windowEnd = monthdelta(windowStart, lookback)
        if cummulative:
            windowStart = min(dataframe.index)
            windowEnd = monthdelta(windowStart, c) + timedelta(days=1)
            c += horizon
        horizonStart = windowEnd + timedelta(days=1)
        horizonEnd = monthdelta(horizonStart, horizon)
        windows.append(dataframe[windowStart:windowEnd])
        horizons.append(dataframe[horizonStart:horizonEnd])
        initial = monthdelta(initial, step)
    return windows, horizons

def actual_return(actual_returns, w):
    mu = mean_returns(actual_returns, actual_returns.shape[0])
    cov = actual_returns.cov()
    port_return = mu.T.dot(w)
    port_variance = w.T.dot(cov).dot(w)
    return port_return, port_variance

def scipy_opt(predicted_returns, actual_returns, lam1, lam2):
    mu = mean_returns(predicted_returns, predicted_returns.shape[0])
    cov = predicted_returns.cov()
    def objective(w):
        return -(mu.T.dot(w) - lam1 * (w.T.dot(cov).dot(w)) + lam2 * norm(w, ord=1))
    bounds = Bounds(0, 1)
    constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
    sol = minimize(objective,
                   x0=np.ones(mu.shape[0]),
                   constraints=constraints,
                   bounds=bounds,
                   options={'disp': False},
                   tol=1e-9)
    w = sol.x
    predicted_port_return = w.dot(mu)
    portfolio_std = w.T.dot(cov).dot(w)
    actual_port_return, actual_port_variance = actual_return(PCA_Actual_Returns, w)
    sharpe_ratio = actual_port_return / np.sqrt(actual_port_variance)
    return {
        'weights': w,
        'predicted_returns': predicted_port_return,
        'predicted_variance': portfolio_std,
        'actual_returns': actual_port_return,
        'actual_variance': actual_port_variance,
        'sharpe_ratio': sharpe_ratio
    }

def metrics(returns_series):
    sharpe = returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0
    annualized_sharpe = sharpe * math.sqrt(252)
    annualized_return = returns_series.mean() * 252
    annualized_vol = returns_series.std() * math.sqrt(252)
    max_drawdown = (returns_series.cumsum() - returns_series.cumsum().cummax()).min()
    return {
        "Annualized Return": round(annualized_return, 4),
        "Annualized Volatility": round(annualized_vol, 4),
        "Annualized Sharpe Ratio": round(annualized_sharpe, 4),
        "Maximum Drawdown": round(max_drawdown, 4)
    }

def plot_equity_curve(equity_series, timestamps):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, equity_series)
    ax.set_title("Portfolio Equity Growth Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    plt.close(fig)
    return base64.b64encode(img_bytes).decode()

def predict_portfolio(lookback, horizon, initial_equity):
    if lookback <= 0 or horizon <= 0 or initial_equity <= 0:
        raise ValueError("Parameters must be positive values")
    pred_windows, pred_horizons = windowGenerator(PCA_Predicted_Returns, lookback, 1, 1)
    act_windows, act_horizons = windowGenerator(PCA_Actual_Returns, lookback, 1, 1)
    if len(act_horizons) < horizon:
        raise ValueError(f"Not enough data for the specified horizon. Maximum horizon available is: {len(act_horizons)}")
    start = len(act_horizons) - horizon
    returns, variance, sharperatio, timestamps, equity = [], [], [], [], [initial_equity]
    weights_history = []
    for i in range(start, start + horizon):
        r = scipy_opt(pred_horizons[i], act_horizons[i], 0.5, 2)
        returns.append(r['actual_returns'])
        variance.append(r['actual_variance'])
        sharperatio.append(r['sharpe_ratio'])
        timestamps.append(act_horizons[i].index[0])
        equity.append(equity[-1] * math.exp(r['actual_returns']))
        weights_history.append(r['weights'])
        print(i, "complete")
    returns_series = pd.Series(returns)
    performance_metrics = metrics(returns_series)
    graph = plot_equity_curve(equity[1:], timestamps)
    final_equity = equity[-1]
    asset_names = PCA_Actual_Returns.columns.tolist()
    formatted_weights = []
    for period_weights in weights_history:
        period_dict = {}
        for j, asset in enumerate(asset_names):
            if period_weights[j] > 0.01:
                period_dict[asset] = round(period_weights[j] * 100, 2)
        formatted_weights.append(period_dict)
    return {
        'portfolio_returns': [round(r, 4) for r in returns],
        'equity_growth': [round(e, 2) for e in equity[1:]],
        'final_equity': round(final_equity, 2),
        'initial_equity': initial_equity,
        'timestamps': [t.strftime('%Y-%m-%d') for t in timestamps],
        'equity_plot_base64': graph,
        'performance_metrics': performance_metrics,
        'weights_history': formatted_weights
    }