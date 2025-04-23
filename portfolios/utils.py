import os
import base64
import io 
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from dateutil.parser import parse
import math

# For GARCH volatility prediction
try:
    from arch import arch_model
except ImportError:
    print("Warning: arch module not found. GARCH volatility prediction will not be available.")
    # Define a placeholder to avoid errors if arch is not installed
    def arch_model(*args, **kwargs):
        raise ImportError("arch module not installed. Install with: pip install arch")

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
    actual_port_return, actual_port_variance = actual_return(actual_returns, w)
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

# New function: GARCH for volatility prediction
def forecast_portfolio_volatility(prices_df, weights_dict):
    """
    Forecast portfolio volatility using GARCH model.
    
    Args:
        prices_df: DataFrame of asset prices
        weights_dict: Dictionary mapping asset names to weights
        
    Returns:
        Dict with portfolio volatility and individual asset volatilities
    """
    try:
        # Filter for assets in weights dictionary
        tickers = [t for t in weights_dict.keys() if t in prices_df.columns]
        price_df = prices_df[tickers].copy()
        
        # Calculate log returns
        log_returns = np.log(price_df / price_df.shift(1)).dropna()
        
        # Normalize weights
        total_weight = sum(weights_dict[t] for t in tickers)
        weight_vector = np.array([weights_dict[t] / total_weight for t in tickers])
        
        # GARCH forecast for each ticker
        forecasted_vols = []
        individual_vols = {}
        
        for i, ticker in enumerate(tickers):
            returns = log_returns[ticker] * 100  # GARCH works better with percentage scale
            model = arch_model(returns, vol='GARCH', p=1, q=1)  # Simplified p,q parameters
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            sigma = np.sqrt(forecast.variance.values[-1][0]) / 100  # Back to raw scale
            forecasted_vols.append(sigma)
            individual_vols[ticker] = sigma
        
        # Correlation matrix from historical returns
        correlation_matrix = log_returns.corr().values
        
        # Construct forecasted covariance matrix
        forecasted_vol_matrix = np.outer(forecasted_vols, forecasted_vols)
        forecasted_cov_matrix = forecasted_vol_matrix * correlation_matrix
        
        # Calculate portfolio volatility
        portfolio_variance = weight_vector.T @ forecasted_cov_matrix @ weight_vector
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            "portfolio_volatility": portfolio_volatility,
            "individual_vols": individual_vols
        }
    except Exception as e:
        print(f"Error in forecast_portfolio_volatility: {str(e)}")
        return {
            "portfolio_volatility": None,
            "individual_vols": {}
        }

def predict_portfolio(lookback, horizon, initial_equity, tickers=None):
    """
    Predict portfolio performance.
    
    Args:
        lookback: Lookback period in months
        horizon: Forecast horizon in months
        initial_equity: Initial portfolio value
        tickers: List of tickers to include (filters the universe)
        
    Returns:
        Dict with portfolio metrics and forecasts
    """
    if lookback <= 0 or horizon <= 0 or initial_equity <= 0:
        raise ValueError("Parameters must be positive values")
    
    # Filter the data for selected tickers if provided
    if tickers and len(tickers) > 0:
        filtered_predicted_returns = PCA_Predicted_Returns[tickers]
        filtered_actual_returns = PCA_Actual_Returns[tickers]
    else:
        filtered_predicted_returns = PCA_Predicted_Returns
        filtered_actual_returns = PCA_Actual_Returns
    
    pred_windows, pred_horizons = windowGenerator(filtered_predicted_returns, lookback, 1, 1)
    act_windows, act_horizons = windowGenerator(filtered_actual_returns, lookback, 1, 1)
    
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
    
    # Create the equity curve graph
    graph = plot_equity_curve(equity[1:], timestamps)
    final_equity = equity[-1]
    
    # Get the asset names we're working with (filtered or all)
    asset_names = filtered_actual_returns.columns.tolist()
    
    # Format the weights history
    formatted_weights = []
    for period_weights in weights_history:
        period_dict = {}
        for j, asset in enumerate(asset_names):
            if period_weights[j] > 0.01:  # Only include significant weights
                period_dict[asset] = round(period_weights[j] * 100, 2)
        formatted_weights.append(period_dict)
    
    # Get the final weights dictionary for volatility forecasting
    final_weights_dict = {asset_names[j]: weights_history[-1][j] for j in range(len(asset_names))}
    
    # Forecast volatility using GARCH if available
    vol_forecast = None
    try:
        # Use the original price data for volatility forecasting
        if tickers and len(tickers) > 0:
            filtered_prices = PCA_Actual_Prices[tickers]
        else:
            filtered_prices = PCA_Actual_Prices
            
        vol_forecast = forecast_portfolio_volatility(filtered_prices, final_weights_dict)
    except Exception as e:
        print(f"Volatility forecasting error: {str(e)}")
    
    result = {
        'portfolio_returns': [round(r, 4) for r in returns],
        'equity_growth': [round(e, 2) for e in equity[1:]],
        'final_equity': round(final_equity, 2),
        'initial_equity': initial_equity,
        'timestamps': [t.strftime('%Y-%m-%d') for t in timestamps],
        'equity_plot_base64': graph,
        'performance_metrics': performance_metrics,
        'weights_history': formatted_weights,
        'final_optimal_weights': {k: round(v*100, 2) for k, v in final_weights_dict.items() if v > 0.01}
    }
    
    # Add volatility forecast if available
    if vol_forecast and vol_forecast["portfolio_volatility"] is not None:
        result['predicted_portfolio_volatility'] = round(vol_forecast["portfolio_volatility"] * 100, 4)  # As percentage
        result['individual_volatilities'] = {k: round(v * 100, 4) for k, v in vol_forecast["individual_vols"].items()}  # As percentage
    
    return result