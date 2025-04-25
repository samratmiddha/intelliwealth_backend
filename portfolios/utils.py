import os
import keras
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.linalg import norm
from datetime import timedelta
from dateutil.parser import parse
import math


# Define stock list
stock_list = ["MMM" , "AOS" , "ABT" , "ABBV" , "ABMD" , "ACN" , "ATVI" , "ADBE" , "AAP" , "AMD" , "AES" , "AFL" , "A" , "APD" , "AKAM" , "ALK" , "ALB" , "ARE" , "ALXN" , "ALGN" , "ALLE" , "LNT" , "ALL" , "GOOGL" , "GOOG" , "MO" , "AMZN" , "AMCR" , "AEE" , "AAL" , "AEP" , "AXP" , "AIG" , "AMT" , "AWK" , "AMP" , "ABC" , "AME" , "AMGN" , "APH" , "ADI" , "ANSS" , "ANTM" , "AON" , "APA" , "AIV" , "AAPL" , "AMAT" , "APTV" , "ADM" , "ANET" , "AJG" , "AIZ" , "T" , "ATO" , "ADSK" , "ADP" , "AZO" , "AVB" , "AVY" , "BKR" , "BLL" , "BAC" , "BAX" , "BDX" ,  "BBY" , "BIO" , "BIIB" , "BLK" , "BA" , "BKNG" , "BWA" , "BXP" , "BSX" , "BMY" , "AVGO" , "BR" ,  "CHRW" , "COG" , "CDNS" , "CPB" , "COF" , "CAH" , "KMX" , "CCL" , "CARR" , "CAT" , "CBOE" , "CBRE" , "CDW" , "CE" , "CNC" , "CNP" , "CTL" , "CERN" , "CF" , "SCHW" , "CHTR" , "CVX" , "CMG" , "CB" , "CHD" , "CI" , "CINF" , "CTAS" , "CSCO" , "C" , "CFG" , "CTXS" , "CME" , "CMS" , "KO" , "CTSH" , "CL" , "CMCSA" , "CMA" , "CAG" , "CXO" , "COP" , "ED" , "STZ" , "CPRT" , "GLW" , "CTVA" , "COST" , "COTY" , "CCI" , "CSX" , "CMI" , "CVS" , "DHI" , "DHR" , "DRI" , "DVA" , "DE" , "DAL" , "XRAY" , "DVN" , "DXCM" , "FANG" , "DLR" , "DFS" , "DISCA" , "DISCK" , "DISH" , "DG" , "DLTR" , "D" , "DPZ" , "DOV" , "DOW" , "DTE" , "DUK" , "DRE" , "DD" , "DXC" , "ETFC" , "EMN" , "ETN" , "EBAY" , "ECL" , "EIX" , "EW" , "EA" , "EMR" , "ETR" , "EOG" , "EFX" , "EQIX" , "EQR" , "ESS" , "EL" , "RE" , "EVRG" , "ES" , "EXC" , "EXPE" , "EXPD" , "EXR" , "XOM" , "FFIV" , "FB" , "FAST" , "FRT" , "FDX" , "FIS" , "FITB" , "FRC" , "FE" , "FISV" , "FLT" , "FLIR" , "FLS" , "FMC" , "F" , "FTNT" , "FTV" , "FBHS" , "FOXA" , "FOX" , "BEN" , "FCX" , "GPS" , "GRMN" , "IT" , "GD" , "GE" , "GIS" , "GM" , "GPC" , "GILD" , "GPN" , "GL" , "GS" , "GWW" , "HRB" , "HAL" , "HBI" , "HIG" , "HAS" , "HCA" , "PEAK" , "HSIC" , "HES" , "HPE" , "HLT" , "HFC" , "HOLX" , "HD" , "HON" , "HRL" , "HST" , "HWM" , "HPQ" , "HUM" , "HBAN" , "HII" , "IEX" , "IDXX" , "INFO" , "ITW" , "ILMN" , "INCY" , "IR" , "INTC" , "ICE" , "IBM" , "IFF" , "IP" , "IPG" , "INTU" , "ISRG" , "IVZ" , "IPGP" , "IQV" , "IRM" , "JBHT" , "JKHY" , "J" , "SJM" , "JNJ" , "JCI" , "JPM" , "JNPR" , "KSU" , "K" , "KEY" , "KEYS" , "KMB" , "KIM" , "KMI" , "KLAC" , "KSS" , "KHC" , "KR" , "LB" , "LHX" , "LH" , "LRCX" , "LW" , "LVS" , "LEG" , "LDOS" , "LEN" , "LLY" , "LNC" , "LIN" , "LYV" , "LKQ" , "LMT" , "L" , "LOW" , "LYB" , "MTB" , "MRO" , "MPC" , "MKTX" , "MAR" , "MMC" , "MLM" , "MAS" , "MA" , "MXIM" , "MKC" , "MCD" , "MCK" , "MDT" , "MRK" , "MET" , "MTD" , "MGM" , "MCHP" , "MU" , "MSFT" , "MAA" , "MHK" , "TAP" , "MDLZ" , "MNST" , "MCO" , "MS" , "MSI" , "MSCI" , "MYL" , "NDAQ" , "NOV" , "NTAP" , "NFLX" , "NWL" , "NEM" , "NWSA" , "NWS" , "NEE" , "NLSN" , "NKE" , "NI" , "NBL" , "NSC" , "NTRS" , "NOC" , "NLOK" , "NCLH" , "NRG" , "NUE" , "NVDA" , "NVR" , "ORLY" , "OXY" , "ODFL" , "OMC" , "OKE" , "ORCL" , "OTIS" , "PCAR" , "PKG" , "PH" , "PAYX" , "PAYC" , "PYPL" , "PNR" , "PBCT" , "PEP" , "PKI" , "PRGO" , "PFE" , "PM" , "PSX" , "PNW" , "PXD" , "PNC" , "PPG" , "PPL" , "PFG" , "PG" , "PGR" , "PLD" , "PRU" , "PEG" , "PSA" , "PHM" , "PVH" , "QRVO" , "QCOM" , "PWR" , "DGX" , "RL" , "RJF" , "RTX" , "O" , "REG" , "REGN" , "RF" , "RSG" , "RMD" , "RHI" , "ROK" , "ROL" , "ROP" , "ROST" , "RCL" , "SPGI" , "CRM" , "SBAC" , "SLB" , "STX" , "SEE" , "SRE" , "NOW" , "SHW" , "SPG" , "SWKS" , "SLG" , "SNA" , "SO" , "LUV" , "SWK" , "SBUX" , "STT" , "STE" , "SYK" , "SIVB" , "SYF" , "SNPS" , "SYY" , "TMUS" , "TROW" , "TTWO" , "TPR" , "TGT" , "TEL" , "FTI" , "TDY" , "TFX" , "TXN" , "TXT" , "BK" , "CLX" , "COO" , "HSY" , "MOS" , "TRV" , "DIS" , "TMO" , "TIF" , "TJX" , "TSCO" , "TT" , "TDG" , "TFC" , "TWTR" , "TYL" , "TSN" , "USB" , "UDR" , "ULTA" , "UAA" , "UA" , "UNP" , "UAL" , "UNH" , "UPS" , "URI" , "UHS" , "UNM" , "VLO" , "VAR" , "VTR" , "VRSN" , "VRSK" , "VZ" , "VRTX" , "VFC" , "VIAC" , "V" , "VNO" , "VMC" , "WRB" , "WAB" , "WBA" , "WMT" , "WM" , "WAT" , "WEC" , "WFC" , "WELL" , "WST" , "WDC" , "WU" , "WRK" , "WY" , "WHR" , "WMB" , "WLTW" , "WYNN" , "XEL" , "XRX" , "XLNX" 
                , "XYL" , "YUM" , "ZBRA" , "ZBH" , "ZION" , "ZTS"]

scl = MinMaxScaler()

def create_df(horizon):
    
    num_days=252+10*horizon
    if num_days<=1000:
        num_days=1000
    end_date = datetime.today()
    start_date = end_date - timedelta(days=num_days)

    all_data = []

    for stock in stock_list:
        print(f"Fetching data for {stock}...")
        try:
            df = yf.download(stock, start=start_date, end=end_date, progress=False)

            if not df.empty:
                df.reset_index(inplace=True) 
                all_data.append(df) 
            else:
                print(f"Warning: No data found for {stock}")

        except Exception as e:
            print(f"Could not retrieve data for {stock}: {e}")
    if all_data:
        master_df = pd.concat(all_data, axis=1)
        master_df.rename(columns={"Date": "Date", "Open": "Open", "High": "High", "Low": "Low",
                                "Close": "Close", "Volume": "Volume", "Adj Close": "Adjusted"}, inplace=True)
        
        return master_df
    else:
        print("No valid stock data found.")
        return pd.DataFrame()

def preprocess_data(df):
    df = df.drop(columns=[col for col in df.columns if col[1] == 'Date' and col != ('', 'Date')])
    df.columns = pd.MultiIndex.from_tuples([('Date', '') if col[1] == 'Date' else (col[1], col[0]) for col in df.columns])

    date_indices = [i for i, col in enumerate(df.columns) if col == ('', 'Date')]
    date_cols = [col for col in df.columns if col == ('', 'Date')]
    print(df.columns)
    print(date_cols)
    new_df = df[[date_cols[0]]]
    new_df = df.iloc[:, :1]
    
    if len(date_indices) > 1:
        cols_to_drop = [df.columns[i] for i in date_indices[1:]]
        df = df.drop(columns=cols_to_drop)

    df = pd.concat([new_df, df], axis=1)

    stocks_to_remove = [
        "ABBV", "ALLE", "AMCR", "AAL", "AWK", "AMP", "APTV", "ANET", "AVGO", "BR",
        "CARR", "CBOE", "CDW", "CF", "CHTR", "CMG", "CFG", "CTVA", "COTY", "DAL",
        "FANG", "DFS", "DG", "DOW", "FTV", "FOXA", "FOX", "GM", "HBI", "HCA",
        "HPE", "HLT", "HWM", "HII", "INFO", "IR", "ICE", "IPGP", "IQV", "KEYS",
        "KMI", "KHC", "LB", "LW", "LDOS", "LYV", "LYB", "MPC", "MA", "MSCI",
        "NWSA", "NWS", "NCLH", "OTIS", "PAYC", "PYPL", "PM", "PSX", "QRVO", "NOW",
        "SYF", "TEL", "TDG", "ULTA", "UA", "UAL", "VRSK", "V", "WU", "XYL", "ZTS"
    ]
    df = df.drop(columns=stocks_to_remove, level=0)

    stock_symbols = df.columns.get_level_values(0).unique()
    if "" in stock_symbols:
        stock_symbols = stock_symbols.drop("")

    for stock in stock_symbols:
        # Daily Return
        df[(stock, 'DailyRet')] = df[(stock, 'Close')].pct_change()

        # 20 Day Return
        df[(stock, '20DayRet')] = df[(stock, 'Close')].pct_change(20)

        # 20 Day Volatility (std of DailyRet over 20 days)
        df[(stock, '20DayVol')] = df[(stock, 'DailyRet')].rolling(window=20).std(ddof=0)

        # Z-normalized 20 Day Return
        rolling_ret = df[(stock, '20DayRet')].rolling(window=252)
        df[(stock, 'Z20DayRet')] = (
            (rolling_ret.mean().shift(1) - df[(stock, '20DayRet')]) / rolling_ret.std(ddof=0).shift(1)
        )

        # Z-normalized 20 Day Volatility
        rolling_vol = df[(stock, '20DayVol')].rolling(window=252)
        df[(stock, 'Z20DayVol')] = (
            (rolling_vol.mean().shift(1) - df[(stock, '20DayVol')]) / rolling_vol.std(ddof=0).shift(1)
        )

    stock_symbols = sorted([col for col in df.columns.get_level_values(0).unique() if col != ''])

    desired_metrics = ['Close', 'High', 'Low', 'Open', 'Volume',
                    'DailyRet', '20DayRet', '20DayVol', 'Z20DayRet', 'Z20DayVol']

    new_columns = [('', 'Date')]
    for stock in stock_symbols:
        for metric in desired_metrics:
            if (stock, metric) in df.columns:
                new_columns.append((stock, metric))

    df = df.loc[:, new_columns]

    full_feature_dataset = df.dropna(axis=0)

    return full_feature_dataset,stock_symbols

def closingPrices(df):
  stock_symbols = df.columns.get_level_values(0).unique()
  if "" in stock_symbols:
        stock_symbols = stock_symbols.drop("")
  close_columns = [(stock, 'Close') for stock in stock_symbols]
  close_columns = [('', 'Date')] + close_columns
  
  df_close = df[close_columns].copy()
  new_columns = ['Date'] + list(stock_symbols)
  df_close.columns = new_columns
  dates = df_close['Date'].copy()
  df_close = df_close.drop(columns=[('Date')])  
  
  return df_close,dates

def processData(data, lookback,jump):
    X= []
    for i in range(0,len(data) -lookback +1, jump):
        X.append(data[i:(i+lookback)])
    return np.array(X)

def prepare_data(dataset,closing_prices,num_stocks):
    pca = PCA(n_components = num_stocks)
    train_scl=MinMaxScaler()

    closing_prices= scl.fit_transform(closing_prices)

    dataset = train_scl.fit_transform(dataset)
    dataset = pca.fit_transform(dataset)

    return dataset

def do_inverse_transform(output_result,num_companies):
    original_matrix_format = []
    for result in output_result:
        original_matrix_format.append(scl.inverse_transform([result[x:x+num_companies] for x in range(0, len(result), num_companies)]))
    original_matrix_format = np.array(original_matrix_format)

    for i in range(len(original_matrix_format)):
        output_result[i] = original_matrix_format[i].ravel()

    return output_result

def prediction_by_step_by_company(raw_model_output, num_companies):
    matrix_prediction = []
    for i in range(0,num_companies):
        matrix_prediction.append([[lista[j] for j in range(i,len(lista),num_companies)] for lista in raw_model_output])
    return np.array(matrix_prediction)

def mean_returns(df, length):
    mu = df.sum(axis=0)/length
    return mu

from scipy.optimize import minimize

def get_ret_vol_sr(weights, log_return): 
    weights = np.array(weights)
    ret = np.sum(log_return.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
    sr = ret / vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights, log_return): 
    return -get_ret_vol_sr(weights, log_return)[2]

def check_sum(weights): 
    return np.sum(weights) - 1

def optimize(log_return,num_companies):
    
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple((0, 1) for _ in range(num_companies))
    init_guess = [1.0 / num_companies] * num_companies  
    
    opt_results = minimize(neg_sharpe, init_guess, args=(log_return,), method='SLSQP', bounds=bounds, constraints=cons)
    
    return opt_results

model_path = os.path.join(os.path.dirname(__file__), "final_model.keras")
model = keras.models.load_model(model_path)

# @app.route('/predict-live', methods=['POST'])
def predict_prices(horizon, initial_equity, selected_stocks=None):
    
    df=create_df(horizon)
    print("hayyy")
    full_feature_dataset,stocks=preprocess_data(df)
    print(full_feature_dataset.shape)
    print("1")
    num_companies=len(stocks)
    print(num_companies)
    print("2")
    df_close,dates=closingPrices(full_feature_dataset)
    full_feature_dataset = full_feature_dataset.drop(columns=[('', 'Date')])
    full_feature_dataset = full_feature_dataset.to_numpy()

    print("dates", dates)
    print("3")
    stocks=df_close.columns
    df_close=df_close.dropna(axis=0)
    closing_prices = df_close.iloc[:full_feature_dataset.shape[0],:]
    num_companies = df_close.shape[1]
    print("4")
    dataset = prepare_data(full_feature_dataset,closing_prices,num_companies)
    print(dataset.shape)

    X = processData(dataset, 252, 22)
    print("X.shape", X.shape)     
    
    total_predictions=None
    c=0
    print("5")
    print(horizon)
    while horizon > 0:
        print("entered")
        Xt = model.predict(X) 
        print("horizon",horizon)

        print(Xt.shape)
        Xt = do_inverse_transform(Xt, num_companies)
        
        print(Xt.shape)
        
        predictions = prediction_by_step_by_company(Xt, num_companies)
        print("predictions.shape ", predictions.shape)
        last_prediction = predictions[:, -1:, :]
        print("last_prediction.shape",last_prediction.shape)
        
        if total_predictions is None:
            total_predictions = last_prediction
        else:
            total_predictions = np.concatenate((total_predictions, last_prediction), axis=2)

        last_prediction = last_prediction.transpose(1, 2, 0)

        first_sample = X[0] 
        tail_samples = [X[i, -22:, :] for i in range(1, X.shape[0])]  
        combined = np.concatenate([first_sample] + tail_samples, axis=0)

        lastprediction_reshaped = last_prediction[0]  
        final_combined = np.concatenate([combined, lastprediction_reshaped], axis=0) 

        X=processData(final_combined,252,22)

        print("final_combined.shape",final_combined.shape)
        print("X.shape",X.shape)
        
        c=c+1
        horizon=horizon-22

    print("total_predictions.shape",total_predictions.shape)
            
    predicted_prices = np.zeros((total_predictions.shape[1]*total_predictions.shape[2], 
                            total_predictions.shape[0]))

    for i in range(total_predictions.shape[0]):
        counter = 0
        for j in range(total_predictions.shape[1]):
            for z in range(total_predictions.shape[2]):
                predicted_prices[counter, i] = total_predictions[i, j, z]
                counter += 1

    print("dates",len(dates))
    print("predicted_prices.shape[0]",predicted_prices.shape[0])

    print("predicted_prices.shape",predicted_prices.shape)

    predicted_dates = []

    x = (len(dates) - 252) / 22
    last_date_index = int(252 + 22 * x)
    last_date = dates[last_date_index]

    days=22*c
    predicted_dates = [last_date + timedelta(days=i) for i in range(1, 1+days)]

    predicted_prices_df = pd.DataFrame(data=predicted_prices, columns=stocks, index=predicted_dates)
    predicted_prices_df = predicted_prices_df.reset_index().rename(columns={"index": "Date"})

    print("predicted dates")
    print(predicted_dates)

    predicted_prices_df['Date'] = pd.to_datetime(predicted_prices_df['Date'])
    predicted_prices_df = predicted_prices_df.set_index('Date')

    print("reached yayyy")
        
    predicted_Returns = predicted_prices_df.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]

    if not selected_stocks:
        selected_stocks = predicted_prices_df.columns.tolist()
    print("reached34")
    Predicted_Returns = predicted_Returns[selected_stocks]

    result=optimize(Predicted_Returns,num_companies)
    weights=result.x
    predicted_returns,predicted_volatilty,predicted_sharperatio=get_ret_vol_sr(weights,Predicted_Returns)

    print("reached 45")
    predicted_equity = initial_equity * math.exp(predicted_returns)

    asset_names = selected_stocks
    print("reached67")
    
    # formatted_weights = {asset_names[i]: round(weights[i] * 100, 2) for i in range(len(weights)) if weights[i] > 0.01}
    formatted_weights = {asset_names[i]: round(weights[i] * 100, 2) for i in range(len(weights))}

    result={
        'initial_equity': initial_equity,
        'predicted_return': round(predicted_returns, 4),
        'predicted_volatility': round(predicted_volatilty, 4),
        'predicted_sharpe_ratio': round(predicted_sharperatio, 4),
        'expected_equity': round(predicted_equity, 2),
        'weights': formatted_weights
    }

    return result




