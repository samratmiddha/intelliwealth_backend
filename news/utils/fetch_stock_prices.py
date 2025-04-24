import pandas as pd
import yfinance as yf

tweets_df = pd.read_csv("tweets.csv")
unique_tickers = tweets_df["Stock_Name"].dropna().unique()
print(unique_tickers)

start_date = "2022-12-31"
end_date = "2025-03-31"
stock_data_list = []

for ticker in unique_tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data = data.reset_index()
            data["Stock_Name"] = ticker
            stock_data_list.append(data[["Date", "Open", "High", "Low", "Close", "Volume", "Stock_Name"]])
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

all_stock_data = pd.concat(stock_data_list, ignore_index=True)

all_stock_data.to_csv("prices.csv", index=False)
print("Stock data saved to prices.csv")
