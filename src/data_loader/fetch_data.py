import yfinance as yf

def fetch_stock_data(ticker, start_date="2015-01-01", save_path=None):
    df = yf.download(ticker, start=start_date, progress=False)
    if save_path:
        df.to_csv(save_path)
    return df
