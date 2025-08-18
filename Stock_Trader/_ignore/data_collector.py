import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.trend import CCIIndicator
import os
import warnings
import time
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings("ignore")

def fetch_nifty100_data(start_date="2014-01-01", end_date="2025-03-31", 
                        test_start="2025-06-15", test_end="2025-08-31"):
    """
    Fetch data for all NIFTY100 stocks with technical indicators
    
    Parameters:
    -----------
    start_date : str
        Start date for training data (YYYY-MM-DD)
    end_date : str
        End date for training data (YYYY-MM-DD)
    test_start : str
        Start date for testing data (YYYY-MM-DD)
    test_end : str
        End date for testing data (YYYY-MM-DD)
    
    Returns:
    --------
    tuple
        (train_data, test_data, market_train, market_test)
    """
    # Complete list of NIFTY100 stocks (as of current composition)
    nifty100_tickers = [
        'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 
        'HINDUNILVR.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'ITC.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
        'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'TATAMOTORS.NS', 'NTPC.NS',
        'ULTRACEMCO.NS', 'ADANIENT.NS', 'JSWSTEEL.NS', 'BAJAJFINSV.NS', 'WIPRO.NS',
        'ONGC.NS', 'NESTLEIND.NS', 'ADANIPORTS.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
        'GRASIM.NS', 'COALINDIA.NS', 'HINDALCO.NS', 'DIVISLAB.NS', 'SBILIFE.NS',
        'TECHM.NS', 'DRREDDY.NS', 'BAJAJ-AUTO.NS', 'BRITANNIA.NS', 'INDUSINDBK.NS',
        'CIPLA.NS', 'EICHERMOT.NS', 'TATACONSUM.NS', 'APOLLOHOSP.NS', 'M&M.NS',
        'HDFCLIFE.NS', 'ADANIGREEN.NS', 'UPL.NS', 'HEROMOTOCO.NS', 'BPCL.NS',
        'DMART.NS', 'SHREECEM.NS', 'LTIM.NS', 'PIDILITIND.NS', 'BANKBARODA.NS',
        'ADANIPOWER.NS', 'GAIL.NS', 'TATAPOWER.NS', 'IOC.NS', 'HAVELLS.NS',
        'GODREJCP.NS', 'SIEMENS.NS', 'DLF.NS', 'DABUR.NS', 'INDIGO.NS',
        'AMBUJACEM.NS', 'BERGEPAINT.NS', 'CHOLAFIN.NS', 'JINDALSTEL.NS', 'ICICIPRULI.NS',
        'NAUKRI.NS', 'BOSCHLTD.NS', 'VEDL.NS', 'MARICO.NS', 'ZYDUSLIFE.NS',
        'BAJAJHLDNG.NS', 'ICICIGI.NS', 'TORNTPHARM.NS', 'COLPAL.NS', 'MUTHOOTFIN.NS',
        'SAIL.NS', 'ATGL.NS', 'PIIND.NS', 'CANBK.NS', 'TRENT.NS',
        'INDUSTOWER.NS', 'GUJGASLTD.NS', 'SBICARD.NS', 'LUPIN.NS', 'NYKAA.NS',
        'PAYTM.NS', 'ZOMATO.NS', 'PNB.NS', 'BIOCON.NS', 'POLICYBZR.NS',
        'HAL.NS', 'BEL.NS', 'AUROPHARMA.NS', 'MPHASIS.NS', 'LODHA.NS'
    ]
    
    # Create directories to save data
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Download market benchmark data (NIFTY 50 index)
    print("Downloading market benchmark data...")
    try:
        market_train = yf.download('^NSEI', start=start_date, end=end_date, interval="1d", auto_adjust=True, prepost=True, threads=False)
        market_test = yf.download('^NSEI', start=test_start, end=test_end, interval="1d", auto_adjust=True, prepost=True, threads=False)
        
        if not market_train.empty:
            # Handle multi-level columns if they exist
            if isinstance(market_train.columns, pd.MultiIndex):
                market_train.columns = market_train.columns.droplevel(1)
            # Ensure Close is 1-dimensional
            market_train['Close'] = pd.Series(market_train['Close'].values.flatten(), index=market_train.index)
            
            # Save market data
            market_train.to_csv('data/train/NIFTY50_benchmark.csv')
            print(f"Market benchmark training data saved: {len(market_train)} records")
        else:
            print("Warning: Market training data is empty")
            market_train = None
            
        if not market_test.empty:
            # Handle multi-level columns if they exist
            if isinstance(market_test.columns, pd.MultiIndex):
                market_test.columns = market_test.columns.droplevel(1)
            # Ensure Close is 1-dimensional
            market_test['Close'] = pd.Series(market_test['Close'].values.flatten(), index=market_test.index)
            
            # Save market data
            market_test.to_csv('data/test/NIFTY50_benchmark.csv')
            print(f"Market benchmark testing data saved: {len(market_test)} records")
        else:
            print("Warning: Market testing data is empty")
            market_test = None
            
    except Exception as e:
        print(f"Error downloading market data: {e}")
        market_train = None
        market_test = None
    
    train_data = {}
    test_data = {}
    successful_downloads = 0
    failed_downloads = 0
    
    # Create a progress bar
    for i, ticker in enumerate(tqdm(nifty100_tickers, desc="Downloading stock data")):
        try:
            # Download training data
            df_train = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, prepost=True, threads=False)
            
            # Download testing data
            df_test = yf.download(ticker, start=test_start, end=test_end, interval="1d", auto_adjust=True, prepost=True, threads=False)
            
            # Check if data is sufficient
            if df_train.empty or len(df_train) < 100:
                print(f"Insufficient training data for {ticker}, skipping...")
                failed_downloads += 1
                continue
                
            if df_test.empty or len(df_test) < 20:
                print(f"Insufficient testing data for {ticker}, skipping...")
                failed_downloads += 1
                continue
            
            # Process training data
            df_train = process_stock_data(df_train, ticker, market_train)
            
            # Process testing data
            df_test = process_stock_data(df_test, ticker, market_test)
            
            # Save to CSV
            df_train.to_csv(f'data/train/{ticker.replace(".NS", "")}.csv')
            df_test.to_csv(f'data/test/{ticker.replace(".NS", "")}.csv')
            
            # Store in dictionaries
            train_data[ticker] = df_train
            test_data[ticker] = df_test
            
            successful_downloads += 1
            
            # Add a small delay to avoid hitting API limits
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            failed_downloads += 1
            continue
    
    if len(train_data) == 0:
        raise ValueError("No valid stock data could be downloaded. Please check your internet connection and try again.")
    
    print(f"\nSuccessfully downloaded data for {successful_downloads} stocks out of {len(nifty100_tickers)} attempted")
    print(f"Failed downloads: {failed_downloads}")
    
    # Save a summary file with all tickers that were successfully downloaded
    with open('data/successful_tickers.txt', 'w') as f:
        for ticker in train_data.keys():
            f.write(f"{ticker}\n")
    
    return train_data, test_data, market_train, market_test

def process_stock_data(df, ticker, market_data=None):
    """Process stock data by adding technical indicators and handling missing values"""
    
    # Handle multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} for {ticker}")
    
    # Convert to Series if needed and ensure proper data types
    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)
    df['High'] = pd.Series(df['High'].values.flatten(), index=df.index)
    df['Low'] = pd.Series(df['Low'].values.flatten(), index=df.index)
    df['Volume'] = pd.Series(df['Volume'].values.flatten(), index=df.index)
    
    # Add technical indicators with error handling
    try:
        # RSI (Relative Strength Index)
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
    except:
        df['RSI'] = 50.0  # Neutral RSI if calculation fails
    
    try:
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['BB_position'] = df['BB_position'].fillna(0.5)  # Neutral if calculation fails
    except:
        df['BB_upper'] = df['Close'] * 1.02
        df['BB_lower'] = df['Close'] * 0.98
        df['BB_position'] = 0.5
    
    try:
        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()  # Difference between MACD and signal line
    except:
        df['MACD'] = 0.0
        df['MACD_signal'] = 0.0
        df['MACD_diff'] = 0.0
    
    try:
        # CCI (Commodity Channel Index)
        df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
    except:
        df['CCI'] = 0.0
    
    try:
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
    except:
        df['ADX'] = 25.0  # Neutral ADX
        df['DI_plus'] = 20.0
        df['DI_minus'] = 20.0
    
    try:
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()
    except:
        df['Stoch_k'] = 50.0
        df['Stoch_d'] = 50.0
    
    # Returns and volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Market correlation - use market data if available
    if market_data is not None and not market_data.empty:
        try:
            # Ensure market_data Close is also 1-dimensional
            market_close = pd.Series(market_data['Close'].values.flatten(), index=market_data.index)
            market_returns = market_close.pct_change()
            
            # Align indices for correlation calculation
            common_idx = df.index.intersection(market_returns.index)
            if len(common_idx) > 20:
                stock_returns = df.loc[common_idx, 'Returns']
                market_returns = market_returns.loc[common_idx]
                
                # Calculate Beta using rolling window
                df['Market_Returns'] = np.nan
                df.loc[common_idx, 'Market_Returns'] = market_returns
                
                # Calculate Beta (20-day rolling)
                df['Beta'] = df['Returns'].rolling(window=20).cov(df['Market_Returns']) / \
                            df['Market_Returns'].rolling(window=20).var()
                
                # Calculate correlation (20-day rolling)
                df['Market_Correlation'] = df['Returns'].rolling(window=20).corr(df['Market_Returns'])
            else:
                df['Market_Returns'] = df['Returns'] * 0.8
                df['Beta'] = 1.0
                df['Market_Correlation'] = 0.5
        except:
            df['Market_Returns'] = df['Returns'] * 0.8
            df['Beta'] = 1.0
            df['Market_Correlation'] = 0.5
    else:
        # Use dummy values if no market data
        df['Market_Returns'] = df['Returns'] * 0.8  # Approximate market correlation
        df['Beta'] = 1.0  # Neutral beta
        df['Market_Correlation'] = 0.5  # Neutral correlation
    
    # Add turbulence index as mentioned in the paper
    # "To control the risk in a worst-case scenario like 2008 global financial crisis, 
    # we employ the financial turbulence index"
    try:
        # Simple approximation of turbulence using rolling volatility relative to historical
        df['Turbulence'] = df['Volatility'] / df['Volatility'].rolling(window=252).mean()
        df['Turbulence'] = df['Turbulence'].fillna(1.0)
    except:
        df['Turbulence'] = 1.0
    
    # Fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.fillna(0)  # Fill any remaining NaNs with 0
    df.dropna(inplace=True)
    
    return df

def create_summary_file(train_data, test_data, market_train, market_test):
    """Create a summary file with statistics about the collected data"""
    
    summary = {
        'ticker': [],
        'train_records': [],
        'test_records': [],
        'train_start': [],
        'train_end': [],
        'test_start': [],
        'test_end': [],
        'avg_price': [],
        'volatility': [],
        'beta': []
    }
    
    for ticker in train_data.keys():
        train_df = train_data[ticker]
        test_df = test_data[ticker]
        
        summary['ticker'].append(ticker)
        summary['train_records'].append(len(train_df))
        summary['test_records'].append(len(test_df))
        summary['train_start'].append(train_df.index[0].strftime('%Y-%m-%d'))
        summary['train_end'].append(train_df.index[-1].strftime('%Y-%m-%d'))
        summary['test_start'].append(test_df.index[0].strftime('%Y-%m-%d'))
        summary['test_end'].append(test_df.index[-1].strftime('%Y-%m-%d'))
        summary['avg_price'].append(train_df['Close'].mean())
        summary['volatility'].append(train_df['Volatility'].mean())
        summary['beta'].append(train_df['Beta'].mean())
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('data/data_summary.csv', index=False)
    
    # Create a more detailed summary with market data
    if market_train is not None:
        market_summary = {
            'dataset': ['Market (Train)', 'Market (Test)'],
            'records': [len(market_train), len(market_test)],
            'start_date': [market_train.index[0].strftime('%Y-%m-%d'), 
                          market_test.index[0].strftime('%Y-%m-%d')],
            'end_date': [market_train.index[-1].strftime('%Y-%m-%d'), 
                        market_test.index[-1].strftime('%Y-%m-%d')],
            'avg_close': [market_train['Close'].mean(), market_test['Close'].mean()],
            'volatility': [market_train['Close'].pct_change().std() * np.sqrt(252), 
                          market_test['Close'].pct_change().std() * np.sqrt(252)]
        }
        market_summary_df = pd.DataFrame(market_summary)
        market_summary_df.to_csv('data/market_summary.csv', index=False)
    
    print(f"Data summary saved to data/data_summary.csv")
    if market_train is not None:
        print(f"Market summary saved to data/market_summary.csv")

def main():
    print("Starting NIFTY100 data collection process...")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch data for all NIFTY100 stocks
    train_data, test_data, market_train, market_test = fetch_nifty100_data(
        start_date="2014-01-01", 
        end_date="2025-03-31",
        test_start="2025-06-15", 
        test_end="2025-08-31"
    )
    
    # Create summary file
    create_summary_file(train_data, test_data, market_train, market_test)
    
    print("\nData collection complete!")
    print(f"Training data: {len(train_data)} stocks from 2014-01-01 to 2025-03-31")
    print(f"Testing data: {len(test_data)} stocks from 2025-06-15 to 2025-08-31")
    print(f"Data saved in data/train and data/test directories")

if __name__ == "__main__":
    main()
