# NIFTY100 Stock Data Collector

This script collects historical data for all 100 NIFTY100 stocks from 2014 to 2025, with training data from January 2014 to March 2025 and testing data from mid-June 2025 to August 2025. The data includes various technical indicators as described in the research paper "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy".

## Features

- Downloads data for all 100 NIFTY100 stocks
- Adds technical indicators:
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
  - CCI (Commodity Channel Index)
  - ADX (Average Directional Index)
  - Stochastic Oscillator
- Calculates market correlation metrics (Beta, Market Correlation)
- Implements turbulence index for risk management
- Handles missing data and errors gracefully
- Saves data to CSV files for easy access

## Technical Indicators

Based on the research paper, the following indicators are included:

1. **RSI (Relative Strength Index)**: Quantifies the extent of recent price changes to identify overbought or oversold conditions
2. **Bollinger Bands**: Identifies volatility and potential price breakouts
3. **MACD (Moving Average Convergence Divergence)**: A momentum indicator that identifies moving averages
4. **CCI (Commodity Channel Index)**: Compares current price to average price over a time window
5. **ADX (Average Directional Index)**: Identifies trend strength by quantifying price movement
6. **Stochastic Oscillator**: Compares a security's closing price to its price range over a specific period

## Usage

To run the data collector:

```bash
python data_collector.py
```

The script will:
1. Create directories for storing data (`data/train` and `data/test`)
2. Download NIFTY50 index data as a market benchmark
3. Download historical data for all NIFTY100 stocks
4. Calculate technical indicators for each stock
5. Save processed data to CSV files
6. Create summary files with data statistics

## Output

The script generates the following files:

- `data/train/`: Directory containing CSV files for each stock's training data (2014-01-01 to 2025-03-31)
- `data/test/`: Directory containing CSV files for each stock's testing data (2025-06-15 to 2025-08-31)
- `data/train/NIFTY50_benchmark.csv`: Market benchmark data for the training period
- `data/test/NIFTY50_benchmark.csv`: Market benchmark data for the testing period
- `data/data_summary.csv`: Summary statistics for all downloaded stocks
- `data/market_summary.csv`: Summary statistics for the market benchmark
- `data/successful_tickers.txt`: List of successfully downloaded stock tickers

## Requirements

- pandas
- numpy
- yfinance
- ta (Technical Analysis library)
- tqdm (for progress bars)

Install the required packages with:

```bash
pip install pandas numpy yfinance ta tqdm
```

## Note

The script includes error handling to deal with potential API limitations and missing data. If you encounter issues with downloading data, try running the script again or adjusting the delay between API calls.
