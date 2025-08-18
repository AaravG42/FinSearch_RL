import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

# ---------- Step 2: Enhanced Trading Environment ----------
class EnhancedTradingEnv(gym.Env):
    def __init__(self, data, tickers, initial_balance=1e6, transaction_cost=0.001):
        super(EnhancedTradingEnv, self).__init__()
        if not data or len(data) == 0:
            raise ValueError("Data dictionary is empty. Cannot create trading environment.")
        
        self.data = data
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.n_assets = len(tickers)
        self.current_step = 0
        self.max_step = min(len(d) for d in data.values()) - 1
        
        if self.max_step < 50:
            raise ValueError(f"Insufficient data: only {self.max_step + 1} days available. Need at least 50 days.")
        
        # Enhanced action space: weights for each asset + cash
        self.action_space = gym.spaces.Box(0, 1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Enhanced observation space: price features + portfolio state
        obs_dim = self.n_assets * 6 + 3  # 6 features per asset + 3 portfolio features
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.current_step = 20  # Start after indicator warmup
        self.total_value = self.initial_balance
        self.prev_total_value = self.initial_balance
        self.transaction_costs = 0
        self.value_history = [self.initial_balance]
        
        # Ensure we have valid data at the starting position
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            obs = self._get_obs()
            if not np.any(np.isnan(obs)) and not np.any(np.isinf(obs)):
                return obs, {}
            else:
                # Move to next step if current observation has issues
                self.current_step = min(self.current_step + 1, self.max_step - 10)
                attempts += 1
        
        # If we still have issues, return a safe default observation
        safe_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return safe_obs, {}

    def _get_obs(self):
        features = []
        
        # Asset features
        for i, ticker in enumerate(self.tickers):
            df = self.data[ticker]
            row = df.iloc[self.current_step]
            
            # Safely extract and normalize features with NaN handling
            rsi = float(row['RSI']) if not pd.isna(row['RSI']) else 50.0
            returns = float(row['Returns']) if not pd.isna(row['Returns']) else 0.0
            bb_pos = float(row['BB_position']) if not pd.isna(row['BB_position']) else 0.5
            macd = float(row['MACD']) if not pd.isna(row['MACD']) else 0.0
            volatility = float(row['Volatility']) if not pd.isna(row['Volatility']) else 0.01
            beta = float(row['Beta']) if not pd.isna(row['Beta']) else 1.0
            close_price = float(row['Close']) if not pd.isna(row['Close']) else 100.0
            
            # Normalized features with safe divisions
            features.extend([
                np.clip(rsi / 100.0, 0.0, 1.0),  # RSI normalized to 0-1
                np.clip(np.tanh(returns * 100), -1.0, 1.0),  # Returns normalized
                np.clip(bb_pos, 0.0, 1.0),  # BB position clamped to 0-1
                np.clip(np.tanh(macd / (close_price + 1e-8)), -1.0, 1.0),  # Safe MACD normalization
                np.clip(np.tanh(volatility * 100), -1.0, 1.0),  # Volatility normalized
                np.clip(np.tanh(beta), -1.0, 1.0)  # Beta normalized
            ])
        
        # Portfolio features with safe calculations
        prices = []
        for t in self.tickers:
            price = self.data[t].iloc[self.current_step]['Close']
            prices.append(float(price) if not pd.isna(price) else 100.0)
        
        total_invested = np.sum(self.positions * np.array(prices))
        cash_ratio = np.clip(self.balance / (self.initial_balance + 1e-8), 0.0, 2.0)
        investment_ratio = np.clip(total_invested / (self.initial_balance + 1e-8), 0.0, 2.0)
        
        # Safe previous return calculation
        if self.prev_total_value > 0:
            prev_return = np.clip((self.total_value - self.prev_total_value) / self.prev_total_value, -1.0, 1.0)
        else:
            prev_return = 0.0
        
        features.extend([cash_ratio, investment_ratio, prev_return])
        
        # Final safety check - replace any remaining NaN or inf values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features

    def step(self, action):
        # Normalize action to ensure sum = 1 with safety checks
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)  # Ensure non-negative
        
        action_sum = np.sum(action)
        if action_sum > 1e-8:
            weights = action / action_sum
        else:
            # Default equal allocation if action is invalid
            weights = np.ones_like(action) / len(action)
        
        # Get current prices with safety checks
        prices = []
        for t in self.tickers:
            price = self.data[t].iloc[self.current_step]['Close']
            prices.append(float(price) if not pd.isna(price) and price > 0 else 1.0)
        prices = np.array(prices)
        
        # Calculate current portfolio value
        current_positions_value = np.sum(self.positions * prices)
        current_total_value = self.balance + current_positions_value
        
        # Ensure current_total_value is positive
        if current_total_value <= 0:
            current_total_value = self.initial_balance * 0.01  # Emergency minimum
        
        # Calculate target allocations
        target_values = current_total_value * weights
        target_positions = target_values[:-1] / (prices + 1e-8)  # Avoid division by zero
        target_cash = target_values[-1]
        
        # Calculate trades and transaction costs
        trades = np.abs(target_positions - self.positions)
        trade_costs = np.sum(trades * prices * self.transaction_cost)
        self.transaction_costs += trade_costs
        
        # Execute trades with safety bounds
        self.positions = np.clip(target_positions, 0, current_total_value / (prices + 1e-8))
        self.balance = max(0, target_cash - trade_costs)  # Ensure non-negative balance
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_step
        
        # Calculate new portfolio value
        if not done:
            new_prices = []
            for t in self.tickers:
                price = self.data[t].iloc[self.current_step]['Close']
                new_prices.append(float(price) if not pd.isna(price) and price > 0 else 1.0)
            new_prices = np.array(new_prices)
            new_total_value = self.balance + np.sum(self.positions * new_prices)
        else:
            new_total_value = current_total_value
        
        # Ensure new_total_value is positive
        if new_total_value <= 0:
            new_total_value = self.initial_balance * 0.01
        
        # Enhanced reward function with safe calculations
        if self.total_value > 0:
            returns = (new_total_value - self.total_value) / self.total_value
        else:
            returns = 0.0
        
        # Risk-adjusted reward with Sharpe-like calculation
        if len(self.value_history) > 5:
            recent_values = self.value_history[-5:]
            recent_returns = pd.Series(recent_values).pct_change().dropna()
            volatility = recent_returns.std() if len(recent_returns) > 1 else 0.01
            volatility = max(volatility, 1e-6)  # Minimum volatility
            reward = returns / volatility - 0.001  # Risk-adjusted with small penalty
        else:
            reward = returns
        
        # Clip reward to reasonable bounds
        reward = np.clip(reward, -1.0, 1.0)
        reward = float(reward) if not pd.isna(reward) else 0.0
        
        self.prev_total_value = self.total_value
        self.total_value = new_total_value
        self.value_history.append(new_total_value)
        
        # Update to return 5 values as per Gymnasium API (obs, reward, terminated, truncated, info)
        return self._get_obs(), reward, done, False, {'portfolio_value': new_total_value}

# ---------- Step 3: Benchmark Models ----------
class BuyHoldStrategy:
    """Equal-weight buy and hold benchmark"""
    def __init__(self, data, tickers, initial_balance=1e6):
        self.data = data
        self.tickers = tickers
        self.initial_balance = initial_balance
        
    def backtest(self):
        # Equal weight allocation
        n_assets = len(self.tickers)
        allocation_per_asset = self.initial_balance / n_assets
        
        start_prices = np.array([self.data[t].iloc[20]['Close'] for t in self.tickers])
        positions = allocation_per_asset / start_prices
        
        values = []
        for i in range(20, min(len(d) for d in self.data.values())):
            prices = np.array([self.data[t].iloc[i]['Close'] for t in self.tickers])
            portfolio_value = np.sum(positions * prices)
            values.append(portfolio_value)
        
        return values

class MomentumStrategy:
    """Simple momentum strategy benchmark"""
    def __init__(self, data, tickers, initial_balance=1e6, lookback=10):
        self.data = data
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.lookback = lookback
        
    def backtest(self):
        values = [self.initial_balance]
        cash = self.initial_balance
        positions = np.zeros(len(self.tickers))
        
        for i in range(20, min(len(d) for d in self.data.values()) - 1):
            # Calculate momentum scores
            momentum_scores = []
            prices = []
            
            for ticker in self.tickers:
                df = self.data[ticker]
                current_price = df.iloc[i]['Close']
                past_price = df.iloc[i - self.lookback]['Close']
                momentum = (current_price - past_price) / past_price
                momentum_scores.append(momentum)
                prices.append(current_price)
            
            momentum_scores = np.array(momentum_scores)
            prices = np.array(prices)
            
            # Sell current positions
            cash += np.sum(positions * prices)
            
            # Select top momentum stocks
            top_indices = np.argsort(momentum_scores)[-3:]  # Top 3 stocks
            allocation_per_stock = cash / 3
            
            new_positions = np.zeros(len(self.tickers))
            for idx in top_indices:
                new_positions[idx] = allocation_per_stock / prices[idx]
            
            positions = new_positions
            cash = 0
            
            # Calculate portfolio value
            next_prices = np.array([self.data[t].iloc[i + 1]['Close'] for t in self.tickers])
            portfolio_value = cash + np.sum(positions * next_prices)
            values.append(portfolio_value)
        
        return values

# ---------- Step 4: Training with Callback ----------
class TradingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.locals.get('episode_rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['episode_rewards'])

# ---------- Step 5: Main Training and Evaluation ----------
def main():
    print("Fetching Nifty100 data...")
    try:
        data, market_data = fetch_nifty100_data(sample_size=8)
        tickers = list(data.keys())
        print(f"Successfully fetched data for {len(tickers)} stocks: {tickers}")
        
        if len(tickers) < 3:
            print("Warning: Less than 3 stocks available. Trying with larger sample...")
            data, market_data = fetch_nifty100_data(sample_size=15)
            tickers = list(data.keys())
            
        if len(tickers) < 2:
            raise ValueError("Insufficient stocks downloaded. Please check your internet connection.")
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Falling back to simple data fetch...")
        
        # Fallback: Simple data fetch with basic error handling
        tickers = ['RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'TCS.NS', 'ITC.NS']
        data = {}
        
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="3mo", interval="1d", threads=False)
                if not df.empty and len(df) > 30:
                    # Handle multi-level columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    
                    # Ensure Close is 1-dimensional
                    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)
                    
                    # Simple technical indicators
                    try:
                        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
                    except:
                        df['RSI'] = 50.0
                    
                    df['BB_position'] = 0.5  # Neutral position
                    df['MACD'] = 0.0  # Neutral MACD
                    df['Returns'] = df['Close'].pct_change()
                    df['Volatility'] = df['Returns'].rolling(window=10).std().fillna(0.01)
                    df['Beta'] = 1.0  # Market beta
                    
                    # Fill NaN values
                    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    df.dropna(inplace=True)
                    
                    if len(df) > 25:
                        data[ticker] = df
                        print(f"Downloaded {ticker}: {len(df)} records")
            except Exception as ex:
                print(f"Failed to download {ticker}: {ex}")
                continue
        
        if len(data) == 0:
            raise ValueError("Could not download any stock data. Please check your internet connection and try again.")
        
        tickers = list(data.keys())
        market_data = None
    
    # Split data for training and testing
    train_data = {}
    test_data = {}
    
    for ticker in tickers:
        df = data[ticker]
        split_point = int(len(df) * 0.7)  # 70% for training
        train_data[ticker] = df.iloc[:split_point].copy()
        test_data[ticker] = df.iloc[split_point:].copy().reset_index(drop=True)
    
    # Training Environment
    print("\nTraining RL models...")
    train_env = DummyVecEnv([lambda: EnhancedTradingEnv(train_data, tickers)])
    
    # Train SAC model
    callback = TradingCallback()
    sac_model = SAC("MlpPolicy", train_env, verbose=1, 
                    learning_rate=3e-4, buffer_size=50000,
                    learning_starts=500, batch_size=128)
    sac_model.learn(total_timesteps=5000, callback=callback)
    
    # Train PPO model for comparison
    ppo_model = PPO("MlpPolicy", train_env, verbose=1,
                    learning_rate=3e-4, n_steps=1024, batch_size=32)
    ppo_model.learn(total_timesteps=5000)
    
    # Testing Environment
    print("\nEvaluating models...")
    test_env = DummyVecEnv([lambda: EnhancedTradingEnv(test_data, tickers)])
    
    # Evaluate SAC
    obs = test_env.reset()
    sac_values = []
    sac_rewards = []
    done = False
    
    while not done:
        action, _ = sac_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        sac_rewards.append(reward[0])
        sac_values.append(info[0]['portfolio_value'])
        done = done[0]  # Ensure loop stops
    
    # Evaluate PPO
    obs = test_env.reset()
    ppo_values = []
    ppo_rewards = []
    done = False
    
    while not done:
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        ppo_rewards.append(reward[0])
        ppo_values.append(info[0]['portfolio_value'])
        done = done[0]  # Ensure loop stops
    
    # Benchmark strategies
    print("Running benchmark strategies...")
    buy_hold_values = BuyHoldStrategy(test_data, tickers).backtest()
    momentum_values = MomentumStrategy(test_data, tickers).backtest()
    
    # Market benchmark (Nifty50 or first stock if market data unavailable)
    if market_data is not None and not market_data.empty:
        market_test = market_data.iloc[int(len(market_data) * 0.7):].copy()
        market_values = (market_test['Close'] / market_test['Close'].iloc[0] * 1e6).tolist()
    else:
        # Use first stock as market proxy if no market data
        first_stock = list(test_data.keys())[0]
        proxy_data = test_data[first_stock]
        market_values = (proxy_data['Close'] / proxy_data['Close'].iloc[0] * 1e6).tolist()
        print("Using first stock as market benchmark proxy")
    
    # ---------- Step 6: Comprehensive Analysis ----------
    results = {
        'SAC': sac_values,
        'PPO': ppo_values,
        'Buy & Hold': buy_hold_values,
        'Momentum': momentum_values,
        'Market Benchmark': market_values[:len(sac_values)]  # Align lengths
    }
    
    # Performance metrics calculation
    def calculate_metrics(values, initial_value=1e6):
        returns = pd.Series(values).pct_change().dropna()
        
        # Return metrics
        total_return = (values[-1] - initial_value) / initial_value
        annualized_return = (1 + total_return) ** (252 / len(values)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        running_max = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR (95%)
        var_95 = np.percentile(returns, 5)
        
        return {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'VaR 95% (%)': var_95 * 100
        }
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS RESULTS")
    print("="*80)
    
    metrics_df = pd.DataFrame()
    for strategy, values in results.items():
        metrics = calculate_metrics(values)
        metrics_df[strategy] = pd.Series(metrics)
        
        print(f"\n{strategy}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    # Statistical significance tests
    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*50)
    
    sac_returns = pd.Series(sac_values).pct_change().dropna()
    market_returns = pd.Series(results['Market Benchmark']).pct_change().dropna()
    
    # T-test for mean return difference
    if len(sac_returns) > 5 and len(market_returns) > 5:
        t_stat, p_value = stats.ttest_ind(sac_returns, market_returns)
        print(f"SAC vs Market Benchmark returns t-test: t-stat={t_stat:.3f}, p-value={p_value:.3f}")
    else:
        print("Insufficient data for statistical significance testing")
    
    # Visualization
    plt.figure(figsize=(15, 12))
    
    # Portfolio value comparison
    plt.subplot(2, 2, 1)
    for strategy, values in results.items():
        plt.plot(values, label=strategy, linewidth=2)
    plt.title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value (INR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns distribution
    plt.subplot(2, 2, 2)
    returns_data = []
    labels = []
    for strategy, values in results.items():
        returns = pd.Series(values).pct_change().dropna() * 100
        returns_data.append(returns)
        labels.append(strategy)
    
    plt.boxplot(returns_data, labels=labels)
    plt.title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Returns (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Drawdown analysis
    plt.subplot(2, 2, 3)
    for strategy, values in results.items():
        running_max = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - running_max) / running_max * 100
        plt.plot(drawdown, label=strategy, linewidth=2)
    plt.title('Drawdown Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Risk-Return scatter
    plt.subplot(2, 2, 4)
    for strategy in results.keys():
        metrics = metrics_df[strategy]
        plt.scatter(metrics['Volatility (%)'], metrics['Annualized Return (%)'], 
                   s=100, label=strategy)
        plt.annotate(strategy, 
                    (metrics['Volatility (%)'], metrics['Annualized Return (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Annualized Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    print(f"\nSaving results...")
    metrics_df.to_csv('trading_performance_metrics.csv')
    print("Results saved to 'trading_performance_metrics.csv'")
    
    return results, metrics_df

if __name__ == "__main__":
    results, metrics = main()