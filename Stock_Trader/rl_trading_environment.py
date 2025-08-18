import gymnasium as gym
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

class StockTradingEnvironment(gym.Env):
    """
    Stock Trading Environment based on the paper methodology
    
    Environment for Multiple Stocks with continuous action space following the paper:
    - State Space: [b_t, p_t, h_t, M_t, R_t, C_t, X_t] (181-dimensional for 30 stocks)
    - Action Space: Continuous actions for buying/selling each stock
    - Reward Function: Change in portfolio value minus transaction costs
    """
    
    def __init__(self, 
                 stock_data: Dict[str, pd.DataFrame],
                 initial_balance: float = 1_000_000,
                 transaction_cost_rate: float = 0.001,  # 0.1% as in paper
                 turbulence_threshold: float = 140,
                 tech_indicators: List[str] = ['RSI', 'MACD', 'CCI', 'ADX'],
                 hmax: int = 100,  # Maximum shares per action
                 lookback_window: int = 20):
        """
        Initialize the trading environment
        
        Args:
            stock_data: Dictionary of stock DataFrames with technical indicators
            initial_balance: Starting cash amount
            transaction_cost_rate: Transaction cost as percentage of trade value
            turbulence_threshold: Threshold for market crash detection
            tech_indicators: List of technical indicators to include in state
            hmax: Maximum number of shares per single action
            lookback_window: Window for calculating historical statistics
        """
        super().__init__()
        
        self.stock_data = stock_data
        self.stock_list = list(stock_data.keys())
        self.n_stocks = len(self.stock_list)
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.turbulence_threshold = turbulence_threshold
        self.tech_indicators = tech_indicators
        self.hmax = hmax
        self.lookback_window = lookback_window
        
        # Find common date range across all stocks
        self._find_common_dates()
        
        # Initialize environment state
        self.current_step = 0
        self.day = self.lookback_window  # Start after lookback period
        
        # State space: [balance, prices, holdings, technical_indicators]
        # Following paper: [b_t, p_t, h_t, M_t, R_t, C_t, X_t]
        self.state_dim = 1 + self.n_stocks * (2 + len(self.tech_indicators))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Action space: normalized to [-1, 1] for each stock as per paper
        # Actions represent number of shares to buy/sell (normalized)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        # Initialize portfolio state
        self.reset()
    
    def _find_common_dates(self):
        """Find common date range across all stocks"""
        date_ranges = []
        for stock_df in self.stock_data.values():
            date_ranges.append(set(stock_df.index))
        
        # Find intersection of all date ranges
        common_dates = set.intersection(*date_ranges)
        self.common_dates = sorted(list(common_dates))
        
        # If insufficient common dates, use union approach with minimum coverage
        min_required_days = self.lookback_window + 30  # Reduced minimum requirement
        
        if len(self.common_dates) < min_required_days:
            print(f"Warning: Only {len(self.common_dates)} common dates found across all stocks.")
            print("Using union approach with stocks that have sufficient data...")
            
            # Find the date range that covers most stocks
            all_dates = set()
            for stock_df in self.stock_data.values():
                all_dates.update(stock_df.index)
            
            all_dates = sorted(list(all_dates))
            
            # Filter stocks to only include those with sufficient data coverage
            filtered_stock_data = {}
            for stock, stock_df in self.stock_data.items():
                # Check if this stock has enough data in the common period
                stock_dates = set(stock_df.index)
                overlap_with_all = len(stock_dates.intersection(set(all_dates[-100:])))  # Check last 100 days
                
                if len(stock_df) >= min_required_days and overlap_with_all >= 30:
                    filtered_stock_data[stock] = stock_df
            
            if len(filtered_stock_data) < 5:  # Need at least 5 stocks
                # Further reduce requirements for test data
                min_required_days = self.lookback_window + 10
                for stock, stock_df in self.stock_data.items():
                    if len(stock_df) >= min_required_days:
                        filtered_stock_data[stock] = stock_df
                        if len(filtered_stock_data) >= 5:
                            break
            
            if len(filtered_stock_data) == 0:
                raise ValueError(f"No stocks have sufficient data (minimum {min_required_days} days)")
            
            self.stock_data = filtered_stock_data
            self.stock_list = list(filtered_stock_data.keys())
            self.n_stocks = len(self.stock_list)
            
            print(f"Using {len(self.stock_list)} stocks with sufficient data: {self.stock_list[:5]}...")
            
            # Recalculate common dates with filtered stocks
            date_ranges = []
            for stock_df in self.stock_data.values():
                date_ranges.append(set(stock_df.index))
            
            common_dates = set.intersection(*date_ranges)
            self.common_dates = sorted(list(common_dates))
            
            if len(self.common_dates) < self.lookback_window + 5:
                # Use the maximum available date range
                self.common_dates = sorted(list(set.union(*date_ranges)))
                print(f"Using union of dates: {len(self.common_dates)} total days")
        
        print(f"Final common trading days: {len(self.common_dates)}")
        
        # Filter stock data to common dates (only for intersection approach)
        if len(set.intersection(*[set(df.index) for df in self.stock_data.values()])) >= self.lookback_window + 5:
            for stock in self.stock_list:
                self.stock_data[stock] = self.stock_data[stock].loc[self.common_dates]
    
    def _calculate_turbulence(self, day: int) -> float:
        """
        Calculate turbulence index as per paper equation (3):
        turbulence_t = (y_t - μ) * Σ^(-1) * (y_t - μ)^T
        
        Where:
        - y_t: current period returns
        - μ: average historical returns  
        - Σ: covariance matrix of historical returns
        """
        if day < self.lookback_window:
            return 0.0
        
        # Get current returns
        current_returns = []
        for stock in self.stock_list:
            current_price = self.stock_data[stock].iloc[day]['Close']
            prev_price = self.stock_data[stock].iloc[day-1]['Close']
            current_returns.append((current_price - prev_price) / prev_price)
        
        current_returns = np.array(current_returns)
        
        # Get historical returns for covariance calculation
        historical_returns = []
        for i in range(day - self.lookback_window, day):
            day_returns = []
            for stock in self.stock_list:
                price = self.stock_data[stock].iloc[i]['Close']
                prev_price = self.stock_data[stock].iloc[i-1]['Close']
                day_returns.append((price - prev_price) / prev_price)
            historical_returns.append(day_returns)
        
        historical_returns = np.array(historical_returns)
        
        # Calculate mean and covariance
        mu = np.mean(historical_returns, axis=0)
        sigma = np.cov(historical_returns.T)
        
        # Add small regularization to avoid singular matrix
        sigma += np.eye(len(sigma)) * 1e-8
        
        try:
            # Calculate turbulence
            diff = current_returns - mu
            turbulence = np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)
            return float(turbulence)
        except:
            return 0.0
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state following paper's state space definition:
        [b_t, p_t, h_t, M_t, R_t, C_t, X_t]
        
        Returns:
            State vector of dimension (1 + n_stocks * (2 + n_indicators))
        """
        state = []
        
        # Balance (normalized by initial balance)
        state.append(self.balance / self.initial_balance)
        
        # For each stock: price, holdings, technical indicators
        for i, stock in enumerate(self.stock_list):
            stock_df = self.stock_data[stock]
            current_data = stock_df.iloc[self.day]
            
            # Stock price (normalized by first price)
            first_price = stock_df.iloc[0]['Close']
            current_price = current_data['Close']
            state.append(current_price / first_price)
            
            # Holdings (normalized by max possible holdings)
            max_holdings = self.initial_balance / first_price
            state.append(self.holdings[i] / max_holdings)
            
            # Technical indicators (already normalized in data processing)
            for indicator in self.tech_indicators:
                if indicator in current_data:
                    value = current_data[indicator]
                    if pd.isna(value):
                        value = 0.0
                    # Normalize indicators to [-1, 1] range
                    if indicator == 'RSI':
                        state.append((value - 50) / 50)  # RSI: 0-100 -> -1 to 1
                    elif indicator == 'MACD':
                        state.append(np.tanh(value / current_price))  # MACD normalization
                    elif indicator == 'CCI':
                        state.append(np.tanh(value / 100))  # CCI normalization
                    elif indicator == 'ADX':
                        state.append((value - 25) / 25)  # ADX: 0-100, neutral at 25
                    else:
                        state.append(np.tanh(value))  # Default tanh normalization
                else:
                    state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_stocks)  # Number of shares for each stock
        self.day = self.lookback_window
        self.current_step = 0
        
        # Portfolio tracking
        self.portfolio_value_history = [self.initial_balance]
        self.transaction_costs_total = 0.0
        self.trades_count = 0
        
        # Get initial state
        state = self._get_state()
        info = {'day': self.day, 'portfolio_value': self.initial_balance}
        
        return state, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute trading actions following the paper's methodology
        
        Args:
            actions: Array of normalized actions [-1, 1] for each stock
            
        Returns:
            next_state, reward, terminated, truncated, info
        """
        # Ensure actions are in valid range
        actions = np.clip(actions, -1, 1)
        
        # Calculate current portfolio value
        current_prices = []
        for i, stock in enumerate(self.stock_list):
            price = self.stock_data[stock].iloc[self.day]['Close']
            current_prices.append(price)
        
        current_prices = np.array(current_prices)
        current_portfolio_value = self.balance + np.sum(self.holdings * current_prices)
        
        # Check for turbulence (market crash detection)
        turbulence = self._calculate_turbulence(self.day)
        market_crash = turbulence > self.turbulence_threshold
        
        if market_crash:
            # Emergency sell all positions as per paper
            sell_proceeds = np.sum(self.holdings * current_prices)
            transaction_costs = sell_proceeds * self.transaction_cost_rate
            self.balance += sell_proceeds - transaction_costs
            self.holdings = np.zeros(self.n_stocks)
            self.transaction_costs_total += transaction_costs
            
            reward_info = {
                'emergency_sell': True,
                'turbulence': turbulence,
                'transaction_costs': transaction_costs
            }
        else:
            # Normal trading
            # Convert normalized actions to actual share amounts
            share_actions = (actions * self.hmax).astype(int)
            
            # Execute trades
            transaction_costs = 0.0
            for i, shares_to_trade in enumerate(share_actions):
                current_price = current_prices[i]
                
                if shares_to_trade > 0:  # Buy
                    max_shares_affordable = int(self.balance / current_price)
                    shares_to_buy = min(shares_to_trade, max_shares_affordable)
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        trade_cost = cost * self.transaction_cost_rate
                        
                        if self.balance >= cost + trade_cost:
                            self.balance -= cost + trade_cost
                            self.holdings[i] += shares_to_buy
                            transaction_costs += trade_cost
                            self.trades_count += 1
                
                elif shares_to_trade < 0:  # Sell
                    shares_to_sell = min(-shares_to_trade, int(self.holdings[i]))
                    
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_price
                        trade_cost = proceeds * self.transaction_cost_rate
                        
                        self.balance += proceeds - trade_cost
                        self.holdings[i] -= shares_to_sell
                        transaction_costs += trade_cost
                        self.trades_count += 1
            
            self.transaction_costs_total += transaction_costs
            
            reward_info = {
                'emergency_sell': False,
                'turbulence': turbulence,
                'transaction_costs': transaction_costs
            }
        
        # Move to next day
        self.day += 1
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.day < len(self.common_dates):
            new_prices = []
            for i, stock in enumerate(self.stock_list):
                price = self.stock_data[stock].iloc[self.day]['Close']
                new_prices.append(price)
            
            new_prices = np.array(new_prices)
            new_portfolio_value = self.balance + np.sum(self.holdings * new_prices)
        else:
            new_portfolio_value = current_portfolio_value
        
        # Calculate reward as change in portfolio value minus transaction costs
        # Following paper equation (4): r(s_t, a_t, s_{t+1}) = (portfolio_value_{t+1} - portfolio_value_t) - c_t
        reward = (new_portfolio_value - current_portfolio_value) / self.initial_balance
        
        # Add to portfolio history
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Check if episode is done
        terminated = self.day >= len(self.common_dates) - 1
        truncated = False
        
        # Get next state
        if not terminated:
            next_state = self._get_state()
        else:
            next_state = self._get_state()  # Return final state
        
        # Info dictionary
        info = {
            'day': self.day,
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'holdings': self.holdings.copy(),
            'transaction_costs': transaction_costs,
            'total_transaction_costs': self.transaction_costs_total,
            'trades_count': self.trades_count,
            'turbulence': turbulence,
            'market_crash': market_crash
        }
        info.update(reward_info)
        
        return next_state, reward, terminated, truncated, info
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if len(self.portfolio_value_history) < 2:
            return {}
        
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        # Performance metrics following the paper
        total_return = (values[-1] - values[0]) / values[0]
        
        if len(returns) > 1:
            annual_return = np.mean(returns) * 252  # 252 trading days per year
            annual_volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            annual_return = 0
            annual_volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_transaction_costs': self.transaction_costs_total,
            'trades_count': self.trades_count
        }
