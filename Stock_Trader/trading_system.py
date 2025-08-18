#!/usr/bin/env python3
"""
Comprehensive Deep Reinforcement Learning Trading System

This script implements the complete trading system from the paper:
"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"

Features:
- Data collection and preprocessing
- Environment setup with proper MDP formulation
- Training of PPO, A2C, DDPG algorithms
- Ensemble strategy with Sharpe ratio selection
- Comprehensive testing and evaluation
- Robust error handling and recovery
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Deep Learning imports
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Data and TA imports
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.trend import CCIIndicator

# Local imports
from rl_trading_environment import StockTradingEnvironment

warnings.filterwarnings("ignore")

class TradingSystem:
    """Complete Deep RL Trading System Implementation"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the trading system with configuration"""
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize directories
        self._setup_directories()
        
        # Algorithm configurations
        self.algorithms = {
            'PPO': {
                'class': PPO,
                'params': {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'verbose': 0
                }
            },
            'A2C': {
                'class': A2C,
                'params': {
                    'policy': 'MlpPolicy',
                    'learning_rate': 7e-4,
                    'n_steps': 5,
                    'gamma': 0.99,
                    'gae_lambda': 1.0,
                    'ent_coef': 0.01,
                    'vf_coef': 0.25,
                    'max_grad_norm': 0.5,
                    'verbose': 0
                }
            },
            'DDPG': {
                'class': DDPG,
                'params': {
                    'policy': 'MlpPolicy',
                    'learning_rate': 1e-3,
                    'buffer_size': 1000000,
                    'learning_starts': 100,
                    'batch_size': 100,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'verbose': 0
                }
            }
        }
        
        # Results storage
        self.training_results = {}
        self.validation_results = {}
        self.test_results = {}
        self.benchmark_results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'initial_balance': 1_000_000,
            'transaction_cost_rate': 0.001,
            'turbulence_threshold': 140,
            'lookback_window': 20,
            'training_timesteps': 50000,
            'validation_episodes': 10,
            'test_episodes': 1,
            'top_stocks': 30,  # Limit to top stocks for faster training
            'train_split': 0.7,
            'val_split': 0.2,
            'data_start_date': '2014-01-01',
            'data_end_date': '2024-12-31',
            'min_data_length': 100,
            'use_pretrained': False,
            'models_dir': 'models',
            'best_algorithm': None,
            'run_benchmarks': True,
            'device': 'auto',
            'n_envs': 1,
            'net_arch': None,
            'ppo_n_steps': None,
            'ppo_batch_size': None,
            'ppo_n_epochs': None,
            'a2c_n_steps': None
        }
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = ['data', 'data/processed', 'models', 'results', 'logs']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def collect_data(self) -> Dict[str, pd.DataFrame]:
        """Collect and preprocess stock data"""
        print("🔄 COLLECTING STOCK DATA")
        print("=" * 50)
        
        # NIFTY100 tickers (top performing ones)
        nifty100_tickers = [
            'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
            'HINDUNILVR.NS', 'BHARTIARTL.NS', 'SBIN.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
            'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'TATAMOTORS.NS', 'NTPC.NS',
            'ULTRACEMCO.NS', 'JSWSTEEL.NS', 'BAJAJFINSV.NS', 'WIPRO.NS',
            'ONGC.NS', 'NESTLEIND.NS', 'ADANIPORTS.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
            'GRASIM.NS', 'COALINDIA.NS', 'HINDALCO.NS', 'DIVISLAB.NS', 'TECHM.NS'
        ]
        
        # Select top stocks based on config
        selected_tickers = nifty100_tickers[:self.config['top_stocks']]
        
        print(f"📈 Downloading data for {len(selected_tickers)} stocks...")
        print(f"📅 Date range: {self.config['data_start_date']} to {self.config['data_end_date']}")
        
        stock_data = {}
        successful_downloads = 0
        
        # Download market benchmark
        try:
            print("📊 Downloading market benchmark (NIFTY50)...")
            market_data = yf.download('^NSEI', 
                                    start=self.config['data_start_date'],
                                    end=self.config['data_end_date'],
                                    interval="1d", auto_adjust=True, prepost=True, threads=False)
            
            if not market_data.empty:
                if isinstance(market_data.columns, pd.MultiIndex):
                    market_data.columns = market_data.columns.droplevel(1)
                market_data['Close'] = pd.Series(market_data['Close'].values.flatten(), index=market_data.index)
                print(f"✅ Market data: {len(market_data)} records")
            else:
                market_data = None
                print("⚠️  Market data unavailable")
        except Exception as e:
            print(f"⚠️  Market data error: {e}")
            market_data = None
        
        # Download individual stocks
        for i, ticker in enumerate(tqdm(selected_tickers, desc="Downloading stocks")):
            try:
                df = yf.download(ticker,
                               start=self.config['data_start_date'],
                               end=self.config['data_end_date'],
                               interval="1d", auto_adjust=True, prepost=True, threads=False)
                
                if df.empty or len(df) < self.config['min_data_length']:
                    continue
                
                # Process stock data
                processed_df = self._process_stock_data(df, ticker, market_data)
                
                if len(processed_df) >= self.config['min_data_length']:
                    stock_data[ticker] = processed_df
                    successful_downloads += 1
                    
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Failed to download {ticker}: {e}")
                continue
        
        print(f"✅ Successfully collected data for {successful_downloads} stocks")
        
        if len(stock_data) < 5:
            raise ValueError("Insufficient stock data collected. Need at least 5 stocks.")
        
        # Save processed data
        self._save_processed_data(stock_data)
        
        return stock_data
    
    def _process_stock_data(self, df: pd.DataFrame, ticker: str, market_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Process individual stock data with technical indicators"""
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} for {ticker}")
        
        # Convert to proper format
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.Series(df[col].values.flatten(), index=df.index)
        
        # Technical indicators with error handling
        try:
            df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        except:
            df['RSI'] = 50.0
        
        try:
            bb = BollingerBands(close=df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        except:
            df['BB_upper'] = df['Close'] * 1.02
            df['BB_lower'] = df['Close'] * 0.98
            df['BB_position'] = 0.5
        
        try:
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
        except:
            df['MACD'] = 0.0
            df['MACD_signal'] = 0.0
            df['MACD_diff'] = 0.0
        
        try:
            df['CCI'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
        except:
            df['CCI'] = 0.0
        
        try:
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
            df['ADX'] = adx.adx()
        except:
            df['ADX'] = 25.0
        
        # Returns and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Market correlation
        if market_data is not None and not market_data.empty:
            try:
                market_close = pd.Series(market_data['Close'].values.flatten(), index=market_data.index)
                market_returns = market_close.pct_change()
                
                # Align indices
                common_idx = df.index.intersection(market_returns.index)
                if len(common_idx) > 20:
                    stock_returns = df.loc[common_idx, 'Returns']
                    market_returns_aligned = market_returns.loc[common_idx]
                    
                    df['Market_Returns'] = np.nan
                    df.loc[common_idx, 'Market_Returns'] = market_returns_aligned
                    
                    # Calculate Beta
                    df['Beta'] = df['Returns'].rolling(window=20).cov(df['Market_Returns']) / \
                                df['Market_Returns'].rolling(window=20).var()
                else:
                    df['Market_Returns'] = df['Returns'] * 0.8
                    df['Beta'] = 1.0
            except:
                df['Market_Returns'] = df['Returns'] * 0.8
                df['Beta'] = 1.0
        else:
            df['Market_Returns'] = df['Returns'] * 0.8
            df['Beta'] = 1.0
        
        # Turbulence index (simplified)
        try:
            df['Turbulence'] = df['Volatility'] / df['Volatility'].rolling(window=252).mean()
        except:
            df['Turbulence'] = 1.0
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        df.dropna(inplace=True)
        
        return df
    
    def _save_processed_data(self, stock_data: Dict[str, pd.DataFrame]):
        """Save processed data to files"""
        print("💾 Saving processed data...")
        
        for ticker, df in stock_data.items():
            filename = ticker.replace('.NS', '') + '.csv'
            df.to_csv(f'data/processed/{filename}')
        
        # Save summary
        summary = {
            'ticker': [],
            'records': [],
            'start_date': [],
            'end_date': [],
            'avg_price': [],
            'volatility': []
        }
        
        for ticker, df in stock_data.items():
            summary['ticker'].append(ticker)
            summary['records'].append(len(df))
            summary['start_date'].append(df.index[0].strftime('%Y-%m-%d'))
            summary['end_date'].append(df.index[-1].strftime('%Y-%m-%d'))
            summary['avg_price'].append(df['Close'].mean())
            summary['volatility'].append(df['Volatility'].mean())
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('data/processed/data_summary.csv', index=False)
        print(f"✅ Data saved for {len(stock_data)} stocks")
    
    def load_or_collect_data(self) -> Dict[str, pd.DataFrame]:
        """Load existing data or collect new data"""
        processed_dir = 'data/processed'
        
        if os.path.exists(f'{processed_dir}/data_summary.csv'):
            print("📂 Loading existing processed data...")
            stock_data = {}
            
            summary_df = pd.read_csv(f'{processed_dir}/data_summary.csv')
            
            for _, row in summary_df.iterrows():
                ticker = row['ticker']
                filename = ticker.replace('.NS', '') + '.csv'
                filepath = f'{processed_dir}/{filename}'
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    if len(df) >= self.config['min_data_length']:
                        stock_data[ticker] = df
            
            if len(stock_data) >= 5:
                print(f"✅ Loaded {len(stock_data)} stocks from cache")
                return stock_data
        
        print("🔄 No suitable cached data found, collecting fresh data...")
        return self.collect_data()
    
    def split_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        """Split data into train, validation, and test sets"""
        print("📊 Splitting data into train/validation/test sets...")
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        train_split = self.config['train_split']
        val_split = self.config['val_split']
        
        for ticker, df in stock_data.items():
            n = len(df)
            train_end = int(n * train_split)
            val_end = int(n * (train_split + val_split))
            
            train_data[ticker] = df.iloc[:train_end].copy()
            val_data[ticker] = df.iloc[train_end:val_end].copy()
            test_data[ticker] = df.iloc[val_end:].copy()
        
        print(f"✅ Data split completed:")
        print(f"   📚 Training: {len(train_data)} stocks")
        print(f"   🔍 Validation: {len(val_data)} stocks") 
        print(f"   🧪 Testing: {len(test_data)} stocks")
        
        return train_data, val_data, test_data
    
    def train_algorithms(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all three algorithms"""
        print("\n🚀 TRAINING ALGORITHMS")
        print("=" * 50)
        
        print("🏗️  Creating training environment...")
        def make_env():
            return StockTradingEnvironment(
                stock_data=train_data,
                initial_balance=self.config['initial_balance'],
                transaction_cost_rate=self.config['transaction_cost_rate'],
                turbulence_threshold=self.config['turbulence_threshold'],
                lookback_window=self.config['lookback_window']
            )
        n_envs = max(1, int(self.config.get('n_envs', 1)))
        if n_envs > 1:
            vec_env = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])
        else:
            vec_env = DummyVecEnv([lambda: make_env()])
        
        print(f"✅ Environment created:")
        temp_env = make_env()
        try:
            print(f"   📈 Stocks: {temp_env.n_stocks}")
            print(f"   📊 Observation space: {temp_env.observation_space.shape}")
            print(f"   🎯 Action space: {temp_env.action_space.shape}")
            print(f"   📅 Trading days: {len(temp_env.common_dates)}")
        finally:
            try:
                temp_env.close()
            except Exception:
                pass
        
        trained_models = {}
        training_times = {}
        
        # Train each algorithm
        for algo_name, algo_config in self.algorithms.items():
            print(f"\n🔧 Training {algo_name}...")
            start_time = time.time()
            
            try:
                # Initialize model
                AlgoClass = algo_config['class']
                params = algo_config['params'].copy()
                # Apply optional overrides
                if self.config.get('net_arch'):
                    params['policy_kwargs'] = params.get('policy_kwargs', {})
                    params['policy_kwargs']['net_arch'] = self.config['net_arch']
                if algo_name == 'PPO':
                    if self.config.get('ppo_n_steps'):
                        params['n_steps'] = int(self.config['ppo_n_steps'])
                    if self.config.get('ppo_batch_size'):
                        params['batch_size'] = int(self.config['ppo_batch_size'])
                    if self.config.get('ppo_n_epochs'):
                        params['n_epochs'] = int(self.config['ppo_n_epochs'])
                if algo_name == 'A2C':
                    if self.config.get('a2c_n_steps'):
                        params['n_steps'] = int(self.config['a2c_n_steps'])

                model = AlgoClass(env=vec_env, device=self.config.get('device', 'auto'), **params)
                
                # Train model
                model.learn(total_timesteps=self.config['training_timesteps'])
                
                # Save model
                model_path = f"models/{algo_name}_model.zip"
                model.save(model_path)
                
                trained_models[algo_name] = model
                training_times[algo_name] = time.time() - start_time
                
                print(f"✅ {algo_name} training completed in {training_times[algo_name]:.1f}s")
                
            except Exception as e:
                print(f"❌ {algo_name} training failed: {e}")
                continue
        
        if len(trained_models) == 0:
            raise RuntimeError("No algorithms trained successfully")
        
        print(f"\n✅ Training completed for {len(trained_models)} algorithms")
        self.training_results = {'models': trained_models, 'times': training_times}
        
        return trained_models

    def load_pretrained_models(self, vec_env: DummyVecEnv) -> Dict[str, Any]:
        """Load pre-trained models from disk"""
        print("\n📦 LOADING PRE-TRAINED MODELS")
        print("=" * 50)
        models_dir = self.config.get('models_dir', 'models')
        loaded_models = {}
        for algo_name, algo_config in self.algorithms.items():
            model_path = os.path.join(models_dir, f"{algo_name}_model.zip")
            if os.path.exists(model_path):
                try:
                    AlgoClass = algo_config['class']
                    model = AlgoClass.load(model_path, env=vec_env, device=self.config.get('device', 'auto'))
                    env_action_shape = vec_env.action_space.shape
                    env_obs_shape = vec_env.observation_space.shape
                    model_action_shape = getattr(model.action_space, 'shape', None)
                    model_obs_shape = getattr(model.observation_space, 'shape', None)
                    if (model_action_shape is not None and model_action_shape != env_action_shape) or \
                       (model_obs_shape is not None and model_obs_shape != env_obs_shape):
                        print(f"⚠️  Shape mismatch for {algo_name}: model action {model_action_shape}, env action {env_action_shape}; model obs {model_obs_shape}, env obs {env_obs_shape}. Skipping.")
                        continue
                    loaded_models[algo_name] = model
                    print(f"✅ Loaded {algo_name} from {model_path}")
                except Exception as e:
                    print(f"❌ Failed to load {algo_name} from {model_path}: {e}")
            else:
                print(f"ℹ️  No saved model found for {algo_name} at {model_path}")
        if len(loaded_models) == 0:
            print("❌ No pre-trained models loaded.")
        else:
            print(f"\n✅ Loaded {len(loaded_models)} model(s) from '{models_dir}'")
        return loaded_models
    
    def validate_algorithms(self, trained_models: Dict[str, Any], val_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Validate all trained algorithms"""
        print("\n🔍 VALIDATING ALGORITHMS")
        print("=" * 50)
        
        validation_results = {}
        
        # Create validation environment
        val_env = StockTradingEnvironment(
            stock_data=val_data,
            initial_balance=self.config['initial_balance'],
            transaction_cost_rate=self.config['transaction_cost_rate'],
            turbulence_threshold=self.config['turbulence_threshold'],
            lookback_window=self.config['lookback_window']
        )
        
        print(f"🏗️  Validation environment: {val_env.n_stocks} stocks, {len(val_env.common_dates)} days")
        
        for algo_name, model in trained_models.items():
            print(f"\n📊 Validating {algo_name}...")
            
            try:
                portfolio_values = []
                episode_rewards = []
                
                for episode in range(self.config['validation_episodes']):
                    obs, _ = val_env.reset()
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = val_env.step(action)
                        done = terminated or truncated
                        episode_reward += reward
                    
                    portfolio_values.append(info['portfolio_value'])
                    episode_rewards.append(episode_reward)
                
                # Calculate metrics
                metrics = val_env.get_portfolio_metrics()
                metrics.update({
                    'mean_episode_reward': np.mean(episode_rewards),
                    'std_episode_reward': np.std(episode_rewards),
                    'mean_portfolio_value': np.mean(portfolio_values),
                    'validation_episodes': self.config['validation_episodes']
                })
                
                validation_results[algo_name] = metrics
                
                print(f"✅ {algo_name} validation completed:")
                print(f"   📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   💰 Avg Portfolio: ${metrics.get('mean_portfolio_value', 0):,.0f}")
                print(f"   📊 Total Return: {metrics.get('total_return', 0)*100:+.2f}%")
                
            except Exception as e:
                print(f"❌ {algo_name} validation failed: {e}")
                validation_results[algo_name] = {'sharpe_ratio': -np.inf, 'error': str(e)}
        
        self.validation_results = validation_results
        return validation_results
    
    def select_best_algorithm(self, validation_results: Dict[str, Dict]) -> str:
        """Select best algorithm based on Sharpe ratio"""
        print("\n🏆 SELECTING BEST ALGORITHM")
        print("=" * 50)
        
        best_algo = None
        best_sharpe = -np.inf
        
        print("📊 Validation Results Summary:")
        for algo_name, metrics in validation_results.items():
            sharpe = metrics.get('sharpe_ratio', -np.inf)
            total_return = metrics.get('total_return', 0) * 100
            
            print(f"   {algo_name:>6}: Sharpe={sharpe:>7.4f}, Return={total_return:>+6.2f}%")
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_algo = algo_name
        
        print(f"\n🥇 Best Algorithm: {best_algo} (Sharpe: {best_sharpe:.4f})")
        return best_algo
    
    def test_ensemble_strategy(self, trained_models: Dict[str, Any], best_algo: str, test_data: Dict[str, pd.DataFrame]) -> Dict:
        """Test the ensemble strategy (using best algorithm)"""
        print(f"\n🧪 TESTING ENSEMBLE STRATEGY")
        print("=" * 50)
        
        # Create test environment
        test_env = StockTradingEnvironment(
            stock_data=test_data,
            initial_balance=self.config['initial_balance'],
            transaction_cost_rate=self.config['transaction_cost_rate'],
            turbulence_threshold=self.config['turbulence_threshold'],
            lookback_window=self.config['lookback_window']
        )
        
        print(f"🏗️  Test environment: {test_env.n_stocks} stocks, {len(test_env.common_dates)} days")
        
        # Test best model
        best_model = trained_models[best_algo]
        print(f"🚀 Testing {best_algo} model...")
        
        try:
            obs, _ = test_env.reset()
            portfolio_history = [test_env.initial_balance]
            actions_history = []
            rewards_history = []
            done = False
            step_count = 0
            
            while not done:
                action, _ = best_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
                portfolio_history.append(info['portfolio_value'])
                actions_history.append(action.copy())
                rewards_history.append(reward)
                step_count += 1
                
                if step_count % 20 == 0:
                    pnl = info['portfolio_value'] - test_env.initial_balance
                    print(f"   Step {step_count:3d}: Portfolio=${info['portfolio_value']:,.0f} (PnL=${pnl:+,.0f})")
            
            # Calculate test metrics
            test_metrics = test_env.get_portfolio_metrics()
            
            test_results = {
                'best_algorithm': best_algo,
                'portfolio_history': portfolio_history,
                'actions_history': actions_history,
                'rewards_history': rewards_history,
                'metrics': test_metrics,
                'steps_executed': step_count
            }
            
            print(f"✅ Testing completed: {step_count} steps")
            print(f"   💰 Final Portfolio: ${portfolio_history[-1]:,.2f}")
            print(f"   📈 Total Return: {test_metrics.get('total_return', 0)*100:+.2f}%")
            print(f"   📊 Sharpe Ratio: {test_metrics.get('sharpe_ratio', 0):.4f}")
            
            self.test_results = test_results
            return test_results
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return {'error': str(e)}
    
    def run_benchmarks(self, test_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run benchmark strategies"""
        print(f"\n📊 RUNNING BENCHMARKS")
        print("=" * 50)
        
        benchmarks = {}
        
        # Buy and Hold Strategy
        print("📈 Running Buy & Hold benchmark...")
        try:
            test_env = StockTradingEnvironment(
                stock_data=test_data,
                initial_balance=self.config['initial_balance'],
                transaction_cost_rate=self.config['transaction_cost_rate'],
                turbulence_threshold=self.config['turbulence_threshold'],
                lookback_window=self.config['lookback_window']
            )
            
            obs, _ = test_env.reset()
            # Equal weight buy at start
            buy_action = np.ones(len(test_env.stock_list)) * 0.5
            obs, _, _, _, info = test_env.step(buy_action)
            
            buy_hold_values = [info['portfolio_value']]
            
            # Hold for rest of period
            while True:
                hold_action = np.zeros(len(test_env.stock_list))
                obs, _, terminated, truncated, info = test_env.step(hold_action)
                buy_hold_values.append(info['portfolio_value'])
                if terminated or truncated:
                    break
            
            buy_hold_metrics = test_env.get_portfolio_metrics()
            benchmarks['Buy_Hold'] = {
                'portfolio_history': buy_hold_values,
                'metrics': buy_hold_metrics
            }
            
            print(f"✅ Buy & Hold: {buy_hold_metrics.get('total_return', 0)*100:+.2f}% return")
            
        except Exception as e:
            print(f"❌ Buy & Hold failed: {e}")
        
        self.benchmark_results = benchmarks
        return benchmarks
    
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        # Compile all results
        all_results = {}
        
        # Add ensemble results
        if hasattr(self, 'test_results') and 'metrics' in self.test_results:
            all_results['Ensemble_Strategy'] = self.test_results['metrics']
        
        # Add individual validation results
        if hasattr(self, 'validation_results'):
            for algo, metrics in self.validation_results.items():
                if 'error' not in metrics:
                    all_results[f'{algo}_Validation'] = metrics
        
        # Add benchmark results
        if hasattr(self, 'benchmark_results'):
            for bench_name, bench_data in self.benchmark_results.items():
                all_results[bench_name] = bench_data['metrics']
        
        # Create summary DataFrame
        if all_results:
            summary_df = pd.DataFrame(all_results).T
            summary_df.to_csv('results/comprehensive_performance_summary.csv')
            
            print("📊 Performance Summary:")
            print(summary_df[['total_return', 'sharpe_ratio', 'max_drawdown']].round(4))
        
        # Create visualizations
        self._create_comprehensive_visualizations()
        
        # Save detailed results
        self._save_detailed_results()
        
        print("✅ Comprehensive report generated!")
        print("📁 Check 'results/' folder for detailed outputs")
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualization plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Portfolio performance comparison
            ax1 = axes[0, 0]
            if hasattr(self, 'test_results') and 'portfolio_history' in self.test_results:
                days = range(len(self.test_results['portfolio_history']))
                ax1.plot(days, self.test_results['portfolio_history'], 
                        label=f"Ensemble ({self.test_results['best_algorithm']})", linewidth=2)
            
            if hasattr(self, 'benchmark_results'):
                for bench_name, bench_data in self.benchmark_results.items():
                    if 'portfolio_history' in bench_data:
                        days = range(len(bench_data['portfolio_history']))
                        ax1.plot(days, bench_data['portfolio_history'], 
                                label=bench_name, alpha=0.7)
            
            ax1.set_title('Portfolio Value Comparison')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Algorithm validation comparison
            ax2 = axes[0, 1]
            if hasattr(self, 'validation_results'):
                algos = []
                sharpe_ratios = []
                
                for algo, metrics in self.validation_results.items():
                    if 'error' not in metrics:
                        algos.append(algo)
                        sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                
                if algos:
                    bars = ax2.bar(algos, sharpe_ratios, alpha=0.7)
                    ax2.set_title('Validation Sharpe Ratios')
                    ax2.set_ylabel('Sharpe Ratio')
                    ax2.grid(True, alpha=0.3)
                    
                    # Highlight best
                    if hasattr(self, 'test_results') and 'best_algorithm' in self.test_results:
                        best_algo = self.test_results['best_algorithm']
                        if best_algo in algos:
                            best_idx = algos.index(best_algo)
                            bars[best_idx].set_color('red')
                            bars[best_idx].set_alpha(1.0)
            
            # Returns distribution
            ax3 = axes[1, 0]
            if hasattr(self, 'test_results') and 'portfolio_history' in self.test_results:
                portfolio_values = self.test_results['portfolio_history']
                if len(portfolio_values) > 1:
                    returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    ax3.hist(returns, bins=30, alpha=0.7, density=True)
                    ax3.set_title('Daily Returns Distribution (Ensemble)')
                    ax3.set_xlabel('Daily Returns')
                    ax3.set_ylabel('Density')
                    ax3.grid(True, alpha=0.3)
            
            # Performance metrics radar (simplified bar chart)
            ax4 = axes[1, 1]
            if hasattr(self, 'test_results') and 'metrics' in self.test_results:
                metrics = self.test_results['metrics']
                metric_names = ['Total Return', 'Sharpe Ratio', 'Ann. Return']
                metric_values = [
                    metrics.get('total_return', 0),
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('annual_return', 0)
                ]
                
                ax4.bar(metric_names, metric_values, alpha=0.7)
                ax4.set_title('Ensemble Performance Metrics')
                ax4.set_ylabel('Value')
                ax4.grid(True, alpha=0.3)
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Visualizations saved to results/comprehensive_analysis.png")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    
    def _save_detailed_results(self):
        """Save detailed results to files"""
        try:
            # Save training results
            if hasattr(self, 'training_results'):
                with open('results/training_summary.txt', 'w') as f:
                    f.write("TRAINING SUMMARY\n")
                    f.write("=" * 50 + "\n")
                    if 'times' in self.training_results:
                        for algo, time_taken in self.training_results['times'].items():
                            f.write(f"{algo}: {time_taken:.1f} seconds\n")
            
            # Save validation results
            if hasattr(self, 'validation_results'):
                val_df = pd.DataFrame(self.validation_results).T
                val_df.to_csv('results/validation_detailed.csv')
            
            # Save test results
            if hasattr(self, 'test_results'):
                if 'portfolio_history' in self.test_results:
                    test_df = pd.DataFrame({
                        'Step': range(len(self.test_results['portfolio_history'])),
                        'Portfolio_Value': self.test_results['portfolio_history']
                    })
                    test_df.to_csv('results/test_portfolio_history.csv', index=False)
            
            print("💾 Detailed results saved to results/ folder")
            
        except Exception as e:
            print(f"⚠️  Save error: {e}")
    
    def run_complete_system(self, **kwargs):
        """Run the complete trading system"""
        start_time = time.time()
        
        print("🚀 DEEP REINFORCEMENT LEARNING TRADING SYSTEM")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update config with any provided arguments
        self.config.update(kwargs)
        
        try:
            # Step 1: Data Collection
            stock_data = self.load_or_collect_data()
            
            # Step 2: Data Splitting
            train_data, val_data, test_data = self.split_data(stock_data)
            
            # Step 3: Algorithm Training or Loading
            print("\n🔧 MODEL PREPARATION")
            print("=" * 50)
            train_env = StockTradingEnvironment(
                stock_data=train_data,
                initial_balance=self.config['initial_balance'],
                transaction_cost_rate=self.config['transaction_cost_rate'],
                turbulence_threshold=self.config['turbulence_threshold'],
                lookback_window=self.config['lookback_window']
            )
            vec_env = DummyVecEnv([lambda: train_env])

            if self.config.get('use_pretrained') or self.config.get('test_only'):
                trained_models = self.load_pretrained_models(vec_env)
                if len(trained_models) == 0:
                    raise RuntimeError("Requested to use pre-trained models, but none were loaded.")
            else:
                trained_models = self.train_algorithms(train_data)
            
            # Step 4: Validation (optional if best algo provided)
            if self.config.get('best_algorithm') in self.algorithms:
                best_algo = self.config['best_algorithm']
                print(f"\n🏆 Using specified best algorithm: {best_algo}")
                validation_results = self.validation_results or {}
            else:
                validation_results = self.validate_algorithms(trained_models, val_data)
                # Step 5: Best Algorithm Selection
                best_algo = self.select_best_algorithm(validation_results)
            
            # Step 6: Testing
            test_results = self.test_ensemble_strategy(trained_models, best_algo, test_data)
            
            # Step 7: Benchmarks
            if self.config.get('run_benchmarks', True):
                benchmark_results = self.run_benchmarks(test_data)
            else:
                benchmark_results = {}
            
            # Step 8: Comprehensive Report
            self.generate_comprehensive_report()
            
            # Summary
            total_time = time.time() - start_time
            print(f"\n🎉 SYSTEM EXECUTION COMPLETED!")
            print("=" * 60)
            print(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            print(f"🏆 Best Algorithm: {best_algo}")
            
            if 'metrics' in test_results:
                metrics = test_results['metrics']
                print(f"📈 Final Return: {metrics.get('total_return', 0)*100:+.2f}%")
                print(f"📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            
            print(f"📁 Results saved in 'results/' directory")
            
            return True
            
        except Exception as e:
            print(f"\n❌ SYSTEM EXECUTION FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Deep RL Trading System')
    parser.add_argument('--stocks', type=int, default=10, help='Number of top stocks to use')
    parser.add_argument('--timesteps', type=int, default=30000, help='Training timesteps per algorithm')
    parser.add_argument('--balance', type=float, default=1000000, help='Initial balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost rate')
    parser.add_argument('--use-pretrained', action='store_true', help='Load and use pre-trained models from models directory')
    parser.add_argument('--test-only', action='store_true', help='Skip training and only test using pre-trained models')
    parser.add_argument('--best-algo', type=str, choices=['PPO', 'A2C', 'DDPG'], help='Manually specify which algorithm to test')
    parser.add_argument('--no-benchmarks', action='store_true', help='Disable running benchmark strategies')
    parser.add_argument('--device', type=str, default='auto', help="Device for training/inference: 'cuda', 'cpu', or 'auto'")
    parser.add_argument('--envs', type=int, default=1, help='Number of parallel environments (SubprocVecEnv)')
    parser.add_argument('--net-arch', type=str, default=None, help='Comma-separated hidden sizes, e.g. 256,256 or 256,256,256')
    parser.add_argument('--ppo-n-steps', type=int, default=None, help='PPO rollout length (n_steps)')
    parser.add_argument('--ppo-batch-size', type=int, default=None, help='PPO batch size')
    parser.add_argument('--ppo-n-epochs', type=int, default=None, help='PPO epochs per update')
    parser.add_argument('--a2c-n-steps', type=int, default=None, help='A2C n_steps')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'top_stocks': args.stocks,
        'training_timesteps': args.timesteps,
        'initial_balance': args.balance,
        'transaction_cost_rate': args.transaction_cost,
        'use_pretrained': args.use_pretrained or args.test_only,
        'test_only': args.test_only,
        'best_algorithm': args.best_algo,
        'run_benchmarks': not args.no_benchmarks,
        'device': args.device,
        'n_envs': args.envs,
        'net_arch': [int(x) for x in args.net_arch.split(',')] if args.net_arch else None,
        'ppo_n_steps': args.ppo_n_steps,
        'ppo_batch_size': args.ppo_batch_size,
        'ppo_n_epochs': args.ppo_n_epochs,
        'a2c_n_steps': args.a2c_n_steps
    }
    
    # Run system
    trading_system = TradingSystem(config)
    success = trading_system.run_complete_system()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

