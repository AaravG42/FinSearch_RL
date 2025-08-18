import os
import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from rl_trading_environment import StockTradingEnvironment

warnings.filterwarnings("ignore")

class TradingCallback(BaseCallback):
    """Callback for monitoring training progress"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.portfolio_values = []
    
    def _on_step(self) -> bool:
        # Get portfolio value from environment info
        if 'portfolio_value' in self.locals.get('infos', [{}])[0]:
            self.portfolio_values.append(self.locals['infos'][0]['portfolio_value'])
        return True

class EnsembleTradingAgent:
    """
    Ensemble Trading Agent implementing the paper's methodology
    
    Trains three actor-critic algorithms (PPO, A2C, DDPG) and selects
    the best performing one based on Sharpe ratio for each trading period.
    """
    
    def __init__(self, 
                 train_data: Dict[str, pd.DataFrame],
                 validation_data: Dict[str, pd.DataFrame],
                 test_data: Dict[str, pd.DataFrame],
                 initial_balance: float = 1_000_000,
                 retraining_window: int = 3,  # months
                 validation_window: int = 3,  # months
                 models_dir: str = "models",
                 results_dir: str = "results"):
        """
        Initialize the Ensemble Trading Agent
        
        Args:
            train_data: Training data for all stocks
            validation_data: Validation data for model selection
            test_data: Testing data for final evaluation
            initial_balance: Starting portfolio value
            retraining_window: How often to retrain models (months)
            validation_window: Window for validation (months)
            models_dir: Directory to save trained models
            results_dir: Directory to save results
        """
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.initial_balance = initial_balance
        self.retraining_window = retraining_window
        self.validation_window = validation_window
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        self.models_dir = models_dir
        self.results_dir = results_dir
        
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
                    'verbose': 1
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
                    'verbose': 1
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
                    'verbose': 1
                }
            }
        }
        
        # Results storage
        self.training_results = {}
        self.validation_results = {}
        self.ensemble_results = {}
        
    def create_environment(self, data: Dict[str, pd.DataFrame]) -> gym.Env:
        """Create trading environment from data"""
        return StockTradingEnvironment(
            stock_data=data,
            initial_balance=self.initial_balance,
            transaction_cost_rate=0.001,  # 0.1% as per paper
            turbulence_threshold=140
        )
    
    def train_single_agent(self, 
                          algorithm_name: str, 
                          train_env: gym.Env,
                          total_timesteps: int = 50000) -> object:
        """
        Train a single RL agent
        
        Args:
            algorithm_name: Name of the algorithm (PPO, A2C, or DDPG)
            train_env: Training environment
            total_timesteps: Number of training timesteps
            
        Returns:
            Trained model
        """
        print(f"Training {algorithm_name} agent...")
        
        # Get algorithm class and parameters
        algo_config = self.algorithms[algorithm_name]
        AlgoClass = algo_config['class']
        params = algo_config['params'].copy()
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: train_env])
        
        # Initialize model
        model = AlgoClass(env=vec_env, **params)
        
        # Train model
        callback = TradingCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{algorithm_name}_model.zip")
        model.save(model_path)
        
        print(f"{algorithm_name} training completed. Model saved to {model_path}")
        
        return model
    
    def validate_agent(self, 
                      model: object, 
                      algorithm_name: str,
                      val_env: gym.Env,
                      num_episodes: int = 10) -> Dict[str, float]:
        """
        Validate a trained agent and calculate performance metrics
        
        Args:
            model: Trained model
            algorithm_name: Name of the algorithm
            val_env: Validation environment
            num_episodes: Number of validation episodes
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"Validating {algorithm_name} agent...")
        
        portfolio_values = []
        episode_rewards = []
        
        for episode in range(num_episodes):
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
        metrics['mean_episode_reward'] = np.mean(episode_rewards)
        metrics['std_episode_reward'] = np.std(episode_rewards)
        metrics['mean_portfolio_value'] = np.mean(portfolio_values)
        
        print(f"{algorithm_name} validation completed.")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Total Return: {metrics.get('total_return', 0):.4f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
        
        return metrics
    
    def select_best_agent(self, validation_results: Dict[str, Dict]) -> str:
        """
        Select the best performing agent based on Sharpe ratio
        
        Args:
            validation_results: Validation results for all agents
            
        Returns:
            Name of the best performing algorithm
        """
        best_algorithm = None
        best_sharpe = -np.inf
        
        for algo_name, metrics in validation_results.items():
            sharpe_ratio = metrics.get('sharpe_ratio', -np.inf)
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_algorithm = algo_name
        
        print(f"Best performing algorithm: {best_algorithm} (Sharpe: {best_sharpe:.4f})")
        return best_algorithm
    
    def train_ensemble(self, total_timesteps: int = 50000):
        """
        Train all three algorithms on the training data
        
        Args:
            total_timesteps: Number of training timesteps for each algorithm
        """
        print("Starting ensemble training...")
        print(f"Training data contains {len(self.train_data)} stocks")
        
        # Create training environment
        train_env = self.create_environment(self.train_data)
        
        # Train all algorithms
        trained_models = {}
        for algorithm_name in self.algorithms.keys():
            model = self.train_single_agent(algorithm_name, train_env, total_timesteps)
            trained_models[algorithm_name] = model
        
        # Store trained models
        self.trained_models = trained_models
        
        print("Ensemble training completed!")
    
    def validate_ensemble(self, num_episodes: int = 10):
        """
        Validate all trained models on validation data
        
        Args:
            num_episodes: Number of episodes for validation
        """
        print("Starting ensemble validation...")
        
        # Create validation environment
        val_env = self.create_environment(self.validation_data)
        
        # Validate all models
        validation_results = {}
        for algorithm_name, model in self.trained_models.items():
            metrics = self.validate_agent(model, algorithm_name, val_env, num_episodes)
            validation_results[algorithm_name] = metrics
        
        # Store validation results
        self.validation_results = validation_results
        
        # Select best agent
        self.best_algorithm = self.select_best_agent(validation_results)
        
        print("Ensemble validation completed!")
        
        return validation_results
    
    def test_ensemble(self, num_episodes: int = 1):
        """
        Test the ensemble strategy on test data
        
        Args:
            num_episodes: Number of test episodes
        """
        print("Testing ensemble strategy...")
        
        # Create test environment
        test_env = self.create_environment(self.test_data)
        
        # Test best model
        best_model = self.trained_models[self.best_algorithm]
        
        # Run test
        obs, _ = test_env.reset()
        portfolio_history = [test_env.initial_balance]
        actions_history = []
        done = False
        
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            portfolio_history.append(info['portfolio_value'])
            actions_history.append(action)
        
        # Calculate test metrics
        test_metrics = test_env.get_portfolio_metrics()
        
        # Store results
        self.ensemble_results = {
            'best_algorithm': self.best_algorithm,
            'portfolio_history': portfolio_history,
            'actions_history': actions_history,
            'metrics': test_metrics
        }
        
        print("Ensemble testing completed!")
        print(f"Best Algorithm: {self.best_algorithm}")
        print(f"Final Portfolio Value: ${portfolio_history[-1]:,.2f}")
        print(f"Total Return: {test_metrics.get('total_return', 0):.4f}")
        print(f"Sharpe Ratio: {test_metrics.get('sharpe_ratio', 0):.4f}")
        
        return self.ensemble_results
    
    def run_benchmark_strategies(self):
        """Run benchmark strategies for comparison"""
        print("Running benchmark strategies...")
        
        # Create test environment for benchmarks
        test_env = self.create_environment(self.test_data)
        
        benchmarks = {}
        
        # 1. Buy and Hold Strategy
        obs, _ = test_env.reset()
        # Equal weight allocation to all stocks
        buy_hold_action = np.ones(len(test_env.stock_list)) * 0.5  # Moderate buy signal
        
        obs, _, _, _, info = test_env.step(buy_hold_action)
        buy_hold_value = [info['portfolio_value']]
        
        # Hold for rest of period
        while True:
            hold_action = np.zeros(len(test_env.stock_list))  # No trading
            obs, _, terminated, truncated, info = test_env.step(hold_action)
            buy_hold_value.append(info['portfolio_value'])
            if terminated or truncated:
                break
        
        benchmarks['Buy_Hold'] = {
            'portfolio_history': buy_hold_value,
            'metrics': test_env.get_portfolio_metrics()
        }
        
        # 2. Random Strategy
        test_env_random = self.create_environment(self.test_data)
        obs, _ = test_env_random.reset()
        random_values = [test_env_random.initial_balance]
        
        while True:
            random_action = np.random.uniform(-0.1, 0.1, len(test_env_random.stock_list))
            obs, _, terminated, truncated, info = test_env_random.step(random_action)
            random_values.append(info['portfolio_value'])
            if terminated or truncated:
                break
        
        benchmarks['Random'] = {
            'portfolio_history': random_values,
            'metrics': test_env_random.get_portfolio_metrics()
        }
        
        self.benchmark_results = benchmarks
        print("Benchmark strategies completed!")
        
        return benchmarks
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("Generating performance report...")
        
        # Create results summary
        summary = {
            'Ensemble_Strategy': self.ensemble_results['metrics'],
            'Validation_Results': self.validation_results,
        }
        
        if hasattr(self, 'benchmark_results'):
            for bench_name, bench_data in self.benchmark_results.items():
                summary[bench_name] = bench_data['metrics']
        
        # Convert to DataFrame for easier viewing
        summary_df = pd.DataFrame(summary).T
        
        # Save summary
        summary_path = os.path.join(self.results_dir, 'performance_summary.csv')
        summary_df.to_csv(summary_path)
        
        # Generate plots
        self._create_visualizations()
        
        print(f"Performance report saved to {self.results_dir}")
        print("\nPerformance Summary:")
        print(summary_df.round(4))
        
        return summary_df
    
    def _create_visualizations(self):
        """Create performance visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Portfolio Value Over Time
        ax1 = axes[0, 0]
        if hasattr(self, 'ensemble_results'):
            days = range(len(self.ensemble_results['portfolio_history']))
            ax1.plot(days, self.ensemble_results['portfolio_history'], 
                    label=f"Ensemble ({self.best_algorithm})", linewidth=2)
        
        if hasattr(self, 'benchmark_results'):
            for bench_name, bench_data in self.benchmark_results.items():
                days = range(len(bench_data['portfolio_history']))
                ax1.plot(days, bench_data['portfolio_history'], 
                        label=bench_name, alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation Results Comparison
        ax2 = axes[0, 1]
        if hasattr(self, 'validation_results'):
            algorithms = list(self.validation_results.keys())
            sharpe_ratios = [self.validation_results[algo].get('sharpe_ratio', 0) 
                           for algo in algorithms]
            
            bars = ax2.bar(algorithms, sharpe_ratios, alpha=0.7)
            ax2.set_title('Validation Sharpe Ratios')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.grid(True, alpha=0.3)
            
            # Highlight best algorithm
            if hasattr(self, 'best_algorithm'):
                best_idx = algorithms.index(self.best_algorithm)
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(1.0)
        
        # 3. Returns Distribution
        ax3 = axes[1, 0]
        if hasattr(self, 'ensemble_results'):
            portfolio_values = self.ensemble_results['portfolio_history']
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            ax3.hist(returns, bins=30, alpha=0.7, density=True)
            ax3.set_title('Daily Returns Distribution')
            ax3.set_xlabel('Daily Returns')
            ax3.set_ylabel('Density')
            ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax4 = axes[1, 1]
        if hasattr(self, 'ensemble_results') and hasattr(self, 'benchmark_results'):
            metrics_data = {'Ensemble': self.ensemble_results['metrics']}
            metrics_data.update({name: data['metrics'] 
                               for name, data in self.benchmark_results.items()})
            
            metric_names = ['total_return', 'sharpe_ratio', 'max_drawdown']
            x_pos = np.arange(len(metric_names))
            width = 0.25
            
            for i, (strategy, metrics) in enumerate(metrics_data.items()):
                values = [metrics.get(metric, 0) for metric in metric_names]
                ax4.bar(x_pos + i*width, values, width, label=strategy, alpha=0.7)
            
            ax4.set_title('Performance Metrics Comparison')
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Value')
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(['Total Return', 'Sharpe Ratio', 'Max Drawdown'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {plot_path}")
    
    def save_results(self):
        """Save all results to files"""
        # Save ensemble results
        with open(os.path.join(self.results_dir, 'ensemble_results.pkl'), 'wb') as f:
            pickle.dump(self.ensemble_results, f)
        
        # Save validation results
        with open(os.path.join(self.results_dir, 'validation_results.pkl'), 'wb') as f:
            pickle.dump(self.validation_results, f)
        
        # Save benchmark results if available
        if hasattr(self, 'benchmark_results'):
            with open(os.path.join(self.results_dir, 'benchmark_results.pkl'), 'wb') as f:
                pickle.dump(self.benchmark_results, f)
        
        print("All results saved successfully!")

def main():
    """Main function to run the ensemble trading system"""
    print("Starting Ensemble Trading System...")
    print("=" * 50)
    
    # Load data (assuming data_collector has been run)
    print("Loading data...")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("Error: Data directory not found. Please run data_collector.py first.")
        return
    
    # Load training data
    train_data = {}
    train_dir = 'data/train'
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.endswith('.csv') and file != 'NIFTY50_benchmark.csv':
                ticker = file.replace('.csv', '') + '.NS'
                df = pd.read_csv(os.path.join(train_dir, file), index_col=0, parse_dates=True)
                if len(df) > 100:  # Minimum data requirement
                    train_data[ticker] = df
    
    # Load test data
    test_data = {}
    test_dir = 'data/test'
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.csv') and file != 'NIFTY50_benchmark.csv':
                ticker = file.replace('.csv', '') + '.NS'
                df = pd.read_csv(os.path.join(test_dir, file), index_col=0, parse_dates=True)
                if len(df) > 20:  # Minimum data requirement
                    test_data[ticker] = df
    
    # Create validation data from last part of training data
    validation_data = {}
    for ticker, df in train_data.items():
        if len(df) > 200:  # Ensure enough data for split
            split_point = int(len(df) * 0.8)  # Use last 20% for validation
            validation_data[ticker] = df.iloc[split_point:].copy()
            train_data[ticker] = df.iloc[:split_point].copy()
    
    print(f"Loaded {len(train_data)} stocks for training")
    print(f"Loaded {len(validation_data)} stocks for validation")
    print(f"Loaded {len(test_data)} stocks for testing")
    
    if len(train_data) == 0:
        print("Error: No training data found. Please run data_collector.py first.")
        return
    
    # Initialize ensemble agent
    ensemble_agent = EnsembleTradingAgent(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        initial_balance=1_000_000
    )
    
    # Train ensemble
    ensemble_agent.train_ensemble(total_timesteps=20000)  # Reduced for faster execution
    
    # Validate ensemble
    ensemble_agent.validate_ensemble(num_episodes=5)
    
    # Test ensemble
    ensemble_agent.test_ensemble()
    
    # Run benchmarks
    ensemble_agent.run_benchmark_strategies()
    
    # Generate report
    ensemble_agent.generate_report()
    
    # Save results
    ensemble_agent.save_results()
    
    print("=" * 50)
    print("Ensemble Trading System completed successfully!")

if __name__ == "__main__":
    main()
