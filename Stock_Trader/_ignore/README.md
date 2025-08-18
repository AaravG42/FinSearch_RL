# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy

This project implements the research paper "Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy" by Yang et al. The system uses three actor-critic based algorithms (PPO, A2C, DDPG) in an ensemble approach to learn optimal stock trading strategies.

## 📋 Overview

The system implements a complete automated trading framework that:

- **Collects and processes** NIFTY100 stock data with technical indicators
- **Models trading** as a Markov Decision Process (MDP) 
- **Trains three algorithms** (PPO, A2C, DDPG) using deep reinforcement learning
- **Selects the best performer** using Sharpe ratio validation
- **Implements risk management** with turbulence index for market crash detection
- **Evaluates performance** against multiple benchmarks

## 🏗️ Architecture

### Components

1. **Data Collection (`data_collector.py`)**
   - Downloads NIFTY100 stocks from 2014-2025
   - Calculates technical indicators (RSI, MACD, CCI, ADX, Bollinger Bands)
   - Handles missing data and preprocessing

2. **Trading Environment (`rl_trading_environment.py`)**
   - Implements OpenAI Gym compatible environment
   - State space: [balance, prices, holdings, technical_indicators]
   - Action space: Continuous actions for each stock [-1, 1]
   - Reward function: Portfolio value change minus transaction costs

3. **Ensemble Agent (`ensemble_trading_agent.py`)**
   - Trains PPO, A2C, and DDPG algorithms
   - Validates models using Sharpe ratio
   - Selects best performer for each period
   - Implements ensemble strategy

4. **Main Runner (`run_trading_system.py`)**
   - Orchestrates the complete pipeline
   - Provides command-line interface
   - Generates comprehensive reports

## 📊 Methodology (Following the Paper)

### State Space (181-dimensional for 30 stocks)
```
[b_t, p_t, h_t, M_t, R_t, C_t, X_t]
```
- `b_t`: Available balance
- `p_t`: Stock prices  
- `h_t`: Current holdings
- `M_t`: MACD indicators
- `R_t`: RSI indicators
- `C_t`: CCI indicators  
- `X_t`: ADX indicators

### Action Space
- Continuous actions normalized to [-1, 1]
- Each action represents shares to buy/sell for each stock
- Actions scaled by maximum shares per trade (`hmax`)

### Reward Function
```
r(s_t, a_t, s_{t+1}) = (portfolio_value_{t+1} - portfolio_value_t) - transaction_costs
```

### Ensemble Strategy
1. **Training Phase**: Train PPO, A2C, DDPG on historical data
2. **Validation Phase**: Evaluate models using 3-month rolling window
3. **Selection Phase**: Choose model with highest Sharpe ratio
4. **Trading Phase**: Use selected model for next quarter

### Risk Management
- **Turbulence Index**: Detects market crashes using covariance analysis
- **Emergency Protocol**: Sell all positions when turbulence exceeds threshold
- **Transaction Costs**: 0.1% of trade value as per paper

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Stock_Trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Complete Pipeline (Recommended)
```bash
python run_trading_system.py --collect-data --train
```

#### Individual Steps

1. **Data Collection Only**:
```bash
python data_collector.py
```

2. **Training Only** (requires existing data):
```bash
python ensemble_trading_agent.py
```

3. **Custom Training**:
```bash
python run_trading_system.py --train --timesteps 100000
```

## 📁 Project Structure

```
Stock_Trader/
├── data_collector.py              # NIFTY100 data collection
├── rl_trading_environment.py      # Trading environment (MDP)
├── ensemble_trading_agent.py      # Main ensemble implementation
├── run_trading_system.py          # Complete system runner
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── paper.md                       # Original research paper
├── data/                          # Generated data directory
│   ├── train/                     # Training data (2014-2025)
│   ├── test/                      # Testing data (June-Aug 2025)
│   └── *.csv                      # Processed stock data
├── models/                        # Trained model files
│   ├── PPO_model.zip
│   ├── A2C_model.zip
│   └── DDPG_model.zip
└── results/                       # Analysis and reports
    ├── performance_summary.csv
    ├── performance_analysis.png
    └── *.pkl                      # Detailed results
```

## 📈 Performance Metrics

The system evaluates performance using multiple metrics:

- **Total Return**: Overall portfolio gain/loss
- **Annualized Return**: Geometric average annual return
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Transaction Costs**: Total trading costs incurred

## 🎯 Key Features

### 1. **Paper-Faithful Implementation**
- Exact MDP formulation from the research
- Same technical indicators and state space
- Identical reward function and constraints

### 2. **Robust Risk Management**
- Turbulence index for crash detection
- Transaction cost modeling
- Position size limits

### 3. **Comprehensive Evaluation**
- Multiple benchmark strategies
- Statistical significance testing
- Detailed performance visualization

### 4. **Production Ready**
- Error handling and data validation
- Modular, extensible design
- Comprehensive logging and monitoring

## 🔧 Configuration

### Algorithm Parameters

**PPO (Proximal Policy Optimization)**
```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'clip_range': 0.2
}
```

**A2C (Advantage Actor-Critic)**
```python
{
    'learning_rate': 7e-4,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 1.0,
    'ent_coef': 0.01
}
```

**DDPG (Deep Deterministic Policy Gradient)**
```python
{
    'learning_rate': 1e-3,
    'buffer_size': 1000000,
    'batch_size': 100,
    'tau': 0.005,
    'gamma': 0.99
}
```

### Environment Parameters
- **Initial Balance**: ₹10,00,000 (1 million)
- **Transaction Cost**: 0.1% per trade
- **Turbulence Threshold**: 140
- **Maximum Shares per Action**: 100

## 📊 Expected Results

Based on the paper's findings, the ensemble strategy should:

- **Outperform individual algorithms** in terms of Sharpe ratio
- **Achieve higher returns** than buy-and-hold strategies  
- **Show lower volatility** than market benchmarks
- **Demonstrate robustness** during market downturns

### Paper Results (Reference)
- **Ensemble Sharpe Ratio**: 1.30
- **Annual Return**: 13.0%
- **Annual Volatility**: 9.7%
- **Max Drawdown**: -9.7%

## 🛠️ Troubleshooting

### Common Issues

1. **Data Download Errors**
   - Check internet connection
   - Verify yfinance is up to date
   - Some stocks may be delisted/renamed

2. **Memory Issues**
   - Reduce number of stocks
   - Lower training timesteps
   - Use smaller batch sizes

3. **Training Convergence**
   - Increase training timesteps
   - Adjust learning rates
   - Check data quality

### Performance Tips

1. **Faster Training**
   - Use GPU if available
   - Reduce validation episodes
   - Parallel environment training

2. **Better Results**
   - Increase training timesteps
   - Tune hyperparameters
   - Add more technical indicators

## 📚 References

1. Yang, H., Liu, X. Y., Zhong, S., & Walid, A. (2020). Deep reinforcement learning for automated stock trading: An ensemble strategy. *ACM International Conference on AI in Finance (ICAIF '20)*.

2. Lillicrap, T. P., et al. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

3. Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

4. Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *International conference on machine learning*.

## 📄 License

This project is for educational and research purposes. Please ensure compliance with relevant financial regulations when using for actual trading.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the paper's methodology
- Comprehensive testing
- Clear documentation
- Performance benchmarks included

---

**Note**: This implementation is for research and educational purposes. Always consult with financial advisors before making investment decisions.
