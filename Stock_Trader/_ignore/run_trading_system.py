#!/usr/bin/env python3
"""
Complete Trading System Runner

This script implements the deep reinforcement learning trading system
described in the paper "Deep Reinforcement Learning for Automated Stock Trading: 
An Ensemble Strategy" by Yang et al.

The system:
1. Collects NIFTY100 stock data with technical indicators
2. Creates a trading environment following the paper's MDP formulation
3. Trains three actor-critic algorithms (PPO, A2C, DDPG)
4. Implements an ensemble strategy based on Sharpe ratio selection
5. Evaluates performance against benchmarks

Usage:
    python run_trading_system.py [--collect-data] [--train] [--test-only]
"""

import argparse
import os
import sys
import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run the Ensemble Trading System')
    parser.add_argument('--collect-data', action='store_true', 
                       help='Collect fresh data before training')
    parser.add_argument('--train', action='store_true',
                       help='Train the ensemble models')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing with existing models')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Number of training timesteps per algorithm')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DEEP REINFORCEMENT LEARNING FOR AUTOMATED STOCK TRADING")
    print("An Ensemble Strategy Implementation")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Data Collection
    if args.collect_data or not os.path.exists('data'):
        print("STEP 1: COLLECTING NIFTY100 DATA")
        print("-" * 40)
        
        try:
            import data_collector
            print("Running data collection...")
            data_collector.main()
            print("✓ Data collection completed successfully!")
        except Exception as e:
            print(f"✗ Error in data collection: {e}")
            return 1
        
        print()
    else:
        print("STEP 1: USING EXISTING DATA")
        print("-" * 40)
        print("✓ Found existing data directory")
        print()
    
    # Step 2: Training (if not test-only)
    if not args.test_only:
        print("STEP 2: TRAINING ENSEMBLE MODELS")
        print("-" * 40)
        
        try:
            from ensemble_trading_agent import main as train_main
            print("Starting ensemble training...")
            train_main()
            print("✓ Training completed successfully!")
        except Exception as e:
            print(f"✗ Error in training: {e}")
            return 1
        
        print()
    
    # Step 3: Results Analysis
    print("STEP 3: ANALYZING RESULTS")
    print("-" * 40)
    
    # Check if results exist
    if os.path.exists('results'):
        print("✓ Results directory found")
        
        # Display key results if available
        try:
            import pandas as pd
            if os.path.exists('results/performance_summary.csv'):
                summary = pd.read_csv('results/performance_summary.csv', index_col=0)
                print("\nPERFORMANCE SUMMARY:")
                print(summary.round(4))
                
                # Highlight key metrics
                if 'Ensemble_Strategy' in summary.index:
                    ensemble_metrics = summary.loc['Ensemble_Strategy']
                    print(f"\nKEY ENSEMBLE RESULTS:")
                    print(f"  Total Return: {ensemble_metrics.get('total_return', 0):.2%}")
                    print(f"  Sharpe Ratio: {ensemble_metrics.get('sharpe_ratio', 0):.4f}")
                    print(f"  Max Drawdown: {ensemble_metrics.get('max_drawdown', 0):.2%}")
                    print(f"  Annual Volatility: {ensemble_metrics.get('annual_volatility', 0):.2%}")
        except Exception as e:
            print(f"Note: Could not load detailed results: {e}")
        
        print("✓ Check 'results/' directory for detailed analysis")
        print("✓ Check 'results/performance_analysis.png' for visualizations")
    else:
        print("✗ No results directory found. Please run training first.")
    
    print()
    print("="*80)
    print("SYSTEM EXECUTION COMPLETED")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
