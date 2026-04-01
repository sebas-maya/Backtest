"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def mock_ohlcv_data():
    """Fixture que retorna datos OHLCV mock para testing."""
    n_days = 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Precios con trend + ruido
    trend = np.linspace(100, 120, n_days)
    noise = np.random.randn(n_days) * 2
    close_prices = trend + noise
    
    df = pd.DataFrame({
        'date': dates,
        'open': close_prices * 0.99,
        'high': close_prices * 1.01,
        'low': close_prices * 0.98,
        'close': close_prices,
        'adj_close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n_days),
    })
    
    df.set_index('date', inplace=True)
    return df


@pytest.fixture
def simple_strategy():
    """Fixture que retorna una estrategia simple para testing."""
    from strategies import Strategy, Signal
    
    return Strategy(
        name="Test_Strategy",
        entry=Signal.above("close", 100),
        exit=Signal.below("close", 95),
        stop_loss=0.05,
        take_profit=0.15,
    )


@pytest.fixture
def backtest_config():
    """Fixture que retorna configuración estándar."""
    from backtest_engine import BacktestConfig
    
    return BacktestConfig(
        initial_capital=100_000,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )
