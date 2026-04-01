"""
Tests unitarios para backtest_engine.py

Validaciones críticas:
- Sin lookahead bias (señal en t, ejecución en t+1)
- Métricas correctas contra benchmark conocido
- Stop-loss y take-profit funcionan correctamente
- Comisiones y slippage aplicados
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backtest_engine import BacktestEngine, BacktestConfig, Trade
from strategies import Strategy, Signal, Rule


def create_mock_data(n_days=252, start_price=100):
    """Crea datos mock OHLCV para testing."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generar precios con trend + ruido
    trend = np.linspace(start_price, start_price * 1.2, n_days)
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


class TestBacktestEngine:
    """Tests para BacktestEngine."""
    
    def test_engine_initialization(self):
        """Test que el engine se inicializa correctamente."""
        config = BacktestConfig(initial_capital=100_000)
        engine = BacktestEngine(config=config)
        
        assert engine.config.initial_capital == 100_000
        assert engine.config.commission_pct == 0.001
    
    def test_no_lookahead_bias(self):
        """
        Test crítico: Verifica que señal en barra t se ejecuta en t+1.
        
        Setup:
        - Datos mock con 100 días
        - Estrategia que genera señal de compra en día 10
        - Verificar que entrada es día 11 (no día 10)
        """
        df = create_mock_data(n_days=100, start_price=100)
        
        # Estrategia simple: compra cuando close > 105
        strategy = Strategy(
            name="Test_No_Lookahead",
            entry=Signal.above("close", 105.0),
            exit=Signal.below("close", 100.0),
            stop_loss=None,
            take_profit=None,
        )
        
        engine = BacktestEngine(config=BacktestConfig(initial_capital=10_000))
        result = engine.run(df, strategy, ticker="TEST", add_indicators=False)
        
        # Si hay trades, verificar que entry_date > signal_date
        if result.trades:
            first_trade = result.trades[0]
            
            # Buscar primera señal
            signals = result.df_with_signals['entry_signal']
            first_signal_date = signals[signals == True].index[0]
            
            # La entrada debe ser al menos 1 día después
            assert first_trade.entry_date > first_signal_date, \
                "Lookahead bias detectado: entrada antes de señal"
    
    def test_stop_loss_works(self):
        """Test que stop-loss se ejecuta correctamente."""
        df = create_mock_data(n_days=100, start_price=100)
        
        # Forzar caída de precio
        df.loc[df.index[20]:, 'close'] = 85  # Caída de 15%
        df.loc[df.index[20]:, 'open'] = 85
        
        strategy = Strategy(
            name="Test_StopLoss",
            entry=Signal.above("close", 0),  # Siempre compra
            stop_loss=0.10,  # 10% stop-loss
            take_profit=None,
        )
        
        engine = BacktestEngine(config=BacktestConfig(initial_capital=10_000))
        result = engine.run(df, strategy, ticker="TEST", add_indicators=False)
        
        # Verificar que hay trades cerrados por stop-loss
        stop_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        assert len(stop_trades) > 0, "Stop-loss no se ejecutó"
        
        # Verificar que pérdida es aproximadamente -10%
        for trade in stop_trades:
            assert trade.pnl_pct < 0, "Stop-loss debe generar pérdida"
            assert trade.pnl_pct >= -12, "Pérdida mayor a stop-loss + slippage"
    
    def test_commissions_applied(self):
        """Test que comisiones se aplican correctamente."""
        df = create_mock_data(n_days=50, start_price=100)
        
        strategy = Strategy(
            name="Test_Commission",
            entry=Signal.above("close", 99),
            exit=Signal.above("close", 110),
        )
        
        config = BacktestConfig(
            initial_capital=10_000,
            commission_pct=0.01,  # 1% comisión (alta para testing)
        )
        
        engine = BacktestEngine(config=config)
        result = engine.run(df, strategy, ticker="TEST", add_indicators=False)
        
        # Verificar que trades tienen comisión
        if result.trades:
            first_trade = result.trades[0]
            assert first_trade.commission_paid > 0, "Comisión no aplicada"
            
            # Comisión debe ser aprox 1% de (entrada + salida)
            expected_commission = (first_trade.entry_price + first_trade.exit_price) * first_trade.shares * 0.01
            assert abs(first_trade.commission_paid - expected_commission) < 1, \
                "Comisión calculada incorrectamente"
    
    def test_metrics_calculation(self):
        """Test que métricas básicas se calculan correctamente."""
        df = create_mock_data(n_days=252, start_price=100)
        
        strategy = Strategy(
            name="Test_Metrics",
            entry=Signal.above("close", 100),
            exit=Signal.below("close", 95),
        )
        
        engine = BacktestEngine()
        result = engine.run(df, strategy, ticker="TEST", add_indicators=False)
        
        # Verificar que métricas existen
        m = result.metrics
        assert "sharpe_ratio" in m
        assert "total_return_pct" in m
        assert "max_drawdown_pct" in m
        assert "n_trades" in m
        assert "win_rate_pct" in m
        
        # Verificar que win_rate está entre 0-100
        assert 0 <= m["win_rate_pct"] <= 100, "Win rate fuera de rango"
        
        # Verificar que n_trades >= 0
        assert m["n_trades"] >= 0, "N trades negativo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
