"""
backtest_engine.py
==================
Motor de backtest riguroso y detallado para estrategias de inversión.

Características:
- Simulación barra a barra (sin lookahead bias)
- Stop-loss, take-profit y trailing stop
- Salida por tiempo máximo de holding
- Gestión de posición dinámica con PositionSizer
- Comisiones y slippage configurables
- Cálculo completo de métricas cuantitativas
- Series temporales: equity curve, drawdown, rolling metrics
- Trade log detallado con toda la información relevante

Autor: Backtest Framework
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

from indicators import add_all_indicators
from strategies import Strategy

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuración del backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Parámetros de configuración del motor de backtest."""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001        # 0.1% por trade (ida y vuelta)
    slippage_pct: float = 0.0005         # 0.05% de slippage
    allow_fractional_shares: bool = True
    max_open_positions: int = 1          # por estrategia mono-activo
    reinvest_profits: bool = True        # reinvertir ganancias
    benchmark_col: str = "adj_close"     # columna de precio
    risk_free_rate: float = 0.04         # tasa libre de riesgo anual
    annualization: int = 252             # días trading al año
    min_data_points: int = 200           # mínimo de datos para calcular indicadores
    execution_price: str = "open"        # precio de ejecución: "open" o "close"


# ---------------------------------------------------------------------------
# Resultado de un trade individual
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Registro completo de un trade."""
    trade_id: int
    ticker: str
    direction: str           # "long" | "short"
    entry_date: pd.Timestamp
    entry_price: float
    entry_signal: str
    shares: float
    capital_at_entry: float

    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "signal", "stop_loss", "take_profit", "trailing_stop", "max_days", "end_of_data"

    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_high: Optional[float] = None

    # Calculados al cerrar
    gross_pnl: float = 0.0
    commission_paid: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    mae: float = 0.0          # Maximum Adverse Excursion
    mfe: float = 0.0          # Maximum Favorable Excursion
    entry_portfolio_value: float = 0.0
    exit_portfolio_value: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "shares": round(self.shares, 4),
            "capital_invested": round(self.entry_price * self.shares, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "commission_paid": round(self.commission_paid, 2),
            "net_pnl": round(self.net_pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 4),
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
            "entry_signal": self.entry_signal,
            "stop_loss_price": round(self.stop_loss_price, 4) if self.stop_loss_price else None,
            "take_profit_price": round(self.take_profit_price, 4) if self.take_profit_price else None,
            "mae_pct": round(self.mae * 100, 4),
            "mfe_pct": round(self.mfe * 100, 4),
            "entry_portfolio_value": round(self.entry_portfolio_value, 2),
            "exit_portfolio_value": round(self.exit_portfolio_value, 2),
        }


# ---------------------------------------------------------------------------
# Métricas del backtest
# ---------------------------------------------------------------------------

def _compute_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    config: BacktestConfig,
) -> Dict[str, Any]:
    """
    Calcula el conjunto completo de métricas de rendimiento.
    """
    if not trades:
        return {"error": "Sin trades"}

    trades_df = pd.DataFrame([t.to_dict() for t in trades])
    closed = trades_df[trades_df["exit_date"].notna()].copy()

    n_trades = len(closed)
    if n_trades == 0:
        return {"error": "Sin trades cerrados"}

    rf = config.risk_free_rate
    ann = config.annualization
    initial = config.initial_capital
    final = equity_curve.iloc[-1]

    # --- Retornos ---
    total_return = (final - initial) / initial
    eq_returns = equity_curve.pct_change().dropna()

    # --- Retornos anualizados ---
    n_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    years = n_days / 365.25
    cagr = (final / initial) ** (1 / max(years, 0.01)) - 1

    # --- Volatilidad ---
    ann_vol = eq_returns.std() * np.sqrt(ann)

    # --- Sharpe ---
    excess = eq_returns - rf / ann
    sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(ann)

    # --- Sortino ---
    downside = eq_returns[eq_returns < 0]
    sortino_denom = downside.std() * np.sqrt(ann) if len(downside) > 0 else 1e-10
    sortino = (eq_returns.mean() * ann - rf) / (sortino_denom + 1e-10)

    # --- Calmar ---
    drawdown = _compute_drawdown(equity_curve)
    max_dd = drawdown.min()
    calmar = cagr / (abs(max_dd) + 1e-10)

    # --- Omega Ratio ---
    threshold = rf / ann
    gains = (eq_returns - threshold).clip(lower=0).sum()
    losses = (threshold - eq_returns).clip(lower=0).sum()
    omega = gains / (losses + 1e-10)

    # --- Win stats ---
    winners = closed[closed["net_pnl"] > 0]
    losers = closed[closed["net_pnl"] <= 0]
    win_rate = len(winners) / n_trades
    avg_win = winners["pnl_pct"].mean() if len(winners) > 0 else 0
    avg_loss = losers["pnl_pct"].mean() if len(losers) > 0 else 0
    profit_factor = (
        winners["net_pnl"].sum() / (abs(losers["net_pnl"].sum()) + 1e-10)
        if len(losers) > 0
        else float("inf")
    )

    # --- Expectancy ---
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # --- Drawdown stats ---
    dd_stats = _drawdown_stats(equity_curve)

    # --- Trade duración ---
    avg_holding = closed["holding_days"].mean()
    median_holding = closed["holding_days"].median()

    # --- MAE/MFE ---
    avg_mae = closed["mae_pct"].mean()
    avg_mfe = closed["mfe_pct"].mean()

    # --- Recovery Factor ---
    recovery_factor = total_return / (abs(max_dd) + 1e-10)

    # --- Consistency ---
    if len(eq_returns) > 0:
        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
        pct_positive_months = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
    else:
        pct_positive_months = 0

    # --- Tail Ratio ---
    p95 = np.percentile(eq_returns.dropna(), 95) if len(eq_returns) > 5 else 0
    p5 = abs(np.percentile(eq_returns.dropna(), 5)) if len(eq_returns) > 5 else 1
    tail_ratio = p95 / (p5 + 1e-10)

    # --- VaR y CVaR ---
    var_95 = np.percentile(eq_returns.dropna(), 5) if len(eq_returns) > 5 else 0
    cvar_95 = eq_returns[eq_returns <= var_95].mean() if len(eq_returns) > 5 else 0

    return {
        # Rendimiento
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "final_capital": round(final, 2),
        "initial_capital": round(initial, 2),
        # Riesgo
        "annual_volatility_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_drawdown_pct": round(dd_stats["avg_drawdown"] * 100, 2),
        "max_drawdown_duration_days": dd_stats["max_duration"],
        "var_95_pct": round(var_95 * 100, 4),
        "cvar_95_pct": round(cvar_95 * 100, 4),
        # Ratios
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "omega_ratio": round(omega, 4),
        "profit_factor": round(profit_factor, 4),
        "recovery_factor": round(recovery_factor, 4),
        "tail_ratio": round(tail_ratio, 4),
        # Trades
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "expectancy_pct": round(expectancy, 4),
        "avg_holding_days": round(avg_holding, 1),
        "median_holding_days": round(median_holding, 1),
        "avg_mae_pct": round(avg_mae, 4),
        "avg_mfe_pct": round(avg_mfe, 4),
        # Consistencia
        "pct_positive_months": round(pct_positive_months * 100, 2),
        # Duración del backtest
        "backtest_years": round(years, 2),
    }


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    """Retorna la serie de drawdown (fracción negativa)."""
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _drawdown_stats(equity: pd.Series) -> Dict:
    """Estadísticas de drawdown: promedio, duración máxima."""
    dd = _compute_drawdown(equity)
    avg_dd = dd.mean()
    # Duración máxima
    in_dd = dd < 0
    max_duration = 0
    current = 0
    for val in in_dd:
        if val:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return {"avg_drawdown": avg_dd, "max_duration": max_duration}


def _compute_rolling_metrics(
    equity: pd.Series,
    window: int = 63,  # ~3 meses
    rf: float = 0.04,
    ann: int = 252,
) -> pd.DataFrame:
    """Calcula métricas rolling: Sharpe, Vol, Drawdown."""
    returns = equity.pct_change()
    roll_ret = returns.rolling(window).mean() * ann
    roll_vol = returns.rolling(window).std() * np.sqrt(ann)
    roll_sharpe = (roll_ret - rf) / (roll_vol + 1e-10)
    roll_dd = _compute_drawdown(equity)

    return pd.DataFrame(
        {
            "rolling_return": roll_ret,
            "rolling_vol": roll_vol,
            "rolling_sharpe": roll_sharpe,
            "drawdown": roll_dd,
        },
        index=equity.index,
    )


# ---------------------------------------------------------------------------
# Motor principal de Backtest
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Motor de backtest barra a barra.

    Uso básico:
        engine = BacktestEngine(config=BacktestConfig())
        result = engine.run(df, strategy)
        print(result.metrics)
        result.trades_df
        result.equity_curve
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df_raw: pd.DataFrame,
        strategy: Strategy,
        ticker: str = "ASSET",
        add_indicators: bool = True,
    ) -> "BacktestResult":
        """
        Ejecuta el backtest completo para una estrategia sobre un activo.

        Parameters
        ----------
        df_raw         : DataFrame con columnas OHLCV (puede ser long o wide)
        strategy       : objeto Strategy
        ticker         : símbolo del activo
        add_indicators : si True, calcula automáticamente los indicadores

        Returns
        -------
        BacktestResult con métricas, trades y series temporales
        """
        # Preparar datos
        df = self._prepare_data(df_raw, ticker, add_indicators)
        if df is None or len(df) < self.config.min_data_points:
            return BacktestResult.empty(strategy.name, ticker)

        # Evaluar señales (sin lookahead: señal en barra t → ejecución en t+1)
        try:
            entry_signals = strategy.get_entry_signals(df)
            exit_signals = strategy.get_exit_signals(df)
        except Exception as e:
            return BacktestResult.empty(strategy.name, ticker, error=str(e))

        # Simular trades barra a barra
        trades, equity_curve, portfolio_series = self._simulate(
            df, strategy, entry_signals, exit_signals, ticker
        )

        # Calcular métricas
        metrics = _compute_metrics(trades, equity_curve, self.config)

        # Series temporales adicionales
        drawdown_series = _compute_drawdown(equity_curve)
        rolling_metrics = _compute_rolling_metrics(
            equity_curve,
            rf=self.config.risk_free_rate,
            ann=self.config.annualization,
        )

        # Buy & Hold benchmark
        bh_equity = self._buy_and_hold(df)

        return BacktestResult(
            strategy_name=strategy.name,
            ticker=ticker,
            trades=trades,
            equity_curve=equity_curve,
            portfolio_series=portfolio_series,
            drawdown_series=drawdown_series,
            rolling_metrics=rolling_metrics,
            buy_and_hold=bh_equity,
            metrics=metrics,
            df_with_signals=df.assign(
                entry_signal=entry_signals,
                exit_signal=exit_signals,
            ),
            config=self.config,
        )

    def _prepare_data(
        self, df_raw: pd.DataFrame, ticker: str, add_indicators: bool
    ) -> Optional[pd.DataFrame]:
        """Filtra y prepara el DataFrame para el backtest."""
        # Si viene en formato long, filtrar el ticker
        if "ticker" in df_raw.columns:
            df = df_raw[df_raw["ticker"] == ticker].copy()
        else:
            df = df_raw.copy()

        if df.empty:
            return None

        # Asegurar índice de fecha
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Columnas mínimas
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                return None

        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0

        # Agregar indicadores técnicos
        if add_indicators:
            try:
                df = add_all_indicators(df)
                # Agregar sma_10 y sma_20 explícitamente si no están
                from indicators import sma
                if "sma_10" not in df.columns:
                    df["sma_10"] = sma(df, 10)
            except Exception as e:
                pass

        df = df.dropna(how="all")
        return df

    def _simulate(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        ticker: str,
    ) -> Tuple[List[Trade], pd.Series, pd.DataFrame]:
        """
        Simulación barra a barra sin lookahead bias.
        La señal del día t se ejecuta al precio de apertura del día t+1.
        """
        cfg = self.config
        capital = cfg.initial_capital
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        trade_counter = 0

        equity_values = []
        portfolio_rows = []

        dates = df.index
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Pre-computar volatilidad para position sizing
        vol_series = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        entry_arr = entry_signals.values.astype(bool)
        exit_arr = exit_signals.values.astype(bool)

        recent_trades_df = pd.DataFrame(columns=["pnl_pct"])

        for i in range(1, len(df)):
            date = dates[i]
            price_open = opens[i]
            price_high = highs[i]
            price_low = lows[i]
            price_close = closes[i]

            if np.isnan(price_close) or price_close <= 0:
                equity_values.append(capital if current_trade is None else
                                     capital + current_trade.shares * price_close)
                continue

            exec_price = price_open if cfg.execution_price == "open" else price_close
            if np.isnan(exec_price) or exec_price <= 0:
                exec_price = price_close

            # ---------------------------------------------------------------
            # Gestión de posición abierta
            # ---------------------------------------------------------------
            if current_trade is not None:
                exit_reason = None
                exit_price = None

                # --- Stop Loss ---
                if strategy.stop_loss and current_trade.stop_loss_price:
                    if price_low <= current_trade.stop_loss_price:
                        exit_reason = "stop_loss"
                        exit_price = min(current_trade.stop_loss_price, exec_price)

                # --- Take Profit ---
                if exit_reason is None and strategy.take_profit and current_trade.take_profit_price:
                    if price_high >= current_trade.take_profit_price:
                        exit_reason = "take_profit"
                        exit_price = current_trade.take_profit_price

                # --- Trailing Stop ---
                if exit_reason is None and strategy.trailing_stop:
                    current_trade.trailing_high = max(
                        current_trade.trailing_high or current_trade.entry_price,
                        price_high,
                    )
                    trail_price = current_trade.trailing_high * (1 - strategy.trailing_stop)
                    if price_low <= trail_price:
                        exit_reason = "trailing_stop"
                        exit_price = min(trail_price, exec_price)

                # --- Max Holding Days ---
                if exit_reason is None and strategy.max_holding_days:
                    holding = (date - current_trade.entry_date).days
                    if holding >= strategy.max_holding_days:
                        exit_reason = "max_days"
                        exit_price = exec_price

                # --- Signal Exit ---
                if exit_reason is None and exit_arr[i - 1]:
                    exit_reason = "signal"
                    exit_price = exec_price

                # --- Cerrar trade ---
                if exit_reason:
                    current_trade = self._close_trade(
                        current_trade, date, exit_price, exit_reason, capital, cfg
                    )
                    capital += current_trade.net_pnl
                    recent_trades_df = pd.concat(
                        [recent_trades_df, pd.DataFrame([{"pnl_pct": current_trade.pnl_pct}])],
                        ignore_index=True,
                    )
                    trades.append(current_trade)
                    current_trade = None
                else:
                    # Actualizar MAE/MFE
                    current_price_ret = (price_close - current_trade.entry_price) / current_trade.entry_price
                    current_trade.mae = min(current_trade.mae, current_price_ret)
                    current_trade.mfe = max(current_trade.mfe, current_price_ret)

            # ---------------------------------------------------------------
            # Señal de entrada (del día anterior)
            # ---------------------------------------------------------------
            if current_trade is None and entry_arr[i - 1]:
                recent_vol = vol_series.iloc[i] if i < len(vol_series) else None

                sl_price = None
                tp_price = None
                if strategy.stop_loss:
                    sl_price = exec_price * (1 - strategy.stop_loss)
                if strategy.take_profit:
                    tp_price = exec_price * (1 + strategy.take_profit)

                shares = strategy.position_sizer.compute(
                    capital=capital,
                    price=exec_price,
                    stop_loss_price=sl_price,
                    recent_trades=recent_trades_df if len(recent_trades_df) > 0 else None,
                    recent_vol=float(recent_vol) if recent_vol and not np.isnan(recent_vol) else None,
                )

                # Comisión de entrada
                commission_entry = exec_price * shares * cfg.commission_pct
                slippage_cost = exec_price * shares * cfg.slippage_pct
                total_cost = exec_price * shares + commission_entry + slippage_cost

                if total_cost > capital * 0.98:
                    shares = (capital * 0.98) / (exec_price * (1 + cfg.commission_pct + cfg.slippage_pct))

                if shares > 0 and exec_price > 0:
                    trade_counter += 1
                    current_trade = Trade(
                        trade_id=trade_counter,
                        ticker=ticker,
                        direction="long",
                        entry_date=date,
                        entry_price=exec_price,
                        entry_signal=f"entry_signal_bar_{i}",
                        shares=shares,
                        capital_at_entry=capital,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        trailing_high=exec_price,
                        entry_portfolio_value=capital,
                    )
                    capital -= commission_entry + slippage_cost

            # Portfolio value
            port_val = capital
            if current_trade is not None:
                port_val += current_trade.shares * price_close

            equity_values.append(port_val)
            portfolio_rows.append({
                "date": date,
                "capital": capital,
                "portfolio_value": port_val,
                "in_trade": current_trade is not None,
                "close": price_close,
            })

        # Fin de datos: cerrar trade abierto
        if current_trade is not None:
            last_date = dates[-1]
            last_price = closes[-1]
            if not np.isnan(last_price) and last_price > 0:
                current_trade = self._close_trade(
                    current_trade, last_date, last_price, "end_of_data", capital, cfg
                )
                capital += current_trade.net_pnl
                trades.append(current_trade)

        # Construir equity curve
        equity_curve = pd.Series(
            equity_values, index=dates[1:], name="equity"
        )
        equity_curve = pd.concat([
            pd.Series([self.config.initial_capital], index=[dates[0]], name="equity"),
            equity_curve,
        ])

        portfolio_df = pd.DataFrame(portfolio_rows).set_index("date") if portfolio_rows else pd.DataFrame()

        return trades, equity_curve, portfolio_df

    def _close_trade(
        self,
        trade: Trade,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        capital: float,
        cfg: BacktestConfig,
    ) -> Trade:
        """Cierra un trade calculando P&L y comisiones."""
        # Slippage en salida (adverso)
        if exit_reason in ("stop_loss", "trailing_stop"):
            exit_price = exit_price * (1 - cfg.slippage_pct)
        else:
            exit_price = exit_price * (1 - cfg.slippage_pct * 0.5)

        commission_exit = exit_price * trade.shares * cfg.commission_pct
        commission_entry = trade.entry_price * trade.shares * cfg.commission_pct

        gross_pnl = (exit_price - trade.entry_price) * trade.shares
        total_commission = commission_entry + commission_exit
        net_pnl = gross_pnl - total_commission

        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price

        holding_days = max((exit_date - trade.entry_date).days, 0)

        trade.exit_date = exit_date
        trade.exit_price = round(exit_price, 4)
        trade.exit_reason = exit_reason
        trade.gross_pnl = round(gross_pnl, 4)
        trade.commission_paid = round(total_commission, 4)
        trade.net_pnl = round(net_pnl, 4)
        trade.pnl_pct = round(pnl_pct, 6)
        trade.holding_days = holding_days
        trade.exit_portfolio_value = round(capital + net_pnl, 2)

        return trade

    def _buy_and_hold(self, df: pd.DataFrame) -> pd.Series:
        """Retorna la equity curve del Buy & Hold para comparación."""
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        prices = df[price_col].dropna()
        if prices.empty:
            return pd.Series(dtype=float)
        normalized = prices / prices.iloc[0] * self.config.initial_capital
        return normalized.rename("buy_and_hold")


# ---------------------------------------------------------------------------
# Resultado del Backtest
# ---------------------------------------------------------------------------

class BacktestResult:
    """
    Contenedor de resultados del backtest.

    Atributos principales:
    - metrics        : dict con todas las métricas
    - trades_df      : DataFrame de trades cerrados
    - equity_curve   : Serie temporal del valor del portfolio
    - drawdown_series: Serie temporal del drawdown
    - rolling_metrics: DataFrame con métricas rolling
    - buy_and_hold   : Equity curve del benchmark
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        trades: List[Trade],
        equity_curve: pd.Series,
        portfolio_series: pd.DataFrame,
        drawdown_series: pd.Series,
        rolling_metrics: pd.DataFrame,
        buy_and_hold: pd.Series,
        metrics: Dict,
        df_with_signals: pd.DataFrame,
        config: BacktestConfig,
    ):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.trades = trades
        self.trades_df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
        self.equity_curve = equity_curve
        self.portfolio_series = portfolio_series
        self.drawdown_series = drawdown_series
        self.rolling_metrics = rolling_metrics
        self.buy_and_hold = buy_and_hold
        self.metrics = metrics
        self.df_with_signals = df_with_signals
        self.config = config

    @classmethod
    def empty(cls, strategy_name: str, ticker: str, error: str = "") -> "BacktestResult":
        return cls(
            strategy_name=strategy_name,
            ticker=ticker,
            trades=[],
            equity_curve=pd.Series(dtype=float),
            portfolio_series=pd.DataFrame(),
            drawdown_series=pd.Series(dtype=float),
            rolling_metrics=pd.DataFrame(),
            buy_and_hold=pd.Series(dtype=float),
            metrics={"error": error or "Datos insuficientes"},
            df_with_signals=pd.DataFrame(),
            config=BacktestConfig(),
        )

    def summary(self) -> pd.Series:
        """Resumen ejecutivo en una línea."""
        m = self.metrics
        if "error" in m:
            return pd.Series({"strategy": self.strategy_name, "ticker": self.ticker, "error": m["error"]})
        return pd.Series({
            "strategy": self.strategy_name,
            "ticker": self.ticker,
            "total_return_%": m.get("total_return_pct"),
            "cagr_%": m.get("cagr_pct"),
            "sharpe": m.get("sharpe_ratio"),
            "sortino": m.get("sortino_ratio"),
            "calmar": m.get("calmar_ratio"),
            "max_dd_%": m.get("max_drawdown_pct"),
            "win_rate_%": m.get("win_rate_pct"),
            "profit_factor": m.get("profit_factor"),
            "n_trades": m.get("n_trades"),
            "avg_holding_days": m.get("avg_holding_days"),
            "annual_vol_%": m.get("annual_volatility_pct"),
        })

    def print_report(self) -> None:
        """Imprime un reporte detallado en consola."""
        m = self.metrics
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  BACKTEST: {self.strategy_name} | {self.ticker}")
        print(sep)

        if "error" in m:
            print(f"  ERROR: {m['error']}")
            return

        print(f"\n  RENDIMIENTO")
        print(f"    Capital inicial:        ${m['initial_capital']:>12,.2f}")
        print(f"    Capital final:          ${m['final_capital']:>12,.2f}")
        print(f"    Retorno total:          {m['total_return_pct']:>10.2f}%")
        print(f"    CAGR:                   {m['cagr_pct']:>10.2f}%")
        print(f"    Años backtest:          {m['backtest_years']:>10.2f}")

        print(f"\n  RIESGO")
        print(f"    Volatilidad anual:      {m['annual_volatility_pct']:>10.2f}%")
        print(f"    Max Drawdown:           {m['max_drawdown_pct']:>10.2f}%")
        print(f"    Avg Drawdown:           {m['avg_drawdown_pct']:>10.2f}%")
        print(f"    Max DD Duration (días): {m['max_drawdown_duration_days']:>10}")
        print(f"    VaR 95%:                {m['var_95_pct']:>10.4f}%")
        print(f"    CVaR 95%:               {m['cvar_95_pct']:>10.4f}%")

        print(f"\n  RATIOS")
        print(f"    Sharpe Ratio:           {m['sharpe_ratio']:>10.4f}")
        print(f"    Sortino Ratio:          {m['sortino_ratio']:>10.4f}")
        print(f"    Calmar Ratio:           {m['calmar_ratio']:>10.4f}")
        print(f"    Omega Ratio:            {m['omega_ratio']:>10.4f}")
        print(f"    Profit Factor:          {m['profit_factor']:>10.4f}")
        print(f"    Recovery Factor:        {m['recovery_factor']:>10.4f}")
        print(f"    Tail Ratio:             {m['tail_ratio']:>10.4f}")

        print(f"\n  TRADES")
        print(f"    Número de trades:       {m['n_trades']:>10}")
        print(f"    Win Rate:               {m['win_rate_pct']:>10.2f}%")
        print(f"    Avg Ganancia:           {m['avg_win_pct']:>10.4f}%")
        print(f"    Avg Pérdida:            {m['avg_loss_pct']:>10.4f}%")
        print(f"    Expectancy:             {m['expectancy_pct']:>10.4f}%")
        print(f"    Avg Holding (días):     {m['avg_holding_days']:>10.1f}")
        print(f"    Avg MAE:                {m['avg_mae_pct']:>10.4f}%")
        print(f"    Avg MFE:                {m['avg_mfe_pct']:>10.4f}%")
        print(f"    Meses positivos:        {m['pct_positive_months']:>10.2f}%")
        print(sep)

    def get_trades_by_reason(self) -> pd.DataFrame:
        """Agrupa los trades por razón de salida."""
        if self.trades_df.empty:
            return pd.DataFrame()
        return (
            self.trades_df.groupby("exit_reason")
            .agg(
                count=("net_pnl", "count"),
                avg_pnl=("pnl_pct", "mean"),
                total_pnl=("net_pnl", "sum"),
                win_rate=("net_pnl", lambda x: (x > 0).mean()),
            )
            .round(4)
        )

    def get_monthly_returns(self) -> pd.DataFrame:
        """Retornos mensuales en formato pivot para análisis de estacionalidad."""
        if self.equity_curve.empty:
            return pd.DataFrame()
        monthly = self.equity_curve.resample("ME").last().pct_change().dropna()
        monthly_df = monthly.reset_index()
        monthly_df.columns = ["date", "return"]
        monthly_df["year"] = monthly_df["date"].dt.year
        monthly_df["month"] = monthly_df["date"].dt.strftime("%b")
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        return pivot.round(4)

    def __repr__(self) -> str:
        n = len(self.trades_df) if not self.trades_df.empty else 0
        sr = self.metrics.get("sharpe_ratio", "N/A")
        tr = self.metrics.get("total_return_pct", "N/A")
        return f"BacktestResult({self.strategy_name} | {self.ticker} | Sharpe={sr} | Return={tr}%)"


# ---------------------------------------------------------------------------
# Ejecución directa: prueba rápida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import download_data
    from strategies import STRATEGY_LIBRARY

    print("Descargando datos...")
    df = download_data(["AAPL"], period="5y", cache_dir="data/cache")

    strategy = STRATEGY_LIBRARY["SMA_Cross_20_50"]
    engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
    result = engine.run(df, strategy, ticker="AAPL")
    result.print_report()

    print("\n--- Trades recientes ---")
    print(result.trades_df.tail(5).to_string())
    print("\n--- Exit reasons ---")
    print(result.get_trades_by_reason())
