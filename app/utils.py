"""
app/utils.py
============
Utilidades compartidas para la app Streamlit.
Manejo de estado global, helpers de UI y wrappers ligeros
que SOLO llaman al modelo ya construido — sin lógica analítica propia.
"""

from __future__ import annotations

import sys
import os

# Asegurar que el directorio raíz del proyecto esté en el path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# ── Imports del modelo ────────────────────────────────────────────────────────
from data_loader import (
    download_data, get_data_summary, get_available_tickers,
    INDICES, EQUITIES_US, ETFS,
)
from strategies import STRATEGY_LIBRARY, list_strategies
from backtest_engine import BacktestEngine, BacktestConfig
from strategy_scanner import StrategyScanner
from optimizer import (
    StrategyOptimizer, ParameterGrid,
    make_sma_crossover_strategy, make_ema_crossover_strategy,
    make_rsi_strategy, make_bollinger_strategy, make_macd_strategy,
    PARAM_GRIDS, STRATEGY_FACTORIES,
)
from visualization import (
    plot_equity_curve, plot_drawdown, plot_trades,
    plot_monthly_returns, plot_rolling_metrics,
    plot_candlestick_signals, plot_scanner_summary,
    plot_optimization_heatmap, plot_monte_carlo,
    plot_wfo_results, plot_returns_distribution,
    plot_full_dashboard, plot_sensitivity,
)

# ── Constantes ────────────────────────────────────────────────────────────────

DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "UNH",
    "^GSPC", "^IXIC", "^DJI",
    "SPY", "QQQ", "GLD", "TLT",
]

TICKER_PRESETS: Dict[str, List[str]] = {
    "Mag-7 + indices": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
                         "^GSPC", "^IXIC", "SPY"],
    "Blue chips US": ["JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK"],
    "ETFs principales": ["SPY", "QQQ", "IWM", "GLD", "TLT", "HYG", "VNQ", "XLE", "EFA", "EEM"],
    "Índices globales": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^FTSE", "^N225", "^HSI"],
}

PERIOD_OPTIONS = {
    "1 año": "1y",
    "2 años": "2y",
    "3 años": "3y",
    "5 años": "5y",
    "10 años": "10y",
    "Máximo disponible": "max",
}

STRATEGY_TYPE_OPTIONS = {
    "SMA Crossover": "sma_crossover",
    "EMA Crossover": "ema_crossover",
    "RSI Mean Reversion": "rsi",
    "Bollinger Bands": "bollinger",
    "MACD": "macd",
}

METRIC_OPTIONS = {
    "Sharpe Ratio": "sharpe_ratio",
    "Sortino Ratio": "sortino_ratio",
    "Calmar Ratio": "calmar_ratio",
    "Retorno Total (%)": "total_return_pct",
    "CAGR (%)": "cagr_pct",
    "Profit Factor": "profit_factor",
}

# ── Session State helpers ─────────────────────────────────────────────────────

def init_state() -> None:
    """Inicializa todos los valores del session state con sus defaults."""
    defaults = {
        "ticker_list": DEFAULT_TICKERS.copy(),
        "long_df": None,
        "data_summary": None,
        "selected_period": "5y",
        "scanner_ticker": None,
        "scanner_results": None,
        "scanner_summary_df": None,
        "selected_strategy_result": None,
        "opt_ticker": None,
        "opt_strategy_type": None,
        "opt_report": None,
        "opt_grid_df": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_long_df() -> Optional[pd.DataFrame]:
    return st.session_state.get("long_df")


def get_ticker_list() -> List[str]:
    return st.session_state.get("ticker_list", DEFAULT_TICKERS.copy())


def set_ticker_list(tickers: List[str]) -> None:
    st.session_state["ticker_list"] = tickers


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_download(tickers_tuple: tuple, period: str) -> pd.DataFrame:
    """Descarga datos con caché de Streamlit (TTL 1h)."""
    return download_data(list(tickers_tuple), period=period, cache_dir="data/cache")


@st.cache_data(show_spinner=False)
def cached_summary(df_hash: int) -> pd.DataFrame:
    df = st.session_state.get("long_df")
    if df is None:
        return pd.DataFrame()
    return get_data_summary(df)


# ── Backtest config desde sidebar ─────────────────────────────────────────────

def backtest_config_sidebar() -> BacktestConfig:
    """
    Renderiza controles de configuración del backtest en el sidebar
    y retorna un BacktestConfig.
    """
    with st.sidebar.expander("⚙️ Config. Backtest", expanded=False):
        capital = st.number_input(
            "Capital inicial ($)", min_value=1_000, max_value=10_000_000,
            value=100_000, step=10_000, format="%d",
        )
        commission = st.slider(
            "Comisión (%)", min_value=0.0, max_value=1.0,
            value=0.1, step=0.05, format="%.2f"
        ) / 100
        slippage = st.slider(
            "Slippage (%)", min_value=0.0, max_value=0.5,
            value=0.05, step=0.01, format="%.2f"
        ) / 100
        rf = st.slider(
            "Tasa libre riesgo (%)", min_value=0.0, max_value=10.0,
            value=4.0, step=0.25, format="%.2f"
        ) / 100
        exec_price = st.selectbox(
            "Precio ejecución", ["open", "close"], index=0
        )
    return BacktestConfig(
        initial_capital=float(capital),
        commission_pct=commission,
        slippage_pct=slippage,
        risk_free_rate=rf,
        execution_price=exec_price,
    )


# ── UI helpers ────────────────────────────────────────────────────────────────

def metric_card(label: str, value, delta=None, prefix: str = "", suffix: str = "") -> None:
    """Renderiza una métrica con estilo."""
    if isinstance(value, float):
        display = f"{prefix}{value:,.4f}{suffix}"
    elif isinstance(value, int):
        display = f"{prefix}{value:,}{suffix}"
    else:
        display = f"{prefix}{value}{suffix}"
    st.metric(label=label, value=display, delta=delta)


def color_metric(value: float, good_positive: bool = True) -> str:
    """Retorna clase CSS de color según si el valor es bueno o malo."""
    if good_positive:
        return "🟢" if value >= 0 else "🔴"
    else:
        return "🔴" if value >= 0 else "🟢"


def format_pct(value: Optional[float], decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_num(value: Optional[float], decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}"


def show_metrics_grid(metrics: dict, n_cols: int = 4) -> None:
    """
    Muestra las métricas del backtest en una cuadrícula de tarjetas.
    Solo recibe el dict de métricas — no hace cálculos propios.
    """
    if not metrics or "error" in metrics:
        st.error(f"Error: {metrics.get('error', 'Sin datos')}")
        return

    rows = [
        ("Retorno Total", format_pct(metrics.get("total_return_pct")), None),
        ("CAGR", format_pct(metrics.get("cagr_pct")), None),
        ("Sharpe Ratio", format_num(metrics.get("sharpe_ratio")), None),
        ("Sortino Ratio", format_num(metrics.get("sortino_ratio")), None),
        ("Calmar Ratio", format_num(metrics.get("calmar_ratio")), None),
        ("Omega Ratio", format_num(metrics.get("omega_ratio")), None),
        ("Max Drawdown", format_pct(metrics.get("max_drawdown_pct")), None),
        ("Avg Drawdown", format_pct(metrics.get("avg_drawdown_pct")), None),
        ("Vol. Anual", format_pct(metrics.get("annual_volatility_pct")), None),
        ("VaR 95%", format_pct(metrics.get("var_95_pct"), 4), None),
        ("Profit Factor", format_num(metrics.get("profit_factor")), None),
        ("Win Rate", format_pct(metrics.get("win_rate_pct")), None),
        ("# Trades", str(int(metrics.get("n_trades", 0))), None),
        ("Avg Holding", f"{metrics.get('avg_holding_days', 0):.1f}d", None),
        ("Expectancy", format_pct(metrics.get("expectancy_pct"), 4), None),
        ("Recovery Factor", format_num(metrics.get("recovery_factor")), None),
        ("Meses positivos", format_pct(metrics.get("pct_positive_months")), None),
        ("CVaR 95%", format_pct(metrics.get("cvar_95_pct"), 4), None),
        ("Profit Factor", format_num(metrics.get("profit_factor")), None),
        ("Años backtest", f"{metrics.get('backtest_years', 0):.1f}", None),
    ]
    # Deduplicar
    seen = set()
    unique_rows = []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            unique_rows.append(r)

    cols = st.columns(n_cols)
    for i, (label, value, delta) in enumerate(unique_rows):
        with cols[i % n_cols]:
            st.metric(label=label, value=value, delta=delta)


def styled_dataframe(df: pd.DataFrame, pct_cols: List[str] = None) -> None:
    """Muestra un DataFrame con formato de colores para columnas clave."""
    if df.empty:
        st.info("Sin datos para mostrar.")
        return

    def color_negative_red(val):
        if isinstance(val, (int, float)) and not np.isnan(val):
            color = "#d4edda" if val > 0 else "#f8d7da" if val < 0 else ""
            return f"background-color: {color}"
        return ""

    pct_cols = pct_cols or []
    try:
        fmt = {col: "{:.2f}%" for col in pct_cols if col in df.columns}
        styled = df.style.format(fmt, na_rep="N/A")
        for col in pct_cols:
            if col in df.columns:
                styled = styled.applymap(color_negative_red, subset=[col])
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(df, use_container_width=True)


def page_header(title: str, subtitle: str = "") -> None:
    """Renderiza el header estándar de cada página."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color: #888; margin-top:-10px'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.divider()
