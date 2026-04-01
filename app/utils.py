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

import json
import logging

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# ── Imports del modelo ────────────────────────────────────────────────────────
from data_loader import (
    download_data, get_data_summary, get_available_tickers,
    INDICES, EQUITIES_US, ETFS,
)
from strategies import (
    STRATEGY_LIBRARY, list_strategies, create_custom_strategy,
    add_strategy_to_library, get_available_columns,
)
from backtest_engine import BacktestEngine, BacktestConfig
from strategy_scanner import StrategyScanner
from optimizer import (
    StrategyOptimizer, ParameterGrid,
    make_sma_crossover_strategy, make_ema_crossover_strategy,
    make_rsi_strategy, make_bollinger_strategy, make_macd_strategy,
    PARAM_GRIDS, STRATEGY_FACTORIES,
    optimize_any_strategy, create_auto_param_grid,
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
    
    # Cargar estrategias rastreadas desde disco (solo la primera vez)
    if "tracked_strategies" not in st.session_state:
        st.session_state["tracked_strategies"] = load_tracked_strategies_from_disk()
    
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
        # tracked_strategies ya inicializado arriba desde disco
        "tracking_cache": {},      # Cache de backtests: {(ticker, strategy_name): BacktestResult}
        "custom_strategies": [],   # Lista de estrategias personalizadas
        "builder_conditions": [],  # Condiciones temporales del constructor
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
def cached_download(tickers_tuple: tuple, period: str, interval: str = "1d") -> pd.DataFrame:
    """Descarga datos con caché de Streamlit (TTL 1h).
    
    Args:
        tickers_tuple: Tupla de tickers para caché
        period: Período histórico (e.g., '5y', '1y')
        interval: Intervalo de datos ('1d', '1wk')
    """
    return download_data(list(tickers_tuple), period=period, 
                        interval=interval, cache_dir="data/cache")


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


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_tracked_strategies_from_disk() -> List[Dict]:
    """
    Carga estrategias rastreadas desde JSON en disco.
    
    Returns
    -------
    Lista de estrategias rastreadas. Lista vacía si no existe o hay error.
    """
    filepath = os.path.join(ROOT, "data", "tracked_strategies.json")
    
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("tracked_strategies", [])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON corrupto en tracked_strategies.json: {e}")
        return []
    except Exception as e:
        logger.warning(f"Error cargando tracked strategies: {e}")
        return []


def save_tracked_strategies_to_disk(tracked: List[Dict]) -> bool:
    """
    Guarda estrategias rastreadas a JSON en disco.
    
    Parameters
    ----------
    tracked : Lista de estrategias rastreadas
    
    Returns
    -------
    True si se guardó exitosamente, False si hubo error.
    """
    filepath = os.path.join(ROOT, "data", "tracked_strategies.json")
    
    # Asegurar que el directorio existe
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    except Exception as e:
        logger.error(f"Error creando directorio data/: {e}")
        return False
    
    try:
        data = {
            "tracked_strategies": tracked,
            "version": "1.0",
            "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Error guardando tracked strategies: {e}")
        return False


# ── Tracking helpers ──────────────────────────────────────────────────────────

def add_tracked_strategy(ticker: str, strategy_name: str) -> bool:
    """Agrega una estrategia al seguimiento y persiste en disco."""
    tracked = st.session_state.get("tracked_strategies", [])
    
    # Verificar si ya existe
    for item in tracked:
        if item["ticker"] == ticker and item["strategy_name"] == strategy_name:
            return False
    
    tracked.append({
        "ticker": ticker,
        "strategy_name": strategy_name,
        "added_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    st.session_state["tracked_strategies"] = tracked
    
    # Guardar a disco
    save_tracked_strategies_to_disk(tracked)
    
    return True


def remove_tracked_strategy(ticker: str, strategy_name: str) -> bool:
    """Elimina una estrategia del seguimiento y actualiza disco."""
    tracked = st.session_state.get("tracked_strategies", [])
    initial_len = len(tracked)
    
    tracked = [
        item for item in tracked
        if not (item["ticker"] == ticker and item["strategy_name"] == strategy_name)
    ]
    
    st.session_state["tracked_strategies"] = tracked
    
    # Limpiar cache
    cache_key = (ticker, strategy_name)
    cache = st.session_state.get("tracking_cache", {})
    if cache_key in cache:
        del cache[cache_key]
        st.session_state["tracking_cache"] = cache
    
    # Guardar a disco si hubo cambios
    if len(tracked) < initial_len:
        save_tracked_strategies_to_disk(tracked)
    
    return len(tracked) < initial_len


def get_tracked_strategies() -> List[Dict]:
    """Retorna la lista de estrategias en seguimiento."""
    return st.session_state.get("tracked_strategies", [])


def run_tracking_backtest(ticker: str, strategy_name: str, df: pd.DataFrame) -> Optional[Any]:
    """Ejecuta backtest para una estrategia en seguimiento (con cache)."""
    cache_key = (ticker, strategy_name)
    cache = st.session_state.get("tracking_cache", {})
    
    # Verificar cache
    if cache_key in cache:
        return cache[cache_key]
    
    # Ejecutar backtest
    try:
        strategy = STRATEGY_LIBRARY.get(strategy_name)
        if strategy is None:
            return None
        
        engine = BacktestEngine(config=BacktestConfig(
            initial_capital=100_000,
            commission_pct=0.001,
            slippage_pct=0.0005,
        ))
        
        result = engine.run(df, strategy, ticker=ticker, add_indicators=True)
        
        # Guardar en cache
        cache[cache_key] = result
        st.session_state["tracking_cache"] = cache
        
        return result
    except Exception as e:
        st.error(f"Error en backtest {ticker}/{strategy_name}: {e}")
        return None


def detect_active_trades(result: Any) -> Optional[Dict]:
    """Detecta si hay un trade activo en el último día del backtest."""
    if result is None or not hasattr(result, "df_with_signals"):
        return None
    
    df = result.df_with_signals
    if df is None or df.empty:
        return None
    
    # Obtener trades cerrados
    trades = result.trades if hasattr(result, "trades") else []
    
    # Si hay trades, verificar el último
    if len(trades) > 0:
        last_trade = trades[-1]
        
        # Si el último trade no tiene exit_date, está activo
        if last_trade.exit_date is None or pd.isna(last_trade.exit_date):
            last_date = df.index[-1]
            last_price = df["close"].iloc[-1]
            
            # Calcular retorno actual
            entry_price = last_trade.entry_price
            pnl_pct = ((last_price - entry_price) / entry_price) * 100
            
            # Calcular días en posición
            days_held = (last_date - last_trade.entry_date).days
            
            return {
                "status": "ACTIVO",
                "entry_date": last_trade.entry_date.strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 2),
                "current_price": round(last_price, 2),
                "shares": round(last_trade.shares, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "stop_loss": round(last_trade.stop_loss_price, 2) if last_trade.stop_loss_price else None,
                "take_profit": round(last_trade.take_profit_price, 2) if last_trade.take_profit_price else None,
            }
    
    return None


def detect_signals_next_bar(result: Any) -> Dict:
    """Detecta señales que se ejecutarían en la próxima vela."""
    signals = {
        "buy_signal": False,
        "sell_signal": False,
        "stop_signal": False,
        "profit_signal": False,
    }
    
    if result is None or not hasattr(result, "df_with_signals"):
        return signals
    
    df = result.df_with_signals
    if df is None or df.empty:
        return signals
    
    # Última barra
    last_idx = df.index[-1]
    last_row = df.loc[last_idx]
    
    # Verificar señales de entrada/salida
    if hasattr(last_row, "entry_signal") and last_row.entry_signal:
        signals["buy_signal"] = True
    
    if hasattr(last_row, "exit_signal") and last_row.exit_signal:
        signals["sell_signal"] = True
    
    # Verificar si hay trade activo para stop/profit
    active_trade = detect_active_trades(result)
    if active_trade:
        current_price = active_trade["current_price"]
        
        if active_trade["stop_loss"]:
            if current_price <= active_trade["stop_loss"]:
                signals["stop_signal"] = True
        
        if active_trade["take_profit"]:
            if current_price >= active_trade["take_profit"]:
                signals["profit_signal"] = True
    
    return signals
