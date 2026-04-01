"""
data_loader.py
==============
Módulo de descarga y gestión de datos financieros desde Yahoo Finance.
Convierte los datos a formato long (tidy) para uso eficiente en el motor de backtest.

Autor: Backtest Framework
"""

import os
import warnings
import logging
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universo de tickers predefinido
# ---------------------------------------------------------------------------

INDICES = [
    "^GSPC",   # S&P 500
    "^IXIC",   # NASDAQ Composite
    "^DJI",    # Dow Jones
    "^RUT",    # Russell 2000
    "^VIX",    # VIX
    "^FTSE",   # FTSE 100
    "^N225",   # Nikkei 225
    "^HSI",    # Hang Seng
    "GC=F",    # Gold Futures
    "CL=F",    # Crude Oil WTI
]

EQUITIES_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA",
    "HD", "CVX", "MRK", "LLY", "ABBV",
]

ETFS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "HYG", "VNQ", "XLE",
]

DEFAULT_UNIVERSE = EQUITIES_US[:10] + INDICES[:5] + ETFS[:5]


# ---------------------------------------------------------------------------
# Funciones de descarga
# ---------------------------------------------------------------------------

def download_data(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "5y",
    interval: str = "1d",
    cache_dir: Optional[str] = "data/cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Descarga datos OHLCV de Yahoo Finance para una lista de tickers
    y los retorna en formato LONG (tidy).

    Columnas del resultado:
        date, ticker, open, high, low, close, volume, adj_close,
        returns, log_returns

    Parameters
    ----------
    tickers       : lista de símbolos de Yahoo Finance
    start / end   : fechas en formato 'YYYY-MM-DD'; si se omiten se usa `period`
    period        : '1y', '2y', '5y', '10y', 'max', etc.
    interval      : '1d', '1wk', '1mo'
    cache_dir     : directorio para caché local (None = sin caché)
    force_refresh : si True, ignora caché y re-descarga
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    results = []

    for ticker in tickers:
        df = _load_single(
            ticker, start, end, period, interval,
            cache_dir, force_refresh
        )
        if df is not None and not df.empty:
            results.append(df)
        else:
            logger.warning(f"Sin datos para {ticker}")

    if not results:
        raise ValueError("No se pudieron descargar datos para ningún ticker.")

    long_df = pd.concat(results, ignore_index=True)
    long_df = _clean_and_enrich(long_df)
    logger.info(f"Dataset final: {long_df.shape[0]:,} filas | {long_df['ticker'].nunique()} tickers")
    return long_df


def _load_single(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: str,
    interval: str,
    cache_dir: Optional[str],
    force_refresh: bool,
) -> Optional[pd.DataFrame]:
    """Descarga o carga desde caché los datos de un solo ticker."""
    cache_path = None
    if cache_dir:
        safe_ticker = ticker.replace("^", "IDX_").replace("=", "_").replace("-", "_")
        cache_path = os.path.join(cache_dir, f"{safe_ticker}_{interval}.parquet")

    # Intentar cargar caché
    if cache_path and os.path.exists(cache_path) and not force_refresh:
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"{ticker}: cargado desde caché ({len(df)} filas)")
            return df
        except Exception as e:
            logger.warning(f"{ticker}: error leyendo caché: {e}")

    # Descargar desde Yahoo Finance
    try:
        logger.info(f"Descargando {ticker} ...")
        t = yf.Ticker(ticker)

        if start and end:
            raw = t.history(start=start, end=end, interval=interval, auto_adjust=False)
        else:
            raw = t.history(period=period, interval=interval, auto_adjust=False)

        if raw.empty:
            return None

        raw = raw.reset_index()
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

        # Normalizar columna de fecha
        date_col = next((c for c in raw.columns if "date" in c or "datetime" in c), None)
        if date_col:
            raw = raw.rename(columns={date_col: "date"})
            raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()

        # Columna ticker
        raw["ticker"] = ticker

        # Renombrar adj close si existe
        if "adj_close" not in raw.columns and "adjclose" in raw.columns:
            raw = raw.rename(columns={"adjclose": "adj_close"})
        elif "adj close" in raw.columns:
            raw = raw.rename(columns={"adj close": "adj_close"})

        # Asegurar columnas mínimas
        required = ["date", "open", "high", "low", "close", "volume", "ticker"]
        for col in required:
            if col not in raw.columns:
                logger.warning(f"{ticker}: falta columna '{col}'")
                return None

        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw["close"]

        df = raw[["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]].copy()
        df = df[df["close"].notna()]
        df = df.sort_values("date").reset_index(drop=True)

        if cache_path:
            df.to_parquet(cache_path, index=False)
            logger.info(f"{ticker}: guardado en caché")

        return df

    except Exception as e:
        logger.error(f"Error descargando {ticker}: {e}")
        return None


def _clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y enriquece el dataset long con columnas derivadas básicas."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Retornos simples y logarítmicos por ticker
    df["returns"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_returns"] = np.log(df.groupby("ticker")["adj_close"].transform(lambda x: x / x.shift(1)))

    # Rango diario normalizado
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Asegurar tipos
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    return df


# ---------------------------------------------------------------------------
# Utilidades de acceso
# ---------------------------------------------------------------------------

def get_ticker_data(long_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extrae los datos de un ticker específico del dataset long,
    retornando un DataFrame wide con índice de fecha.
    """
    df = long_df[long_df["ticker"] == ticker].copy()
    df = df.set_index("date").sort_index()
    return df


def get_available_tickers(long_df: pd.DataFrame) -> List[str]:
    """Lista de tickers disponibles en el dataset."""
    return sorted(long_df["ticker"].unique().tolist())


def get_data_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen del dataset: ticker, fecha inicio, fecha fin, # observaciones,
    retorno total, volatilidad anualizada.
    """
    summary = []
    for ticker, group in long_df.groupby("ticker"):
        group = group.sort_values("date")
        ret = group["adj_close"].iloc[-1] / group["adj_close"].iloc[0] - 1
        vol = group["returns"].std() * np.sqrt(252)
        summary.append({
            "ticker": ticker,
            "start": group["date"].min().date(),
            "end": group["date"].max().date(),
            "observations": len(group),
            "total_return": round(ret, 4),
            "annual_vol": round(vol, 4),
        })
    return pd.DataFrame(summary).sort_values("ticker").reset_index(drop=True)


def load_default_universe(
    period: str = "5y",
    interval: str = "1d",
    cache_dir: str = "data/cache",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Carga el universo predefinido de tickers (acciones US + índices + ETFs).
    Conveniente para pruebas rápidas.
    """
    logger.info(f"Cargando universo por defecto: {len(DEFAULT_UNIVERSE)} tickers")
    return download_data(
        tickers=DEFAULT_UNIVERSE,
        period=period,
        interval=interval,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# Ejecución directa: prueba rápida
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "^GSPC", "SPY"]
    df = download_data(tickers, period="2y", cache_dir="data/cache")
    print("\n--- Primeras filas ---")
    print(df.head(10).to_string())
    print("\n--- Resumen del dataset ---")
    print(get_data_summary(df).to_string())
