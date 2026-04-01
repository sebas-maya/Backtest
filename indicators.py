"""
indicators.py
=============
Librería de indicadores técnicos implementados sobre pandas.
Todos los indicadores operan sobre un DataFrame con columnas OHLCV estándar
y retornan Series o DataFrames con nombres descriptivos.

Convención de nombres de columnas generadas:
    sma_{n}, ema_{n}, wma_{n}
    rsi_{n}
    macd, macd_signal, macd_hist
    bb_upper_{n}, bb_mid_{n}, bb_lower_{n}, bb_width_{n}, bb_pct_{n}
    atr_{n}
    stoch_k_{n}, stoch_d_{n}
    adx_{n}, di_plus_{n}, di_minus_{n}
    obv
    cci_{n}
    williams_r_{n}
    mfi_{n}
    dema_{n}, tema_{n}
    vwap
    psar
    donchian_upper_{n}, donchian_lower_{n}, donchian_mid_{n}
    roc_{n}
    cmf_{n}
    keltner_upper_{n}, keltner_lower_{n}, keltner_mid_{n}
    ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a, ichimoku_senkou_b
    supertrend_{n}_{m}

Autor: Backtest Framework
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Tendencia
# ---------------------------------------------------------------------------

def sma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Simple Moving Average."""
    _validate(df, col)
    return df[col].rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def ema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Exponential Moving Average."""
    _validate(df, col)
    return _ema(df[col], period).rename(f"ema_{period}")


def wma(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Weighted Moving Average."""
    _validate(df, col)
    weights = np.arange(1, period + 1)

    def _wma(x):
        if len(x) < period:
            return np.nan
        return np.dot(x, weights) / weights.sum()

    return df[col].rolling(period).apply(_wma, raw=True).rename(f"wma_{period}")


def dema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Double Exponential Moving Average: 2*EMA - EMA(EMA)."""
    _validate(df, col)
    e = _ema(df[col], period)
    return (2 * e - _ema(e, period)).rename(f"dema_{period}")


def tema(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Triple Exponential Moving Average."""
    _validate(df, col)
    e1 = _ema(df[col], period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return (3 * e1 - 3 * e2 + e3).rename(f"tema_{period}")


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (acumulado por día; para series largas
    se reinicia por fecha si hay columna 'date').
    """
    _validate(df, "high", "low", "close", "volume")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]

    if "date" in df.columns:
        date_key = pd.to_datetime(df["date"]).dt.date
        cum_pv = pv.groupby(date_key).cumsum()
        cum_vol = df["volume"].groupby(date_key).cumsum()
    else:
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()

    return (cum_pv / cum_vol).rename("vwap")


def psar(
    df: pd.DataFrame,
    initial_af: float = 0.02,
    max_af: float = 0.20,
    step_af: float = 0.02,
) -> pd.Series:
    """Parabolic SAR."""
    _validate(df, "high", "low", "close")
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    sar = np.full(n, np.nan)
    bull = True
    af = initial_af
    ep = lows[0]
    sar[0] = highs[0]

    for i in range(1, n):
        if bull:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = min(sar[i], lows[i - 1], lows[max(0, i - 2)])
            if lows[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = lows[i]
                af = initial_af
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + step_af, max_af)
        else:
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            sar[i] = max(sar[i], highs[i - 1], highs[max(0, i - 2)])
            if highs[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = highs[i]
                af = initial_af
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + step_af, max_af)

    return pd.Series(sar, index=df.index, name="psar")


def donchian_channel(
    df: pd.DataFrame, period: int = 20
) -> pd.DataFrame:
    """Donchian Channel (upper, lower, mid)."""
    _validate(df, "high", "low")
    upper = df["high"].rolling(period).max().rename(f"donchian_upper_{period}")
    lower = df["low"].rolling(period).min().rename(f"donchian_lower_{period}")
    mid = ((upper + lower) / 2).rename(f"donchian_mid_{period}")
    return pd.concat([upper, lower, mid], axis=1)


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Retorna columnas: supertrend_{period}_{m}, supertrend_dir_{period}_{m}
    donde dir = 1 (alcista) ó -1 (bajista).
    """
    _validate(df, "high", "low", "close")
    atr_series = atr(df, period)
    m = multiplier
    col = f"supertrend_{period}_{int(m)}"

    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + m * atr_series
    lower_band = hl2 - m * atr_series

    close = df["close"].values
    ub = upper_band.values
    lb = lower_band.values
    n = len(df)
    st = np.full(n, np.nan)
    direction = np.zeros(n)

    for i in range(1, n):
        if np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        # Adjust bands
        if lb[i] > lb[i - 1] or close[i - 1] < lb[i - 1]:
            lb[i] = lb[i]
        else:
            lb[i] = lb[i - 1]

        if ub[i] < ub[i - 1] or close[i - 1] > ub[i - 1]:
            ub[i] = ub[i]
        else:
            ub[i] = ub[i - 1]

        if not np.isnan(st[i - 1]):
            if st[i - 1] == ub[i - 1]:
                st[i] = lb[i] if close[i] > ub[i] else ub[i]
            else:
                st[i] = ub[i] if close[i] < lb[i] else lb[i]
        else:
            st[i] = lb[i] if close[i] > ub[i] else ub[i]

        direction[i] = 1 if close[i] > st[i] else -1

    return pd.DataFrame(
        {col: st, f"supertrend_dir_{period}_{int(m)}": direction},
        index=df.index,
    )


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku Cloud (parámetros estándar 9/26/52).
    Columnas: ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
              ichimoku_senkou_b, ichimoku_chikou
    """
    _validate(df, "high", "low", "close")

    def midpoint(h, l, p):
        return ((h.rolling(p).max() + l.rolling(p).min()) / 2)

    tenkan = midpoint(df["high"], df["low"], 9).rename("ichimoku_tenkan")
    kijun = midpoint(df["high"], df["low"], 26).rename("ichimoku_kijun")
    senkou_a = ((tenkan + kijun) / 2).shift(26).rename("ichimoku_senkou_a")
    senkou_b = midpoint(df["high"], df["low"], 52).shift(26).rename("ichimoku_senkou_b")
    chikou = df["close"].shift(-26).rename("ichimoku_chikou")

    return pd.concat([tenkan, kijun, senkou_a, senkou_b, chikou], axis=1)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    """Relative Strength Index."""
    _validate(df, col)
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> pd.DataFrame:
    """MACD, Signal y Histogram."""
    _validate(df, col)
    fast_ema = _ema(df[col], fast)
    slow_ema = _ema(df[col], slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
        index=df.index,
    )


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Stochastic Oscillator %K y %D."""
    _validate(df, "high", "low", "close")
    lowest = df["low"].rolling(k_period).min()
    highest = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest + 1e-10)
    d = k.rolling(d_period).mean()
    return pd.DataFrame(
        {f"stoch_k_{k_period}": k, f"stoch_d_{k_period}": d},
        index=df.index,
    )


def roc(df: pd.DataFrame, period: int = 12, col: str = "close") -> pd.Series:
    """Rate of Change (%)."""
    _validate(df, col)
    return (df[col].pct_change(period) * 100).rename(f"roc_{period}")


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    _validate(df, "high", "low", "close")
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    return (-(highest - df["close"]) / (highest - lowest + 1e-10) * 100).rename(
        f"williams_r_{period}"
    )


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    _validate(df, "high", "low", "close")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean_tp = typical.rolling(period).mean()
    mean_dev = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((typical - mean_tp) / (0.015 * mean_dev + 1e-10)).rename(f"cci_{period}")


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    _validate(df, "high", "low", "close", "volume")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mf = typical * df["volume"]
    delta = typical.diff()

    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta <= 0, 0).rolling(period).sum()

    mfr = pos_mf / (neg_mf + 1e-10)
    return (100 - 100 / (1 + mfr)).rename(f"mfi_{period}")


# ---------------------------------------------------------------------------
# Volatilidad
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    _validate(df, "high", "low", "close")
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename(f"atr_{period}")


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands (upper, mid, lower, width, %B)."""
    _validate(df, col)
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / (mid + 1e-10)
    pct_b = (df[col] - lower) / (upper - lower + 1e-10)
    suffix = f"{period}"
    return pd.DataFrame(
        {
            f"bb_upper_{suffix}": upper,
            f"bb_mid_{suffix}": mid,
            f"bb_lower_{suffix}": lower,
            f"bb_width_{suffix}": width,
            f"bb_pct_{suffix}": pct_b,
        },
        index=df.index,
    )


def keltner_channel(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0
) -> pd.DataFrame:
    """Keltner Channel."""
    _validate(df, "high", "low", "close")
    mid = _ema(df["close"], period)
    atr_val = atr(df, period)
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return pd.DataFrame(
        {
            f"keltner_upper_{period}": upper,
            f"keltner_mid_{period}": mid,
            f"keltner_lower_{period}": lower,
        },
        index=df.index,
    )


def historical_volatility(
    df: pd.DataFrame, period: int = 20, col: str = "close", annualize: int = 252
) -> pd.Series:
    """Volatilidad histórica (desv. estándar de retornos logarítmicos anualizada)."""
    _validate(df, col)
    log_ret = np.log(df[col] / df[col].shift(1))
    return (log_ret.rolling(period).std() * np.sqrt(annualize)).rename(f"hvol_{period}")


# ---------------------------------------------------------------------------
# Volumen
# ---------------------------------------------------------------------------

def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    _validate(df, "close", "volume")
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum().rename("obv")


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    _validate(df, "high", "low", "close", "volume")
    hl_range = df["high"] - df["low"] + 1e-10
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mfv = clv * df["volume"]
    return (mfv.rolling(period).sum() / (df["volume"].rolling(period).sum() + 1e-10)).rename(
        f"cmf_{period}"
    )


# ---------------------------------------------------------------------------
# Tendencia (ADX)
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index + DI+, DI-.
    Columnas: adx_{n}, di_plus_{n}, di_minus_{n}
    """
    _validate(df, "high", "low", "close")
    atr_val = atr(df, period)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    smooth_pos = pos_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    smooth_neg = neg_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    di_plus = 100 * smooth_pos / (atr_val + 1e-10)
    di_minus = 100 * smooth_neg / (atr_val + 1e-10)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame(
        {
            f"adx_{period}": adx_val,
            f"di_plus_{period}": di_plus,
            f"di_minus_{period}": di_minus,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Función maestra: agregar TODOS los indicadores a un DataFrame
# ---------------------------------------------------------------------------

def add_all_indicators(
    df: pd.DataFrame,
    sma_periods: list = [10, 20, 50, 100, 200],
    ema_periods: list = [9, 21, 50, 100, 200],
    rsi_periods: list = [14],
    atr_periods: list = [14],
    bb_periods: list = [20],
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    Agrega un conjunto completo de indicadores técnicos al DataFrame.
    Retorna el DataFrame original con columnas adicionales.
    """
    out = df.copy()

    # Medias móviles
    for p in sma_periods:
        out[f"sma_{p}"] = sma(out, p)
    for p in ema_periods:
        out[f"ema_{p}"] = ema(out, p)

    # Momentum
    for p in rsi_periods:
        out[f"rsi_{p}"] = rsi(out, p)

    macd_df = macd(out)
    out = pd.concat([out, macd_df], axis=1)

    stoch_df = stochastic(out)
    out = pd.concat([out, stoch_df], axis=1)

    out["roc_12"] = roc(out, 12)
    out["williams_r_14"] = williams_r(out, 14)
    out["cci_20"] = cci(out, 20)

    # Volatilidad
    for p in atr_periods:
        out[f"atr_{p}"] = atr(out, p)

    for p in bb_periods:
        bb_df = bollinger_bands(out, p)
        out = pd.concat([out, bb_df], axis=1)

    out["hvol_20"] = historical_volatility(out, 20)

    # Volumen
    out["obv"] = obv(out)
    out["cmf_20"] = cmf(out, 20)
    out["mfi_14"] = mfi(out, 14)

    if include_advanced:
        adx_df = adx(out, 14)
        out = pd.concat([out, adx_df], axis=1)

        kc_df = keltner_channel(out, 20)
        out = pd.concat([out, kc_df], axis=1)

        out["wma_20"] = wma(out, 20)
        out["dema_20"] = dema(out, 20)
        out["tema_20"] = tema(out, 20)

        don_df = donchian_channel(out, 20)
        out = pd.concat([out, don_df], axis=1)

        # SuperTrend
        try:
            st_df = supertrend(out, period=10, multiplier=3.0)
            out = pd.concat([out, st_df], axis=1)
        except Exception:
            pass

        # VWAP
        try:
            out["vwap"] = vwap(out)
        except Exception:
            pass

    return out


def get_indicator_catalog() -> dict:
    """Retorna un catálogo de todos los indicadores disponibles con su descripción."""
    return {
        "sma": "Simple Moving Average - promedio simple de precios",
        "ema": "Exponential Moving Average - promedio ponderado exponencialmente",
        "wma": "Weighted Moving Average - promedio ponderado lineal",
        "dema": "Double EMA - reduce lag del EMA",
        "tema": "Triple EMA - mayor reducción de lag",
        "rsi": "Relative Strength Index - momentum 0-100, sobrecompra/venta",
        "macd": "MACD - divergencia convergencia de medias móviles",
        "stochastic": "Stochastic Oscillator - posición del cierre en rango HL",
        "roc": "Rate of Change - velocidad del cambio de precio",
        "williams_r": "Williams %R - momentum negativo",
        "cci": "Commodity Channel Index - desviación del precio típico",
        "mfi": "Money Flow Index - RSI ponderado por volumen",
        "atr": "Average True Range - volatilidad basada en rango",
        "bollinger_bands": "Bandas de Bollinger - canal de volatilidad",
        "keltner_channel": "Canal Keltner - canal basado en ATR",
        "historical_volatility": "Volatilidad histórica anualizada",
        "obv": "On-Balance Volume - volumen acumulado por dirección",
        "cmf": "Chaikin Money Flow - flujo de dinero",
        "adx": "Average Directional Index - fuerza de tendencia",
        "vwap": "VWAP - precio promedio ponderado por volumen",
        "psar": "Parabolic SAR - trailing stop de tendencia",
        "donchian_channel": "Canal Donchian - máximos/mínimos del período",
        "supertrend": "SuperTrend - señal de tendencia con ATR",
        "ichimoku": "Ichimoku Cloud - sistema japonés multi-componente",
    }


if __name__ == "__main__":
    # Prueba rápida
    import yfinance as yf
    raw = yf.Ticker("AAPL").history(period="1y")
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    out = add_all_indicators(raw)
    print(f"Columnas generadas: {len(out.columns)}")
    print(out[["close", "sma_20", "ema_20", "rsi_14", "macd", "atr_14"]].tail(5))
