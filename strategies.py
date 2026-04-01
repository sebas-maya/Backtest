"""
strategies.py
=============
Módulo de definición flexible de estrategias de inversión.

Arquitectura:
- `Signal`: condición atómica evaluable sobre un DataFrame (crossover, threshold, etc.)
- `Rule`: combina señales con operadores lógicos (AND, OR, NOT)
- `Strategy`: encapsula reglas de entrada + salida + gestión de riesgo + sizing

DSL de uso rápido:
    from strategies import Strategy, Signal, Rule, STRATEGY_LIBRARY

Ejemplo:
    strategy = Strategy(
        name="SMA_Crossover_100_20",
        entry=Rule([
            Signal.crossover("close", "sma_100"),
            Signal.above_threshold("volume", "volume_sma_20", factor=1.2),
        ]),
        exit=Rule([
            Signal.crossunder("close", "sma_20"),
        ], combinator="OR"),
        stop_loss=0.05,       # 5%
        take_profit=0.15,     # 15%
        max_holding_days=30,
        position_sizing="fixed",
    )

Autor: Backtest Framework
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union, Literal, Dict, Any

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Signal: condición atómica evaluable
# ---------------------------------------------------------------------------

SignalFn = Callable[[pd.DataFrame], pd.Series]


class Signal:
    """
    Condición atómica que evalúa sobre un DataFrame y retorna
    una Serie booleana.

    Se puede crear directamente pasando una función, o usando
    los constructores estáticos de conveniencia.
    """

    def __init__(self, fn: SignalFn, description: str = ""):
        self._fn = fn
        self.description = description

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        result = self._fn(df)
        return result.fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # Constructores estáticos
    # ------------------------------------------------------------------

    @staticmethod
    def crossover(col_a: str, col_b: Union[str, float]) -> "Signal":
        """col_a cruza HACIA ARRIBA col_b (o un valor fijo)."""
        if isinstance(col_b, (int, float)):
            threshold = col_b
            fn = lambda df: (df[col_a] > threshold) & (df[col_a].shift(1) <= threshold)
            desc = f"{col_a} cross up {threshold}"
        else:
            fn = lambda df: (df[col_a] > df[col_b]) & (df[col_a].shift(1) <= df[col_b].shift(1))
            desc = f"{col_a} cross up {col_b}"
        return Signal(fn, desc)

    @staticmethod
    def crossunder(col_a: str, col_b: Union[str, float]) -> "Signal":
        """col_a cruza HACIA ABAJO col_b (o un valor fijo)."""
        if isinstance(col_b, (int, float)):
            threshold = col_b
            fn = lambda df: (df[col_a] < threshold) & (df[col_a].shift(1) >= threshold)
            desc = f"{col_a} cross down {threshold}"
        else:
            fn = lambda df: (df[col_a] < df[col_b]) & (df[col_a].shift(1) >= df[col_b].shift(1))
            desc = f"{col_a} cross down {col_b}"
        return Signal(fn, desc)

    @staticmethod
    def above(col_a: str, col_b: Union[str, float]) -> "Signal":
        """col_a ESTÁ POR ENCIMA de col_b."""
        if isinstance(col_b, (int, float)):
            fn = lambda df: df[col_a] > col_b
        else:
            fn = lambda df: df[col_a] > df[col_b]
        return Signal(fn, f"{col_a} > {col_b}")

    @staticmethod
    def below(col_a: str, col_b: Union[str, float]) -> "Signal":
        """col_a ESTÁ POR DEBAJO de col_b."""
        if isinstance(col_b, (int, float)):
            fn = lambda df: df[col_a] < col_b
        else:
            fn = lambda df: df[col_a] < df[col_b]
        return Signal(fn, f"{col_a} < {col_b}")

    @staticmethod
    def above_threshold(col: str, ref_col: str, factor: float = 1.0) -> "Signal":
        """col > factor * ref_col  (útil para volumen > media_volumen * 1.5)."""
        fn = lambda df: df[col] > factor * df[ref_col]
        return Signal(fn, f"{col} > {factor}x {ref_col}")

    @staticmethod
    def between(col: str, low: float, high: float) -> "Signal":
        """low < col < high."""
        fn = lambda df: (df[col] > low) & (df[col] < high)
        return Signal(fn, f"{low} < {col} < {high}")

    @staticmethod
    def rising(col: str, periods: int = 1) -> "Signal":
        """col está subiendo respecto a `periods` velas atrás."""
        fn = lambda df: df[col] > df[col].shift(periods)
        return Signal(fn, f"{col} rising ({periods}p)")

    @staticmethod
    def falling(col: str, periods: int = 1) -> "Signal":
        """col está bajando respecto a `periods` velas atrás."""
        fn = lambda df: df[col] < df[col].shift(periods)
        return Signal(fn, f"{col} falling ({periods}p)")

    @staticmethod
    def pct_change_above(col: str, pct: float, periods: int = 1) -> "Signal":
        """Retorno en `periods` > pct (0.02 = 2%)."""
        fn = lambda df: df[col].pct_change(periods) > pct
        return Signal(fn, f"{col} change > {pct*100:.1f}% in {periods}p")

    @staticmethod
    def pct_change_below(col: str, pct: float, periods: int = 1) -> "Signal":
        """Retorno en `periods` < pct."""
        fn = lambda df: df[col].pct_change(periods) < pct
        return Signal(fn, f"{col} change < {pct*100:.1f}% in {periods}p")

    @staticmethod
    def value_in_range(col: str, low_col: str, high_col: str) -> "Signal":
        """col está entre low_col y high_col (e.g., precio dentro de Bollinger)."""
        fn = lambda df: (df[col] > df[low_col]) & (df[col] < df[high_col])
        return Signal(fn, f"{col} in [{low_col}, {high_col}]")

    @staticmethod
    def divergence_bull(
        price_col: str, indicator_col: str, lookback: int = 14
    ) -> "Signal":
        """
        Divergencia alcista simplificada:
        precio hace mínimo más bajo pero indicador hace mínimo más alto.
        """
        def fn(df: pd.DataFrame) -> pd.Series:
            p_min = df[price_col].rolling(lookback).min()
            i_min = df[indicator_col].rolling(lookback).min()
            price_lower = df[price_col] < p_min.shift(lookback)
            ind_higher = df[indicator_col] > i_min.shift(lookback)
            return price_lower & ind_higher
        return Signal(fn, f"Bullish divergence {price_col}/{indicator_col}")

    @staticmethod
    def n_days_after_entry(entry_series: pd.Series, n: int) -> "Signal":
        """
        Señal que se activa exactamente n días después de una entrada.
        Útil como condición de salida temporal.
        Nota: se evalúa en el backtest, no aquí directamente.
        """
        def fn(df: pd.DataFrame) -> pd.Series:
            result = pd.Series(False, index=df.index)
            entry_idx = entry_series[entry_series].index
            for idx in entry_idx:
                try:
                    pos = df.index.get_loc(idx)
                    target_pos = pos + n
                    if target_pos < len(df):
                        result.iloc[target_pos] = True
                except Exception:
                    pass
            return result
        return Signal(fn, f"Exit after {n} days")

    @staticmethod
    def custom(fn: SignalFn, description: str = "custom") -> "Signal":
        """Señal completamente personalizada con función lambda."""
        return Signal(fn, description)

    # ------------------------------------------------------------------
    # Operadores lógicos entre señales
    # ------------------------------------------------------------------

    def __and__(self, other: "Signal") -> "Signal":
        fn = lambda df: self.evaluate(df) & other.evaluate(df)
        return Signal(fn, f"({self.description} AND {other.description})")

    def __or__(self, other: "Signal") -> "Signal":
        fn = lambda df: self.evaluate(df) | other.evaluate(df)
        return Signal(fn, f"({self.description} OR {other.description})")

    def __invert__(self) -> "Signal":
        fn = lambda df: ~self.evaluate(df)
        return Signal(fn, f"NOT ({self.description})")

    def __repr__(self) -> str:
        return f"Signal({self.description})"


# ---------------------------------------------------------------------------
# Rule: combina múltiples señales
# ---------------------------------------------------------------------------

class Rule:
    """
    Combina una lista de Signal con operador AND u OR.
    Permite construir reglas de entrada/salida complejas de forma legible.
    """

    def __init__(
        self,
        signals: List[Signal],
        combinator: Literal["AND", "OR"] = "AND",
        description: str = "",
    ):
        self.signals = signals
        self.combinator = combinator.upper()
        self.description = description or f" {combinator} ".join(
            s.description for s in signals
        )

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if not self.signals:
            return pd.Series(False, index=df.index)
        result = self.signals[0].evaluate(df)
        for sig in self.signals[1:]:
            if self.combinator == "AND":
                result = result & sig.evaluate(df)
            else:
                result = result | sig.evaluate(df)
        return result

    def __and__(self, other: "Rule") -> "Rule":
        combined = Signal(
            lambda df, a=self, b=other: a.evaluate(df) & b.evaluate(df),
            f"{self.description} AND {other.description}",
        )
        return Rule([combined])

    def __or__(self, other: "Rule") -> "Rule":
        combined = Signal(
            lambda df, a=self, b=other: a.evaluate(df) | b.evaluate(df),
            f"{self.description} OR {other.description}",
        )
        return Rule([combined])

    def __repr__(self) -> str:
        return f"Rule({self.description})"


# ---------------------------------------------------------------------------
# PositionSizer: gestión de tamaño de posición
# ---------------------------------------------------------------------------

@dataclass
class PositionSizer:
    """
    Define cómo calcular el tamaño de cada posición.

    Modos:
    - "fixed"        : fracción fija del capital (default 10%)
    - "equal_weight" : 1/n del capital por posición
    - "kelly"        : criterio de Kelly ajustado
    - "volatility"   : inversamente proporcional a la volatilidad reciente
    - "dynamic"      : ajuste basado en racha reciente (aumenta en éxito)
    - "percent_risk" : arriesga un % fijo del capital por trade (requiere SL)
    """
    mode: Literal["fixed", "equal_weight", "kelly", "volatility", "dynamic", "percent_risk"] = "fixed"
    fraction: float = 0.10        # Para modo "fixed" o "equal_weight"
    max_fraction: float = 0.30    # Límite máximo de posición
    min_fraction: float = 0.02    # Límite mínimo de posición
    risk_per_trade: float = 0.01  # Para "percent_risk" (1% del capital)
    vol_target: float = 0.15      # Volatilidad objetivo anual para "volatility"
    kelly_lookback: int = 50      # Ventana para estimar win_rate en Kelly
    dynamic_lookback: int = 10    # Ventana para racha en "dynamic"
    dynamic_scale: float = 0.5    # Multiplicador de ajuste dinámico

    def compute(
        self,
        capital: float,
        price: float,
        stop_loss_price: Optional[float] = None,
        recent_trades: Optional[pd.DataFrame] = None,
        recent_vol: Optional[float] = None,
    ) -> float:
        """
        Retorna el número de unidades/acciones a comprar.
        """
        if self.mode == "fixed":
            fraction = self.fraction

        elif self.mode == "equal_weight":
            fraction = self.fraction

        elif self.mode == "percent_risk":
            if stop_loss_price and stop_loss_price > 0 and price > 0:
                risk_per_share = abs(price - stop_loss_price)
                if risk_per_share > 0:
                    max_loss = capital * self.risk_per_trade
                    shares = max_loss / risk_per_share
                    return max(0, min(shares, capital * self.max_fraction / price))
            fraction = self.fraction

        elif self.mode == "volatility":
            if recent_vol and recent_vol > 0:
                fraction = self.vol_target / (recent_vol * np.sqrt(252))
                fraction = np.clip(fraction, self.min_fraction, self.max_fraction)
            else:
                fraction = self.fraction

        elif self.mode == "kelly":
            if recent_trades is not None and len(recent_trades) >= 5:
                wins = recent_trades["pnl_pct"] > 0
                win_rate = wins.mean()
                avg_win = recent_trades.loc[wins, "pnl_pct"].mean() if wins.any() else 0.01
                avg_loss = abs(recent_trades.loc[~wins, "pnl_pct"].mean()) if (~wins).any() else 0.01
                odds = avg_win / (avg_loss + 1e-10)
                kelly = win_rate - (1 - win_rate) / (odds + 1e-10)
                fraction = np.clip(kelly * 0.5, self.min_fraction, self.max_fraction)
            else:
                fraction = self.fraction

        elif self.mode == "dynamic":
            if recent_trades is not None and len(recent_trades) > 0:
                last_n = recent_trades.tail(self.dynamic_lookback)
                win_rate = (last_n["pnl_pct"] > 0).mean()
                # Aumenta fracción cuando win_rate es alto
                adj = 1 + self.dynamic_scale * (win_rate - 0.5) * 2
                fraction = np.clip(self.fraction * adj, self.min_fraction, self.max_fraction)
            else:
                fraction = self.fraction
        else:
            fraction = self.fraction

        fraction = np.clip(fraction, self.min_fraction, self.max_fraction)
        invested = capital * fraction
        if price <= 0:
            return 0
        return invested / price


# ---------------------------------------------------------------------------
# Strategy: clase principal
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """
    Define una estrategia de inversión completa.

    Parámetros principales
    ----------------------
    name           : nombre único de la estrategia
    entry          : Rule o Signal que genera señal de COMPRA
    exit           : Rule o Signal que genera señal de VENTA (opcional)
    stop_loss      : fracción de pérdida máxima (0.05 = 5%)
    take_profit    : fracción de ganancia objetivo (0.15 = 15%)
    trailing_stop  : trailing stop como % del máximo alcanzado
    max_holding_days: salida forzada después de N días
    position_sizer : objeto PositionSizer (default: fixed 10%)
    direction      : "long", "short", "both"
    required_indicators: lista de indicadores que deben agregarse al df
    params         : dict de parámetros libres (para optimización)
    """

    name: str
    entry: Union[Rule, Signal]
    exit: Optional[Union[Rule, Signal]] = None
    stop_loss: Optional[float] = None          # e.g. 0.05 = 5%
    take_profit: Optional[float] = None        # e.g. 0.15 = 15%
    trailing_stop: Optional[float] = None      # e.g. 0.08 = 8% desde máximo
    max_holding_days: Optional[int] = None
    position_sizer: PositionSizer = field(default_factory=PositionSizer)
    direction: Literal["long", "short", "both"] = "long"
    required_indicators: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    category: str = "general"

    def get_entry_signals(self, df: pd.DataFrame) -> pd.Series:
        """Evalúa y retorna la señal de entrada para cada barra."""
        if isinstance(self.entry, (Rule, Signal)):
            return self.entry.evaluate(df)
        return pd.Series(False, index=df.index)

    def get_exit_signals(self, df: pd.DataFrame) -> pd.Series:
        """Evalúa y retorna la señal de salida para cada barra."""
        if self.exit is None:
            return pd.Series(False, index=df.index)
        if isinstance(self.exit, (Rule, Signal)):
            return self.exit.evaluate(df)
        return pd.Series(False, index=df.index)

    def clone_with_params(self, **new_params) -> "Strategy":
        """Retorna una copia de la estrategia con parámetros modificados."""
        import copy
        new_strategy = copy.deepcopy(self)
        new_strategy.params.update(new_params)
        return new_strategy

    def __repr__(self) -> str:
        return (
            f"Strategy(name='{self.name}', "
            f"SL={self.stop_loss}, TP={self.take_profit}, "
            f"max_days={self.max_holding_days})"
        )


# ---------------------------------------------------------------------------
# Librería de estrategias predefinidas
# ---------------------------------------------------------------------------

def _make_sma_crossover(fast: int = 20, slow: int = 50) -> Strategy:
    """Golden Cross / Death Cross entre dos SMAs."""
    return Strategy(
        name=f"SMA_Cross_{fast}_{slow}",
        description=f"Compra cuando SMA{fast} cruza arriba SMA{slow}",
        category="trend_following",
        entry=Signal.crossover(f"sma_{fast}", f"sma_{slow}"),
        exit=Signal.crossunder(f"sma_{fast}", f"sma_{slow}"),
        stop_loss=0.06,
        take_profit=0.18,
        max_holding_days=None,
        required_indicators=[f"sma_{fast}", f"sma_{slow}"],
    )


def _make_ema_crossover(fast: int = 9, slow: int = 21) -> Strategy:
    """EMA crossover."""
    return Strategy(
        name=f"EMA_Cross_{fast}_{slow}",
        description=f"Compra cuando EMA{fast} cruza arriba EMA{slow}",
        category="trend_following",
        entry=Signal.crossover(f"ema_{fast}", f"ema_{slow}"),
        exit=Signal.crossunder(f"ema_{fast}", f"ema_{slow}"),
        stop_loss=0.05,
        take_profit=0.15,
        required_indicators=[f"ema_{fast}", f"ema_{slow}"],
    )


def _make_macd_strategy() -> Strategy:
    """MACD Signal crossover."""
    return Strategy(
        name="MACD_Signal_Cross",
        description="Compra cuando MACD cruza arriba la señal",
        category="momentum",
        entry=Signal.crossover("macd", "macd_signal"),
        exit=Signal.crossunder("macd", "macd_signal"),
        stop_loss=0.06,
        take_profit=0.18,
        required_indicators=["macd", "macd_signal"],
    )


def _make_rsi_mean_reversion(oversold: float = 30, overbought: float = 70) -> Strategy:
    """RSI mean reversion: compra en sobreventa, vende en sobrecompra."""
    return Strategy(
        name=f"RSI_MeanReversion_{int(oversold)}_{int(overbought)}",
        description=f"Compra cuando RSI cruza arriba {oversold}, vende cuando cruza {overbought}",
        category="mean_reversion",
        entry=Signal.crossover("rsi_14", oversold),
        exit=Signal.crossover("rsi_14", overbought),
        stop_loss=0.04,
        take_profit=0.10,
        max_holding_days=20,
        required_indicators=["rsi_14"],
    )


def _make_rsi_trend(upper: float = 50) -> Strategy:
    """RSI trend following: compra cuando RSI supera 50."""
    return Strategy(
        name="RSI_Trend_50",
        description="Compra cuando RSI cruza arriba 50 (momentum positivo)",
        category="momentum",
        entry=Signal.crossover("rsi_14", upper),
        exit=Signal.crossunder("rsi_14", upper),
        stop_loss=0.06,
        take_profit=0.15,
        required_indicators=["rsi_14"],
    )


def _make_bollinger_breakout() -> Strategy:
    """Precio cruza banda superior de Bollinger (breakout)."""
    return Strategy(
        name="BB_Breakout_Upper",
        description="Compra cuando precio rompe banda superior de Bollinger",
        category="breakout",
        entry=Signal.crossover("close", "bb_upper_20"),
        exit=Signal.crossunder("close", "bb_mid_20"),
        stop_loss=0.05,
        take_profit=0.12,
        max_holding_days=15,
        required_indicators=["bb_upper_20", "bb_mid_20", "bb_lower_20"],
    )


def _make_bollinger_reversal() -> Strategy:
    """Precio toca banda inferior de Bollinger (reversión)."""
    return Strategy(
        name="BB_Reversal_Lower",
        description="Compra cuando precio toca banda inferior (reversión a la media)",
        category="mean_reversion",
        entry=Signal.crossover("close", "bb_lower_20"),
        exit=Signal.crossover("close", "bb_mid_20"),
        stop_loss=0.04,
        take_profit=0.08,
        max_holding_days=20,
        required_indicators=["bb_upper_20", "bb_mid_20", "bb_lower_20"],
    )


def _make_stochastic_strategy(oversold: float = 20, overbought: float = 80) -> Strategy:
    """Stochastic crossover en zonas extremas."""
    return Strategy(
        name=f"Stoch_Cross_{int(oversold)}_{int(overbought)}",
        description="Compra cuando Stoch K cruza arriba D en zona de sobreventa",
        category="mean_reversion",
        entry=Rule([
            Signal.crossover("stoch_k_14", "stoch_d_14"),
            Signal.below("stoch_k_14", oversold + 10),
        ]),
        exit=Rule([
            Signal.crossunder("stoch_k_14", "stoch_d_14"),
            Signal.above("stoch_k_14", overbought - 10),
        ]),
        stop_loss=0.04,
        take_profit=0.10,
        required_indicators=["stoch_k_14", "stoch_d_14"],
    )


def _make_volume_breakout() -> Strategy:
    """Precio rompe máximo de 20 días con alto volumen."""
    return Strategy(
        name="Volume_Price_Breakout",
        description="Compra cuando precio cierra arriba de SMA50 con volumen > 1.5x promedio",
        category="breakout",
        entry=Rule([
            Signal.crossover("close", "sma_50"),
            Signal.above_threshold("volume", "sma_20", factor=1.5),
        ], combinator="AND"),
        exit=Signal.crossunder("close", "sma_50"),
        stop_loss=0.06,
        take_profit=0.18,
        required_indicators=["sma_50", "sma_20"],
    )


def _make_golden_cross_volume() -> Strategy:
    """Golden Cross (50/200) con confirmación de volumen."""
    return Strategy(
        name="Golden_Cross_50_200_Vol",
        description="Golden Cross SMA50/200 con confirmación de volumen elevado",
        category="trend_following",
        entry=Rule([
            Signal.crossover("sma_50", "sma_200"),
            Signal.above_threshold("volume", "sma_20", factor=1.2),
        ]),
        exit=Signal.crossunder("sma_50", "sma_200"),
        stop_loss=0.08,
        take_profit=0.25,
        required_indicators=["sma_50", "sma_200", "sma_20"],
    )


def _make_macd_rsi_combo() -> Strategy:
    """MACD + RSI combinados para mayor fiabilidad."""
    return Strategy(
        name="MACD_RSI_Combo",
        description="MACD cruza señal Y RSI entre 40-70 (tendencia + momentum)",
        category="combo",
        entry=Rule([
            Signal.crossover("macd", "macd_signal"),
            Signal.between("rsi_14", 40, 65),
        ]),
        exit=Rule([
            Signal.crossunder("macd", "macd_signal"),
        ], combinator="OR"),
        stop_loss=0.06,
        take_profit=0.16,
        required_indicators=["macd", "macd_signal", "macd_hist", "rsi_14"],
    )


def _make_ema_ribbon() -> Strategy:
    """EMA ribbon: EMA9 > EMA21 > EMA50 (alineación completa de tendencia)."""
    return Strategy(
        name="EMA_Ribbon_9_21_50",
        description="Compra cuando EMA9 > EMA21 > EMA50 (tendencia alcista fuerte)",
        category="trend_following",
        entry=Rule([
            Signal.crossover("ema_9", "ema_21"),
            Signal.above("ema_21", "ema_50"),
        ]),
        exit=Signal.crossunder("ema_9", "ema_21"),
        stop_loss=0.06,
        take_profit=0.20,
        required_indicators=["ema_9", "ema_21", "ema_50"],
    )


def _make_donchian_breakout(period: int = 20) -> Strategy:
    """Breakout del canal Donchian."""
    return Strategy(
        name=f"Donchian_Breakout_{period}",
        description=f"Compra cuando precio rompe el máximo del canal Donchian {period}",
        category="breakout",
        entry=Signal.crossover("close", f"donchian_upper_{period}"),
        exit=Signal.crossunder("close", f"donchian_mid_{period}"),
        stop_loss=0.06,
        take_profit=0.18,
        trailing_stop=0.08,
        required_indicators=[f"donchian_upper_{period}", f"donchian_lower_{period}", f"donchian_mid_{period}"],
    )


def _make_cci_strategy() -> Strategy:
    """CCI reversión: compra cuando CCI sale de zona de sobreventa."""
    return Strategy(
        name="CCI_Reversal",
        description="Compra cuando CCI cruza -100 hacia arriba",
        category="mean_reversion",
        entry=Signal.crossover("cci_20", -100),
        exit=Signal.crossover("cci_20", 100),
        stop_loss=0.05,
        take_profit=0.12,
        max_holding_days=20,
        required_indicators=["cci_20"],
    )


def _make_williams_r_strategy() -> Strategy:
    """Williams %R reversión."""
    return Strategy(
        name="WilliamsR_Reversal",
        description="Compra cuando Williams %R cruza -80 hacia arriba",
        category="mean_reversion",
        entry=Signal.crossover("williams_r_14", -80),
        exit=Signal.crossover("williams_r_14", -20),
        stop_loss=0.04,
        take_profit=0.10,
        max_holding_days=15,
        required_indicators=["williams_r_14"],
    )


def _make_adx_trend() -> Strategy:
    """ADX para confirmar tendencia fuerte + DI+ > DI-."""
    return Strategy(
        name="ADX_Trend_Filter",
        description="Compra cuando ADX>25 y DI+ cruza arriba DI- (tendencia fuerte alcista)",
        category="trend_following",
        entry=Rule([
            Signal.crossover("di_plus_14", "di_minus_14"),
            Signal.above("adx_14", 25),
        ]),
        exit=Signal.crossunder("di_plus_14", "di_minus_14"),
        stop_loss=0.06,
        take_profit=0.18,
        required_indicators=["adx_14", "di_plus_14", "di_minus_14"],
    )


def _make_obv_crossover() -> Strategy:
    """OBV cruza su media móvil (volumen confirma tendencia)."""
    return Strategy(
        name="OBV_SMA_Cross",
        description="Compra cuando OBV cruza arriba su SMA20",
        category="volume",
        entry=Rule([
            Signal.crossover("obv", "sma_50"),   # placeholder: se necesita OBV_SMA
            Signal.above("close", "sma_50"),
        ]),
        exit=Signal.crossunder("close", "sma_50"),
        stop_loss=0.06,
        take_profit=0.15,
        required_indicators=["obv", "sma_50"],
    )


def _make_mfi_reversal() -> Strategy:
    """MFI (RSI de volumen) reversión."""
    return Strategy(
        name="MFI_Reversal",
        description="Compra cuando MFI cruza 20 hacia arriba (dinero entrando)",
        category="volume",
        entry=Signal.crossover("mfi_14", 20),
        exit=Signal.crossover("mfi_14", 80),
        stop_loss=0.05,
        take_profit=0.12,
        max_holding_days=25,
        required_indicators=["mfi_14"],
    )


def _make_supertrend_strategy() -> Strategy:
    """SuperTrend: señal cuando dirección cambia a alcista."""
    return Strategy(
        name="SuperTrend_10_3",
        description="Compra cuando SuperTrend cambia de bajista a alcista",
        category="trend_following",
        entry=Rule([
            Signal.crossover("supertrend_dir_10_3", 0),
        ]),
        exit=Rule([
            Signal.crossunder("supertrend_dir_10_3", 0),
        ]),
        stop_loss=0.07,
        take_profit=0.20,
        trailing_stop=0.08,
        required_indicators=["supertrend_10_3", "supertrend_dir_10_3"],
    )


def _make_keltner_breakout() -> Strategy:
    """Breakout del Canal Keltner."""
    return Strategy(
        name="Keltner_Breakout",
        description="Compra cuando precio rompe canal Keltner superior",
        category="breakout",
        entry=Signal.crossover("close", "keltner_upper_20"),
        exit=Signal.crossunder("close", "keltner_mid_20"),
        stop_loss=0.05,
        take_profit=0.15,
        required_indicators=["keltner_upper_20", "keltner_mid_20", "keltner_lower_20"],
    )


def _make_price_sma200() -> Strategy:
    """Precio sobre/bajo SMA200 (filtro de largo plazo)."""
    return Strategy(
        name="Price_Above_SMA200",
        description="Compra cuando precio cruza arriba SMA200 (tendencia de largo plazo)",
        category="trend_following",
        entry=Signal.crossover("close", "sma_200"),
        exit=Signal.crossunder("close", "sma_200"),
        stop_loss=0.08,
        take_profit=0.25,
        required_indicators=["sma_200"],
    )


def _make_rsi_bb_combo() -> Strategy:
    """RSI en sobreventa + precio en banda inferior de Bollinger."""
    return Strategy(
        name="RSI_BB_Reversal",
        description="RSI<30 Y precio toca BB inferior (doble confirmación de sobreventa)",
        category="mean_reversion",
        entry=Rule([
            Signal.below("rsi_14", 35),
            Signal.below("close", "bb_lower_20"),
        ]),
        exit=Rule([
            Signal.above("rsi_14", 60),
        ]),
        stop_loss=0.05,
        take_profit=0.12,
        max_holding_days=20,
        required_indicators=["rsi_14", "bb_lower_20", "bb_upper_20", "bb_mid_20"],
    )


def _make_three_ma_trend() -> Strategy:
    """Triple media móvil: alineación alcista completa."""
    return Strategy(
        name="Triple_SMA_10_30_100",
        description="Compra con alineación alcista: SMA10 > SMA30 > SMA100",
        category="trend_following",
        entry=Rule([
            Signal.crossover("sma_10", "sma_20"),
            Signal.above("sma_20", "sma_100"),
            Signal.above("close", "sma_100"),
        ]),
        exit=Signal.crossunder("sma_10", "sma_20"),
        stop_loss=0.06,
        take_profit=0.20,
        required_indicators=["sma_10", "sma_20", "sma_100"],
    )


def _make_momentum_roc() -> Strategy:
    """Rate of Change positivo como señal de momentum."""
    return Strategy(
        name="ROC_Momentum",
        description="Compra cuando ROC-12 cruza cero hacia arriba",
        category="momentum",
        entry=Signal.crossover("roc_12", 0),
        exit=Signal.crossunder("roc_12", 0),
        stop_loss=0.06,
        take_profit=0.15,
        required_indicators=["roc_12"],
    )


def _make_cmf_volume_trend() -> Strategy:
    """CMF positivo + precio sobre SMA50."""
    return Strategy(
        name="CMF_Volume_Trend",
        description="Compra cuando CMF cruza cero arriba y precio está sobre SMA50",
        category="volume",
        entry=Rule([
            Signal.crossover("cmf_20", 0),
            Signal.above("close", "sma_50"),
        ]),
        exit=Rule([
            Signal.crossunder("cmf_20", 0),
        ]),
        stop_loss=0.06,
        take_profit=0.15,
        required_indicators=["cmf_20", "sma_50"],
    )


def _make_vwap_cross() -> Strategy:
    """Cruce del precio sobre VWAP intradiario."""
    return Strategy(
        name="VWAP_Cross",
        description="Compra cuando precio cruza arriba VWAP",
        category="momentum",
        entry=Signal.crossover("close", "vwap"),
        exit=Signal.crossunder("close", "vwap"),
        stop_loss=0.04,
        take_profit=0.10,
        max_holding_days=10,
        required_indicators=["vwap"],
    )


def _make_macd_zero_cross() -> Strategy:
    """MACD cruza cero (cambio de tendencia)."""
    return Strategy(
        name="MACD_Zero_Cross",
        description="Compra cuando MACD cruza cero hacia arriba",
        category="trend_following",
        entry=Signal.crossover("macd", 0),
        exit=Signal.crossunder("macd", 0),
        stop_loss=0.06,
        take_profit=0.18,
        required_indicators=["macd"],
    )


def _make_dema_tema_cross() -> Strategy:
    """DEMA cruza TEMA (medias rápidas con menor lag)."""
    return Strategy(
        name="DEMA_TEMA_Cross",
        description="Compra cuando DEMA20 cruza arriba TEMA20",
        category="trend_following",
        entry=Signal.crossover("dema_20", "tema_20"),
        exit=Signal.crossunder("dema_20", "tema_20"),
        stop_loss=0.05,
        take_profit=0.15,
        required_indicators=["dema_20", "tema_20"],
    )


# ---------------------------------------------------------------------------
# Registro de biblioteca de estrategias
# ---------------------------------------------------------------------------

STRATEGY_LIBRARY: Dict[str, Strategy] = {
    s.name: s for s in [
        _make_sma_crossover(20, 50),
        _make_sma_crossover(50, 200),
        _make_sma_crossover(10, 30),
        _make_ema_crossover(9, 21),
        _make_ema_crossover(21, 55),
        _make_macd_strategy(),
        _make_macd_zero_cross(),
        _make_macd_rsi_combo(),
        _make_rsi_mean_reversion(30, 70),
        _make_rsi_mean_reversion(25, 75),
        _make_rsi_trend(50),
        _make_bollinger_breakout(),
        _make_bollinger_reversal(),
        _make_rsi_bb_combo(),
        _make_stochastic_strategy(20, 80),
        _make_volume_breakout(),
        _make_golden_cross_volume(),
        _make_ema_ribbon(),
        _make_donchian_breakout(20),
        _make_cci_strategy(),
        _make_williams_r_strategy(),
        _make_adx_trend(),
        _make_mfi_reversal(),
        _make_keltner_breakout(),
        _make_price_sma200(),
        _make_three_ma_trend(),
        _make_momentum_roc(),
        _make_cmf_volume_trend(),
        _make_dema_tema_cross(),
        _make_supertrend_strategy(),
    ]
}


def list_strategies(category: Optional[str] = None) -> pd.DataFrame:
    """Lista todas las estrategias en la biblioteca con sus metadatos."""
    rows = []
    for name, s in STRATEGY_LIBRARY.items():
        if category and s.category != category:
            continue
        rows.append({
            "name": s.name,
            "category": s.category,
            "description": s.description,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "max_holding_days": s.max_holding_days,
            "trailing_stop": s.trailing_stop,
        })
    return pd.DataFrame(rows)


def get_strategy(name: str) -> Strategy:
    """Obtiene una estrategia de la biblioteca por nombre."""
    if name not in STRATEGY_LIBRARY:
        available = ", ".join(STRATEGY_LIBRARY.keys())
        raise KeyError(f"Estrategia '{name}' no encontrada. Disponibles: {available}")
    return STRATEGY_LIBRARY[name]


def add_strategy_to_library(strategy: Strategy) -> None:
    """Agrega una nueva estrategia a la biblioteca global."""
    STRATEGY_LIBRARY[strategy.name] = strategy


def get_strategy_parameters(strategy: Strategy) -> Dict[str, Any]:
    """Extrae los parámetros clave de una estrategia para optimización."""
    params = {
        "stop_loss": strategy.stop_loss,
        "take_profit": strategy.take_profit,
        "max_holding_days": strategy.max_holding_days,
        "trailing_stop": strategy.trailing_stop,
    }
    
    # Agregar parámetros custom si existen
    if hasattr(strategy, "params") and strategy.params:
        params.update(strategy.params)
    
    return {k: v for k, v in params.items() if v is not None}


def create_custom_strategy(
    name: str,
    entry_conditions: List[Dict[str, Any]],
    exit_conditions: Optional[List[Dict[str, Any]]] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    max_holding_days: Optional[int] = None,
    trailing_stop: Optional[float] = None,
    description: str = "",
    category: str = "custom",
) -> Strategy:
    """
    Crea una estrategia personalizada desde una definición estructurada.
    
    Parameters
    ----------
    entry_conditions : Lista de condiciones de entrada
        Cada condición es un dict con:
        {
            "type": "crossover" | "crossunder" | "above" | "below",
            "col_a": str,  # columna o indicador
            "col_b": str | float,  # columna, indicador o valor
        }
    exit_conditions : Lista de condiciones de salida (opcional)
    
    Example
    -------
    strategy = create_custom_strategy(
        name="Mi_SMA_Custom",
        entry_conditions=[
            {"type": "crossover", "col_a": "sma_20", "col_b": "sma_50"},
            {"type": "above", "col_a": "close", "col_b": "sma_100"},
        ],
        stop_loss=0.05,
        take_profit=0.15,
    )
    """
    # Construir señales de entrada
    entry_signals = []
    for cond in entry_conditions:
        cond_type = cond["type"]
        col_a = cond["col_a"]
        col_b = cond["col_b"]
        
        if cond_type == "crossover":
            entry_signals.append(Signal.crossover(col_a, col_b))
        elif cond_type == "crossunder":
            entry_signals.append(Signal.crossunder(col_a, col_b))
        elif cond_type == "above":
            entry_signals.append(Signal.above(col_a, col_b))
        elif cond_type == "below":
            entry_signals.append(Signal.below(col_a, col_b))
    
    # Si no hay señales, crear una señal dummy que nunca se activa
    if not entry_signals:
        entry_signals.append(Signal(lambda df: pd.Series(False, index=df.index), "No entry"))
    
    entry_rule = Rule(entry_signals, combinator="AND")
    
    # Construir señales de salida
    exit_rule = None
    if exit_conditions:
        exit_signals = []
        for cond in exit_conditions:
            cond_type = cond["type"]
            col_a = cond["col_a"]
            col_b = cond["col_b"]
            
            if cond_type == "crossover":
                exit_signals.append(Signal.crossover(col_a, col_b))
            elif cond_type == "crossunder":
                exit_signals.append(Signal.crossunder(col_a, col_b))
            elif cond_type == "above":
                exit_signals.append(Signal.above(col_a, col_b))
            elif cond_type == "below":
                exit_signals.append(Signal.below(col_a, col_b))
        
        exit_rule = Rule(exit_signals, combinator="OR") if exit_signals else None
    
    return Strategy(
        name=name,
        entry=entry_rule,
        exit=exit_rule,
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        trailing_stop=trailing_stop,
        description=description or f"Estrategia personalizada: {name}",
        category=category,
    )


def get_available_columns() -> List[str]:
    """Retorna lista de columnas e indicadores disponibles para estrategias."""
    return [
        # Precio
        "open", "high", "low", "close", "adj_close", "volume",
        # SMAs
        "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        # EMAs
        "ema_9", "ema_12", "ema_21", "ema_26", "ema_50", "ema_100", "ema_200",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # RSI
        "rsi_14",
        # Bollinger Bands
        "bb_upper_20", "bb_mid_20", "bb_lower_20", "bb_width_20",
        # Stochastic
        "stoch_k", "stoch_d",
        # ATR
        "atr_14",
        # ADX
        "adx_14", "plus_di_14", "minus_di_14",
        # Volume
        "volume_sma_20", "obv", "mfi_14", "cmf_20",
        # Otros
        "cci_20", "williams_r_14", "roc_10",
        "dema_21", "tema_21",
        "keltner_upper_20", "keltner_mid_20", "keltner_lower_20",
        "donchian_upper_20", "donchian_mid_20", "donchian_lower_20",
        "supertrend_10", "supertrend_signal_10",
    ]


if __name__ == "__main__":
    print("=== Biblioteca de Estrategias ===")
    df = list_strategies()
    print(df[["name", "category", "stop_loss", "take_profit"]].to_string())
    print(f"\nTotal: {len(df)} estrategias")
