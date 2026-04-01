"""
optimizer.py
============
Módulo de análisis profundo y optimización de parámetros de estrategias.

Funcionalidades:
- Grid Search sobre parámetros de la estrategia
- Walk-Forward Optimization (WFO) para evitar overfitting
- Monte Carlo para robustez estadística
- Análisis de sensibilidad de parámetros
- Out-of-sample validation
- Reporte detallado con heatmaps y curvas de optimización

El módulo retorna:
- DataFrame completo de todas las combinaciones evaluadas
- Parámetros óptimos
- Análisis de robustez
- Trades detallados del mejor modelo

Autor: Backtest Framework
"""

from __future__ import annotations

import warnings
import itertools
import logging
import copy
from typing import Dict, List, Optional, Callable, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from strategies import Strategy, Signal, Rule, PositionSizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Espacio de parámetros
# ---------------------------------------------------------------------------

class ParameterGrid:
    """
    Define el espacio de búsqueda de parámetros.

    Ejemplo:
        grid = ParameterGrid({
            "fast": [5, 10, 20],
            "slow": [30, 50, 100, 200],
            "stop_loss": [0.03, 0.05, 0.08],
            "take_profit": [0.10, 0.15, 0.20],
        })
        for params in grid:
            print(params)
    """

    def __init__(self, param_space: Dict[str, List]):
        self.param_space = param_space
        self._keys = list(param_space.keys())
        self._values = list(param_space.values())

    def __iter__(self):
        for combo in itertools.product(*self._values):
            yield dict(zip(self._keys, combo))

    def __len__(self):
        n = 1
        for v in self._values:
            n *= len(v)
        return n

    def __repr__(self):
        return f"ParameterGrid({self.param_space}) -> {len(self)} combinations"


# ---------------------------------------------------------------------------
# Optimizador principal
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """
    Optimizador de parámetros para una estrategia de inversión.

    Métodos:
    - grid_search()          : búsqueda exhaustiva en cuadrícula
    - walk_forward()         : Walk-Forward Optimization
    - monte_carlo()          : simulación Monte Carlo de trades
    - sensitivity_analysis() : análisis de sensibilidad univariado
    - full_analysis()        : análisis completo (grid + WFO + MC)

    Uso típico:
        optimizer = StrategyOptimizer(
            strategy_factory=make_sma_crossover,
            param_grid=ParameterGrid({"fast": [10,20], "slow": [50,100]}),
            config=BacktestConfig(),
            optimize_metric="sharpe_ratio",
        )
        results = optimizer.grid_search(df, ticker="AAPL")
        print(optimizer.best_params)
    """

    def __init__(
        self,
        strategy_factory: Callable[..., Strategy],
        param_grid: ParameterGrid,
        config: Optional[BacktestConfig] = None,
        optimize_metric: str = "sharpe_ratio",
        n_jobs: int = 1,
    ):
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.config = config or BacktestConfig()
        self.optimize_metric = optimize_metric
        self.n_jobs = n_jobs

        self.grid_results_df: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict] = None
        self.best_result: Optional[BacktestResult] = None
        self.wfo_results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------------------

    def grid_search(
        self,
        df: pd.DataFrame,
        ticker: str,
        min_trades: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Búsqueda exhaustiva en cuadrícula de parámetros.

        Returns
        -------
        DataFrame con todas las combinaciones evaluadas y sus métricas.
        """
        total = len(self.param_grid)
        if verbose:
            print(f"\n{'='*60}")
            print(f"  GRID SEARCH: {ticker} | {total} combinaciones")
            print(f"  Métrica objetivo: {self.optimize_metric}")
            print(f"{'='*60}")

        engine = BacktestEngine(self.config)
        rows = []

        for i, params in enumerate(self.param_grid, 1):
            try:
                strategy = self.strategy_factory(**params)
                result = engine.run(df, strategy, ticker=ticker)
                m = result.metrics

                if "error" in m or m.get("n_trades", 0) < min_trades:
                    continue

                row = {"params_str": str(params), **params, **{
                    k: m.get(k) for k in [
                        "total_return_pct", "cagr_pct", "sharpe_ratio",
                        "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
                        "win_rate_pct", "profit_factor", "n_trades",
                        "avg_holding_days", "annual_volatility_pct",
                        "expectancy_pct", "omega_ratio", "recovery_factor",
                    ]
                }}
                rows.append(row)

                if verbose and i % max(1, total // 10) == 0:
                    best_val = max((r.get(self.optimize_metric, -999) for r in rows), default=-999)
                    print(f"  [{i:4d}/{total}] Mejor {self.optimize_metric}: {best_val:.4f}")

            except Exception as e:
                if verbose:
                    logger.debug(f"Error con {params}: {e}")
                continue

        if not rows:
            logger.warning("Sin resultados válidos en grid search")
            return pd.DataFrame()

        df_results = pd.DataFrame(rows)
        df_results = df_results.sort_values(
            self.optimize_metric, ascending=False, na_position="last"
        ).reset_index(drop=True)

        self.grid_results_df = df_results

        # Encontrar mejores parámetros
        best_row = df_results.iloc[0]
        param_keys = list(self.param_grid._keys)
        self.best_params = {k: best_row[k] for k in param_keys if k in best_row}

        # Re-ejecutar con mejores parámetros para obtener BacktestResult completo
        try:
            best_strategy = self.strategy_factory(**self.best_params)
            self.best_result = engine.run(df, best_strategy, ticker=ticker)
        except Exception:
            pass

        if verbose:
            print(f"\n  Mejores parámetros: {self.best_params}")
            print(f"  Mejor {self.optimize_metric}: {best_row.get(self.optimize_metric):.4f}")
            print(f"  Total combinaciones válidas: {len(df_results)}")

        return df_results

    # ------------------------------------------------------------------
    # Walk-Forward Optimization
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        df: pd.DataFrame,
        ticker: str,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        min_trades_per_fold: int = 3,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Walk-Forward Optimization: entrena en ventana in-sample,
        evalúa en out-of-sample, avanzando progresivamente.

        Retorna DataFrame con métricas por fold y parámetros óptimos
        in-sample aplicados al out-of-sample.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  WALK-FORWARD OPTIMIZATION: {ticker} | {n_splits} folds")
            print(f"  Train/Test ratio: {train_ratio:.0%}/{1-train_ratio:.0%}")
            print(f"{'='*60}")

        # Preparar datos
        engine = BacktestEngine(self.config)
        df_prep = engine._prepare_data(df, ticker, add_indicators=True)
        if df_prep is None or len(df_prep) < 200:
            logger.warning("Datos insuficientes para WFO")
            return pd.DataFrame()

        n = len(df_prep)
        fold_size = n // n_splits
        wfo_rows = []

        for fold_idx in range(n_splits - 1):
            # Ventana in-sample
            train_end = int(fold_size * (fold_idx + 1))
            test_end = min(train_end + int(fold_size * (1 - train_ratio) / train_ratio), n)

            train_df = df_prep.iloc[:train_end].copy()
            test_df = df_prep.iloc[train_end:test_end].copy()

            if len(train_df) < 100 or len(test_df) < 30:
                continue

            # Optimizar en in-sample
            best_is_params = self._optimize_on_window(
                train_df, ticker, engine, min_trades_per_fold
            )

            if best_is_params is None:
                continue

            # Evaluar en out-of-sample
            try:
                oos_strategy = self.strategy_factory(**best_is_params)
                oos_result = engine.run(test_df, oos_strategy, ticker=ticker, add_indicators=False)
                oos_m = oos_result.metrics

                is_strategy = self.strategy_factory(**best_is_params)
                is_result = engine.run(train_df, is_strategy, ticker=ticker, add_indicators=False)
                is_m = is_result.metrics

                row = {
                    "fold": fold_idx + 1,
                    "train_start": train_df.index[0],
                    "train_end": train_df.index[-1],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                    "best_params": str(best_is_params),
                    # In-sample
                    "is_sharpe": is_m.get("sharpe_ratio"),
                    "is_return_%": is_m.get("total_return_pct"),
                    "is_n_trades": is_m.get("n_trades"),
                    # Out-of-sample
                    "oos_sharpe": oos_m.get("sharpe_ratio"),
                    "oos_return_%": oos_m.get("total_return_pct"),
                    "oos_max_dd_%": oos_m.get("max_drawdown_pct"),
                    "oos_win_rate_%": oos_m.get("win_rate_pct"),
                    "oos_n_trades": oos_m.get("n_trades"),
                    # Eficiencia OOS/IS
                    "efficiency": (
                        oos_m.get("sharpe_ratio", 0) / (abs(is_m.get("sharpe_ratio", 1)) + 1e-10)
                    ),
                }
                wfo_rows.append(row)

                if verbose:
                    print(f"  Fold {fold_idx+1}: IS Sharpe={is_m.get('sharpe_ratio', 0):.3f} "
                          f"| OOS Sharpe={oos_m.get('sharpe_ratio', 0):.3f} "
                          f"| Params: {best_is_params}")

            except Exception as e:
                logger.debug(f"WFO fold {fold_idx+1} error: {e}")
                continue

        if not wfo_rows:
            return pd.DataFrame()

        self.wfo_results = pd.DataFrame(wfo_rows)

        if verbose:
            avg_eff = self.wfo_results["efficiency"].mean()
            avg_oos = self.wfo_results["oos_sharpe"].mean()
            print(f"\n  WFO Summary:")
            print(f"    Avg OOS Sharpe:   {avg_oos:.4f}")
            print(f"    Avg IS/OOS Ratio: {avg_eff:.4f}")
            print(f"    {'ROBUSTO ✓' if avg_eff > 0.5 else 'SOBREAJUSTE POSIBLE ⚠'}")

        return self.wfo_results

    def _optimize_on_window(
        self,
        df_window: pd.DataFrame,
        ticker: str,
        engine: BacktestEngine,
        min_trades: int,
    ) -> Optional[Dict]:
        """Encuentra los mejores parámetros en una ventana de datos."""
        best_val = -np.inf
        best_params = None

        for params in self.param_grid:
            try:
                strategy = self.strategy_factory(**params)
                result = engine.run(df_window, strategy, ticker=ticker, add_indicators=False)
                m = result.metrics
                if "error" in m or m.get("n_trades", 0) < min_trades:
                    continue
                val = m.get(self.optimize_metric, -np.inf)
                if isinstance(val, (int, float)) and val > best_val:
                    best_val = val
                    best_params = params
            except Exception:
                continue

        return best_params

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        confidence_levels: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95],
        verbose: bool = True,
    ) -> Dict:
        """
        Simulación Monte Carlo para evaluar robustez estadística.
        Reordena aleatoriamente los trades y calcula la distribución
        de métricas de desempeño.

        Returns
        -------
        Dict con distribuciones de métricas y percentiles.
        """
        if result.trades_df.empty:
            return {"error": "Sin trades para Monte Carlo"}

        trades = result.trades_df[result.trades_df["exit_date"].notna()].copy()
        pnl_pcts = trades["pnl_pct"].values

        if len(pnl_pcts) < 5:
            return {"error": "Insuficientes trades"}

        if verbose:
            print(f"\n  Monte Carlo: {n_simulations} simulaciones | {len(pnl_pcts)} trades")

        mc_metrics = {
            "total_return": [],
            "sharpe": [],
            "max_drawdown": [],
            "win_rate": [],
            "profit_factor": [],
        }
        equity_paths = []
        initial = self.config.initial_capital

        for _ in range(n_simulations):
            shuffled = np.random.choice(pnl_pcts, size=len(pnl_pcts), replace=True)

            # Reconstruir equity
            equity = [initial]
            for ret in shuffled:
                equity.append(equity[-1] * (1 + ret))
            equity = np.array(equity)
            equity_paths.append(equity)

            total_ret = (equity[-1] - initial) / initial
            log_ret = np.diff(np.log(equity + 1e-10))
            sharpe = (log_ret.mean() / (log_ret.std() + 1e-10)) * np.sqrt(252)

            rolling_max = np.maximum.accumulate(equity)
            dd = (equity - rolling_max) / (rolling_max + 1e-10)
            max_dd = dd.min()

            wins = shuffled > 0
            win_rate = wins.mean()
            pf = shuffled[wins].sum() / (abs(shuffled[~wins].sum()) + 1e-10) if (~wins).any() else 99

            mc_metrics["total_return"].append(total_ret * 100)
            mc_metrics["sharpe"].append(sharpe)
            mc_metrics["max_drawdown"].append(max_dd * 100)
            mc_metrics["win_rate"].append(win_rate * 100)
            mc_metrics["profit_factor"].append(pf)

        # Calcular percentiles
        mc_stats = {}
        for metric, values in mc_metrics.items():
            arr = np.array(values)
            mc_stats[metric] = {
                f"p{int(cl*100)}": round(np.percentile(arr, cl * 100), 4)
                for cl in confidence_levels
            }
            mc_stats[metric]["mean"] = round(arr.mean(), 4)
            mc_stats[metric]["std"] = round(arr.std(), 4)

        # Probabilidad de pérdida
        prob_loss = (np.array(mc_metrics["total_return"]) < 0).mean()
        mc_stats["prob_loss"] = round(prob_loss * 100, 2)
        mc_stats["n_simulations"] = n_simulations

        # Equity paths para visualización (percentiles)
        equity_array = np.array(equity_paths)
        mc_stats["equity_percentiles"] = {
            f"p{int(cl*100)}": equity_array.T[np.arange(equity_array.shape[1])].mean(axis=1)
            for cl in [0.05, 0.25, 0.50, 0.75, 0.95]
        }

        if verbose:
            print(f"  Retorno median (MC):    {mc_stats['total_return']['p50']:.2f}%")
            print(f"  Sharpe median (MC):     {mc_stats['sharpe']['p50']:.4f}")
            print(f"  Max DD median (MC):     {mc_stats['max_drawdown']['p50']:.2f}%")
            print(f"  Prob. pérdida:          {mc_stats['prob_loss']:.2f}%")

        return mc_stats

    # ------------------------------------------------------------------
    # Análisis de sensibilidad
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        base_params: Optional[Dict] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Análisis de sensibilidad univariado: varía un parámetro a la vez
        manteniendo los demás en su valor base (o mejor encontrado).

        Returns
        -------
        DataFrame con el efecto de cada parámetro sobre la métrica objetivo.
        """
        if base_params is None:
            base_params = self.best_params or next(iter(self.param_grid))

        if verbose:
            print(f"\n  ANÁLISIS DE SENSIBILIDAD | Base params: {base_params}")

        engine = BacktestEngine(self.config)
        rows = []

        for param_name, param_values in self.param_grid.param_space.items():
            for val in param_values:
                test_params = {**base_params, param_name: val}
                try:
                    strategy = self.strategy_factory(**test_params)
                    result = engine.run(df, strategy, ticker=ticker)
                    m = result.metrics
                    if "error" in m:
                        continue
                    rows.append({
                        "param_name": param_name,
                        "param_value": val,
                        "is_base": val == base_params.get(param_name),
                        self.optimize_metric: m.get(self.optimize_metric),
                        "total_return_pct": m.get("total_return_pct"),
                        "sharpe_ratio": m.get("sharpe_ratio"),
                        "max_drawdown_pct": m.get("max_drawdown_pct"),
                        "n_trades": m.get("n_trades"),
                    })
                except Exception:
                    continue

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Análisis completo
    # ------------------------------------------------------------------

    def full_analysis(
        self,
        df: pd.DataFrame,
        ticker: str,
        run_wfo: bool = True,
        run_mc: bool = True,
        n_mc: int = 1000,
        verbose: bool = True,
    ) -> "OptimizationReport":
        """
        Ejecuta el análisis completo:
        1. Grid Search
        2. Walk-Forward Optimization
        3. Monte Carlo sobre el mejor modelo
        4. Análisis de sensibilidad

        Returns
        -------
        OptimizationReport con todos los resultados
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ANÁLISIS COMPLETO: {ticker}")
            print(f"{'='*60}")

        # 1. Grid Search
        grid_df = self.grid_search(df, ticker, verbose=verbose)

        # 2. WFO
        wfo_df = pd.DataFrame()
        if run_wfo and self.best_params:
            wfo_df = self.walk_forward(df, ticker, verbose=verbose)

        # 3. Monte Carlo
        mc_stats = {}
        if run_mc and self.best_result:
            mc_stats = self.monte_carlo(self.best_result, n_simulations=n_mc, verbose=verbose)

        # 4. Sensibilidad
        sensitivity_df = pd.DataFrame()
        if self.best_params:
            sensitivity_df = self.sensitivity_analysis(df, ticker, verbose=verbose)

        return OptimizationReport(
            ticker=ticker,
            optimize_metric=self.optimize_metric,
            best_params=self.best_params,
            best_result=self.best_result,
            grid_results=grid_df,
            wfo_results=wfo_df,
            mc_stats=mc_stats,
            sensitivity_results=sensitivity_df,
        )


# ---------------------------------------------------------------------------
# Reporte de optimización
# ---------------------------------------------------------------------------

class OptimizationReport:
    """
    Contenedor del reporte completo de optimización de una estrategia.
    """

    def __init__(
        self,
        ticker: str,
        optimize_metric: str,
        best_params: Optional[Dict],
        best_result: Optional[BacktestResult],
        grid_results: pd.DataFrame,
        wfo_results: pd.DataFrame,
        mc_stats: Dict,
        sensitivity_results: pd.DataFrame,
    ):
        self.ticker = ticker
        self.optimize_metric = optimize_metric
        self.best_params = best_params
        self.best_result = best_result
        self.grid_results = grid_results
        self.wfo_results = wfo_results
        self.mc_stats = mc_stats
        self.sensitivity_results = sensitivity_results

    def print_report(self) -> None:
        """Imprime el reporte completo."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  REPORTE DE OPTIMIZACIÓN: {self.ticker}")
        print(sep)

        print(f"\n  MEJORES PARÁMETROS: {self.best_params}")
        print(f"  Métrica optimizada: {self.optimize_metric}")

        if self.best_result:
            self.best_result.print_report()

        if not self.grid_results.empty:
            print(f"\n  TOP 10 COMBINACIONES (Grid Search):")
            cols = ["params_str", self.optimize_metric, "sharpe_ratio",
                    "total_return_pct", "max_drawdown_pct", "n_trades"]
            cols = [c for c in cols if c in self.grid_results.columns]
            print(self.grid_results[cols].head(10).to_string(index=False))

        if not self.wfo_results.empty:
            print(f"\n  WALK-FORWARD RESULTS:")
            wfo_cols = ["fold", "is_sharpe", "oos_sharpe", "oos_return_%",
                        "oos_max_dd_%", "efficiency", "best_params"]
            wfo_cols = [c for c in wfo_cols if c in self.wfo_results.columns]
            print(self.wfo_results[wfo_cols].to_string(index=False))
            avg_eff = self.wfo_results["efficiency"].mean()
            print(f"\n  Eficiencia promedio IS→OOS: {avg_eff:.4f}")
            print(f"  {'Sistema ROBUSTO ✓' if avg_eff > 0.5 else 'POSIBLE SOBREAJUSTE ⚠'}")

        if self.mc_stats and "prob_loss" in self.mc_stats:
            print(f"\n  MONTE CARLO ({self.mc_stats.get('n_simulations')} simulaciones):")
            for metric in ["total_return", "sharpe", "max_drawdown"]:
                if metric in self.mc_stats:
                    s = self.mc_stats[metric]
                    print(f"    {metric:20s}: p5={s.get('p5'):.3f}  "
                          f"median={s.get('p50'):.3f}  p95={s.get('p95'):.3f}")
            print(f"    Probabilidad de pérdida: {self.mc_stats['prob_loss']:.2f}%")

    def get_trades_df(self) -> pd.DataFrame:
        """Retorna el DataFrame de trades del mejor modelo."""
        if self.best_result:
            return self.best_result.trades_df
        return pd.DataFrame()

    def get_time_series(self) -> pd.DataFrame:
        """
        Retorna las series de tiempo más importantes del mejor modelo.
        """
        if self.best_result is None:
            return pd.DataFrame()

        result = self.best_result
        series_dict = {}

        if not result.equity_curve.empty:
            series_dict["equity"] = result.equity_curve
        if not result.drawdown_series.empty:
            series_dict["drawdown"] = result.drawdown_series
        if not result.buy_and_hold.empty:
            series_dict["buy_and_hold"] = result.buy_and_hold
        if not result.rolling_metrics.empty:
            for col in result.rolling_metrics.columns:
                series_dict[col] = result.rolling_metrics[col]

        if not series_dict:
            return pd.DataFrame()

        return pd.DataFrame(series_dict)

    def export(self, path: str = "results", ticker: str = "") -> None:
        """Exporta todos los resultados a Excel."""
        import os
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"{ticker or self.ticker}_optimization.xlsx")

        with pd.ExcelWriter(fname, engine="openpyxl") as writer:
            if not self.grid_results.empty:
                self.grid_results.head(50).to_excel(writer, sheet_name="GridSearch", index=False)
            if not self.wfo_results.empty:
                self.wfo_results.to_excel(writer, sheet_name="WalkForward", index=False)
            if self.best_result and not self.best_result.trades_df.empty:
                self.best_result.trades_df.to_excel(writer, sheet_name="BestTrades", index=False)
            ts = self.get_time_series()
            if not ts.empty:
                ts.to_excel(writer, sheet_name="TimeSeries")
            if not self.sensitivity_results.empty:
                self.sensitivity_results.to_excel(writer, sheet_name="Sensitivity", index=False)
            # MC stats summary
            if self.mc_stats:
                mc_rows = []
                for metric, stats in self.mc_stats.items():
                    if isinstance(stats, dict):
                        row = {"metric": metric, **stats}
                        mc_rows.append(row)
                if mc_rows:
                    pd.DataFrame(mc_rows).to_excel(writer, sheet_name="MonteCarlo", index=False)

        logger.info(f"Reporte exportado: {fname}")
        print(f"  Reporte guardado en: {fname}")


# ---------------------------------------------------------------------------
# Factories de estrategias parametrizadas para optimización
# ---------------------------------------------------------------------------

def make_sma_crossover_strategy(
    fast: int = 20,
    slow: int = 50,
    stop_loss: float = 0.05,
    take_profit: float = 0.15,
    max_holding_days: Optional[int] = None,
) -> Strategy:
    """Factory para SMA crossover parametrizado."""
    from strategies import Strategy, Signal
    from indicators import sma as _sma

    fast_col = f"sma_{fast}"
    slow_col = f"sma_{slow}"

    return Strategy(
        name=f"SMA_{fast}_{slow}_SL{int(stop_loss*100)}_TP{int(take_profit*100)}",
        entry=Signal.crossover(fast_col, slow_col),
        exit=Signal.crossunder(fast_col, slow_col),
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        required_indicators=[fast_col, slow_col],
        category="trend_following",
    )


def make_ema_crossover_strategy(
    fast: int = 9,
    slow: int = 21,
    stop_loss: float = 0.05,
    take_profit: float = 0.15,
) -> Strategy:
    """Factory para EMA crossover parametrizado."""
    from strategies import Strategy, Signal
    return Strategy(
        name=f"EMA_{fast}_{slow}_SL{int(stop_loss*100)}_TP{int(take_profit*100)}",
        entry=Signal.crossover(f"ema_{fast}", f"ema_{slow}"),
        exit=Signal.crossunder(f"ema_{fast}", f"ema_{slow}"),
        stop_loss=stop_loss,
        take_profit=take_profit,
        required_indicators=[f"ema_{fast}", f"ema_{slow}"],
        category="trend_following",
    )


def make_rsi_strategy(
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    stop_loss: float = 0.04,
    take_profit: float = 0.10,
    max_holding_days: int = 20,
) -> Strategy:
    """Factory para RSI mean reversion parametrizado."""
    from strategies import Strategy, Signal
    rsi_col = f"rsi_{period}"
    return Strategy(
        name=f"RSI_{period}_OS{int(oversold)}_OB{int(overbought)}",
        entry=Signal.crossover(rsi_col, oversold),
        exit=Signal.crossover(rsi_col, overbought),
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        required_indicators=[rsi_col],
        category="mean_reversion",
    )


def make_bollinger_strategy(
    period: int = 20,
    std_dev: float = 2.0,
    stop_loss: float = 0.05,
    take_profit: float = 0.12,
    max_holding_days: int = 20,
) -> Strategy:
    """Factory para Bollinger Bands parametrizado."""
    from strategies import Strategy, Signal
    return Strategy(
        name=f"BB_{period}_{std_dev}_SL{int(stop_loss*100)}",
        entry=Signal.crossover("close", f"bb_lower_{period}"),
        exit=Signal.crossover("close", f"bb_mid_{period}"),
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_holding_days=max_holding_days,
        required_indicators=[f"bb_upper_{period}", f"bb_mid_{period}", f"bb_lower_{period}"],
        category="mean_reversion",
    )


def make_macd_strategy(
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    stop_loss: float = 0.06,
    take_profit: float = 0.18,
) -> Strategy:
    """Factory para MACD parametrizado."""
    from strategies import Strategy, Signal, Rule
    return Strategy(
        name=f"MACD_{fast}_{slow}_{signal}_SL{int(stop_loss*100)}",
        entry=Signal.crossover("macd", "macd_signal"),
        exit=Signal.crossunder("macd", "macd_signal"),
        stop_loss=stop_loss,
        take_profit=take_profit,
        required_indicators=["macd", "macd_signal", "macd_hist"],
        category="momentum",
    )


# ---------------------------------------------------------------------------
# Grids predefinidos para optimización rápida
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "sma_crossover": ParameterGrid({
        "fast": [10, 20, 30, 50],
        "slow": [50, 100, 150, 200],
        "stop_loss": [0.04, 0.06, 0.08],
        "take_profit": [0.10, 0.15, 0.20, 0.25],
    }),
    "ema_crossover": ParameterGrid({
        "fast": [5, 9, 12, 21],
        "slow": [21, 34, 50, 89],
        "stop_loss": [0.03, 0.05, 0.07],
        "take_profit": [0.10, 0.15, 0.20],
    }),
    "rsi": ParameterGrid({
        "period": [9, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [65, 70, 75, 80],
        "stop_loss": [0.03, 0.05],
        "take_profit": [0.08, 0.12, 0.15],
        "max_holding_days": [15, 20, 30],
    }),
    "bollinger": ParameterGrid({
        "period": [15, 20, 25],
        "std_dev": [1.5, 2.0, 2.5],
        "stop_loss": [0.03, 0.05, 0.07],
        "take_profit": [0.08, 0.12, 0.15],
        "max_holding_days": [10, 15, 20, 30],
    }),
    "macd": ParameterGrid({
        "fast": [8, 12, 16],
        "slow": [21, 26, 34],
        "signal": [7, 9, 12],
        "stop_loss": [0.04, 0.06, 0.08],
        "take_profit": [0.12, 0.18, 0.24],
    }),
}

STRATEGY_FACTORIES = {
    "sma_crossover": make_sma_crossover_strategy,
    "ema_crossover": make_ema_crossover_strategy,
    "rsi": make_rsi_strategy,
    "bollinger": make_bollinger_strategy,
    "macd": make_macd_strategy,
}


def optimize_strategy(
    strategy_type: str,
    df: pd.DataFrame,
    ticker: str,
    config: Optional[BacktestConfig] = None,
    optimize_metric: str = "sharpe_ratio",
    run_wfo: bool = True,
    run_mc: bool = True,
    verbose: bool = True,
) -> OptimizationReport:
    """
    Función de conveniencia para optimizar una estrategia predefinida.

    Parámetros
    ----------
    strategy_type   : "sma_crossover", "ema_crossover", "rsi", "bollinger", "macd"
    df              : DataFrame de datos
    ticker          : símbolo del activo
    optimize_metric : métrica a maximizar

    Ejemplo:
        report = optimize_strategy("rsi", df, "AAPL")
        report.print_report()
    """
    if strategy_type not in STRATEGY_FACTORIES:
        raise ValueError(f"Tipo no soportado: {strategy_type}. Opciones: {list(STRATEGY_FACTORIES)}")

    factory = STRATEGY_FACTORIES[strategy_type]
    grid = PARAM_GRIDS[strategy_type]

    optimizer = StrategyOptimizer(
        strategy_factory=factory,
        param_grid=grid,
        config=config or BacktestConfig(),
        optimize_metric=optimize_metric,
    )

    return optimizer.full_analysis(
        df, ticker, run_wfo=run_wfo, run_mc=run_mc, verbose=verbose
    )


def create_auto_param_grid(strategy: Strategy, granularity: str = "medium") -> ParameterGrid:
    """
    Crea automáticamente un grid de parámetros para optimizar una estrategia.
    
    Parameters
    ----------
    strategy : Strategy object
    granularity : "coarse" | "medium" | "fine"
        - coarse: 2-3 valores por parámetro (~16-27 combinaciones)
        - medium: 3-4 valores por parámetro (~81-256 combinaciones)
        - fine: 5+ valores por parámetro (>625 combinaciones)
    
    Returns
    -------
    ParameterGrid con stop_loss, take_profit, max_holding_days y trailing_stop
    """
    if granularity == "coarse":
        param_space = {
            "stop_loss": [0.03, 0.06, 0.10] if strategy.stop_loss else [None],
            "take_profit": [0.08, 0.15, 0.25] if strategy.take_profit else [None],
            "max_holding_days": [10, 30, 60] if strategy.max_holding_days else [None],
        }
    elif granularity == "fine":
        param_space = {
            "stop_loss": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
            "take_profit": [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
            "max_holding_days": [5, 10, 15, 20, 30, 45, 60, 90],
        }
    else:  # medium
        param_space = {
            "stop_loss": [0.03, 0.04, 0.05, 0.06, 0.08],
            "take_profit": [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            "max_holding_days": [10, 20, 30, 45, 60],
        }
    
    # Agregar trailing_stop si la estrategia lo usa
    if strategy.trailing_stop:
        if granularity == "coarse":
            param_space["trailing_stop"] = [0.05, 0.10, 0.15]
        elif granularity == "fine":
            param_space["trailing_stop"] = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
        else:
            param_space["trailing_stop"] = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    # Filtrar None values
    param_space = {k: [x for x in v if x is not None] for k, v in param_space.items()}
    param_space = {k: v for k, v in param_space.items() if v}
    
    return ParameterGrid(param_space)


def optimize_any_strategy(
    strategy: Strategy,
    df: pd.DataFrame,
    ticker: str,
    param_grid: Optional[ParameterGrid] = None,
    config: Optional[BacktestConfig] = None,
    optimize_metric: str = "sharpe_ratio",
    run_wfo: bool = True,
    run_mc: bool = True,
    granularity: str = "medium",
    verbose: bool = True,
) -> OptimizationReport:
    """
    Optimiza CUALQUIER estrategia del STRATEGY_LIBRARY o personalizada.
    
    Si no se proporciona param_grid, se crea automáticamente basado en
    los parámetros de la estrategia (stop_loss, take_profit, max_holding_days).
    
    Parameters
    ----------
    strategy : Strategy
        Cualquier estrategia (de la biblioteca o personalizada)
    df : DataFrame
        Datos históricos
    ticker : str
        Símbolo del activo
    param_grid : ParameterGrid (opcional)
        Si no se proporciona, se genera automáticamente
    granularity : str
        "coarse" | "medium" | "fine" - solo si param_grid es None
    
    Example
    -------
    from strategies import get_strategy
    strategy = get_strategy("ROC_Momentum")
    report = optimize_any_strategy(strategy, df, "AAPL")
    """
    # Si no hay grid, crear uno automático
    if param_grid is None:
        param_grid = create_auto_param_grid(strategy, granularity=granularity)
        if verbose:
            print(f"📊 Grid automático: {len(param_grid)} combinaciones")
            print(f"   Parámetros: {list(param_grid.param_space.keys())}")
    
    # Crear factory que modifica la estrategia base
    def strategy_factory(**params):
        # Clonar estrategia
        import copy
        new_strat = copy.deepcopy(strategy)
        
        # Actualizar parámetros
        if "stop_loss" in params:
            new_strat.stop_loss = params["stop_loss"]
        if "take_profit" in params:
            new_strat.take_profit = params["take_profit"]
        if "max_holding_days" in params:
            new_strat.max_holding_days = params["max_holding_days"]
        if "trailing_stop" in params:
            new_strat.trailing_stop = params["trailing_stop"]
        
        # Actualizar nombre para reflejar params
        param_str = "_".join([f"{k[:2]}{v}" for k, v in params.items() if v is not None])
        new_strat.name = f"{strategy.name}_{param_str}"
        
        return new_strat
    
    # Crear optimizador
    optimizer = StrategyOptimizer(
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        config=config or BacktestConfig(),
        optimize_metric=optimize_metric,
    )
    
    return optimizer.full_analysis(
        df, ticker, run_wfo=run_wfo, run_mc=run_mc, verbose=verbose
    )


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import download_data
    from backtest_engine import BacktestConfig

    ticker = "AAPL"
    print(f"Descargando datos para {ticker}...")
    df = download_data([ticker], period="5y", cache_dir="data/cache")

    cfg = BacktestConfig(initial_capital=100_000)

    # Grid Search para RSI
    grid = ParameterGrid({
        "period": [9, 14],
        "oversold": [25, 30],
        "overbought": [70, 75],
        "stop_loss": [0.04, 0.06],
        "take_profit": [0.10, 0.15],
        "max_holding_days": [20],
    })

    optimizer = StrategyOptimizer(
        strategy_factory=make_rsi_strategy,
        param_grid=grid,
        config=cfg,
        optimize_metric="sharpe_ratio",
    )

    report = optimizer.full_analysis(df, ticker, run_wfo=True, run_mc=True, n_mc=500)
    report.print_report()
    print("\nTrades del mejor modelo:")
    print(report.get_trades_df().tail(10).to_string())
