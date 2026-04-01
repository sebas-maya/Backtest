"""
strategy_scanner.py
===================
Módulo de análisis general: ejecuta el backtest de múltiples estrategias
sobre un ticker y genera un resumen ejecutivo comparativo.

Funcionalidades:
- Scan de todas las estrategias de la librería (o un subconjunto)
- Resumen ejecutivo tabular ordenado por métrica seleccionada
- Agrupación por categoría de estrategia
- Comparación vs Buy & Hold
- Filtros de calidad (mínimo de trades, Sharpe mínimo, etc.)
- Exportación a CSV/Excel

Autor: Backtest Framework
"""

from __future__ import annotations

import warnings
import logging
from typing import List, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from strategies import Strategy, STRATEGY_LIBRARY

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class StrategyScanner:
    """
    Ejecuta múltiples estrategias sobre un activo y genera
    un resumen comparativo ejecutivo.

    Uso:
        scanner = StrategyScanner(config=BacktestConfig())
        results = scanner.scan(df, ticker="AAPL")
        summary = scanner.get_summary()
        scanner.print_executive_report()
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        strategies: Optional[List[Strategy]] = None,
        n_workers: int = 4,
    ):
        self.config = config or BacktestConfig()
        self.strategies = strategies or list(STRATEGY_LIBRARY.values())
        self.n_workers = n_workers
        self._results: Dict[str, BacktestResult] = {}
        self._ticker: str = ""
        self._bh_return: float = 0.0

    def scan(
        self,
        df: pd.DataFrame,
        ticker: str,
        strategy_names: Optional[List[str]] = None,
        parallel: bool = True,
        verbose: bool = True,
    ) -> Dict[str, BacktestResult]:
        """
        Ejecuta el backtest de todas las estrategias.

        Parameters
        ----------
        df             : DataFrame en formato long o wide con datos del activo
        ticker         : símbolo del activo
        strategy_names : lista de nombres de estrategias; None = todas
        parallel       : si True, usa ThreadPoolExecutor para paralelizar
        verbose        : mostrar progreso

        Returns
        -------
        dict {strategy_name: BacktestResult}
        """
        self._ticker = ticker
        strategies_to_run = [
            s for s in self.strategies
            if (strategy_names is None or s.name in strategy_names)
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  SCAN: {ticker} | {len(strategies_to_run)} estrategias")
            print(f"{'='*60}")

        engine = BacktestEngine(self.config)

        # Calcular Buy & Hold una sola vez
        self._compute_bh(df, ticker, engine)

        if parallel and len(strategies_to_run) > 1:
            results = self._run_parallel(engine, df, ticker, strategies_to_run, verbose)
        else:
            results = self._run_sequential(engine, df, ticker, strategies_to_run, verbose)

        self._results = results

        valid = sum(1 for r in results.values() if "error" not in r.metrics)
        if verbose:
            print(f"\n  Completado: {valid}/{len(strategies_to_run)} estrategias exitosas")

        return results

    def _compute_bh(self, df: pd.DataFrame, ticker: str, engine: BacktestEngine) -> None:
        """Calcula el retorno de Buy & Hold para referencia."""
        try:
            prep = engine._prepare_data(df, ticker, add_indicators=False)
            if prep is not None and not prep.empty:
                col = "adj_close" if "adj_close" in prep.columns else "close"
                prices = prep[col].dropna()
                if len(prices) > 1:
                    self._bh_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        except Exception:
            self._bh_return = 0.0

    def _run_sequential(
        self,
        engine: BacktestEngine,
        df: pd.DataFrame,
        ticker: str,
        strategies: List[Strategy],
        verbose: bool,
    ) -> Dict[str, BacktestResult]:
        results = {}
        for i, strategy in enumerate(strategies, 1):
            if verbose:
                print(f"  [{i:02d}/{len(strategies)}] {strategy.name:<40}", end=" ")
            try:
                result = engine.run(df, strategy, ticker=ticker)
                results[strategy.name] = result
                if verbose:
                    sr = result.metrics.get("sharpe_ratio", "ERR")
                    tr = result.metrics.get("total_return_pct", "ERR")
                    print(f"✓ Sharpe={sr:.3f}  Return={tr:.1f}%" if isinstance(sr, float) else f"✗ {result.metrics.get('error', '')}")
            except Exception as e:
                results[strategy.name] = BacktestResult.empty(strategy.name, ticker, str(e))
                if verbose:
                    print(f"✗ {e}")
        return results

    def _run_parallel(
        self,
        engine: BacktestEngine,
        df: pd.DataFrame,
        ticker: str,
        strategies: List[Strategy],
        verbose: bool,
    ) -> Dict[str, BacktestResult]:
        results = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_name = {
                executor.submit(engine.run, df, s, ticker): s.name
                for s in strategies
            }
            completed = 0
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                completed += 1
                try:
                    result = future.result()
                    results[name] = result
                    if verbose:
                        sr = result.metrics.get("sharpe_ratio", "ERR")
                        tr = result.metrics.get("total_return_pct", "ERR")
                        status = (f"✓ Sharpe={sr:.3f}  Return={tr:.1f}%"
                                  if isinstance(sr, float) else
                                  f"✗ {result.metrics.get('error', '')}")
                        print(f"  [{completed:02d}/{len(strategies)}] {name:<40} {status}")
                except Exception as e:
                    results[name] = BacktestResult.empty(name, ticker, str(e))
                    if verbose:
                        print(f"  [{completed:02d}/{len(strategies)}] {name:<40} ✗ {e}")
        return results

    def get_summary(
        self,
        sort_by: str = "sharpe_ratio",
        min_trades: int = 5,
        min_sharpe: float = -99.0,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Genera el DataFrame resumen ejecutivo con todas las estrategias.

        Parameters
        ----------
        sort_by    : métrica para ordenar ("sharpe_ratio", "total_return_pct",
                     "calmar_ratio", "profit_factor", etc.)
        min_trades : filtro mínimo de trades
        min_sharpe : filtro mínimo de Sharpe
        ascending  : orden descendente por defecto
        """
        rows = []
        for name, result in self._results.items():
            m = result.metrics
            if "error" in m:
                continue
            row = {
                "strategy": name,
                "category": _get_category(name),
                "total_return_%": m.get("total_return_pct"),
                "cagr_%": m.get("cagr_pct"),
                "sharpe_ratio": m.get("sharpe_ratio"),
                "sortino_ratio": m.get("sortino_ratio"),
                "calmar_ratio": m.get("calmar_ratio"),
                "omega_ratio": m.get("omega_ratio"),
                "profit_factor": m.get("profit_factor"),
                "max_dd_%": m.get("max_drawdown_pct"),
                "annual_vol_%": m.get("annual_volatility_pct"),
                "win_rate_%": m.get("win_rate_pct"),
                "n_trades": m.get("n_trades"),
                "avg_holding_days": m.get("avg_holding_days"),
                "expectancy_%": m.get("expectancy_pct"),
                "recovery_factor": m.get("recovery_factor"),
                "pct_pos_months_%": m.get("pct_positive_months"),
                "var_95_%": m.get("var_95_pct"),
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Filtros
        df = df[df["n_trades"] >= min_trades]
        df = df[df["sharpe_ratio"] >= min_sharpe]

        # Agregar columna de alpha vs Buy & Hold
        df["alpha_vs_bh_%"] = df["total_return_%"].apply(
            lambda x: round(x - self._bh_return, 2) if pd.notna(x) else None
        )

        # Rank compuesto (normalizar métricas clave)
        df = _add_composite_score(df)

        # Ordenar
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending, na_position="last")

        return df.reset_index(drop=True)

    def get_summary_by_category(self, **kwargs) -> pd.DataFrame:
        """Resumen agrupado por categoría de estrategia."""
        summary = self.get_summary(**kwargs)
        if summary.empty:
            return pd.DataFrame()
        numeric_cols = ["total_return_%", "sharpe_ratio", "sortino_ratio",
                        "max_dd_%", "win_rate_%", "n_trades", "profit_factor"]
        agg = summary.groupby("category")[
            [c for c in numeric_cols if c in summary.columns]
        ].agg(["mean", "max", "count"]).round(3)
        return agg

    def get_top_strategies(self, n: int = 10, sort_by: str = "composite_score") -> pd.DataFrame:
        """Retorna las top N estrategias."""
        return self.get_summary(sort_by=sort_by).head(n)

    def get_result(self, strategy_name: str) -> Optional[BacktestResult]:
        """Obtiene el BacktestResult de una estrategia específica."""
        return self._results.get(strategy_name)

    def print_executive_report(
        self,
        top_n: int = 15,
        sort_by: str = "composite_score",
    ) -> None:
        """
        Imprime el resumen ejecutivo en consola con formato tabular.
        """
        summary = self.get_summary(sort_by=sort_by)
        if summary.empty:
            print("Sin resultados disponibles.")
            return

        sep = "=" * 100
        print(f"\n{sep}")
        print(f"  RESUMEN EJECUTIVO: {self._ticker} | Top {top_n} Estrategias")
        print(f"  Buy & Hold retorno: {self._bh_return:.2f}%")
        print(sep)

        display_cols = [
            "strategy", "category", "total_return_%", "cagr_%",
            "sharpe_ratio", "sortino_ratio", "max_dd_%",
            "win_rate_%", "profit_factor", "n_trades", "composite_score",
        ]
        display_cols = [c for c in display_cols if c in summary.columns]
        top = summary[display_cols].head(top_n)

        # Formatear como tabla
        pd.set_option("display.max_colwidth", 30)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.3f}".format)
        print(top.to_string(index=False))

        print(f"\n  Total estrategias evaluadas: {len(summary)}")
        print(f"  Estrategias con alpha positivo vs B&H: "
              f"{(summary['alpha_vs_bh_%'] > 0).sum()}")

        # Mejor por categoría
        print(f"\n  MEJOR POR CATEGORÍA:")
        best_by_cat = summary.groupby("category").apply(
            lambda x: x.nlargest(1, "composite_score")[["strategy", "sharpe_ratio", "total_return_%"]]
        ).reset_index(level=1, drop=True)
        print(best_by_cat.to_string())
        print(sep)

    def export_results(
        self,
        path: str = "backtest_results",
        format: str = "csv",
    ) -> str:
        """
        Exporta los resultados a CSV o Excel.

        Returns
        -------
        Ruta del archivo generado
        """
        import os
        os.makedirs(path, exist_ok=True)
        summary = self.get_summary()

        if format == "excel":
            filepath = os.path.join(path, f"{self._ticker}_strategy_scan.xlsx")
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="Summary", index=False)
                # Trades de top 5
                top5 = self.get_top_strategies(5)
                for _, row in top5.iterrows():
                    name = row["strategy"]
                    result = self.get_result(name)
                    if result and not result.trades_df.empty:
                        sheet_name = name[:30]
                        result.trades_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            filepath = os.path.join(path, f"{self._ticker}_strategy_scan.csv")
            summary.to_csv(filepath, index=False)

        logger.info(f"Resultados exportados: {filepath}")
        return filepath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_category(strategy_name: str) -> str:
    """Infiere la categoría desde el nombre si no está disponible."""
    from strategies import STRATEGY_LIBRARY
    if strategy_name in STRATEGY_LIBRARY:
        return STRATEGY_LIBRARY[strategy_name].category
    name_lower = strategy_name.lower()
    if any(k in name_lower for k in ["sma", "ema", "golden", "ribbon", "price_above"]):
        return "trend_following"
    if any(k in name_lower for k in ["rsi", "bb_rev", "stoch", "cci", "williams"]):
        return "mean_reversion"
    if any(k in name_lower for k in ["macd", "roc", "momentum"]):
        return "momentum"
    if any(k in name_lower for k in ["break", "donchian", "keltner"]):
        return "breakout"
    if any(k in name_lower for k in ["volume", "obv", "mfi", "cmf"]):
        return "volume"
    return "general"


def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega un score compuesto normalizado (0-100) basado en:
    Sharpe (40%) + Profit Factor (20%) + Max DD (20%) + Win Rate (10%) + CAGR (10%)
    """
    df = df.copy()

    def normalize(series, ascending=True):
        s = pd.to_numeric(series, errors="coerce")
        s = s.clip(lower=s.quantile(0.05), upper=s.quantile(0.95))
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(50.0, index=s.index)
        norm = (s - s.min()) / rng * 100
        return norm if ascending else 100 - norm

    weights = {
        "sharpe_ratio": (0.35, True),
        "profit_factor": (0.20, True),
        "max_dd_%": (0.20, False),      # menor drawdown = mejor
        "win_rate_%": (0.10, True),
        "cagr_%": (0.15, True),
    }

    score = pd.Series(0.0, index=df.index)
    total_w = 0
    for col, (w, asc) in weights.items():
        if col in df.columns:
            score += w * normalize(df[col], ascending=asc)
            total_w += w

    if total_w > 0:
        score = score / total_w

    df["composite_score"] = score.round(2)
    return df


# ---------------------------------------------------------------------------
# Quick scan function
# ---------------------------------------------------------------------------

def quick_scan(
    ticker: str,
    period: str = "5y",
    top_n: int = 20,
    sort_by: str = "composite_score",
    config: Optional[BacktestConfig] = None,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Función de conveniencia: descarga datos, ejecuta scan y retorna resumen.

    Ejemplo:
        df = quick_scan("AAPL", period="5y", top_n=20)
        print(df)
    """
    from data_loader import download_data

    print(f"Descargando datos para {ticker}...")
    data = download_data([ticker], period=period, cache_dir=cache_dir)

    scanner = StrategyScanner(config=config)
    scanner.scan(data, ticker=ticker)
    scanner.print_executive_report(top_n=top_n, sort_by=sort_by)

    return scanner.get_summary(sort_by=sort_by)


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import download_data
    from backtest_engine import BacktestConfig

    ticker = "AAPL"
    print(f"Descargando datos para {ticker}...")
    df = download_data([ticker], period="5y", cache_dir="data/cache")

    cfg = BacktestConfig(initial_capital=100_000, commission_pct=0.001)
    scanner = StrategyScanner(config=cfg, n_workers=4)
    scanner.scan(df, ticker=ticker)
    scanner.print_executive_report(top_n=20)

    summary = scanner.get_summary()
    print("\nTop 5 por Sharpe:")
    print(summary[["strategy", "sharpe_ratio", "total_return_%", "max_dd_%", "n_trades"]].head(5))
