"""
main.py
=======
Script principal de demostración del framework de backtest.

Modos de uso:
1. SCAN GENERAL     : Evalúa todas las estrategias sobre un ticker
2. ANÁLISIS PROFUNDO: Optimiza una estrategia específica
3. MULTI-TICKER     : Scan rápido sobre un universo de activos
4. CUSTOM STRATEGY  : Ejemplo de cómo definir estrategias personalizadas

Ejecución:
    python main.py --mode scan --ticker AAPL --period 5y
    python main.py --mode deep --ticker AAPL --strategy rsi
    python main.py --mode multi --tickers AAPL,MSFT,GOOGL
    python main.py --mode demo
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Imports del framework
# ---------------------------------------------------------------------------
from data_loader import download_data, get_data_summary
from indicators import add_all_indicators, get_indicator_catalog
from strategies import (
    Strategy, Signal, Rule, PositionSizer,
    STRATEGY_LIBRARY, list_strategies, get_strategy
)
from backtest_engine import BacktestEngine, BacktestConfig
from strategy_scanner import StrategyScanner
from optimizer import (
    StrategyOptimizer, ParameterGrid, OptimizationReport,
    make_rsi_strategy, make_sma_crossover_strategy,
    make_ema_crossover_strategy, make_bollinger_strategy,
    make_macd_strategy, optimize_strategy, PARAM_GRIDS
)
from visualization import (
    plot_equity_curve, plot_drawdown, plot_trades,
    plot_monthly_returns, plot_rolling_metrics,
    plot_candlestick_signals, plot_scanner_summary,
    plot_optimization_heatmap, plot_monte_carlo,
    plot_wfo_results, plot_returns_distribution,
    plot_full_dashboard, plot_sensitivity, save_figure
)


# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = BacktestConfig(
    initial_capital=100_000,
    commission_pct=0.001,   # 0.1% por trade
    slippage_pct=0.0005,    # 0.05%
    risk_free_rate=0.04,
    annualization=252,
    min_data_points=150,
    execution_price="open",
)


# ---------------------------------------------------------------------------
# MODO 1: Scan general
# ---------------------------------------------------------------------------

def run_scan(
    ticker: str,
    period: str = "5y",
    top_n: int = 20,
    save_results: bool = True,
    show_plots: bool = True,
) -> None:
    """
    Ejecuta el backtest de todas las estrategias predefinidas sobre un ticker
    y muestra un resumen ejecutivo comparativo.
    """
    print(f"\n{'#'*60}")
    print(f"  MODO: SCAN GENERAL")
    print(f"  Ticker: {ticker} | Período: {period} | Top {top_n} estrategias")
    print(f"{'#'*60}")

    # 1. Descargar datos
    print("\n[1/3] Descargando datos...")
    df = download_data([ticker], period=period, cache_dir="data/cache")
    print(get_data_summary(df).to_string(index=False))

    # 2. Ejecutar scan
    print("\n[2/3] Ejecutando scan de estrategias...")
    scanner = StrategyScanner(config=DEFAULT_CONFIG, n_workers=4)
    scanner.scan(df, ticker=ticker, verbose=True)

    # 3. Mostrar resumen
    print("\n[3/3] Resultados...")
    scanner.print_executive_report(top_n=top_n)
    summary = scanner.get_summary()

    # Guardar resultados
    if save_results:
        os.makedirs("results", exist_ok=True)
        filepath = scanner.export_results("results", format="csv")
        print(f"\n  Resultados guardados en: {filepath}")

    # Visualizaciones
    if show_plots:
        print("\n  Generando visualizaciones...")

        # Bubble chart comparativo
        fig_scan = plot_scanner_summary(
            summary,
            top_n=min(top_n, len(summary)),
            x_metric="sharpe_ratio",
            y_metric="total_return_%",
            show=show_plots,
        )
        save_figure(fig_scan, f"results/{ticker}_scan_summary.html")

        # Dashboard del mejor modelo
        best_name = summary.iloc[0]["strategy"] if not summary.empty else None
        if best_name:
            best_result = scanner.get_result(best_name)
            if best_result and not best_result.equity_curve.empty:
                fig_dash = plot_full_dashboard(best_result, show=show_plots)
                save_figure(fig_dash, f"results/{ticker}_{best_name}_dashboard.html")

    print("\nScan completado.")


# ---------------------------------------------------------------------------
# MODO 2: Análisis profundo + optimización
# ---------------------------------------------------------------------------

def run_deep_analysis(
    ticker: str,
    strategy_type: str = "rsi",
    period: str = "5y",
    optimize_metric: str = "sharpe_ratio",
    run_wfo: bool = True,
    run_mc: bool = True,
    n_mc: int = 1000,
    show_plots: bool = True,
) -> OptimizationReport:
    """
    Análisis profundo de una estrategia:
    Grid Search + Walk-Forward + Monte Carlo + Sensibilidad.
    """
    print(f"\n{'#'*60}")
    print(f"  MODO: ANÁLISIS PROFUNDO")
    print(f"  Ticker: {ticker} | Estrategia: {strategy_type}")
    print(f"  Métrica: {optimize_metric}")
    print(f"{'#'*60}")

    # 1. Descargar datos
    print("\n[1/4] Descargando datos...")
    df = download_data([ticker], period=period, cache_dir="data/cache")

    # 2. Optimizar
    print("\n[2/4] Optimizando parámetros...")
    report = optimize_strategy(
        strategy_type=strategy_type,
        df=df,
        ticker=ticker,
        config=DEFAULT_CONFIG,
        optimize_metric=optimize_metric,
        run_wfo=run_wfo,
        run_mc=run_mc,
        verbose=True,
    )

    # 3. Reporte detallado
    print("\n[3/4] Generando reporte...")
    report.print_report()

    print("\n  Trades del mejor modelo (últimos 10):")
    trades_df = report.get_trades_df()
    if not trades_df.empty:
        print(trades_df.tail(10)[
            ["trade_id", "entry_date", "exit_date", "entry_price", "exit_price",
             "pnl_pct", "net_pnl", "holding_days", "exit_reason"]
        ].to_string(index=False))

    print("\n  Series de tiempo (últimas 5 filas):")
    ts = report.get_time_series()
    if not ts.empty:
        print(ts.tail(5).to_string())

    # 4. Visualizaciones
    if show_plots and report.best_result:
        print("\n[4/4] Generando visualizaciones interactivas...")
        os.makedirs("results", exist_ok=True)
        result = report.best_result

        fig1 = plot_full_dashboard(result, show=show_plots)
        save_figure(fig1, f"results/{ticker}_{strategy_type}_dashboard.html")

        fig2 = plot_candlestick_signals(result, n_last=180, show=show_plots)
        save_figure(fig2, f"results/{ticker}_{strategy_type}_signals.html")

        if not report.grid_results.empty and len(report.grid_results.columns) >= 2:
            param_cols = [c for c in report.grid_results.columns
                          if c not in ["params_str", "total_return_pct", "cagr_pct",
                                       "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                                       "max_drawdown_pct", "win_rate_pct", "profit_factor",
                                       "n_trades", "avg_holding_days", "annual_volatility_pct",
                                       "expectancy_pct", "omega_ratio", "recovery_factor"]]
            if len(param_cols) >= 2:
                fig3 = plot_optimization_heatmap(
                    report.grid_results,
                    param_x=param_cols[0],
                    param_y=param_cols[1],
                    metric=optimize_metric,
                    show=show_plots,
                )
                save_figure(fig3, f"results/{ticker}_{strategy_type}_heatmap.html")

        if report.mc_stats and "equity_percentiles" in report.mc_stats:
            fig4 = plot_monte_carlo(
                report.mc_stats,
                equity_curve=result.equity_curve,
                show=show_plots,
            )
            save_figure(fig4, f"results/{ticker}_{strategy_type}_montecarlo.html")

        if not report.wfo_results.empty:
            fig5 = plot_wfo_results(report.wfo_results, show=show_plots)
            save_figure(fig5, f"results/{ticker}_{strategy_type}_wfo.html")

        if not report.sensitivity_results.empty:
            fig6 = plot_sensitivity(report.sensitivity_results, metric=optimize_metric,
                                    show=show_plots)
            save_figure(fig6, f"results/{ticker}_{strategy_type}_sensitivity.html")

    # Exportar a Excel
    report.export("results", ticker=ticker)
    print("\nAnálisis profundo completado.")
    return report


# ---------------------------------------------------------------------------
# MODO 3: Multi-ticker scan
# ---------------------------------------------------------------------------

def run_multi_ticker(
    tickers: List[str],
    period: str = "3y",
    strategy_names: Optional[List[str]] = None,
    sort_by: str = "sharpe_ratio",
) -> None:
    """
    Scan rápido sobre un universo de activos y una selección de estrategias.
    """
    print(f"\n{'#'*60}")
    print(f"  MODO: MULTI-TICKER")
    print(f"  Tickers: {tickers}")
    print(f"{'#'*60}")

    print("\n[1/2] Descargando datos del universo...")
    df = download_data(tickers, period=period, cache_dir="data/cache")
    summary = get_data_summary(df)
    print(summary.to_string(index=False))

    all_results = []

    print("\n[2/2] Ejecutando scan...")
    for ticker in tickers:
        if ticker not in df["ticker"].unique():
            print(f"  {ticker}: sin datos, omitido")
            continue

        scanner = StrategyScanner(config=DEFAULT_CONFIG, n_workers=2)
        scanner.scan(df, ticker=ticker, strategy_names=strategy_names, verbose=False)
        sub_summary = scanner.get_summary(sort_by=sort_by)

        if not sub_summary.empty:
            sub_summary["ticker"] = ticker
            all_results.append(sub_summary.head(5))
            print(f"  {ticker}: top estrategia = {sub_summary.iloc[0]['strategy']} "
                  f"| Sharpe={sub_summary.iloc[0].get('sharpe_ratio', 0):.3f}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        filepath = "results/multi_ticker_scan.csv"
        os.makedirs("results", exist_ok=True)
        combined.to_csv(filepath, index=False)
        print(f"\n  Resultados combinados guardados en: {filepath}")
        print("\n  Resumen multi-ticker:")
        print(combined[["ticker", "strategy", "sharpe_ratio", "total_return_%",
                        "max_dd_%", "n_trades"]].to_string(index=False))


# ---------------------------------------------------------------------------
# MODO 4: Demo de estrategia personalizada
# ---------------------------------------------------------------------------

def run_custom_strategy_demo(
    ticker: str = "AAPL",
    period: str = "5y",
    show_plots: bool = True,
) -> None:
    """
    Demuestra cómo crear estrategias personalizadas usando el DSL del framework.
    """
    print(f"\n{'#'*60}")
    print(f"  MODO: DEMO - ESTRATEGIA PERSONALIZADA")
    print(f"{'#'*60}")

    print("\nEjemplos de estrategias personalizadas:")

    # -------------------------------------------------------
    # EJEMPLO 1: Estrategia que mencionaste en el enunciado
    # "Compra cuando el cierre cruza la media móvil 100
    #  con volumen elevado; venta después de 10 días
    #  o cuando cruza abajo la media de 20"
    # -------------------------------------------------------
    strategy_ejemplo1 = Strategy(
        name="Ejemplo1_SMA100_Vol_Exit20",
        description="Compra: close cruza SMA100 + volumen > 1.5x media; "
                    "Venta: 10 días o close baja SMA20",
        category="trend_following",
        entry=Rule([
            Signal.crossover("close", "sma_100"),
            Signal.above_threshold("volume", "sma_20", factor=1.5),
        ], combinator="AND"),
        exit=Rule([
            Signal.crossunder("close", "sma_20"),
        ]),
        stop_loss=0.06,
        take_profit=0.18,
        max_holding_days=10,
        position_sizer=PositionSizer(mode="fixed", fraction=0.10),
        required_indicators=["sma_100", "sma_20"],
    )

    # -------------------------------------------------------
    # EJEMPLO 2: Estrategia con position sizing dinámico
    # -------------------------------------------------------
    strategy_ejemplo2 = Strategy(
        name="Ejemplo2_MACD_RSI_DynamicSize",
        description="MACD + RSI combo con tamaño dinámico según racha reciente",
        category="combo",
        entry=Rule([
            Signal.crossover("macd", "macd_signal"),
            Signal.between("rsi_14", 40, 65),
            Signal.above("close", "sma_50"),
        ]),
        exit=Rule([
            Signal.crossunder("macd", "macd_signal"),
            Signal.above("rsi_14", 70),
        ], combinator="OR"),
        stop_loss=0.05,
        take_profit=0.15,
        trailing_stop=0.07,
        position_sizer=PositionSizer(
            mode="dynamic",
            fraction=0.10,
            dynamic_lookback=10,
            dynamic_scale=0.4,
            max_fraction=0.25,
        ),
        required_indicators=["macd", "macd_signal", "rsi_14", "sma_50"],
    )

    # -------------------------------------------------------
    # EJEMPLO 3: Gestión de riesgo con percent_risk sizing
    # -------------------------------------------------------
    strategy_ejemplo3 = Strategy(
        name="Ejemplo3_RSI_PercentRisk",
        description="RSI con position sizing basado en riesgo por trade (1%)",
        category="mean_reversion",
        entry=Signal.crossover("rsi_14", 30),
        exit=Signal.crossover("rsi_14", 70),
        stop_loss=0.04,
        take_profit=0.12,
        max_holding_days=25,
        position_sizer=PositionSizer(
            mode="percent_risk",
            risk_per_trade=0.01,  # 1% del capital por trade
            max_fraction=0.20,
        ),
        required_indicators=["rsi_14"],
    )

    # -------------------------------------------------------
    # EJEMPLO 4: Estrategia con Kelly criterion
    # -------------------------------------------------------
    strategy_ejemplo4 = Strategy(
        name="Ejemplo4_BB_Kelly",
        description="Bollinger reversal con Kelly sizing adaptativo",
        category="mean_reversion",
        entry=Rule([
            Signal.below("close", "bb_lower_20"),
            Signal.below("rsi_14", 35),
        ]),
        exit=Signal.crossover("close", "bb_mid_20"),
        stop_loss=0.05,
        take_profit=0.12,
        position_sizer=PositionSizer(
            mode="kelly",
            kelly_lookback=20,
            max_fraction=0.25,
            min_fraction=0.02,
        ),
        required_indicators=["bb_lower_20", "bb_mid_20", "bb_upper_20", "rsi_14"],
    )

    custom_strategies = [
        strategy_ejemplo1,
        strategy_ejemplo2,
        strategy_ejemplo3,
        strategy_ejemplo4,
    ]

    print(f"\nEstrategias personalizadas creadas: {len(custom_strategies)}")
    for s in custom_strategies:
        print(f"  - {s.name}: {s.description[:60]}...")

    print(f"\nDescargando datos para {ticker}...")
    df = download_data([ticker], period=period, cache_dir="data/cache")

    print("\nEjecutando backtest de estrategias personalizadas...")
    engine = BacktestEngine(DEFAULT_CONFIG)
    results = []

    for strategy in custom_strategies:
        print(f"  Evaluando: {strategy.name}...")
        result = engine.run(df, strategy, ticker=ticker)
        result.print_report()
        results.append(result)

    print("\nResumen comparativo:")
    import pandas as pd
    summary_rows = [r.summary() for r in results]
    print(pd.DataFrame(summary_rows).to_string(index=False))

    if show_plots and results:
        print("\nGenerando visualizaciones...")
        os.makedirs("results", exist_ok=True)
        for result in results:
            if not result.equity_curve.empty:
                fig = plot_full_dashboard(result, show=show_plots)
                save_figure(fig, f"results/{ticker}_{result.strategy_name}_dashboard.html")


# ---------------------------------------------------------------------------
# MODO 5: Demo completo automático
# ---------------------------------------------------------------------------

def run_demo(show_plots: bool = True) -> None:
    """Demo completo: descarga, scan, optimización y visualización."""
    import pandas as pd

    print("\n" + "=" * 70)
    print("  DEMO COMPLETO DEL FRAMEWORK DE BACKTEST")
    print("=" * 70)

    ticker = "AAPL"
    period = "5y"

    print("\n[PASO 1] Librería de indicadores técnicos disponibles:")
    catalog = get_indicator_catalog()
    for name, desc in list(catalog.items())[:8]:
        print(f"  {name:25s}: {desc}")
    print(f"  ... y {len(catalog) - 8} más")

    print("\n[PASO 2] Librería de estrategias predefinidas:")
    strat_df = list_strategies()
    print(strat_df[["name", "category", "stop_loss", "take_profit"]].to_string(index=False))

    print("\n[PASO 3] Descargando datos...")
    df = download_data([ticker, "^GSPC", "SPY"], period=period, cache_dir="data/cache")
    print(get_data_summary(df).to_string(index=False))

    print("\n[PASO 4] Scan general de estrategias...")
    scanner = StrategyScanner(config=DEFAULT_CONFIG, n_workers=4)
    scanner.scan(df, ticker=ticker, verbose=True)
    scanner.print_executive_report(top_n=10)
    summary = scanner.get_summary()

    if show_plots and not summary.empty:
        os.makedirs("results", exist_ok=True)
        fig = plot_scanner_summary(summary, show=show_plots)
        save_figure(fig, f"results/{ticker}_scan_bubble.html")

    print("\n[PASO 5] Análisis profundo de la mejor estrategia...")
    if not summary.empty:
        best_name = summary.iloc[0]["strategy"]
        print(f"  Mejor estrategia: {best_name}")
        best_result = scanner.get_result(best_name)
        if best_result:
            best_result.print_report()
            if show_plots:
                fig2 = plot_full_dashboard(best_result, show=show_plots)
                save_figure(fig2, f"results/{ticker}_{best_name}_dashboard.html")
                fig3 = plot_candlestick_signals(best_result, n_last=200, show=show_plots)
                save_figure(fig3, f"results/{ticker}_{best_name}_signals.html")

    print("\n[PASO 6] Optimización de RSI strategy...")
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
        config=DEFAULT_CONFIG,
        optimize_metric="sharpe_ratio",
    )
    report = optimizer.full_analysis(
        df, ticker, run_wfo=True, run_mc=True, n_mc=500, verbose=True
    )
    report.print_report()

    if show_plots and report.best_result:
        fig4 = plot_monte_carlo(
            report.mc_stats,
            equity_curve=report.best_result.equity_curve,
            show=show_plots,
        )
        save_figure(fig4, f"results/{ticker}_rsi_montecarlo.html")

    print("\n[PASO 7] Demo de estrategia personalizada...")
    run_custom_strategy_demo(ticker=ticker, period="3y", show_plots=show_plots)

    print("\n" + "=" * 70)
    print("  DEMO COMPLETADO")
    print(f"  Resultados guardados en: ./results/")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Framework de Backtest de Estrategias de Inversión",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --mode demo
  python main.py --mode scan --ticker AAPL --period 5y
  python main.py --mode deep --ticker MSFT --strategy sma_crossover
  python main.py --mode multi --tickers AAPL,MSFT,GOOGL,TSLA
  python main.py --mode custom --ticker SPY
        """,
    )
    parser.add_argument("--mode", choices=["demo", "scan", "deep", "multi", "custom"],
                        default="demo", help="Modo de ejecución")
    parser.add_argument("--ticker", default="AAPL", help="Símbolo del activo")
    parser.add_argument("--tickers", default="AAPL,MSFT,GOOGL",
                        help="Símbolos separados por coma (modo multi)")
    parser.add_argument("--period", default="5y", help="Período: 1y,2y,5y,10y,max")
    parser.add_argument("--strategy", default="rsi",
                        choices=["sma_crossover", "ema_crossover", "rsi", "bollinger", "macd"],
                        help="Tipo de estrategia para optimizar (modo deep)")
    parser.add_argument("--metric", default="sharpe_ratio",
                        help="Métrica a optimizar: sharpe_ratio, calmar_ratio, total_return_pct")
    parser.add_argument("--top", type=int, default=20, help="Top N estrategias en el resumen")
    parser.add_argument("--no-plots", action="store_true", help="Desactivar visualizaciones")
    parser.add_argument("--no-wfo", action="store_true", help="Omitir Walk-Forward")
    parser.add_argument("--no-mc", action="store_true", help="Omitir Monte Carlo")
    parser.add_argument("--n-mc", type=int, default=1000, help="Simulaciones Monte Carlo")
    return parser.parse_args()


def main():
    import pandas as pd
    args = parse_args()
    show_plots = not args.no_plots

    if args.mode == "demo":
        run_demo(show_plots=show_plots)

    elif args.mode == "scan":
        run_scan(
            ticker=args.ticker,
            period=args.period,
            top_n=args.top,
            show_plots=show_plots,
        )

    elif args.mode == "deep":
        run_deep_analysis(
            ticker=args.ticker,
            strategy_type=args.strategy,
            period=args.period,
            optimize_metric=args.metric,
            run_wfo=not args.no_wfo,
            run_mc=not args.no_mc,
            n_mc=args.n_mc,
            show_plots=show_plots,
        )

    elif args.mode == "multi":
        tickers = [t.strip() for t in args.tickers.split(",")]
        run_multi_ticker(
            tickers=tickers,
            period=args.period,
        )

    elif args.mode == "custom":
        run_custom_strategy_demo(
            ticker=args.ticker,
            period=args.period,
            show_plots=show_plots,
        )


if __name__ == "__main__":
    main()
