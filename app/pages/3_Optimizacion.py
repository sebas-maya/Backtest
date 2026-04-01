"""
app/pages/3_Optimizacion.py
============================
Página 3 — Optimización de estrategias.
Grid Search, Walk-Forward Optimization, Monte Carlo y Análisis de Sensibilidad.
Sin lógica analítica propia — solo llama al modelo existente.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from app.utils import (
    init_state, backtest_config_sidebar,
    get_long_df, METRIC_OPTIONS, STRATEGY_TYPE_OPTIONS,
    show_metrics_grid, page_header,
)
from data_loader import get_available_tickers
from optimizer import (
    StrategyOptimizer, ParameterGrid,
    make_sma_crossover_strategy, make_ema_crossover_strategy,
    make_rsi_strategy, make_bollinger_strategy, make_macd_strategy,
    STRATEGY_FACTORIES, optimize_any_strategy, create_auto_param_grid,
)
from strategies import STRATEGY_LIBRARY, list_strategies, get_strategy
from visualization import (
    plot_full_dashboard, plot_equity_curve, plot_drawdown,
    plot_optimization_heatmap, plot_monte_carlo,
    plot_wfo_results, plot_sensitivity, plot_trades,
    plot_monthly_returns, plot_rolling_metrics,
    plot_returns_distribution, plot_candlestick_signals,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Optimización | Backtest", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")
init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Optimización")
    st.divider()
    if st.button("🏠 Inicio", use_container_width=True):
        st.switch_page("Home.py")
    st.page_link("pages/1_Datos.py", label="🗂️ Datos & Tickers")
    st.page_link("pages/2_Scanner.py", label="🔍 Scanner")
    st.page_link("pages/3_Optimizacion.py", label="⚡ Optimización")
    st.page_link("pages/4_Seguimiento.py", label="🎯 Seguimiento")
    st.page_link("pages/5_Constructor.py", label="🏗️ Constructor")
    st.divider()
    cfg = backtest_config_sidebar()

page_header("⚡ Optimización de Estrategias",
            "Encuentra los parámetros óptimos con Grid Search, Walk-Forward y Monte Carlo.")

df = get_long_df()
if df is None:
    st.warning("⚠️ No hay datos cargados. Ve a **🗂️ Datos & Tickers** para descargar primero.")
    st.page_link("1_Datos.py", label="Ir a Datos →")
    st.stop()

available_tickers = get_available_tickers(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Configuración de la optimización
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Configurar Optimización")

# Selector de modo: optimizar estrategia de biblioteca o tipo específico
optimization_mode = st.radio(
    "Modo de Optimización",
    ["🎯 Cualquier Estrategia de la Biblioteca", "🔧 Tipo Específico con Parámetros Personalizados"],
    key="opt_mode",
    help="Biblioteca: optimiza cualquier estrategia existente. Tipo Específico: define grids de parámetros personalizados.",
)

use_library_mode = optimization_mode.startswith("🎯")

if use_library_mode:
    # ── MODO: Optimizar cualquier estrategia ────────────────────────────────
    oc1, oc2, oc3, oc4 = st.columns([2, 2, 2, 1])
    with oc1:
        opt_ticker = st.selectbox("Ticker", options=available_tickers, key="opt_ticker_sel")
    with oc2:
        strategy_names = list(STRATEGY_LIBRARY.keys())
        selected_strategy_name = st.selectbox(
            "Estrategia",
            strategy_names,
            key="opt_strategy_lib",
        )
    with oc3:
        metric_label = st.selectbox("Métrica a optimizar",
                                      list(METRIC_OPTIONS.keys()),
                                      key="opt_metric_sel")
        opt_metric = METRIC_OPTIONS[metric_label]
    with oc4:
        granularity = st.selectbox(
            "Granularidad",
            ["coarse", "medium", "fine"],
            index=1,
            key="opt_granularity",
            help="coarse: ~20 combos | medium: ~100 combos | fine: ~400 combos"
        )
    
    # Mostrar info de la estrategia seleccionada
    selected_strategy = get_strategy(selected_strategy_name)
    st.info(
        f"**{selected_strategy.name}**  \n"
        f"Categoría: {selected_strategy.category} | "
        f"SL: {selected_strategy.stop_loss or 'N/A'} | "
        f"TP: {selected_strategy.take_profit or 'N/A'} | "
        f"Max Days: {selected_strategy.max_holding_days or 'N/A'}"
    )
    
    # Se optimizarán automáticamente: stop_loss, take_profit, max_holding_days
    st.caption("⚙️ Se optimizarán automáticamente: stop_loss, take_profit, max_holding_days")
    
else:
    # ── MODO: Tipo específico con parámetros personalizados ──────────────────
    oc1, oc2, oc3 = st.columns([2, 2, 2])
    with oc1:
        opt_ticker = st.selectbox("Ticker", options=available_tickers, key="opt_ticker_sel")
    with oc2:
        strat_label = st.selectbox("Tipo de estrategia",
                                    list(STRATEGY_TYPE_OPTIONS.keys()),
                                    key="opt_strat_type")
        strat_type = STRATEGY_TYPE_OPTIONS[strat_label]
    with oc3:
        metric_label = st.selectbox("Métrica a optimizar",
                                      list(METRIC_OPTIONS.keys()),
                                      key="opt_metric_sel")
        opt_metric = METRIC_OPTIONS[metric_label]
    
    # ── Grid de parámetros personalizable ────────────────────────────────────────
    st.markdown("#### Espacio de parámetros")
    st.caption("Ingresa los valores como lista separada por comas. El motor evaluará todas las combinaciones.")
    
    # Parámetros específicos por tipo de estrategia
    if strat_type == "sma_crossover":
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            fast_vals = st.text_input("fast (períodos MA rápida)", "10, 20, 30, 50", key="p_fast")
        with col_p2:
            slow_vals = st.text_input("slow (períodos MA lenta)", "50, 100, 150, 200", key="p_slow")
        with col_p3:
            sl_vals = st.text_input("stop_loss", "0.04, 0.06, 0.08", key="p_sl")
        with col_p4:
            tp_vals = st.text_input("take_profit", "0.10, 0.15, 0.20, 0.25", key="p_tp")
    
        def parse_list(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]
    
        def parse_int_list(s):
            return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    
        param_space = {
            "fast": parse_int_list(fast_vals),
            "slow": parse_int_list(slow_vals),
            "stop_loss": parse_list(sl_vals),
            "take_profit": parse_list(tp_vals),
        }
        factory = make_sma_crossover_strategy
    
    elif strat_type == "ema_crossover":
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            fast_vals = st.text_input("fast", "5, 9, 12, 21", key="p_fast_ema")
        with col_p2:
            slow_vals = st.text_input("slow", "21, 34, 50, 89", key="p_slow_ema")
        with col_p3:
            sl_vals = st.text_input("stop_loss", "0.03, 0.05, 0.07", key="p_sl_ema")
        with col_p4:
            tp_vals = st.text_input("take_profit", "0.10, 0.15, 0.20", key="p_tp_ema")
    
        def parse_list(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        def parse_int_list(s):
            return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    
        param_space = {
            "fast": parse_int_list(fast_vals),
            "slow": parse_int_list(slow_vals),
            "stop_loss": parse_list(sl_vals),
            "take_profit": parse_list(tp_vals),
        }
        factory = make_ema_crossover_strategy
    
    elif strat_type == "rsi":
        col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
        with col_p1:
            period_vals = st.text_input("period", "9, 14, 21", key="p_period_rsi")
        with col_p2:
            os_vals = st.text_input("oversold", "20, 25, 30", key="p_os_rsi")
        with col_p3:
            ob_vals = st.text_input("overbought", "65, 70, 75, 80", key="p_ob_rsi")
        with col_p4:
            sl_vals = st.text_input("stop_loss", "0.03, 0.05", key="p_sl_rsi")
        with col_p5:
            tp_vals = st.text_input("take_profit", "0.08, 0.12, 0.15", key="p_tp_rsi")
        with col_p6:
            hold_vals = st.text_input("max_holding_days", "15, 20, 30", key="p_hold_rsi")
    
        def parse_list(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        def parse_int_list(s):
            return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    
        param_space = {
            "period": parse_int_list(period_vals),
            "oversold": parse_list(os_vals),
            "overbought": parse_list(ob_vals),
            "stop_loss": parse_list(sl_vals),
            "take_profit": parse_list(tp_vals),
            "max_holding_days": parse_int_list(hold_vals),
        }
        factory = make_rsi_strategy
    
    elif strat_type == "bollinger":
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1:
            period_vals = st.text_input("period", "15, 20, 25", key="p_period_bb")
        with col_p2:
            std_vals = st.text_input("std_dev", "1.5, 2.0, 2.5", key="p_std_bb")
        with col_p3:
            sl_vals = st.text_input("stop_loss", "0.03, 0.05, 0.07", key="p_sl_bb")
        with col_p4:
            tp_vals = st.text_input("take_profit", "0.08, 0.12, 0.15", key="p_tp_bb")
        with col_p5:
            hold_vals = st.text_input("max_holding_days", "10, 15, 20, 30", key="p_hold_bb")
    
        def parse_list(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        def parse_int_list(s):
            return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    
        param_space = {
            "period": parse_int_list(period_vals),
            "std_dev": parse_list(std_vals),
            "stop_loss": parse_list(sl_vals),
            "take_profit": parse_list(tp_vals),
            "max_holding_days": parse_int_list(hold_vals),
        }
        factory = make_bollinger_strategy
    
    else:  # macd
        col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
        with col_p1:
            fast_vals = st.text_input("fast", "8, 12, 16", key="p_fast_macd")
        with col_p2:
            slow_vals = st.text_input("slow", "21, 26, 34", key="p_slow_macd")
        with col_p3:
            sig_vals  = st.text_input("signal", "7, 9, 12", key="p_sig_macd")
        with col_p4:
            sl_vals   = st.text_input("stop_loss", "0.04, 0.06, 0.08", key="p_sl_macd")
        with col_p5:
            tp_vals   = st.text_input("take_profit", "0.12, 0.18, 0.24", key="p_tp_macd")
    
        def parse_list(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        def parse_int_list(s):
            return [int(float(x.strip())) for x in s.split(",") if x.strip()]
    
        param_space = {
            "fast": parse_int_list(fast_vals),
            "slow": parse_int_list(slow_vals),
            "signal": parse_int_list(sig_vals),
            "stop_loss": parse_list(sl_vals),
            "take_profit": parse_list(tp_vals),
        }
        factory = make_macd_strategy

# Calcular tamaño del grid
if not use_library_mode:
    try:
        grid = ParameterGrid(param_space)
        grid_size = len(grid)
    except Exception:
        grid_size = 0
    st.info(f"**Grid size: {grid_size} combinaciones** a evaluar.")
else:
    # En modo biblioteca, el grid se crea automáticamente
    temp_grid = create_auto_param_grid(selected_strategy, granularity=granularity)
    grid_size = len(temp_grid)
    st.info(f"**Grid automático: {grid_size} combinaciones** a evaluar (stop_loss, take_profit, max_holding_days).")

# ── Opciones avanzadas ────────────────────────────────────────────────────────
with st.expander("⚙️ Opciones avanzadas"):
    adv1, adv2, adv3, adv4 = st.columns(4)
    with adv1:
        run_wfo = st.checkbox("Walk-Forward Optimization", value=True)
        n_splits = st.number_input("N splits WFO", 3, 10, 5, key="wfo_splits")
    with adv2:
        run_mc = st.checkbox("Monte Carlo", value=True)
        n_mc = st.number_input("Simulaciones MC", 100, 5000, 1000, step=100, key="mc_sims")
    with adv3:
        run_sensitivity = st.checkbox("Análisis de sensibilidad", value=True)
        min_trades = st.number_input("Mín. trades por combo", 1, 20, 5, key="opt_min_trades")
    with adv4:
        st.markdown("**Filtros post-grid:**")
        min_sharpe = st.number_input("Sharpe mínimo", -5.0, 5.0, -5.0, step=0.1,
                                      key="filter_sharpe",
                                      help="Filtrar resultados por Sharpe mínimo. Usa -5.0 para no filtrar.")

# ── Botón ejecutar ────────────────────────────────────────────────────────────
st.divider()
run_btn = st.button(
    f"⚡ Ejecutar Optimización ({grid_size} combinaciones)",
    type="primary",
    use_container_width=True,
    disabled=(grid_size == 0),
)

if run_btn:
    if grid_size == 0:
        st.error("El grid de parámetros está vacío o es inválido.")
        st.stop()

    # ── Progreso ──────────────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Iniciando optimización...")
    status_box = st.empty()

    with st.spinner(""):
        try:
            if use_library_mode:
                # ── MODO BIBLIOTECA: Usar optimize_any_strategy ──────────────
                status_box.info(f"🔎 Optimizando {selected_strategy_name}...")
                progress_bar.progress(10, text="Optimización iniciada...")
                
                # Ejecutar optimización automática
                report = optimize_any_strategy(
                    strategy=selected_strategy,
                    df=df,
                    ticker=opt_ticker,
                    param_grid=None,  # Se crea automáticamente
                    config=cfg,
                    optimize_metric=opt_metric,
                    run_wfo=run_wfo,
                    run_mc=run_mc,
                    granularity=granularity,
                    verbose=False,
                )
                
                progress_bar.progress(80, text="Optimización completada.")
                
                # Extraer resultados del report
                grid_df = report.grid_results
                wfo_df = report.wfo_results if run_wfo else pd.DataFrame()
                # mc_stats está en report.mc_stats (Dict, se usa directamente)
                # best_result está en report.best_result (se usa directamente)
                
                progress_bar.progress(100, text="✅ Optimización completa!")
                status_box.success(f"✅ Optimización completada para {selected_strategy_name}")
                
                # Guardar resultados en session_state (modo Biblioteca)
                st.session_state["opt_report"] = report
                st.session_state["opt_ticker"] = opt_ticker
                st.session_state["opt_strategy_type"] = "library"
                st.session_state["opt_grid_df"] = grid_df
            
            else:
                # ── MODO TIPO ESPECÍFICO: Usar lógica original ───────────────
                grid_obj = ParameterGrid(param_space)
                optimizer = StrategyOptimizer(
                    strategy_factory=factory,
                    param_grid=grid_obj,
                    config=cfg,
                    optimize_metric=opt_metric,
                )

                # Grid Search
                status_box.info("🔎 Ejecutando Grid Search...")
                progress_bar.progress(10, text="Grid Search...")
                grid_df = optimizer.grid_search(df, ticker=opt_ticker,
                                                 min_trades=int(min_trades), verbose=False)
                progress_bar.progress(40, text="Grid Search completado.")

                # WFO
                wfo_df = pd.DataFrame()
                if run_wfo and optimizer.best_params:
                    status_box.info("🔄 Walk-Forward Optimization...")
                    progress_bar.progress(50, text="Walk-Forward...")
                    wfo_df = optimizer.walk_forward(
                        df, ticker=opt_ticker,
                        n_splits=int(n_splits), verbose=False
                    )
                    progress_bar.progress(70, text="WFO completado.")

                # Monte Carlo
                mc_df = pd.DataFrame()
                mc_stats = {}
                if run_mc and optimizer.best_result:
                    status_box.info("🎲 Monte Carlo...")
                    progress_bar.progress(75, text="Monte Carlo...")
                    mc_stats = optimizer.monte_carlo(
                        optimizer.best_result,
                        n_simulations=int(n_mc),
                        verbose=False,
                    )
                    progress_bar.progress(90, text="Monte Carlo completado.")

                # Sensibilidad
                sensitivity_df = pd.DataFrame()
                if run_sensitivity and optimizer.best_params:
                    status_box.info("📊 Análisis de sensibilidad...")
                    sensitivity_df = optimizer.sensitivity_analysis(
                        df, ticker=opt_ticker, verbose=False
                    )
                    progress_bar.progress(95, text="Sensibilidad completada.")
                
                # Variables para compatibilidad
                # best_strategy no existe como atributo, reconstruir desde best_params
                if optimizer.best_params:
                    best_strategy = factory(**optimizer.best_params)
                else:
                    best_strategy = None
                best_result = optimizer.best_result
                progress_bar.progress(100, text="✅ Optimización completa!")
                status_box.success(f"✅ Optimización completada: {best_strategy.name if best_strategy else 'N/A'}")
                
                # Crear report y guardar en session_state (modo Tipo Específico)
                from optimizer import OptimizationReport
                report = OptimizationReport(
                    ticker=opt_ticker,
                    optimize_metric=opt_metric,
                    best_params=optimizer.best_params,
                    best_result=optimizer.best_result,
                    grid_results=grid_df,
                    wfo_results=wfo_df,
                    mc_stats=mc_stats,
                    sensitivity_results=sensitivity_df,
                )
                st.session_state["opt_report"] = report
                st.session_state["opt_ticker"] = opt_ticker
                st.session_state["opt_strategy_type"] = strat_type
                st.session_state["opt_grid_df"] = grid_df

        except Exception as e:
            st.error(f"Error durante la optimización: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Recuperar resultados del estado ──────────────────────────────────────────
report = st.session_state.get("opt_report")
if report is None:
    st.info("Configura y ejecuta la optimización para ver los resultados.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Resultados del Grid Search
# ══════════════════════════════════════════════════════════════════════════════
opt_ticker_saved = st.session_state.get("opt_ticker", "")
opt_strat_type_saved = st.session_state.get("opt_strategy_type", "strategy")
st.markdown(f"### 2 · Resultados del Grid Search — {opt_ticker_saved}")

if report.best_params:
    bp_col, bp_val = st.columns([1, 2])
    with bp_col:
        st.markdown("**Mejores parámetros**")
    with bp_val:
        param_chips = " · ".join(
            f"**{k}** = `{v}`" for k, v in report.best_params.items()
        )
        st.markdown(param_chips)

# KPIs del mejor resultado
if report.best_result and "error" not in report.best_result.metrics:
    best_m = report.best_result.metrics
    gk1, gk2, gk3, gk4, gk5 = st.columns(5)
    gk1.metric("Mejor " + metric_label, f"{best_m.get(opt_metric, 0):.4f}")
    gk2.metric("Total Return", f"{best_m.get('total_return_pct', 0):.2f}%")
    gk3.metric("Sharpe", f"{best_m.get('sharpe_ratio', 0):.4f}")
    gk4.metric("Max DD", f"{best_m.get('max_drawdown_pct', 0):.2f}%")
    gk5.metric("# Trades", int(best_m.get("n_trades", 0)))

st.divider()

# ── Tabla completa del grid ───────────────────────────────────────────────────
if not report.grid_results.empty:
    st.markdown("#### Todas las combinaciones evaluadas")

    grid_disp = report.grid_results.copy()
    # Filtro de Sharpe
    if "sharpe_ratio" in grid_disp.columns and min_sharpe > -4.99:
        grid_disp = grid_disp[grid_disp["sharpe_ratio"] >= min_sharpe]

    top_grid = st.number_input("Mostrar top N combinaciones",
                                min_value=5, max_value=len(grid_disp),
                                value=min(20, len(grid_disp)), key="grid_top_n")

    metric_cols = ["total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
                   "calmar_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor",
                   "n_trades", "avg_holding_days"]
    param_cols_g = [c for c in grid_disp.columns
                    if c not in metric_cols and c != "params_str"]
    show_cols = param_cols_g + [c for c in metric_cols if c in grid_disp.columns]
    show_cols = [c for c in show_cols if c in grid_disp.columns]

    fmt_g = {c: "{:.4f}" for c in grid_disp[show_cols].select_dtypes("float").columns}
    fmt_g.update({c: "{:.2f}%" for c in ["total_return_pct", "cagr_pct", "max_drawdown_pct",
                                           "win_rate_pct"] if c in show_cols})

    def color_row(s):
        if opt_metric in s.index and isinstance(s[opt_metric], float):
            bg = "background-color: rgba(0,204,102,0.12)" if s[opt_metric] > 0 else ""
            return [bg] * len(s)
        return [""] * len(s)

    try:
        styled_g = (grid_disp[show_cols].head(int(top_grid)).style
                    .format(fmt_g, na_rep="—")
                    .apply(color_row, axis=1))
        st.dataframe(styled_g, use_container_width=True, height=420)
    except Exception:
        st.dataframe(grid_disp[show_cols].head(int(top_grid)),
                     use_container_width=True, height=420)

    csv_grid = report.grid_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar grid completo CSV",
        data=csv_grid,
        file_name=f"{opt_ticker_saved}_{opt_strat_type_saved}_grid.csv",
        mime="text/csv",
    )

st.divider()

# ── Heatmaps de optimización ──────────────────────────────────────────────────
if not report.grid_results.empty:
    st.markdown("#### Heatmaps de parámetros")
    param_cols_heat = [c for c in report.grid_results.columns
                       if c not in ["params_str", "total_return_pct", "cagr_pct",
                                    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                                    "max_drawdown_pct", "win_rate_pct", "profit_factor",
                                    "n_trades", "avg_holding_days", "annual_volatility_pct",
                                    "expectancy_pct", "omega_ratio", "recovery_factor"]]

    if len(param_cols_heat) >= 2:
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            px_col = st.selectbox("Eje X (param)", param_cols_heat, key="heat_x")
        with hc2:
            py_opts = [p for p in param_cols_heat if p != px_col]
            py_col = st.selectbox("Eje Y (param)", py_opts, key="heat_y") if py_opts else None
        with hc3:
            heat_metric_label = st.selectbox("Métrica", list(METRIC_OPTIONS.keys()),
                                              key="heat_metric")
            heat_metric = METRIC_OPTIONS[heat_metric_label]

        if py_col:
            fig_heat = plot_optimization_heatmap(
                report.grid_results,
                param_x=px_col,
                param_y=py_col,
                metric=heat_metric,
                show=False,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Análisis del mejor modelo
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · Análisis Completo del Mejor Modelo")

if report.best_result is None or report.best_result.equity_curve.empty:
    st.warning("Sin resultados válidos para el mejor modelo.")
    st.stop()

result = report.best_result

# Métricas completas
st.markdown("#### Métricas de rendimiento")
show_metrics_grid(result.metrics, n_cols=5)
st.divider()

# ── Tabs del mejor modelo ──────────────────────────────────────────────────────
t_dash, t_equity, t_trades, t_candles, t_monthly, t_rolling, t_dist = st.tabs([
    "📊 Dashboard", "📈 Equity & DD", "💰 Trades",
    "🕯️ Señales", "📅 Mensuales", "📉 Rolling", "📊 Distribución",
])

with t_dash:
    st.plotly_chart(plot_full_dashboard(result, show=False), use_container_width=True)

with t_equity:
    st.plotly_chart(plot_equity_curve(result, show_trades=True, show=False),
                    use_container_width=True)
    st.plotly_chart(plot_drawdown(result, show=False), use_container_width=True)

with t_trades:
    kind_opt = st.radio("Tipo", ["waterfall", "scatter", "bar"],
                         horizontal=True, key="opt_trade_kind")
    st.plotly_chart(plot_trades(result, kind=kind_opt, show=False),
                    use_container_width=True)
    if not result.trades_df.empty:
        st.markdown("#### Log de trades del mejor modelo")
        fmt_t = {
            "entry_price": "{:.2f}", "exit_price": "{:.2f}",
            "pnl_pct": "{:.4f}", "net_pnl": "{:.2f}",
            "commission_paid": "{:.2f}", "mae_pct": "{:.4f}", "mfe_pct": "{:.4f}",
        }
        def color_pnl(val):
            if isinstance(val, (int, float)) and not np.isnan(val):
                return "color: #00cc66" if val > 0 else "color: #ff3333"
            return ""
        try:
            styled_t = result.trades_df.style.format(fmt_t, na_rep="—").applymap(
                color_pnl, subset=["net_pnl", "pnl_pct"])
            st.dataframe(styled_t, use_container_width=True, height=400)
        except Exception:
            st.dataframe(result.trades_df, use_container_width=True, height=400)

        # Exit reasons
        exit_df = result.get_trades_by_reason()
        if not exit_df.empty:
            st.markdown("#### Resumen por razón de salida")
            st.dataframe(exit_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

        csv_t = result.trades_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar trades CSV", data=csv_t,
                           file_name=f"{opt_ticker_saved}_{opt_strat_type_saved}_best_trades.csv",
                           mime="text/csv")

with t_candles:
    n_bars_opt = st.slider("Barras a mostrar", 60, 500, 252, step=20, key="opt_candles_n")
    inds_opt = st.multiselect(
        "Indicadores",
        ["sma_20", "sma_50", "sma_100", "ema_20", "ema_50", "bb_upper_20", "bb_lower_20"],
        default=["sma_20", "sma_50"],
        key="opt_candle_indicators",
    )
    st.plotly_chart(
        plot_candlestick_signals(result, n_last=n_bars_opt,
                                  show_indicators=inds_opt, show=False),
        use_container_width=True,
    )

with t_monthly:
    st.plotly_chart(plot_monthly_returns(result, show=False), use_container_width=True)
    monthly_tbl = result.get_monthly_returns()
    if not monthly_tbl.empty:
        fmt_m = {c: "{:.2%}" for c in monthly_tbl.columns}
        st.dataframe(monthly_tbl.style.format(fmt_m, na_rep="—")
                     .background_gradient(cmap="RdYlGn", axis=None),
                     use_container_width=True)

with t_rolling:
    st.plotly_chart(plot_rolling_metrics(result, show=False), use_container_width=True)

with t_dist:
    st.plotly_chart(plot_returns_distribution(result, show=False), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Walk-Forward Optimization
# ══════════════════════════════════════════════════════════════════════════════
if not report.wfo_results.empty:
    st.markdown("### 4 · Walk-Forward Optimization")

    wk1, wk2, wk3 = st.columns(3)
    avg_oos = report.wfo_results["oos_sharpe"].mean() if "oos_sharpe" in report.wfo_results.columns else 0
    avg_is  = report.wfo_results["is_sharpe"].mean()  if "is_sharpe"  in report.wfo_results.columns else 0
    avg_eff = report.wfo_results["efficiency"].mean() if "efficiency" in report.wfo_results.columns else 0

    wk1.metric("Sharpe IS (promedio)", f"{avg_is:.4f}")
    wk2.metric("Sharpe OOS (promedio)", f"{avg_oos:.4f}")
    wk3.metric(
        "Eficiencia IS→OOS",
        f"{avg_eff:.4f}",
        delta="ROBUSTO ✓" if avg_eff > 0.5 else "SOBREAJUSTE ⚠",
        delta_color="normal" if avg_eff > 0.5 else "inverse",
    )

    st.plotly_chart(plot_wfo_results(report.wfo_results, show=False),
                    use_container_width=True)

    st.markdown("#### Detalle por fold")
    wfo_cols = ["fold", "train_start", "train_end", "test_start", "test_end",
                "is_sharpe", "oos_sharpe", "oos_return_%", "oos_max_dd_%",
                "efficiency", "best_params"]
    wfo_cols = [c for c in wfo_cols if c in report.wfo_results.columns]
    st.dataframe(report.wfo_results[wfo_cols].style.format(
        {c: "{:.4f}" for c in ["is_sharpe", "oos_sharpe", "efficiency"]
         if c in wfo_cols}, na_rep="—"),
        use_container_width=True)

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════
if report.mc_stats and "prob_loss" in report.mc_stats:
    st.markdown("### 5 · Análisis Monte Carlo")

    mc = report.mc_stats
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Prob. de pérdida", f"{mc.get('prob_loss', 0):.2f}%")

    if "total_return" in mc:
        mc2.metric("Retorno mediana (MC)", f"{mc['total_return'].get('p50', 0):.2f}%")
    if "sharpe" in mc:
        mc3.metric("Sharpe mediana (MC)", f"{mc['sharpe'].get('p50', 0):.4f}")
    if "max_drawdown" in mc:
        mc4.metric("Max DD mediana (MC)", f"{mc['max_drawdown'].get('p50', 0):.2f}%")

    st.plotly_chart(
        plot_monte_carlo(mc, equity_curve=result.equity_curve, show=False),
        use_container_width=True,
    )

    # Tabla de percentiles
    st.markdown("#### Distribución de métricas (percentiles)")
    mc_rows = []
    for metric_key in ["total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor"]:
        if metric_key in mc and isinstance(mc[metric_key], dict):
            s = mc[metric_key]
            mc_rows.append({
                "Métrica": metric_key,
                "p5": s.get("p5"), "p25": s.get("p25"),
                "Mediana (p50)": s.get("p50"), "Media": s.get("mean"),
                "p75": s.get("p75"), "p95": s.get("p95"),
                "Desv. Std.": s.get("std"),
            })
    if mc_rows:
        mc_df = pd.DataFrame(mc_rows)
        st.dataframe(mc_df.style.format("{:.4f}", na_rep="—", subset=mc_df.columns[1:]),
                     use_container_width=True)
    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — Análisis de sensibilidad
# ══════════════════════════════════════════════════════════════════════════════
if not report.sensitivity_results.empty:
    st.markdown("### 6 · Análisis de Sensibilidad de Parámetros")
    st.caption("Efecto de variar cada parámetro individualmente manteniendo los demás en su valor óptimo.")

    sens_metric_label = st.selectbox(
        "Métrica",
        list(METRIC_OPTIONS.keys()),
        key="sens_metric_sel",
    )
    sens_metric = METRIC_OPTIONS[sens_metric_label]

    st.plotly_chart(
        plot_sensitivity(report.sensitivity_results, metric=sens_metric, show=False),
        use_container_width=True,
    )

    with st.expander("Ver tabla de sensibilidad"):
        st.dataframe(
            report.sensitivity_results.style.format(
                {c: "{:.4f}" for c in report.sensitivity_results.select_dtypes("float").columns},
                na_rep="—"),
            use_container_width=True,
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — Series temporales + exportar
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 7 · Series de Tiempo del Mejor Modelo")

with st.expander("Ver series de tiempo completas"):
    ts_df = report.get_time_series()
    if not ts_df.empty:
        fmt_ts = {c: "{:.6f}" for c in ts_df.select_dtypes("float").columns}
        fmt_ts.update({c: "${:,.2f}" for c in ["equity", "buy_and_hold"]
                       if c in ts_df.columns})
        st.dataframe(ts_df.style.format(fmt_ts, na_rep="—"),
                     use_container_width=True, height=400)
        csv_ts = ts_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Descargar series de tiempo CSV",
            data=csv_ts,
            file_name=f"{opt_ticker_saved}_{opt_strat_type_saved}_timeseries.csv",
            mime="text/csv",
        )

st.markdown("#### Exportar reporte completo")
if st.button("📥 Exportar reporte a Excel", use_container_width=True):
    with st.spinner("Generando Excel..."):
        try:
            os.makedirs("results", exist_ok=True)
            report.export("results", ticker=opt_ticker_saved)
            fname = f"results/{opt_ticker_saved}_optimization.xlsx"
            with open(fname, "rb") as f:
                st.download_button(
                    "⬇️ Descargar Excel",
                    data=f,
                    file_name=os.path.basename(fname),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Error exportando: {e}")
