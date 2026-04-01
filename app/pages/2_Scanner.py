"""
app/pages/2_Scanner.py
======================
Página 2 — Scanner de estrategias.
Evalúa todas las estrategias predefinidas sobre un ticker seleccionado,
muestra el resumen comparativo y permite profundizar en una estrategia particular.
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
    get_long_df, get_ticker_list,
    METRIC_OPTIONS, page_header,
    show_metrics_grid,
)
from data_loader import get_available_tickers
from strategies import STRATEGY_LIBRARY, list_strategies
from backtest_engine import BacktestEngine, BacktestConfig
from strategy_scanner import StrategyScanner
from visualization import (
    plot_scanner_summary, plot_full_dashboard,
    plot_equity_curve, plot_drawdown, plot_trades,
    plot_monthly_returns, plot_rolling_metrics,
    plot_candlestick_signals, plot_returns_distribution,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scanner | Backtest", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")
init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Scanner")
    st.divider()
    if st.button("🏠 Inicio", use_container_width=True):
        st.switch_page("Home.py")
    st.page_link("pages/1_Datos.py", label="🗂️ Datos & Tickers")
    st.page_link("pages/2_Scanner.py", label="🔍 Scanner")
    st.page_link("pages/3_Optimizacion.py", label="⚡ Optimización")
    st.divider()
    cfg = backtest_config_sidebar()

page_header("🔍 Scanner de Estrategias",
            "Compara todas las estrategias predefinidas sobre un activo y profundiza en la que prefieras.")

df = get_long_df()
if df is None:
    st.warning("⚠️ No hay datos cargados. Ve a **🗂️ Datos & Tickers** para descargar primero.")
    st.page_link("1_Datos.py", label="Ir a Datos →")
    st.stop()

available_tickers = get_available_tickers(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Configuración del scan
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Configurar Scan")

sc1, sc2, sc3, sc4 = st.columns([2, 2, 1, 1])

with sc1:
    scan_ticker = st.selectbox(
        "Ticker a analizar",
        options=available_tickers,
        key="scan_ticker_select",
    )

with sc2:
    sort_metric_label = st.selectbox(
        "Ordenar resultados por",
        options=list(METRIC_OPTIONS.keys()),
        index=0,
        key="scan_sort_metric",
    )
    sort_metric = METRIC_OPTIONS[sort_metric_label]

with sc3:
    min_trades = st.number_input("Mín. trades", min_value=1, max_value=50, value=5)

with sc4:
    top_n = st.number_input("Top N", min_value=5, max_value=30, value=20)

# Filtro por categoría
all_cats = ["Todas"] + sorted(list_strategies()["category"].unique().tolist())
selected_cats = st.multiselect(
    "Filtrar por categoría de estrategia",
    options=all_cats,
    default=["Todas"],
    key="scan_category_filter",
)

# Filtrar estrategias
strategy_names_to_run = None
if "Todas" not in selected_cats and selected_cats:
    strategy_names_to_run = [
        name for name, s in STRATEGY_LIBRARY.items()
        if s.category in selected_cats
    ]

st.info(
    f"**{len(strategy_names_to_run) if strategy_names_to_run else len(STRATEGY_LIBRARY)}** estrategias "
    f"serán evaluadas sobre **{scan_ticker}**. "
    f"El proceso puede tardar ~30-60 segundos."
)

# ── Botón ejecutar ────────────────────────────────────────────────────────────
run_scan = st.button("▶️ Ejecutar Scan", type="primary", use_container_width=True)

if run_scan:
    with st.spinner(f"Ejecutando scan sobre {scan_ticker}..."):
        scanner = StrategyScanner(config=cfg, n_workers=1)
        scanner.scan(df, ticker=scan_ticker,
                     strategy_names=strategy_names_to_run, verbose=False)
        summary = scanner.get_summary(sort_by=sort_metric, min_trades=int(min_trades))

        st.session_state["scanner_results"] = scanner
        st.session_state["scanner_summary_df"] = summary
        st.session_state["scanner_ticker"] = scan_ticker
        st.session_state["selected_strategy_result"] = None

    st.success(f"✅ Scan completado — {len(summary)} estrategias válidas encontradas")

# ── Recuperar resultados del estado ──────────────────────────────────────────
scanner: StrategyScanner = st.session_state.get("scanner_results")
summary_df: pd.DataFrame = st.session_state.get("scanner_summary_df")
scan_ticker_saved = st.session_state.get("scanner_ticker")

if scanner is None or summary_df is None or summary_df.empty:
    st.info("Ejecuta el scan para ver los resultados.")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Resumen comparativo general
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"### 2 · Resumen Comparativo — {scan_ticker_saved}")

# KPIs del scanner
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Estrategias evaluadas", len(summary_df))
bh_ret = scanner._bh_return
k2.metric("Buy & Hold (período)", f"{bh_ret:.2f}%")
best_row = summary_df.iloc[0]
k3.metric("Mejor estrategia", best_row["strategy"][:22])
k4.metric(f"Mejor {sort_metric_label}", f"{best_row.get(sort_metric, 0):.3f}")
alpha_positive = (summary_df["alpha_vs_bh_%"] > 0).sum() if "alpha_vs_bh_%" in summary_df.columns else 0
k5.metric("Con alpha positivo vs B&H", alpha_positive)

st.divider()

# ── Tabla resumen ─────────────────────────────────────────────────────────────
st.markdown("#### Tabla de resultados")

pct_cols = ["total_return_%", "cagr_%", "max_dd_%", "annual_vol_%",
            "win_rate_%", "pct_pos_months_%", "alpha_vs_bh_%"]
display_cols = [
    "strategy", "category", "total_return_%", "cagr_%",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "max_dd_%", "win_rate_%", "profit_factor",
    "n_trades", "avg_holding_days", "composite_score", "alpha_vs_bh_%",
]
display_cols = [c for c in display_cols if c in summary_df.columns]

top_df = summary_df[display_cols].head(int(top_n)).copy()

# Colorear la tabla
def highlight_metric(val, col_name):
    if not isinstance(val, (int, float)):
        return ""
    good_positive = col_name not in ["max_dd_%", "annual_vol_%"]
    if good_positive:
        color = "rgba(0,204,102,0.15)" if val > 0 else "rgba(255,51,51,0.15)"
    else:
        color = "rgba(255,51,51,0.15)" if val < -5 else ""
    return f"background-color: {color}"

fmt = {c: "{:.3f}" for c in top_df.select_dtypes("float").columns}
fmt.update({c: "{:.2f}%" for c in pct_cols if c in top_df.columns})

try:
    styled = top_df.style.format(fmt, na_rep="N/A")
    for col in [c for c in pct_cols + ["sharpe_ratio", "sortino_ratio"] if c in top_df.columns]:
        styled = styled.applymap(lambda v, c=col: highlight_metric(v, c), subset=[col])
    st.dataframe(styled, use_container_width=True, height=420)
except Exception:
    st.dataframe(top_df, use_container_width=True, height=420)

# Descargar resultados
csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar resultados CSV",
    data=csv_bytes,
    file_name=f"{scan_ticker_saved}_scan_results.csv",
    mime="text/csv",
)

st.divider()

# ── Bubble chart comparativo ──────────────────────────────────────────────────
st.markdown("#### Visualización comparativa")

vc1, vc2, vc3 = st.columns(3)
with vc1:
    x_metric = st.selectbox("Eje X", list(METRIC_OPTIONS.keys()), index=0, key="viz_x")
with vc2:
    y_opts = [k for k in METRIC_OPTIONS.keys() if k != x_metric]
    y_metric = st.selectbox("Eje Y", y_opts, index=0, key="viz_y")
with vc3:
    size_col_opts = ["n_trades", "avg_holding_days", "win_rate_%"]
    size_col_opts = [c for c in size_col_opts if c in summary_df.columns]
    size_metric = st.selectbox("Tamaño burbuja", size_col_opts, key="viz_size")

fig_bubble = plot_scanner_summary(
    summary_df,
    top_n=int(top_n),
    x_metric=METRIC_OPTIONS[x_metric],
    y_metric=METRIC_OPTIONS[y_metric],
    size_metric=size_metric,
    show=False,
)
st.plotly_chart(fig_bubble, use_container_width=True)

st.divider()

# ── Mejor por categoría ───────────────────────────────────────────────────────
st.markdown("#### Mejor estrategia por categoría")
if "category" in summary_df.columns:
    best_by_cat = (
        summary_df.groupby("category")
        .apply(lambda x: x.nlargest(1, sort_metric))
        .reset_index(drop=True)
    )[["category", "strategy", "total_return_%", "sharpe_ratio",
       "sortino_ratio", "max_dd_%", "win_rate_%", "n_trades"]]
    st.dataframe(best_by_cat, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Análisis profundo de una estrategia individual
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · Análisis Profundo de Estrategia Individual")

all_strategy_names = summary_df["strategy"].tolist()
default_strategy = all_strategy_names[0] if all_strategy_names else None

selected_strategy_name = st.selectbox(
    "Selecciona una estrategia para análisis detallado",
    options=all_strategy_names,
    index=0,
    key="detail_strategy_select",
)

load_detail = st.button("🔬 Cargar análisis detallado", type="primary",
                         use_container_width=True)

if load_detail or st.session_state.get("selected_strategy_result") is not None:
    if load_detail:
        with st.spinner(f"Cargando análisis de {selected_strategy_name}..."):
            result = scanner.get_result(selected_strategy_name)
            st.session_state["selected_strategy_result"] = result

    result = st.session_state.get("selected_strategy_result")
    if result is None:
        st.error("No se encontró el resultado para esta estrategia.")
        st.stop()

    m = result.metrics
    if "error" in m:
        st.error(f"Error: {m['error']}")
        st.stop()

    # ── Info de la estrategia ─────────────────────────────────────────────────
    strat_obj = STRATEGY_LIBRARY.get(selected_strategy_name)
    if strat_obj:
        scol1, scol2 = st.columns([2, 1])
        with scol1:
            st.markdown(f"**{strat_obj.name}**")
            st.caption(strat_obj.description)
        with scol2:
            cat_colors = {
                "trend_following": "badge-trend", "mean_reversion": "badge-rev",
                "momentum": "badge-mom", "breakout": "badge-break",
                "volume": "badge-vol", "combo": "badge-break",
            }
            badge_class = cat_colors.get(strat_obj.category, "badge-vol")
            st.markdown(
                f'<span class="badge {badge_class}">{strat_obj.category}</span> '
                f'<span class="badge badge-vol">SL: {(strat_obj.stop_loss or 0)*100:.0f}%</span> '
                f'<span class="badge badge-mom">TP: {(strat_obj.take_profit or 0)*100:.0f}%</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── KPIs principales ──────────────────────────────────────────────────────
    st.markdown("#### Métricas de rendimiento")
    show_metrics_grid(m, n_cols=5)

    st.divider()

    # ── Tabs con gráficos ────────────────────────────────────────────────────
    tab_dash, tab_equity, tab_candles, tab_trades, tab_monthly, tab_rolling, tab_dist = st.tabs([
        "📊 Dashboard", "📈 Equity Curve", "🕯️ Señales",
        "💰 Trades", "📅 Retornos Mensuales", "📉 Métricas Rolling", "📊 Distribución",
    ])

    with tab_dash:
        st.plotly_chart(
            plot_full_dashboard(result, show=False),
            use_container_width=True,
        )

    with tab_equity:
        st.plotly_chart(
            plot_equity_curve(result, show_trades=True, show=False),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_drawdown(result, show=False),
            use_container_width=True,
        )

    with tab_candles:
        n_bars = st.slider("Barras a mostrar", 60, 500, 252, step=20,
                           key="candle_n_bars")
        inds_to_show = st.multiselect(
            "Indicadores a superponer",
            ["sma_20", "sma_50", "sma_100", "ema_20", "ema_50",
             "bb_upper_20", "bb_lower_20"],
            default=["sma_20", "sma_50"],
            key="candle_indicators",
        )
        st.plotly_chart(
            plot_candlestick_signals(result, n_last=n_bars,
                                     show_indicators=inds_to_show, show=False),
            use_container_width=True,
        )

    with tab_trades:
        trade_kind = st.radio("Tipo de gráfico",
                              ["waterfall", "scatter", "bar"],
                              horizontal=True, key="trades_kind")
        st.plotly_chart(
            plot_trades(result, kind=trade_kind, show=False),
            use_container_width=True,
        )

        if not result.trades_df.empty:
            st.markdown("#### Log de trades")
            trades_display = result.trades_df.copy()
            fmt_trade = {
                "entry_price": "{:.2f}", "exit_price": "{:.2f}",
                "pnl_pct": "{:.4f}", "net_pnl": "{:.2f}",
                "gross_pnl": "{:.2f}", "commission_paid": "{:.2f}",
                "mae_pct": "{:.4f}", "mfe_pct": "{:.4f}",
            }
            # Colorear P&L
            def color_pnl(val):
                if isinstance(val, (int, float)) and not np.isnan(val):
                    return "color: #00cc66" if val > 0 else "color: #ff3333"
                return ""
            try:
                styled_trades = (
                    trades_display.style
                    .format(fmt_trade, na_rep="—")
                    .applymap(color_pnl, subset=["net_pnl", "pnl_pct"])
                )
                st.dataframe(styled_trades, use_container_width=True, height=400)
            except Exception:
                st.dataframe(trades_display, use_container_width=True, height=400)

            # Exit reasons
            st.markdown("#### Resumen por razón de salida")
            exit_df = result.get_trades_by_reason()
            if not exit_df.empty:
                st.dataframe(exit_df.style.format("{:.4f}", na_rep="—"),
                             use_container_width=True)

            # Descarga
            csv_trades = trades_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar trades CSV",
                data=csv_trades,
                file_name=f"{scan_ticker_saved}_{selected_strategy_name}_trades.csv",
                mime="text/csv",
            )

    with tab_monthly:
        st.plotly_chart(
            plot_monthly_returns(result, show=False),
            use_container_width=True,
        )
        monthly_tbl = result.get_monthly_returns()
        if not monthly_tbl.empty:
            fmt_m = {c: "{:.2%}" for c in monthly_tbl.columns}
            st.dataframe(monthly_tbl.style.format(fmt_m, na_rep="—")
                         .background_gradient(cmap="RdYlGn", axis=None),
                         use_container_width=True)

    with tab_rolling:
        st.plotly_chart(
            plot_rolling_metrics(result, show=False),
            use_container_width=True,
        )

    with tab_dist:
        st.plotly_chart(
            plot_returns_distribution(result, show=False),
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### Series de tiempo detalladas")
    with st.expander("Ver todas las series temporales"):
        if not result.equity_curve.empty:
            ts_df = pd.DataFrame({
                "equity": result.equity_curve,
                "buy_and_hold": result.buy_and_hold.reindex(result.equity_curve.index),
                "drawdown_%": result.drawdown_series * 100,
            })
            if not result.rolling_metrics.empty:
                ts_df = ts_df.join(result.rolling_metrics, how="left")
            fmt_ts = {c: "{:.4f}" for c in ts_df.select_dtypes("float").columns}
            fmt_ts["equity"] = "${:,.2f}"
            fmt_ts["buy_and_hold"] = "${:,.2f}"
            st.dataframe(ts_df.tail(252).style.format(fmt_ts, na_rep="—"),
                         use_container_width=True, height=400)
