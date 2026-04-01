"""
app/pages/1_Datos.py
====================
Página 1 — Gestión de datos & Tickers.
Permite al usuario agregar/remover tickers, descargar datos y
explorar el dataset resultante.
No contiene lógica analítica — solo llama al modelo existente.
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
import plotly.graph_objects as go
import plotly.express as px

from app.utils import (
    init_state, backtest_config_sidebar,
    get_long_df, get_ticker_list, set_ticker_list,
    TICKER_PRESETS, PERIOD_OPTIONS,
    INDICES, EQUITIES_US, ETFS,
    cached_download, page_header,
)
from data_loader import get_data_summary, get_available_tickers

# ── Setup ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Datos & Tickers | Backtest", page_icon="🗂️",
                   layout="wide", initial_sidebar_state="expanded")
init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Datos & Tickers")
    st.divider()
    if st.button("🏠 Inicio", use_container_width=True):
        st.switch_page("Home.py")
    st.page_link("pages/1_Datos.py", label="🗂️ Datos & Tickers")
    st.page_link("pages/2_Scanner.py", label="🔍 Scanner")
    st.page_link("pages/3_Optimizacion.py", label="⚡ Optimización")

page_header("🗂️ Datos & Tickers",
            "Configura el universo de activos, descarga datos históricos y explora el dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Gestión de la lista de tickers
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Configurar lista de tickers")

tab_manual, tab_preset, tab_catalogo = st.tabs(
    ["✏️ Editor manual", "📦 Presets", "📖 Catálogo Yahoo"]
)

# ── Tab: Editor manual ────────────────────────────────────────────────────────
with tab_manual:
    current_tickers = get_ticker_list()

    col_edit, col_preview = st.columns([1, 1], gap="large")

    with col_edit:
        st.markdown("**Agregar ticker(s)**")
        new_raw = st.text_input(
            "Símbolo(s) — separar por coma o espacio",
            placeholder="AAPL, MSFT, ^GSPC, SPY ...",
            key="add_ticker_input",
        )
        cadd, cremove = st.columns(2)
        with cadd:
            if st.button("➕ Agregar", use_container_width=True, type="primary"):
                tokens = [t.strip().upper() for t in new_raw.replace(",", " ").split() if t.strip()]
                if tokens:
                    updated = list(dict.fromkeys(current_tickers + tokens))  # mantiene orden, no duplica
                    set_ticker_list(updated)
                    st.success(f"Agregados: {', '.join(tokens)}")
                    st.rerun()

        st.divider()
        st.markdown("**Eliminar tickers seleccionados**")
        to_remove = st.multiselect(
            "Selecciona para eliminar",
            options=get_ticker_list(),
            default=[],
            key="remove_multiselect",
        )
        if st.button("🗑️ Eliminar seleccionados", use_container_width=True,
                     disabled=not to_remove):
            updated = [t for t in get_ticker_list() if t not in to_remove]
            set_ticker_list(updated)
            st.rerun()

        st.divider()
        if st.button("🔄 Restaurar lista por defecto", use_container_width=True):
            from app.utils import DEFAULT_TICKERS
            set_ticker_list(DEFAULT_TICKERS.copy())
            st.rerun()

    with col_preview:
        st.markdown("**Lista actual de tickers**")
        tlist = get_ticker_list()

        # Mostrar chips por tipo
        indices_in   = [t for t in tlist if t.startswith("^") or t.endswith("=F")]
        equities_in  = [t for t in tlist if t not in indices_in and t not in ETFS]
        etfs_in      = [t for t in tlist if t in ETFS]

        def chip_row(items: list, color: str) -> str:
            return " ".join(
                f'<span style="background:{color}; padding:3px 10px; border-radius:12px; '
                f'font-size:0.8rem; margin:2px; display:inline-block">{t}</span>'
                for t in items
            )

        if indices_in:
            st.markdown("**Índices / Futuros**")
            st.markdown(chip_row(indices_in, "#1a3a5c"), unsafe_allow_html=True)
        if equities_in:
            st.markdown("**Acciones**")
            st.markdown(chip_row(equities_in, "#1a3a2c"), unsafe_allow_html=True)
        if etfs_in:
            st.markdown("**ETFs**")
            st.markdown(chip_row(etfs_in, "#3a2a1a"), unsafe_allow_html=True)

        st.markdown(f"**Total: {len(tlist)} tickers**")

# ── Tab: Presets ──────────────────────────────────────────────────────────────
with tab_preset:
    st.markdown("Carga grupos predefinidos de tickers con un clic.")
    for preset_name, preset_tickers in TICKER_PRESETS.items():
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(f"**{preset_name}** — {len(preset_tickers)} tickers")
            st.caption(", ".join(preset_tickers[:10]) + ("..." if len(preset_tickers) > 10 else ""))
        with col_btn:
            action = st.selectbox("", ["Reemplazar", "Agregar"], key=f"preset_action_{preset_name}",
                                  label_visibility="collapsed")
            if st.button("Cargar", key=f"preset_load_{preset_name}", use_container_width=True):
                if action == "Reemplazar":
                    set_ticker_list(preset_tickers.copy())
                else:
                    combined = list(dict.fromkeys(get_ticker_list() + preset_tickers))
                    set_ticker_list(combined)
                st.success(f"Cargado: {preset_name}")
                st.rerun()
        st.divider()

# ── Tab: Catálogo ─────────────────────────────────────────────────────────────
with tab_catalogo:
    st.markdown("Navega el catálogo de símbolos disponibles y agrégalos a tu lista.")
    cat_tab1, cat_tab2, cat_tab3 = st.tabs(["Índices & Futuros", "Acciones US", "ETFs"])

    def catalog_adder(catalog: list, key_prefix: str):
        selected = st.multiselect("Selecciona tickers", catalog, key=f"cat_{key_prefix}")
        if st.button("➕ Agregar seleccionados", key=f"cat_add_{key_prefix}",
                     disabled=not selected, use_container_width=True):
            updated = list(dict.fromkeys(get_ticker_list() + selected))
            set_ticker_list(updated)
            st.success(f"Agregados: {', '.join(selected)}")
            st.rerun()

    with cat_tab1:
        catalog_adder(INDICES, "indices")
    with cat_tab2:
        catalog_adder(EQUITIES_US, "equities")
    with cat_tab3:
        catalog_adder(ETFS, "etfs")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Descarga de datos
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 2 · Descargar datos históricos")

dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 1])
with dl_col1:
    period_label = st.selectbox(
        "Período histórico",
        options=list(PERIOD_OPTIONS.keys()),
        index=list(PERIOD_OPTIONS.keys()).index("5 años"),
    )
    period_val = PERIOD_OPTIONS[period_label]

with dl_col2:
    interval = st.selectbox("Intervalo", ["1d", "1wk"], index=0)

with dl_col3:
    force_refresh = st.checkbox("Forzar re-descarga", value=False)

tickers_to_dl = get_ticker_list()
st.info(f"Se descargarán **{len(tickers_to_dl)} tickers**: {', '.join(tickers_to_dl[:8])}{'...' if len(tickers_to_dl) > 8 else ''}")

if st.button("⬇️ Descargar datos", type="primary", use_container_width=True):
    if not tickers_to_dl:
        st.error("La lista de tickers está vacía.")
    else:
        with st.spinner(f"Descargando {len(tickers_to_dl)} tickers desde Yahoo Finance..."):
            try:
                df = cached_download(tuple(tickers_to_dl), period_val)
                if force_refresh:
                    # Si fuerza refresh, limpiar caché y re-descargar
                    cached_download.clear()
                    df = cached_download(tuple(tickers_to_dl), period_val)
                st.session_state["long_df"] = df
                st.session_state["selected_period"] = period_val
                st.success(f"✅ Descargados {df['ticker'].nunique()} tickers · {len(df):,} filas")
            except Exception as e:
                st.error(f"Error en descarga: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Exploración del dataset
# ══════════════════════════════════════════════════════════════════════════════
df = get_long_df()

if df is None:
    st.info("Descarga los datos primero para explorar el dataset.")
    st.stop()

st.markdown("### 3 · Exploración del dataset")

# Resumen global
summary_df = get_data_summary(df)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("Tickers", df["ticker"].nunique())
with col_s2:
    st.metric("Observaciones totales", f"{len(df):,}")
with col_s3:
    st.metric("Fecha inicio", str(df["date"].min().date()))
with col_s4:
    st.metric("Fecha fin", str(df["date"].max().date()))

st.markdown("#### Tabla resumen por ticker")

# Formatear para display
disp = summary_df.copy()
disp["total_return"] = (disp["total_return"] * 100).round(2).astype(str) + "%"
disp["annual_vol"]   = (disp["annual_vol"]   * 100).round(2).astype(str) + "%"
st.dataframe(disp, use_container_width=True, height=300)

st.divider()

# ── Gráfico interactivo: precio normalizado ───────────────────────────────────
st.markdown("#### Comparación de precios normalizados")

available = get_available_tickers(df)
selected_plot = st.multiselect(
    "Selecciona tickers a comparar",
    options=available,
    default=available[:min(5, len(available))],
    key="price_compare_select",
)

if selected_plot:
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, ticker in enumerate(selected_plot):
        sub = df[df["ticker"] == ticker].sort_values("date")
        if sub.empty:
            continue
        normalized = sub["adj_close"] / sub["adj_close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=sub["date"],
            y=normalized,
            name=ticker,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{ticker}</b><br>Fecha: %{{x}}<br>Retorno índice: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
        font=dict(color="#EEEEEE"),
        xaxis=dict(showgrid=True, gridcolor="#2d2d2d"),
        yaxis=dict(showgrid=True, gridcolor="#2d2d2d", title="Precio base 100"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        hovermode="x unified",
        height=450,
        title="Rendimiento relativo (base 100)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Tabla detalle de un ticker ────────────────────────────────────────────────
st.markdown("#### Datos OHLCV detallados por ticker")

ticker_detail = st.selectbox(
    "Selecciona ticker",
    options=available,
    key="detail_ticker_select",
)

if ticker_detail:
    sub = df[df["ticker"] == ticker_detail].sort_values("date", ascending=False)
    n_rows = st.slider("Filas a mostrar", 10, 500, 50, step=10, key="detail_rows_slider")

    display_cols = ["date", "open", "high", "low", "close", "volume", "adj_close",
                    "returns", "log_returns"]
    display_cols = [c for c in display_cols if c in sub.columns]

    fmt = {
        "open": "{:.2f}", "high": "{:.2f}", "low": "{:.2f}",
        "close": "{:.2f}", "adj_close": "{:.2f}",
        "volume": "{:,.0f}",
        "returns": "{:.4f}", "log_returns": "{:.4f}",
    }
    st.dataframe(
        sub[display_cols].head(n_rows).style.format(fmt, na_rep="—"),
        use_container_width=True,
        height=400,
    )

    # Descarga CSV
    csv = sub[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV",
        data=csv,
        file_name=f"{ticker_detail}_data.csv",
        mime="text/csv",
    )

st.divider()

# ── Distribución de retornos por ticker ──────────────────────────────────────
st.markdown("#### Distribución de retornos diarios")

ret_ticker = st.selectbox("Ticker", options=available, key="ret_dist_ticker")
if ret_ticker:
    sub_ret = df[df["ticker"] == ret_ticker]["returns"].dropna() * 100
    if not sub_ret.empty:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=sub_ret,
            nbinsx=60,
            name="Retornos diarios",
            marker_color="#00d4ff",
            opacity=0.75,
            histnorm="probability density",
        ))
        # Curva normal superpuesta
        mu, sigma = float(sub_ret.mean()), float(sub_ret.std())
        import numpy as np
        x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        from scipy.stats import norm
        y_norm = norm.pdf(x_norm, mu, sigma)
        fig_ret.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            name="Normal",
            line=dict(color="#FFD700", width=2),
        ))
        fig_ret.add_vline(x=0, line_dash="dash", line_color="gray")

        skew_val = float(sub_ret.skew())
        kurt_val = float(sub_ret.kurtosis())
        fig_ret.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
            font=dict(color="#EEEEEE"),
            xaxis=dict(showgrid=True, gridcolor="#2d2d2d", title="Retorno diario (%)"),
            yaxis=dict(showgrid=True, gridcolor="#2d2d2d"),
            title=f"{ret_ticker} — Distribución de retornos | Skew={skew_val:.3f} | Kurt={kurt_val:.3f}",
            height=400,
            legend=dict(bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        # Stats rápidas
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Media diaria", f"{mu:.4f}%")
        sc2.metric("Desv. Std.", f"{sigma:.4f}%")
        sc3.metric("Skewness", f"{skew_val:.4f}")
        sc4.metric("Kurtosis", f"{kurt_val:.4f}")
        sc5.metric("# días", f"{len(sub_ret):,}")
