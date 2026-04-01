"""
app/Home.py
===========
Página de inicio de la app de Backtest.
Punto de entrada principal para `streamlit run app/Home.py`.
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
from app.utils import (
    init_state, get_long_df, get_ticker_list, cached_download,
    PERIOD_OPTIONS,
)

# ── Configuración global de la página ─────────────────────────────────────────
st.set_page_config(
    page_title="Backtest Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

# ── Estilos CSS personalizados ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo del sidebar */
    section[data-testid="stSidebar"] { background: #0f1117; }

    /* Cards de métricas */
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Títulos de sección */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 4px;
    }

    /* Badges de categoría */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-trend   { background: #1a3a5c; color: #64b5f6; }
    .badge-rev     { background: #1a3a2c; color: #81c784; }
    .badge-mom     { background: #3a2a1a; color: #ffb74d; }
    .badge-break   { background: #3a1a2c; color: #f48fb1; }
    .badge-vol     { background: #2a1a3a; color: #ce93d8; }

    /* Línea divisoria con color */
    hr { border-color: #2d3250; }

    /* Table headers */
    thead tr th { background-color: #1e2130 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar global ─────────────────────────────────────────────────────────────
with st.sidebar:
    
    st.markdown("## 📈 Backtest Framework")
    st.divider()

    df = get_long_df()
    tickers = get_ticker_list()

    if df is not None:
        st.success(f"✅ Datos cargados")
        st.markdown(f"**{df['ticker'].nunique()} tickers** · **{len(df):,} filas**")
        date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"
        st.caption(date_range)
        
        # ── Descarga rápida ──────────────────────────────────────────────
        with st.expander("⬇️ Re-descargar datos"):
            period_label_q = st.selectbox(
                "Período histórico",
                options=list(PERIOD_OPTIONS.keys()),
                index=list(PERIOD_OPTIONS.keys()).index("5 años"),
                key="home_period_select",
            )
            period_val_q = PERIOD_OPTIONS[period_label_q]
            
            interval_q = st.selectbox(
                "Intervalo",
                ["1d", "1wk"],
                index=0,
                key="home_interval_select",
            )
            
            force_refresh = st.checkbox(
                "Forzar re-descarga (ignorar caché)",
                value=False,
                key="home_force_refresh",
            )
            
            if st.button("⬇️ Descargar", type="primary", use_container_width=True):
                tickers = get_ticker_list()
                if not tickers:
                    st.error("No hay tickers en la lista. Agrega tickers en **Datos**.")
                else:
                    with st.spinner(f"Descargando {len(tickers)} tickers..."):
                        try:
                            if force_refresh:
                                cached_download.clear()
                            df = cached_download(tuple(tickers), period_val_q, interval_q)
                            st.session_state["long_df"] = df
                            st.session_state["selected_period"] = period_val_q
                            st.success(f"✅ {df['ticker'].nunique()} tickers · {len(df):,} filas descargados")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Sin datos descargados")
        st.caption("Descarga datos rápidamente desde aquí:")
        
        # ── Descarga rápida ──────────────────────────────────────────────
        with st.expander("⬇️ Descargar datos"):
            period_label_q = st.selectbox(
                "Período histórico",
                options=list(PERIOD_OPTIONS.keys()),
                index=list(PERIOD_OPTIONS.keys()).index("5 años"),
                key="home_period_select",
            )
            period_val_q = PERIOD_OPTIONS[period_label_q]
            
            interval_q = st.selectbox(
                "Intervalo",
                ["1d", "1wk"],
                index=0,
                key="home_interval_select",
            )
            
            tickers = get_ticker_list()
            st.info(f"**{len(tickers)} tickers** listos para descargar")
            
            if st.button("⬇️ Descargar", type="primary", use_container_width=True):
                if not tickers:
                    st.error("No hay tickers en la lista. Agrega tickers en **Datos**.")
                else:
                    with st.spinner(f"Descargando {len(tickers)} tickers desde Yahoo Finance..."):
                        try:
                            df = cached_download(tuple(tickers), period_val_q, interval_q)
                            st.session_state["long_df"] = df
                            st.session_state["selected_period"] = period_val_q
                            st.success(f"✅ {df['ticker'].nunique()} tickers · {len(df):,} filas")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error en descarga: {e}")

    st.divider()
    st.caption("**Navegación**")
    st.page_link("Home.py",                       label="Inicio",         icon="🏠")
    st.page_link("pages/1_Datos.py",              label="Datos & Tickers",   icon="🗂️")
    st.page_link("pages/2_Scanner.py",            label="Scanner",           icon="🔍")
    st.page_link("pages/3_Optimizacion.py",       label="Optimización",      icon="⚡")
    st.divider()
    st.caption("v1.0 · Backtest Framework")

# ── Contenido principal ────────────────────────────────────────────────────────
st.markdown("# 📈 Backtest Framework")
st.markdown(
    "<p style='color:#888; font-size:1.1rem;'>"
    "Plataforma cuantitativa para evaluar y optimizar estrategias de inversión.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Cards de navegación
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div style='background:#1e2130; border:1px solid #2d3250; border-radius:12px; padding:24px;'>
        <div style='font-size:2.5rem; margin-bottom:8px'>🗂️</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; margin-bottom:8px'>
            Datos & Tickers
        </div>
        <div style='color:#aaa; font-size:0.9rem;'>
            Configura la lista de tickers. Agrega o elimina activos de índices,
            acciones y ETFs. Descarga y visualiza los datos históricos desde Yahoo Finance.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Datos.py", label="Ir a Datos →", use_container_width=True)

with col2:
    st.markdown("""
    <div style='background:#1e2130; border:1px solid #2d3250; border-radius:12px; padding:24px;'>
        <div style='font-size:2.5rem; margin-bottom:8px'>🔍</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; margin-bottom:8px'>
            Scanner de Estrategias
        </div>
        <div style='color:#aaa; font-size:0.9rem;'>
            Evalúa las 30 estrategias predefinidas sobre un ticker.
            Compara métricas clave, filtra por categoría y profundiza
            en el análisis de una estrategia específica.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Scanner.py", label="Ir a Scanner →", use_container_width=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown("""
    <div style='background:#1e2130; border:1px solid #2d3250; border-radius:12px; padding:24px;'>
        <div style='font-size:2.5rem; margin-bottom:8px'>⚡</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; margin-bottom:8px'>
            Optimización
        </div>
        <div style='color:#aaa; font-size:0.9rem;'>
            Grid Search, Walk-Forward Optimization y Monte Carlo para encontrar
            los parámetros óptimos de una estrategia y evaluar su robustez estadística.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Optimizacion.py", label="Ir a Optimización →", use_container_width=True)

with col4:
    st.markdown("""
    <div style='background:#1e2130; border:1px solid #2d3250; border-radius:12px; padding:24px;'>
        <div style='font-size:2.5rem; margin-bottom:8px'>🎯</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; margin-bottom:8px'>
            Seguimiento
        </div>
        <div style='color:#aaa; font-size:0.9rem;'>
            Monitorea trades activos, gestiona estrategias en seguimiento
            y recibe alertas de señales de compra, venta, stop y profit en tiempo real.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/4_Seguimiento.py", label="Ir a Seguimiento →", use_container_width=True)

st.markdown("")  # Spacer

col5, col6 = st.columns(2, gap="large")

with col5:
    st.markdown("""
    <div style='background:#1e2130; border:1px solid #2d3250; border-radius:12px; padding:24px;'>
        <div style='font-size:2.5rem; margin-bottom:8px'>🏗️</div>
        <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; margin-bottom:8px'>
            Constructor
        </div>
        <div style='color:#aaa; font-size:0.9rem;'>
            Crea estrategias personalizadas definiendo condiciones de entrada, salida
            y gestión de riesgo. Simula, analiza y agrega a la biblioteca.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/5_Constructor.py", label="Ir a Constructor →", use_container_width=True)

st.divider()

# Resumen del modelo
st.markdown("### 🧱 Arquitectura del Framework")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**📥 Data Loader**")
    st.markdown("- Yahoo Finance (yfinance)\n- Formato long (tidy)\n- Caché local Parquet\n- Universo predefinido")
with c2:
    st.markdown("**📊 Indicadores**")
    st.markdown("- 24 indicadores técnicos\n- SMA, EMA, RSI, MACD\n- Bollinger, ATR, ADX\n- OBV, CMF, SuperTrend")
with c3:
    st.markdown("**🎯 Estrategias**")
    st.markdown("- 30 estrategias predefinidas\n- DSL Signal + Rule\n- 6 modos de sizing\n- Stop-loss / Take-profit / Trailing")
with c4:
    st.markdown("**⚙️ Backtest Engine**")
    st.markdown("- Sin lookahead bias\n- 20+ métricas\n- Grid Search / WFO / MC\n- Visualizaciones Plotly")

st.divider()

# Estado actual del sistema
st.markdown("### 📋 Estado del Sistema")
scol1, scol2, scol3 = st.columns(3)

from strategies import STRATEGY_LIBRARY
from indicators import get_indicator_catalog

with scol1:
    st.metric("Estrategias disponibles", len(STRATEGY_LIBRARY))
with scol2:
    st.metric("Indicadores técnicos", len(get_indicator_catalog()))
with scol3:
    if df is not None:
        st.metric("Tickers cargados", df["ticker"].nunique())
    else:
        st.metric("Tickers cargados", 0)
