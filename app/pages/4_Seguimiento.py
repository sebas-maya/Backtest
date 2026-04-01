"""
app/pages/4_Seguimiento.py
===========================
Módulo de seguimiento de estrategias activas con sistema de alertas.

Funcionalidades:
- Agregar/remover estrategias para tickers específicos
- Visualizar trades activos con métricas clave
- Sistema de alertas para señales de compra/venta/stop/profit
- Panel de seguimiento en tiempo real
"""

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
from datetime import datetime

from app.utils import (
    init_state, page_header, get_long_df,
    add_tracked_strategy, remove_tracked_strategy, get_tracked_strategies,
    run_tracking_backtest, detect_active_trades, detect_signals_next_bar,
)
from strategies import list_strategies

# ── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Seguimiento - Backtest App",
    page_icon="📊",
    layout="wide",
)

init_state()

# ── Navegación ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧭 Navegación")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")
    st.page_link("pages/1_Datos.py", label="📈 Datos", use_container_width=True)
    st.page_link("pages/2_Scanner.py", label="🔍 Scanner", use_container_width=True)
    st.page_link("pages/3_Optimizacion.py", label="⚙️ Optimización", use_container_width=True)
    st.page_link("pages/4_Seguimiento.py", label="🎯 Seguimiento", use_container_width=True)
    st.page_link("pages/5_Constructor.py", label="🏗️ Constructor", use_container_width=True)
    st.divider()

# ── Header ────────────────────────────────────────────────────────────────────

page_header(
    "🎯 Seguimiento de Estrategias",
    "Monitorea trades activos y recibe alertas de señales en tiempo real"
)

# ── Verificar datos ───────────────────────────────────────────────────────────

df = get_long_df()
if df is None or df.empty:
    st.warning("⚠️ No hay datos cargados. Ve a **Datos** para descargar información.")
    st.stop()

# ── Sección 1: Agregar/Remover Estrategias ───────────────────────────────────

st.markdown("### ➕ Gestionar Seguimiento")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    available_tickers = sorted(df["ticker"].unique().tolist())
    selected_ticker = st.selectbox(
        "Ticker",
        options=available_tickers,
        key="track_ticker_select",
    )

with col2:
    available_strategies = list_strategies()
    selected_strategy = st.selectbox(
        "Estrategia",
        options=available_strategies,
        key="track_strategy_select",
    )

with col3:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("➕ Agregar", type="primary", use_container_width=True):
        if add_tracked_strategy(selected_ticker, selected_strategy):
            st.success(f"✅ {selected_strategy} agregada para {selected_ticker}")
            st.rerun()
        else:
            st.warning(f"⚠️ Esta combinación ya está en seguimiento")

st.divider()

# ── Sección 2: Estrategias en Seguimiento ────────────────────────────────────

tracked = get_tracked_strategies()

if not tracked:
    st.info("📋 No hay estrategias en seguimiento. Agrega una arriba para comenzar.")
    st.stop()

st.markdown(f"### 📊 Estrategias Activas ({len(tracked)})")

# ── Tabla de estrategias rastreadas ──────────────────────────────────────────

track_data = []
for item in tracked:
    track_data.append({
        "Ticker": item["ticker"],
        "Estrategia": item["strategy_name"],
        "Fecha Agregado": item["added_date"],
    })

track_df = pd.DataFrame(track_data)
st.dataframe(track_df, use_container_width=True, hide_index=True)

# ── Botón para remover ───────────────────────────────────────────────────────

st.markdown("#### 🗑️ Remover Estrategia")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    remove_ticker = st.selectbox(
        "Ticker a remover",
        options=[item["ticker"] for item in tracked],
        key="remove_ticker_select",
    )

with col2:
    # Filtrar estrategias para el ticker seleccionado
    strategies_for_ticker = [
        item["strategy_name"] for item in tracked
        if item["ticker"] == remove_ticker
    ]
    remove_strategy = st.selectbox(
        "Estrategia a remover",
        options=strategies_for_ticker,
        key="remove_strategy_select",
    )

with col3:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("🗑️ Remover", type="secondary", use_container_width=True):
        if remove_tracked_strategy(remove_ticker, remove_strategy):
            st.success(f"✅ {remove_strategy} removida de {remove_ticker}")
            st.rerun()
        else:
            st.error("❌ Error al remover")

st.divider()

# ── Sección 3: Análisis de Trades Activos ────────────────────────────────────

st.markdown("### 💼 Trades Activos")

# Ejecutar backtests y detectar trades activos
active_trades_data = []

with st.spinner("🔄 Analizando estrategias..."):
    for item in tracked:
        ticker = item["ticker"]
        strategy_name = item["strategy_name"]
        
        # Filtrar datos del ticker
        ticker_df = df[df["ticker"] == ticker].copy()
        if ticker_df.empty:
            continue
        
        # Ejecutar backtest
        result = run_tracking_backtest(ticker, strategy_name, ticker_df)
        if result is None:
            continue
        
        # Detectar trade activo
        active = detect_active_trades(result)
        
        if active:
            active_trades_data.append({
                "Ticker": ticker,
                "Estrategia": strategy_name,
                "Estado": active["status"],
                "Entrada": active["entry_date"],
                "Precio Entrada": f"${active['entry_price']:,.2f}",
                "Precio Actual": f"${active['current_price']:,.2f}",
                "Shares": active["shares"],
                "Retorno %": f"{active['pnl_pct']:+.2f}%",
                "Días": active["days_held"],
                "Stop Loss": f"${active['stop_loss']:,.2f}" if active['stop_loss'] else "N/A",
                "Take Profit": f"${active['take_profit']:,.2f}" if active['take_profit'] else "N/A",
            })

if active_trades_data:
    active_df = pd.DataFrame(active_trades_data)
    
    # Aplicar formato de color según retorno
    def color_return(val):
        if isinstance(val, str) and "%" in val:
            num = float(val.replace("%", "").replace("+", ""))
            if num > 0:
                return "background-color: rgba(0, 255, 0, 0.2)"
            elif num < 0:
                return "background-color: rgba(255, 0, 0, 0.2)"
        return ""
    
    styled_df = active_df.style.applymap(color_return, subset=["Retorno %"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Métricas agregadas
    st.markdown("#### 📈 Métricas Consolidadas")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calcular retornos numéricos
    returns = []
    for item in active_trades_data:
        ret_str = item["Retorno %"].replace("%", "").replace("+", "")
        returns.append(float(ret_str))
    
    with col1:
        st.metric("Total Trades Activos", len(active_trades_data))
    with col2:
        avg_return = sum(returns) / len(returns) if returns else 0
        st.metric("Retorno Promedio", f"{avg_return:+.2f}%")
    with col3:
        winning = sum(1 for r in returns if r > 0)
        st.metric("Trades Ganadores", f"{winning}/{len(returns)}")
    with col4:
        max_return = max(returns) if returns else 0
        st.metric("Mejor Trade", f"{max_return:+.2f}%")
else:
    st.info("📭 No hay trades activos en este momento.")

st.divider()

# ── Sección 4: Sistema de Alertas ────────────────────────────────────────────

st.markdown("### 🚨 Centro de Alertas")
st.markdown("*Señales detectadas para ejecutar en la próxima vela*")

alerts = []

with st.spinner("🔍 Escaneando señales..."):
    for item in tracked:
        ticker = item["ticker"]
        strategy_name = item["strategy_name"]
        
        # Filtrar datos
        ticker_df = df[df["ticker"] == ticker].copy()
        if ticker_df.empty:
            continue
        
        # Obtener resultado del backtest
        result = run_tracking_backtest(ticker, strategy_name, ticker_df)
        if result is None:
            continue
        
        # Detectar señales
        signals = detect_signals_next_bar(result)
        
        # Agregar alertas
        if signals["buy_signal"]:
            alerts.append({
                "Ticker": ticker,
                "Estrategia": strategy_name,
                "Tipo": "🟢 COMPRA",
                "Mensaje": "Señal de entrada detectada",
                "Prioridad": "ALTA",
            })
        
        if signals["sell_signal"]:
            alerts.append({
                "Ticker": ticker,
                "Estrategia": strategy_name,
                "Tipo": "🔴 VENTA",
                "Mensaje": "Señal de salida detectada",
                "Prioridad": "ALTA",
            })
        
        if signals["stop_signal"]:
            alerts.append({
                "Ticker": ticker,
                "Estrategia": strategy_name,
                "Tipo": "🛑 STOP LOSS",
                "Mensaje": "Precio alcanzó stop loss",
                "Prioridad": "CRÍTICA",
            })
        
        if signals["profit_signal"]:
            alerts.append({
                "Ticker": ticker,
                "Estrategia": strategy_name,
                "Tipo": "✅ TAKE PROFIT",
                "Mensaje": "Precio alcanzó take profit",
                "Prioridad": "ALTA",
            })

if alerts:
    st.warning(f"⚠️ **{len(alerts)} alerta(s) detectada(s)**")
    
    alerts_df = pd.DataFrame(alerts)
    
    # Aplicar colores según prioridad
    def color_priority(val):
        if val == "CRÍTICA":
            return "background-color: rgba(255, 0, 0, 0.3)"
        elif val == "ALTA":
            return "background-color: rgba(255, 165, 0, 0.3)"
        return ""
    
    styled_alerts = alerts_df.style.applymap(color_priority, subset=["Prioridad"])
    st.dataframe(styled_alerts, use_container_width=True, hide_index=True)
    
    # Sonido de alerta (opcional)
    st.markdown(
        """
        <script>
        const audio = new Audio('https://freesound.org/data/previews/316/316847_4939433-lq.mp3');
        audio.play();
        </script>
        """,
        unsafe_allow_html=True
    )
else:
    st.success("✅ No hay alertas en este momento. Todo tranquilo.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────

st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.info("💡 **Tip:** Esta página se actualiza cada vez que interactúas. Considera refrescar manualmente para ver nuevas señales.")
with col2:
    if st.button("🔄 Refrescar Ahora", type="primary", use_container_width=True):
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f"<p style='text-align: center; color: #666; font-size: 0.9em;'>"
    f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</p>",
    unsafe_allow_html=True
)
