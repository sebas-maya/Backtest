"""
app/pages/5_Constructor.py
===========================
Módulo constructor de estrategias personalizadas.

Funcionalidades:
- Definir nombre y descripción
- Agregar condiciones de entrada (crossover, above, below, etc.)
- Agregar condiciones de salida
- Configurar stop loss, take profit, max holding days
- Simular estrategia en ticker específico
- Ver análisis detallado con bitácora de trades
- Guardar estrategia en biblioteca
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
)
from strategies import (
    create_custom_strategy, add_strategy_to_library,
    get_available_columns, STRATEGY_LIBRARY,
)
from backtest_engine import BacktestEngine, BacktestConfig
from visualization import (
    plot_full_dashboard, plot_candlestick_signals,
)

# ── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Constructor - Backtest App",
    page_icon="🏗️",
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
    "🏗️ Constructor de Estrategias",
    "Crea estrategias personalizadas y agrégalas a la biblioteca"
)

# ── Verificar datos ───────────────────────────────────────────────────────────

df = get_long_df()
if df is None or df.empty:
    st.warning("⚠️ No hay datos cargados. Ve a **Datos** para descargar información.")
    st.stop()

# ── Sección 1: Definición Básica ─────────────────────────────────────────────

st.markdown("### 📝 Definición Básica")

col1, col2 = st.columns(2)

with col1:
    strategy_name = st.text_input(
        "Nombre de la Estrategia",
        value="Mi_Estrategia_Custom",
        help="Nombre único para identificar la estrategia",
        key="strat_name",
    )

with col2:
    strategy_category = st.selectbox(
        "Categoría",
        ["custom", "trend_following", "mean_reversion", "momentum", "breakout", "volume"],
        key="strat_category",
    )

strategy_description = st.text_area(
    "Descripción",
    value="Estrategia personalizada creada con el constructor",
    height=80,
    key="strat_desc",
)

st.divider()

# ── Sección 2: Condiciones de Entrada ────────────────────────────────────────

st.markdown("### ✅ Condiciones de Entrada")
st.markdown("*Todas las condiciones deben cumplirse (AND lógico)*")

# Inicializar lista de condiciones
if "entry_conditions" not in st.session_state:
    st.session_state.entry_conditions = []

# Agregar nueva condición
with st.expander("➕ Agregar Condición de Entrada", expanded=len(st.session_state.entry_conditions) == 0):
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        condition_type = st.selectbox(
            "Tipo",
            ["crossover", "crossunder", "above", "below"],
            key="entry_type_new",
            help="crossover: cruza hacia arriba | crossunder: cruza hacia abajo | above: está por encima | below: está por debajo"
        )
    
    with col2:
        available_cols = get_available_columns()
        col_a = st.selectbox(
            "Columna A",
            available_cols,
            index=available_cols.index("close") if "close" in available_cols else 0,
            key="entry_col_a_new",
        )
    
    with col3:
        # Permitir selector de columna o input manual
        use_column = st.checkbox("Usar columna/indicador", value=True, key="entry_use_col_new")
        if use_column:
            col_b = st.selectbox(
                "Columna B",
                available_cols,
                index=available_cols.index("sma_50") if "sma_50" in available_cols else 0,
                key="entry_col_b_new",
            )
        else:
            col_b = st.number_input(
                "Valor",
                value=50.0,
                step=1.0,
                key="entry_val_b_new",
            )
    
    with col4:
        st.write("")
        st.write("")
        if st.button("➕ Agregar", type="primary", key="add_entry_cond"):
            new_cond = {
                "type": condition_type,
                "col_a": col_a,
                "col_b": col_b,
            }
            st.session_state.entry_conditions.append(new_cond)
            st.rerun()

# Mostrar condiciones actuales
if st.session_state.entry_conditions:
    st.markdown("**Condiciones actuales:**")
    for idx, cond in enumerate(st.session_state.entry_conditions):
        col1, col2 = st.columns([5, 1])
        with col1:
            cond_str = f"{idx+1}. `{cond['col_a']}` **{cond['type']}** `{cond['col_b']}`"
            st.markdown(cond_str)
        with col2:
            if st.button("🗑️", key=f"del_entry_{idx}"):
                st.session_state.entry_conditions.pop(idx)
                st.rerun()
else:
    st.info("📋 No hay condiciones de entrada definidas")

st.divider()

# ── Sección 3: Condiciones de Salida ─────────────────────────────────────────

st.markdown("### ❌ Condiciones de Salida (Opcional)")
st.markdown("*Al menos una condición debe cumplirse (OR lógico)*")

# Inicializar lista de condiciones de salida
if "exit_conditions" not in st.session_state:
    st.session_state.exit_conditions = []

with st.expander("➕ Agregar Condición de Salida"):
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        exit_type = st.selectbox(
            "Tipo",
            ["crossover", "crossunder", "above", "below"],
            key="exit_type_new",
        )
    
    with col2:
        exit_col_a = st.selectbox(
            "Columna A",
            available_cols,
            index=available_cols.index("close") if "close" in available_cols else 0,
            key="exit_col_a_new",
        )
    
    with col3:
        use_column_exit = st.checkbox("Usar columna/indicador", value=True, key="exit_use_col_new")
        if use_column_exit:
            exit_col_b = st.selectbox(
                "Columna B",
                available_cols,
                index=available_cols.index("sma_20") if "sma_20" in available_cols else 0,
                key="exit_col_b_new",
            )
        else:
            exit_col_b = st.number_input(
                "Valor",
                value=30.0,
                step=1.0,
                key="exit_val_b_new",
            )
    
    with col4:
        st.write("")
        st.write("")
        if st.button("➕ Agregar", type="primary", key="add_exit_cond"):
            new_cond = {
                "type": exit_type,
                "col_a": exit_col_a,
                "col_b": exit_col_b,
            }
            st.session_state.exit_conditions.append(new_cond)
            st.rerun()

# Mostrar condiciones de salida
if st.session_state.exit_conditions:
    st.markdown("**Condiciones de salida actuales:**")
    for idx, cond in enumerate(st.session_state.exit_conditions):
        col1, col2 = st.columns([5, 1])
        with col1:
            cond_str = f"{idx+1}. `{cond['col_a']}` **{cond['type']}** `{cond['col_b']}`"
            st.markdown(cond_str)
        with col2:
            if st.button("🗑️", key=f"del_exit_{idx}"):
                st.session_state.exit_conditions.pop(idx)
                st.rerun()
else:
    st.info("📋 No hay condiciones de salida definidas (solo se usará gestión de riesgo)")

st.divider()

# ── Sección 4: Gestión de Riesgo ─────────────────────────────────────────────

st.markdown("### 🛡️ Gestión de Riesgo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    use_stop_loss = st.checkbox("Stop Loss", value=True, key="use_sl")
    if use_stop_loss:
        stop_loss = st.number_input(
            "Stop Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="stop_loss_val",
        ) / 100
    else:
        stop_loss = None

with col2:
    use_take_profit = st.checkbox("Take Profit", value=True, key="use_tp")
    if use_take_profit:
        take_profit = st.number_input(
            "Take Profit (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            key="take_profit_val",
        ) / 100
    else:
        take_profit = None

with col3:
    use_max_days = st.checkbox("Max Holding Days", value=True, key="use_max_days")
    if use_max_days:
        max_holding_days = st.number_input(
            "Días Máximos",
            min_value=1,
            max_value=365,
            value=30,
            step=5,
            key="max_days_val",
        )
    else:
        max_holding_days = None

with col4:
    use_trailing = st.checkbox("Trailing Stop", value=False, key="use_trailing")
    if use_trailing:
        trailing_stop = st.number_input(
            "Trailing Stop (%)",
            min_value=0.0,
            max_value=50.0,
            value=8.0,
            step=0.5,
            key="trailing_val",
        ) / 100
    else:
        trailing_stop = None

st.divider()

# ── Sección 5: Simulación ────────────────────────────────────────────────────

st.markdown("### 🎮 Simulación y Prueba")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    available_tickers = sorted(df["ticker"].unique().tolist())
    test_ticker = st.selectbox(
        "Ticker para Simular",
        available_tickers,
        index=0 if len(available_tickers) > 0 else None,
        key="test_ticker",
    )

with col2:
    st.write("")
    st.write("")

with col3:
    st.write("")
    st.write("")
    if st.button("🚀 Ejecutar Simulación", type="primary", use_container_width=True):
        # Validar que haya al menos una condición de entrada
        if not st.session_state.entry_conditions:
            st.error("❌ Debes definir al menos una condición de entrada")
        else:
            with st.spinner("⏳ Ejecutando backtest..."):
                try:
                    # Crear estrategia
                    strategy = create_custom_strategy(
                        name=strategy_name,
                        entry_conditions=st.session_state.entry_conditions,
                        exit_conditions=st.session_state.exit_conditions if st.session_state.exit_conditions else None,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        max_holding_days=max_holding_days,
                        trailing_stop=trailing_stop,
                        description=strategy_description,
                        category=strategy_category,
                    )
                    
                    # Filtrar datos del ticker
                    ticker_df = df[df["ticker"] == test_ticker].copy()
                    
                    # Ejecutar backtest
                    engine = BacktestEngine(config=BacktestConfig(
                        initial_capital=100_000,
                        commission_pct=0.001,
                        slippage_pct=0.0005,
                    ))
                    
                    result = engine.run(ticker_df, strategy, ticker=test_ticker, add_indicators=True)
                    
                    # Guardar resultado
                    st.session_state.builder_result = result
                    st.session_state.builder_strategy = strategy
                    
                    st.success(f"✅ Simulación completada: {len(result.trades)} trades ejecutados")
                    
                except Exception as e:
                    st.error(f"❌ Error en simulación: {str(e)}")

# ── Sección 6: Resultados ────────────────────────────────────────────────────

if "builder_result" in st.session_state and st.session_state.builder_result:
    st.divider()
    st.markdown("### 📊 Resultados del Backtest")
    
    result = st.session_state.builder_result
    strategy = st.session_state.builder_strategy
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "📋 Bitácora de Trades", 
        "📍 Señales de Trading",
        "💾 Guardar Estrategia"
    ])
    
    with tab1:
        st.markdown("#### Análisis Completo")
        
        # Métricas principales
        m = result.metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Trades", m.get("n_trades", 0))
        with col2:
            st.metric("Win Rate", f"{m.get('win_rate_pct', 0):.1f}%")
        with col3:
            st.metric("Retorno Total", f"{m.get('total_return_pct', 0):.2f}%")
        with col4:
            st.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.2f}")
        with col5:
            st.metric("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.2f}%")
        with col6:
            st.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        
        # Dashboard completo
        st.plotly_chart(
            plot_full_dashboard(result, show=False),
            use_container_width=True,
        )
    
    with tab2:
        st.markdown("#### 📋 Bitácora Detallada de Trades")
        
        if result.trades:
            trades_data = []
            for t in result.trades:
                trades_data.append({
                    "ID": t.trade_id,
                    "Entrada": t.entry_date.strftime("%Y-%m-%d"),
                    "Salida": t.exit_date.strftime("%Y-%m-%d") if t.exit_date else "N/A",
                    "Precio Entrada": f"${t.entry_price:.2f}",
                    "Precio Salida": f"${t.exit_price:.2f}" if t.exit_price else "N/A",
                    "Shares": f"{t.shares:.2f}",
                    "PnL $": f"${t.net_pnl:.2f}",
                    "PnL %": f"{t.pnl_pct:.2f}%",
                    "Días": t.holding_days,
                    "Razón Salida": t.exit_reason or "N/A",
                })
            
            trades_df = pd.DataFrame(trades_data)
            
            # Aplicar colores
            def color_pnl(val):
                if isinstance(val, str) and "%" in val:
                    num = float(val.replace("%", ""))
                    if num > 0:
                        return "background-color: rgba(0, 255, 0, 0.2)"
                    elif num < 0:
                        return "background-color: rgba(255, 0, 0, 0.2)"
                return ""
            
            styled = trades_df.style.applymap(color_pnl, subset=["PnL %"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
            
            # Estadísticas de trades
            st.markdown("#### 📊 Estadísticas de Trades")
            col1, col2, col3, col4 = st.columns(4)
            
            winning_trades = [t for t in result.trades if t.pnl_pct > 0]
            losing_trades = [t for t in result.trades if t.pnl_pct < 0]
            
            with col1:
                st.metric("Trades Ganadores", len(winning_trades))
            with col2:
                st.metric("Trades Perdedores", len(losing_trades))
            with col3:
                avg_win = sum(t.pnl_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
                st.metric("Ganancia Promedio", f"{avg_win:.2f}%")
            with col4:
                avg_loss = sum(t.pnl_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0
                st.metric("Pérdida Promedio", f"{avg_loss:.2f}%")
        else:
            st.info("📭 No se ejecutaron trades en el período simulado")
    
    with tab3:
        st.markdown("#### 📍 Señales de Entrada y Salida")
        st.markdown("Gráfico de velas con marcadores visuales de las señales de compra/venta generadas por la estrategia.")
        
        if result.df_with_signals is not None and not result.df_with_signals.empty:
            st.plotly_chart(
                plot_candlestick_signals(result, show=False),
                use_container_width=True,
            )
        else:
            st.info("📭 No hay señales disponibles para visualizar")
    
    with tab4:
        st.markdown("#### 💾 Guardar Estrategia en Biblioteca")
        st.markdown("Si la estrategia muestra buenos resultados, puedes agregarla a la biblioteca para usarla en Scanner, Optimización y Seguimiento.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"**Estrategia:** {strategy.name}\n\n**Descripción:** {strategy.description}")
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Guardar en Biblioteca", type="primary", use_container_width=True):
                # Verificar si ya existe
                if strategy.name in STRATEGY_LIBRARY:
                    st.warning(f"⚠️ Ya existe una estrategia con el nombre '{strategy.name}'")
                else:
                    add_strategy_to_library(strategy)
                    st.success(f"✅ Estrategia '{strategy.name}' guardada exitosamente!")
                    st.info("Ya puedes usar esta estrategia en las otras páginas de la app")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.9em;'>"
    "💡 Tip: Combina múltiples indicadores para crear estrategias más robustas"
    "</p>",
    unsafe_allow_html=True
)
