"""
visualization.py
================
Módulo de visualización interactiva con Plotly para el framework de backtest.

Gráficos disponibles:
1.  plot_equity_curve()        - Equity curve vs Buy & Hold con trades marcados
2.  plot_drawdown()            - Drawdown en el tiempo
3.  plot_trades()              - Scatter de P&L por trade (waterfall)
4.  plot_monthly_returns()     - Heatmap de retornos mensuales
5.  plot_rolling_metrics()     - Sharpe, Vol y Retorno rolling
6.  plot_candlestick_signals() - OHLCV con señales de entrada/salida
7.  plot_scanner_summary()     - Comparación de múltiples estrategias
8.  plot_optimization_heatmap()- Heatmap de grid search
9.  plot_monte_carlo()         - Distribución y paths de Monte Carlo
10. plot_wfo_results()         - Resultados Walk-Forward
11. plot_returns_distribution() - Histograma de retornos
12. plot_full_dashboard()       - Dashboard completo multi-panel

Autor: Backtest Framework
"""

from __future__ import annotations

import warnings
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.templates.default = "plotly_dark"
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly no disponible. Instalar con: pip install plotly")

warnings.filterwarnings("ignore")

# Paleta de colores corporativa
COLORS = {
    "equity": "#00D4FF",
    "bh": "#FFD700",
    "drawdown": "#FF4444",
    "entry": "#00FF88",
    "exit": "#FF6B6B",
    "win": "#00CC66",
    "loss": "#FF3333",
    "neutral": "#888888",
    "bg": "#1a1a2e",
    "grid": "#2d2d2d",
    "text": "#EEEEEE",
}

LAYOUT_BASE = dict(
    plot_bgcolor=COLORS["bg"],
    paper_bgcolor="#16213e",
    font=dict(color=COLORS["text"], size=12),
    xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor=COLORS["grid"]),
    hovermode="x unified",
)


def _check_plotly():
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly requerido. Instalar: pip install plotly")


# ---------------------------------------------------------------------------
# 1. Equity Curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    result,
    show_trades: bool = True,
    title: Optional[str] = None,
    height: int = 600,
    show: bool = True,
) -> "go.Figure":
    """
    Gráfico de equity curve vs Buy & Hold con marcadores de trades.

    Parameters
    ----------
    result      : BacktestResult
    show_trades : marcar entradas y salidas en el gráfico
    """
    _check_plotly()

    fig = go.Figure()

    # Equity curve
    if not result.equity_curve.empty:
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            name=f"Estrategia: {result.strategy_name}",
            line=dict(color=COLORS["equity"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)",
        ))

    # Buy & Hold
    if not result.buy_and_hold.empty:
        fig.add_trace(go.Scatter(
            x=result.buy_and_hold.index,
            y=result.buy_and_hold.values,
            name="Buy & Hold",
            line=dict(color=COLORS["bh"], width=1.5, dash="dash"),
        ))

    # Marcadores de trades
    if show_trades and not result.trades_df.empty:
        closed = result.trades_df[result.trades_df["exit_date"].notna()]

        wins = closed[closed["net_pnl"] > 0]
        losses = closed[closed["net_pnl"] <= 0]

        if not wins.empty and not result.equity_curve.empty:
            entry_vals = result.equity_curve.reindex(
                pd.to_datetime(wins["entry_date"]), method="nearest"
            )
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(wins["entry_date"]),
                y=entry_vals.values,
                mode="markers",
                name="Entrada ganadora",
                marker=dict(symbol="triangle-up", size=10, color=COLORS["win"],
                            line=dict(color="white", width=1)),
                hovertemplate="Entrada: %{x}<br>P&L: %{customdata:.2f}%",
                customdata=wins["pnl_pct"].values,
            ))

        if not losses.empty and not result.equity_curve.empty:
            entry_vals_l = result.equity_curve.reindex(
                pd.to_datetime(losses["entry_date"]), method="nearest"
            )
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(losses["entry_date"]),
                y=entry_vals_l.values,
                mode="markers",
                name="Entrada perdedora",
                marker=dict(symbol="triangle-down", size=10, color=COLORS["loss"],
                            line=dict(color="white", width=1)),
                hovertemplate="Entrada: %{x}<br>P&L: %{customdata:.2f}%",
                customdata=losses["pnl_pct"].values,
            ))

    m = result.metrics
    title_str = title or (
        f"{result.strategy_name} | {result.ticker} | "
        f"Return={m.get('total_return_pct', 0):.1f}% | "
        f"Sharpe={m.get('sharpe_ratio', 0):.3f} | "
        f"MaxDD={m.get('max_drawdown_pct', 0):.1f}%"
    )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title_str, font=dict(size=14)),
        yaxis_title="Capital ($)",
        xaxis_title="Fecha",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Drawdown
# ---------------------------------------------------------------------------

def plot_drawdown(
    result,
    height: int = 400,
    show: bool = True,
) -> "go.Figure":
    """Gráfico de drawdown en el tiempo con área sombreada."""
    _check_plotly()

    if result.drawdown_series.empty:
        return go.Figure()

    dd = result.drawdown_series * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        name="Drawdown",
        fill="tozeroy",
        fillcolor="rgba(255,68,68,0.3)",
        line=dict(color=COLORS["drawdown"], width=1.5),
    ))

    # Línea de max drawdown
    max_dd = dd.min()
    fig.add_hline(
        y=max_dd,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Max DD: {max_dd:.2f}%",
        annotation_position="top right",
    )

    m = result.metrics
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Drawdown | {result.strategy_name} | MaxDD={m.get('max_drawdown_pct', 0):.2f}%",
        yaxis_title="Drawdown (%)",
        xaxis_title="Fecha",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 3. Distribución de Trades (waterfall / scatter)
# ---------------------------------------------------------------------------

def plot_trades(
    result,
    kind: str = "waterfall",
    height: int = 500,
    show: bool = True,
) -> "go.Figure":
    """
    Visualización de P&L por trade.

    kind: "waterfall" | "scatter" | "bar"
    """
    _check_plotly()

    if result.trades_df.empty:
        return go.Figure()

    closed = result.trades_df[result.trades_df["exit_date"].notna()].copy()
    if closed.empty:
        return go.Figure()

    closed["cumulative_pnl"] = closed["net_pnl"].cumsum()
    colors = [COLORS["win"] if v > 0 else COLORS["loss"] for v in closed["net_pnl"]]

    fig = go.Figure()

    if kind == "waterfall":
        fig.add_trace(go.Waterfall(
            x=list(range(len(closed))),
            y=closed["net_pnl"].values,
            name="P&L por Trade",
            connector={"line": {"color": COLORS["grid"]}},
            increasing={"marker": {"color": COLORS["win"]}},
            decreasing={"marker": {"color": COLORS["loss"]}},
            text=[f"${v:.0f}" for v in closed["net_pnl"]],
            hovertemplate=(
                "Trade #%{x}<br>P&L: $%{y:.2f}<br>"
                "Fecha entrada: %{customdata[0]}<br>"
                "Exit reason: %{customdata[1]}"
            ),
            customdata=closed[["entry_date", "exit_reason"]].values,
        ))
    elif kind == "scatter":
        fig.add_trace(go.Scatter(
            x=list(range(len(closed))),
            y=closed["cumulative_pnl"].values,
            name="P&L Acumulado",
            mode="lines+markers",
            line=dict(color=COLORS["equity"], width=2),
            marker=dict(color=colors, size=8),
        ))
    else:
        fig.add_trace(go.Bar(
            x=list(range(len(closed))),
            y=closed["pnl_pct"].values,
            name="Retorno por Trade (%)",
            marker_color=colors,
        ))

    m = result.metrics
    fig.update_layout(
        **LAYOUT_BASE,
        title=(f"Análisis de Trades | {result.strategy_name} | "
               f"Win Rate={m.get('win_rate_pct', 0):.1f}% | "
               f"Profit Factor={m.get('profit_factor', 0):.2f}"),
        xaxis_title="Trade #",
        yaxis_title="P&L ($)" if kind != "bar" else "P&L (%)",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 4. Heatmap de retornos mensuales
# ---------------------------------------------------------------------------

def plot_monthly_returns(
    result,
    height: int = 400,
    show: bool = True,
) -> "go.Figure":
    """Heatmap de retornos mensuales por año."""
    _check_plotly()

    monthly = result.get_monthly_returns()
    if monthly.empty:
        return go.Figure()

    z = monthly.values * 100
    text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=monthly.columns.tolist(),
        y=monthly.index.tolist(),
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Retorno (%)"),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Retornos Mensuales | {result.strategy_name}",
        xaxis_title="Mes",
        yaxis_title="Año",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 5. Métricas Rolling
# ---------------------------------------------------------------------------

def plot_rolling_metrics(
    result,
    height: int = 700,
    show: bool = True,
) -> "go.Figure":
    """Gráfico multi-panel con métricas rolling."""
    _check_plotly()

    if result.rolling_metrics.empty:
        return go.Figure()

    rm = result.rolling_metrics

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Sharpe Rolling", "Volatilidad Rolling (%)", "Retorno Rolling (%)"),
        vertical_spacing=0.08,
    )

    if "rolling_sharpe" in rm.columns:
        fig.add_trace(go.Scatter(
            x=rm.index, y=rm["rolling_sharpe"],
            name="Sharpe", line=dict(color=COLORS["equity"], width=1.5),
        ), row=1, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="green",
                      annotation_text="Sharpe=1", row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    if "rolling_vol" in rm.columns:
        fig.add_trace(go.Scatter(
            x=rm.index, y=rm["rolling_vol"] * 100,
            name="Volatilidad", line=dict(color="orange", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,165,0,0.1)",
        ), row=2, col=1)

    if "rolling_return" in rm.columns:
        colors_ret = [COLORS["win"] if v >= 0 else COLORS["loss"]
                      for v in rm["rolling_return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=rm.index, y=rm["rolling_return"] * 100,
            name="Retorno Anual.", marker_color=colors_ret,
        ), row=3, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Métricas Rolling | {result.strategy_name}",
        height=height,
        showlegend=True,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 6. Candlestick con señales
# ---------------------------------------------------------------------------

def plot_candlestick_signals(
    result,
    n_last: int = 252,
    show_indicators: List[str] = None,
    height: int = 800,
    show: bool = True,
) -> "go.Figure":
    """
    Candlestick OHLCV con señales de entrada/salida y indicadores técnicos.

    Parameters
    ----------
    n_last            : número de barras recientes a mostrar
    show_indicators   : lista de columnas de indicadores a superponer
    """
    _check_plotly()

    df = result.df_with_signals
    if df.empty:
        return go.Figure()

    df_plot = df.tail(n_last).copy()
    show_indicators = show_indicators or ["sma_20", "sma_50", "ema_20"]

    rows = 3 if "volume" in df_plot.columns else 2
    row_heights = [0.6, 0.2, 0.2] if rows == 3 else [0.7, 0.3]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=("Precio", "Volumen", "RSI") if rows == 3 else ("Precio", "Volumen"),
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="OHLC",
        increasing_line_color=COLORS["win"],
        decreasing_line_color=COLORS["loss"],
    ), row=1, col=1)

    # Indicadores técnicos
    ind_colors = [COLORS["equity"], "orange", "#FF69B4", "cyan", "lime"]
    for i, ind in enumerate(show_indicators):
        if ind in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot[ind],
                name=ind.upper(),
                line=dict(color=ind_colors[i % len(ind_colors)], width=1.2),
            ), row=1, col=1)

    # Bollinger Bands
    if "bb_upper_20" in df_plot.columns and "bb_lower_20" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["bb_upper_20"],
            name="BB Upper", line=dict(color="rgba(255,255,0,0.5)", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["bb_lower_20"],
            name="BB Lower",
            fill="tonexty",
            fillcolor="rgba(255,255,0,0.04)",
            line=dict(color="rgba(255,255,0,0.5)", width=1),
        ), row=1, col=1)

    # Señales de entrada/salida
    if "entry_signal" in df_plot.columns:
        entries = df_plot[df_plot["entry_signal"] == True]
        if not entries.empty:
            fig.add_trace(go.Scatter(
                x=entries.index,
                y=entries["low"] * 0.99,
                mode="markers",
                name="COMPRA",
                marker=dict(symbol="triangle-up", size=12,
                            color=COLORS["entry"], line=dict(color="white", width=1)),
            ), row=1, col=1)

    if "exit_signal" in df_plot.columns:
        exits = df_plot[df_plot["exit_signal"] == True]
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits.index,
                y=exits["high"] * 1.01,
                mode="markers",
                name="VENTA",
                marker=dict(symbol="triangle-down", size=12,
                            color=COLORS["exit"], line=dict(color="white", width=1)),
            ), row=1, col=1)

    # Volumen
    if "volume" in df_plot.columns:
        vol_colors = [COLORS["win"] if c >= o else COLORS["loss"]
                      for c, o in zip(df_plot["close"], df_plot["open"])]
        fig.add_trace(go.Bar(
            x=df_plot.index,
            y=df_plot["volume"],
            name="Volumen",
            marker_color=vol_colors,
            opacity=0.7,
        ), row=2, col=1)

    # RSI
    if "rsi_14" in df_plot.columns and rows == 3:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["rsi_14"],
            name="RSI(14)", line=dict(color="purple", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="70", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                      annotation_text="30", row=3, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{result.ticker} | {result.strategy_name} | Señales de Trading",
        height=height,
        xaxis_rangeslider_visible=False,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 7. Scanner Summary (comparación de estrategias)
# ---------------------------------------------------------------------------

def plot_scanner_summary(
    summary_df: pd.DataFrame,
    top_n: int = 20,
    x_metric: str = "sharpe_ratio",
    y_metric: str = "total_return_%",
    size_metric: str = "n_trades",
    height: int = 600,
    show: bool = True,
) -> "go.Figure":
    """
    Bubble chart comparativo de múltiples estrategias.
    Ejes configurables para analizar trade-offs entre métricas.
    """
    _check_plotly()

    if summary_df.empty:
        return go.Figure()

    df = summary_df.head(top_n).copy()

    # Color por categoría
    categories = df["category"].unique() if "category" in df.columns else ["general"]
    color_map = {cat: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                 for i, cat in enumerate(categories)}
    df["color"] = df["category"].map(color_map) if "category" in df.columns else COLORS["equity"]

    fig = go.Figure()

    for cat in (df["category"].unique() if "category" in df.columns else ["general"]):
        sub = df[df["category"] == cat] if "category" in df.columns else df
        fig.add_trace(go.Scatter(
            x=sub[x_metric] if x_metric in sub.columns else sub.index,
            y=sub[y_metric] if y_metric in sub.columns else sub.index,
            mode="markers+text",
            name=cat,
            text=sub["strategy"].str[:15] if "strategy" in sub.columns else sub.index,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(
                size=sub[size_metric].clip(5, 50) if size_metric in sub.columns else 15,
                color=color_map.get(cat, COLORS["equity"]),
                line=dict(width=1, color="white"),
                opacity=0.8,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{x_metric}: %{{x:.3f}}<br>"
                f"{y_metric}: %{{y:.3f}}<br>"
            ),
        ))

    # Línea de referencia
    if x_metric == "sharpe_ratio":
        fig.add_vline(x=1.0, line_dash="dash", line_color="green",
                      annotation_text="Sharpe=1")
        fig.add_vline(x=0.0, line_dash="dot", line_color="gray")
    if y_metric in ["total_return_%", "cagr_%"]:
        fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Comparación de Estrategias | {x_metric} vs {y_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 8. Heatmap de optimización
# ---------------------------------------------------------------------------

def plot_optimization_heatmap(
    grid_results: pd.DataFrame,
    param_x: str = "fast",
    param_y: str = "slow",
    metric: str = "sharpe_ratio",
    height: int = 500,
    show: bool = True,
) -> "go.Figure":
    """
    Heatmap 2D del grid search (param_x vs param_y → métrica).
    """
    _check_plotly()

    if grid_results.empty:
        return go.Figure()
    if param_x not in grid_results.columns or param_y not in grid_results.columns:
        return go.Figure()

    pivot = grid_results.pivot_table(
        values=metric,
        index=param_y,
        columns=param_x,
        aggfunc="mean",
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(v) for v in pivot.columns],
        y=[str(v) for v in pivot.index],
        colorscale="Viridis",
        colorbar=dict(title=metric),
        text=[[f"{v:.3f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
    ))

    # Marcar el máximo
    if not np.all(np.isnan(pivot.values)):
        max_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
        fig.add_annotation(
            x=str(pivot.columns[max_idx[1]]),
            y=str(pivot.index[max_idx[0]]),
            text="★",
            showarrow=False,
            font=dict(size=20, color="white"),
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Heatmap Optimización | {metric} | {param_y} vs {param_x}",
        xaxis_title=param_x,
        yaxis_title=param_y,
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 9. Monte Carlo
# ---------------------------------------------------------------------------

def plot_monte_carlo(
    mc_stats: Dict,
    equity_curve: Optional[pd.Series] = None,
    n_paths_display: int = 100,
    height: int = 600,
    show: bool = True,
) -> "go.Figure":
    """
    Visualización de resultados Monte Carlo:
    - Fan chart de percentiles de equity
    - Distribución de retornos totales
    """
    _check_plotly()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fan Chart Monte Carlo", "Distribución Retorno Total"),
    )

    # Panel 1: Fan chart
    if "equity_percentiles" in mc_stats:
        percs = mc_stats["equity_percentiles"]
        x = list(range(len(list(percs.values())[0])))

        fills = [
            ("p5", "p95", "rgba(0,212,255,0.1)"),
            ("p25", "p75", "rgba(0,212,255,0.2)"),
        ]
        for p_low, p_high, fill_color in fills:
            if p_low in percs and p_high in percs:
                fig.add_trace(go.Scatter(
                    x=x, y=percs[p_high],
                    fill=None, mode="lines",
                    line=dict(color="rgba(0,212,255,0.3)", width=0),
                    showlegend=False, name=f"p{p_high}",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=x, y=percs[p_low],
                    fill="tonexty",
                    fillcolor=fill_color,
                    mode="lines",
                    line=dict(color="rgba(0,212,255,0.3)", width=0),
                    name=f"{p_low}-{p_high}",
                ), row=1, col=1)

        if "p50" in percs:
            fig.add_trace(go.Scatter(
                x=x, y=percs["p50"],
                name="Mediana",
                line=dict(color=COLORS["equity"], width=2),
            ), row=1, col=1)

    if equity_curve is not None and not equity_curve.empty:
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve.values,
            name="Estrategia real",
            line=dict(color=COLORS["bh"], width=2, dash="dash"),
        ), row=1, col=1)

    # Panel 2: distribución de retornos totales
    if "total_return" in mc_stats and "p50" in mc_stats["total_return"]:
        s = mc_stats["total_return"]
        mu = s.get("mean", 0)
        sigma = s.get("std", 1)
        x_dist = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        from scipy.stats import norm
        y_dist = norm.pdf(x_dist, mu, sigma)

        fig.add_trace(go.Scatter(
            x=x_dist, y=y_dist,
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.15)",
            line=dict(color=COLORS["equity"], width=2),
            name="Distribución",
        ), row=1, col=2)

        fig.add_vline(x=0, line_dash="dash", line_color="red",
                      annotation_text="Pérdida=0%", row=1, col=2)
        fig.add_vline(x=mu, line_dash="dash", line_color=COLORS["bh"],
                      annotation_text=f"Media={mu:.1f}%", row=1, col=2)

    prob_loss = mc_stats.get("prob_loss", 0)
    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Análisis Monte Carlo | Prob. pérdida: {prob_loss:.1f}%",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 10. Walk-Forward Results
# ---------------------------------------------------------------------------

def plot_wfo_results(
    wfo_df: pd.DataFrame,
    height: int = 500,
    show: bool = True,
) -> "go.Figure":
    """Gráfico de barras comparando IS vs OOS por fold."""
    _check_plotly()

    if wfo_df.empty:
        return go.Figure()

    folds = wfo_df["fold"].astype(str).values

    fig = go.Figure()

    if "is_sharpe" in wfo_df.columns:
        fig.add_trace(go.Bar(
            x=folds,
            y=wfo_df["is_sharpe"],
            name="In-Sample Sharpe",
            marker_color=COLORS["equity"],
            opacity=0.7,
        ))

    if "oos_sharpe" in wfo_df.columns:
        fig.add_trace(go.Bar(
            x=folds,
            y=wfo_df["oos_sharpe"],
            name="Out-of-Sample Sharpe",
            marker_color=COLORS["bh"],
            opacity=0.7,
        ))

    if "efficiency" in wfo_df.columns:
        fig.add_trace(go.Scatter(
            x=folds,
            y=wfo_df["efficiency"],
            name="Eficiencia OOS/IS",
            mode="lines+markers",
            line=dict(color="white", width=2, dash="dot"),
            yaxis="y2",
        ))

    avg_eff = wfo_df["efficiency"].mean() if "efficiency" in wfo_df.columns else 0

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Walk-Forward Optimization | Eficiencia media: {avg_eff:.3f}",
        xaxis_title="Fold",
        yaxis_title="Sharpe Ratio",
        yaxis2=dict(
            title="Eficiencia",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        barmode="group",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 11. Distribución de retornos
# ---------------------------------------------------------------------------

def plot_returns_distribution(
    result,
    height: int = 500,
    show: bool = True,
) -> "go.Figure":
    """
    Histograma de retornos diarios con curva normal superpuesta.
    """
    _check_plotly()

    if result.equity_curve.empty:
        return go.Figure()

    returns = result.equity_curve.pct_change().dropna() * 100

    try:
        from scipy.stats import norm, jarque_bera
        jb_stat, jb_p = jarque_bera(returns)
        jb_text = f"Jarque-Bera p={jb_p:.4f}"
    except ImportError:
        jb_text = ""

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Histograma de Retornos", "QQ-Plot"))

    # Histograma
    fig.add_trace(go.Histogram(
        x=returns,
        name="Retornos",
        nbinsx=50,
        marker_color=COLORS["equity"],
        opacity=0.7,
        histnorm="probability density",
    ), row=1, col=1)

    mu, sigma = returns.mean(), returns.std()
    x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    try:
        from scipy.stats import norm
        y_norm = norm.pdf(x_norm, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            name="Normal",
            line=dict(color=COLORS["bh"], width=2),
        ), row=1, col=1)
    except ImportError:
        pass

    # QQ-Plot
    try:
        from scipy.stats import probplot
        osm, osr = probplot(returns, dist="norm", fit=False)
        fig.add_trace(go.Scatter(
            x=osm, y=osr,
            mode="markers",
            name="QQ",
            marker=dict(color=COLORS["equity"], size=4),
        ), row=1, col=2)
        lim = max(abs(min(osm)), abs(max(osm)))
        fig.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines",
            name="Ideal",
            line=dict(color=COLORS["bh"], dash="dash"),
        ), row=1, col=2)
    except ImportError:
        pass

    skew = returns.skew()
    kurt = returns.kurtosis()
    title_str = (f"Distribución de Retornos | {result.strategy_name} | "
                 f"Skew={skew:.3f} | Kurt={kurt:.3f} | {jb_text}")

    fig.update_layout(
        **LAYOUT_BASE,
        title=title_str,
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 12. Dashboard completo
# ---------------------------------------------------------------------------

def plot_full_dashboard(
    result,
    height: int = 1400,
    show: bool = True,
) -> "go.Figure":
    """
    Dashboard multi-panel completo:
    - Equity curve
    - Drawdown
    - RSI
    - Distribución de trades
    - Retornos mensuales (texto)
    - Métricas clave (tabla)
    """
    _check_plotly()

    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "table"}],  # row 1: equity curve, metrics table
            [{"type": "xy"}, {"type": "xy"}],     # row 2: drawdown, P&L trades
            [{"type": "xy"}, {"type": "xy"}],     # row 3: rolling sharpe, returns dist
            [{"type": "xy"}, {"type": "xy"}],     # row 4: cumulative returns, exit reasons
        ],
        subplot_titles=(
            "Equity Curve vs Buy & Hold",
            "Métricas Clave",
            "Drawdown",
            "P&L por Trade",
            "Rolling Sharpe (63 días)",
            "Distribución de Retornos",
            "Retornos Acumulados",
            "Análisis de Exit Reasons",
        ),
        row_heights=[0.3, 0.25, 0.25, 0.2],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # --- 1. Equity Curve ---
    if not result.equity_curve.empty:
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index, y=result.equity_curve.values,
            name="Estrategia", line=dict(color=COLORS["equity"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
        ), row=1, col=1)

    if not result.buy_and_hold.empty:
        fig.add_trace(go.Scatter(
            x=result.buy_and_hold.index, y=result.buy_and_hold.values,
            name="B&H", line=dict(color=COLORS["bh"], width=1.5, dash="dash"),
        ), row=1, col=1)

    # --- 2. Tabla de métricas ---
    m = result.metrics
    if "error" not in m:
        key_metrics = {
            "CAGR": f"{m.get('cagr_pct', 0):.2f}%",
            "Sharpe": f"{m.get('sharpe_ratio', 0):.4f}",
            "Sortino": f"{m.get('sortino_ratio', 0):.4f}",
            "Calmar": f"{m.get('calmar_ratio', 0):.4f}",
            "Max DD": f"{m.get('max_drawdown_pct', 0):.2f}%",
            "Vol Anual": f"{m.get('annual_volatility_pct', 0):.2f}%",
            "Win Rate": f"{m.get('win_rate_pct', 0):.2f}%",
            "Profit Factor": f"{m.get('profit_factor', 0):.4f}",
            "N Trades": str(m.get("n_trades", 0)),
            "Avg Holding": f"{m.get('avg_holding_days', 0):.1f}d",
        }
        fig.add_trace(go.Table(
            header=dict(
                values=["Métrica", "Valor"],
                fill_color="#1e3a5f",
                font=dict(color="white", size=12),
            ),
            cells=dict(
                values=[list(key_metrics.keys()), list(key_metrics.values())],
                fill_color=[["#1a2a3a"] * len(key_metrics)] * 2,
                font=dict(color="white", size=11),
                align="left",
            ),
        ), row=1, col=2)

    # --- 3. Drawdown ---
    if not result.drawdown_series.empty:
        dd = result.drawdown_series * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown", fill="tozeroy",
            fillcolor="rgba(255,68,68,0.3)",
            line=dict(color=COLORS["drawdown"], width=1.5),
        ), row=2, col=1)

    # --- 4. P&L por trade ---
    if not result.trades_df.empty:
        closed = result.trades_df[result.trades_df["exit_date"].notna()]
        if not closed.empty:
            colors_pnl = [COLORS["win"] if v > 0 else COLORS["loss"]
                          for v in closed["net_pnl"]]
            fig.add_trace(go.Bar(
                x=list(range(len(closed))),
                y=closed["net_pnl"].values,
                name="P&L Trade",
                marker_color=colors_pnl,
                opacity=0.8,
            ), row=2, col=2)

    # --- 5. Rolling Sharpe ---
    if not result.rolling_metrics.empty and "rolling_sharpe" in result.rolling_metrics.columns:
        rs_trace = go.Scatter(
            x=result.rolling_metrics.index,
            y=result.rolling_metrics["rolling_sharpe"],
            name="Rolling Sharpe",
            line=dict(color="purple", width=1.5),
        )
        fig.add_trace(rs_trace, row=3, col=1)
        # Add horizontal reference lines as shapes (compatible with table subplots)
        fig.add_shape(type="line", x0=0, x1=1, y0=1, y1=1,
                     line=dict(dash="dash", color="green", width=1),
                     xref="x5 domain", yref="y5")
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=0,
                     line=dict(dash="dot", color="gray", width=1),
                     xref="x5 domain", yref="y5")

    # --- 6. Distribución de retornos ---
    if not result.equity_curve.empty:
        returns = result.equity_curve.pct_change().dropna() * 100
        fig.add_trace(go.Histogram(
            x=returns,
            name="Retornos",
            nbinsx=40,
            marker_color=COLORS["equity"],
            opacity=0.7,
        ), row=3, col=2)

    # --- 7. Retornos acumulados comparados ---
    if not result.equity_curve.empty:
        eq_ret = ((result.equity_curve / result.equity_curve.iloc[0]) - 1) * 100
        fig.add_trace(go.Scatter(
            x=eq_ret.index, y=eq_ret.values,
            name="Ret. Acum. Est.",
            line=dict(color=COLORS["equity"], width=2),
        ), row=4, col=1)
    if not result.buy_and_hold.empty:
        bh_ret = ((result.buy_and_hold / result.buy_and_hold.iloc[0]) - 1) * 100
        fig.add_trace(go.Scatter(
            x=bh_ret.index, y=bh_ret.values,
            name="Ret. Acum. B&H",
            line=dict(color=COLORS["bh"], width=1.5, dash="dash"),
        ), row=4, col=1)

    # --- 8. Exit Reasons ---
    exit_reasons = result.get_trades_by_reason()
    if not exit_reasons.empty:
        fig.add_trace(go.Bar(
            x=exit_reasons.index,
            y=exit_reasons["count"],
            name="Exit Reasons",
            marker_color=[COLORS["equity"], COLORS["win"], COLORS["loss"],
                          "orange", "purple"][:len(exit_reasons)],
            text=exit_reasons["count"],
            textposition="outside",
        ), row=4, col=2)

    title = (f"DASHBOARD BACKTEST | {result.strategy_name} | {result.ticker} | "
             f"CAGR={m.get('cagr_pct', 0):.1f}% | "
             f"Sharpe={m.get('sharpe_ratio', 0):.3f}")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14)),
        height=height,
        showlegend=False,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# 13. Sensibilidad de parámetros
# ---------------------------------------------------------------------------

def plot_sensitivity(
    sensitivity_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    height: int = 500,
    show: bool = True,
) -> "go.Figure":
    """Gráfico de líneas de sensibilidad univariada de parámetros."""
    _check_plotly()

    if sensitivity_df.empty:
        return go.Figure()

    params = sensitivity_df["param_name"].unique()
    colors_list = px.colors.qualitative.Set2

    fig = make_subplots(
        rows=1,
        cols=len(params),
        subplot_titles=[f"Efecto de {p}" for p in params],
    )

    for i, param in enumerate(params, 1):
        sub = sensitivity_df[sensitivity_df["param_name"] == param].sort_values("param_value")
        base = sub[sub["is_base"] == True]

        fig.add_trace(go.Scatter(
            x=sub["param_value"],
            y=sub[metric] if metric in sub.columns else sub["sharpe_ratio"],
            name=param,
            mode="lines+markers",
            line=dict(color=colors_list[i % len(colors_list)], width=2),
            marker=dict(size=8),
        ), row=1, col=i)

        if not base.empty:
            fig.add_trace(go.Scatter(
                x=base["param_value"],
                y=base[metric] if metric in base.columns else base["sharpe_ratio"],
                name=f"{param} base",
                mode="markers",
                marker=dict(symbol="star", size=15, color="gold"),
                showlegend=i == 1,
            ), row=1, col=i)

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Análisis de Sensibilidad | Métrica: {metric}",
        height=height,
    )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Helper: guardar figura como HTML
# ---------------------------------------------------------------------------

def save_figure(fig: "go.Figure", path: str, auto_open: bool = False) -> str:
    """Guarda una figura Plotly como archivo HTML interactivo."""
    _check_plotly()
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path, auto_open=auto_open)
    print(f"  Figura guardada: {path}")
    return path


def save_figure_png(fig: "go.Figure", path: str, width: int = 1400, height: int = 800) -> str:
    """Guarda una figura como PNG (requiere kaleido)."""
    _check_plotly()
    try:
        fig.write_image(path, width=width, height=height)
        print(f"  Imagen guardada: {path}")
    except Exception as e:
        print(f"  No se pudo guardar PNG (instala kaleido): {e}")
    return path


if __name__ == "__main__":
    from data_loader import download_data
    from strategies import STRATEGY_LIBRARY
    from backtest_engine import BacktestEngine, BacktestConfig

    ticker = "AAPL"
    df = download_data([ticker], period="3y", cache_dir="data/cache")
    strategy = STRATEGY_LIBRARY["SMA_Cross_20_50"]
    engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
    result = engine.run(df, strategy, ticker=ticker)

    print("Generando dashboard...")
    fig = plot_full_dashboard(result, show=True)
    save_figure(fig, "results/dashboard.html")
