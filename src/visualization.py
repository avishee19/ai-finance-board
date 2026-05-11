"""
visualization.py — All Plotly chart factories.
Pure functions: DataFrame in, go.Figure out. No Streamlit imports.
Tokens aligned with app.py CSS vars for visual consistency.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Design tokens — must match app.py CSS :root vars exactly ─────────────────
T = {
    "bg":      "#0A0A0A",
    "surface": "#111111",
    "surf2":   "#171717",
    "border":  "#2A2A2A",
    "border2": "#333333",
    "text":    "#E8E3D8",
    "muted":   "#6B6456",
    "blue":    "#2980B9",
    "green":   "#27AE60",
    "red":     "#C0392B",
    "amber":   "#C9A84C",
    "purple":  "#8E44AD",
}

_BASE = dict(
    paper_bgcolor=T["bg"],
    plot_bgcolor=T["surface"],
    font=dict(family="IBM Plex Mono, monospace", color=T["text"], size=11),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(
        bgcolor=T["surf2"],
        bordercolor=T["border2"],
        borderwidth=1,
        font=dict(size=10),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1,
    ),
    xaxis=dict(
        gridcolor=T["border"], showgrid=True,
        zeroline=False, tickfont=dict(color=T["muted"], size=10),
        showspikes=True, spikecolor=T["border2"],
        spikethickness=1, spikedash="dot",
    ),
    yaxis=dict(
        gridcolor=T["border"], showgrid=True,
        zeroline=False, tickfont=dict(color=T["muted"], size=10),
    ),
    hoverlabel=dict(
        bgcolor=T["surf2"], bordercolor=T["border2"],
        font=dict(color=T["text"], family="IBM Plex Mono, monospace", size=11),
    ),
    hovermode="x unified",
)


def _fig(title: str, height: int = 480, rows: int = 1, row_heights=None,
         shared_x: bool = True) -> go.Figure:
    """Create a base figure with shared layout."""
    if rows > 1:
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=shared_x,
            row_heights=row_heights or ([1/rows]*rows),
            vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()

    layout = dict(**_BASE)
    layout.update(dict(
        height=height,
        title=dict(text=title, font=dict(size=12, color=T["muted"]),
                   x=0, xanchor="left", pad=dict(l=4)),
    ))
    fig.update_layout(**layout)
    return fig


def price_chart(df: pd.DataFrame, short: int, long: int) -> go.Figure:
    """Candlestick + MA overlays + Bollinger + buy/sell markers + volume."""
    has_volume = "Volume" in df.columns and df["Volume"].notna().any()
    row_h = [0.72, 0.28] if has_volume else [1.0]
    n_rows = 2 if has_volume else 1
    fig = _fig("", height=520, rows=n_rows, row_heights=row_h)

    # Candlestick
    has_ohlc = all(c in df.columns for c in ["Open","High","Low","Close"])
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"],  close=df["Close"],
            name="Price",
            increasing=dict(line=dict(color=T["green"], width=1),
                           fillcolor=T["green"]),
            decreasing=dict(line=dict(color=T["red"],   width=1),
                           fillcolor=T["red"]),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"], name="Price",
            line=dict(color=T["blue"], width=1.5),
        ), row=1, col=1)

    # Bollinger Bands
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_Upper"],
            line=dict(color=T["muted"], width=0.6, dash="dash"),
            name="BB", showlegend=True, legendgroup="bb",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["BB_Lower"],
            line=dict(color=T["muted"], width=0.6, dash="dash"),
            fill="tonexty", fillcolor="rgba(92,122,148,0.05)",
            name="BB lower", showlegend=False, legendgroup="bb",
        ), row=1, col=1)

    # Moving averages
    short_col, long_col = f"SMA_{short}", f"SMA_{long}"
    if short_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[short_col], name=f"SMA {short}",
            line=dict(color=T["amber"], width=1.2, dash="dot"),
        ), row=1, col=1)
    if long_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[long_col], name=f"SMA {long}",
            line=dict(color=T["purple"], width=1.2, dash="dot"),
        ), row=1, col=1)

    # Buy / sell signals
    if "Position" in df.columns:
        buys  = df[df["Position"] == 2]
        sells = df[df["Position"] == -2]
        if len(buys):
            fig.add_trace(go.Scatter(
                x=buys["Date"], y=buys["Close"], mode="markers",
                marker=dict(symbol="triangle-up", size=9,
                            color=T["green"], line=dict(color="#fff", width=0.8)),
                name="Buy",
            ), row=1, col=1)
        if len(sells):
            fig.add_trace(go.Scatter(
                x=sells["Date"], y=sells["Close"], mode="markers",
                marker=dict(symbol="triangle-down", size=9,
                            color=T["red"], line=dict(color="#fff", width=0.8)),
                name="Sell",
            ), row=1, col=1)

    # Volume
    if has_volume:
        vol_colors = []
        for i in range(len(df)):
            o = df["Open"].iloc[i]  if "Open" in df.columns else df["Close"].iloc[i]
            c = df["Close"].iloc[i]
            vol_colors.append(T["green"] if c >= o else T["red"])
        fig.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=vol_colors, opacity=0.5,
            name="Volume", showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         tickfont=dict(size=9, color=T["muted"]))

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     tickprefix="$", tickfont=dict(size=10))
    return fig


def equity_chart(df: pd.DataFrame) -> go.Figure:
    """Strategy vs buy-and-hold equity curves."""
    fig = _fig("Equity Curve · $10,000 initial", height=360)

    if "Equity_Strategy" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Equity_Strategy"],
            name="MA Strategy",
            line=dict(color=T["blue"], width=2),
            fill="tozeroy", fillcolor="rgba(68,147,248,0.05)",
        ))
    if "Equity_BuyHold" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Equity_BuyHold"],
            name="Buy & Hold",
            line=dict(color=T["muted"], width=1.2, dash="dot"),
        ))

    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI with overbought/oversold bands."""
    fig = _fig("RSI (14)", height=220)
    if "RSI" not in df.columns:
        return fig

    fig.add_hrect(y0=70, y1=100, fillcolor=T["red"],   opacity=0.04, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=T["green"], opacity=0.04, line_width=0)
    fig.add_hline(y=70, line=dict(color=T["red"],   width=0.8, dash="dash"))
    fig.add_hline(y=30, line=dict(color=T["green"], width=0.8, dash="dash"))
    fig.add_hline(y=50, line=dict(color=T["muted"], width=0.6, dash="dot"))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"],
        name="RSI",
        line=dict(color=T["amber"], width=1.5),
        fill="tozeroy", fillcolor="rgba(227,179,65,0.04)",
    ))

    fig.update_layout(yaxis=dict(range=[0,100]), showlegend=False)
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal, histogram."""
    fig = _fig("MACD", height=220)
    if "MACD" not in df.columns:
        return fig

    hist = df["MACD_Hist"].fillna(0) if "MACD_Hist" in df.columns else \
           (df["MACD"] - df["MACD_Signal"])
    colors = [T["green"] if v >= 0 else T["red"] for v in hist]

    fig.add_trace(go.Bar(
        x=df["Date"], y=hist,
        name="Histogram", marker_color=colors, opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD"],
        name="MACD", line=dict(color=T["blue"], width=1.4),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD_Signal"],
        name="Signal", line=dict(color=T["red"], width=1.4),
    ))
    return fig


def volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Rolling annualised volatility."""
    fig = _fig("Rolling Volatility (21d Ann.)", height=220)
    if "Rolling_Vol" not in df.columns:
        return fig

    avg = df["Rolling_Vol"].mean()
    fig.add_hline(y=avg,
                  line=dict(color=T["muted"], width=0.8, dash="dot"),
                  annotation_text=f"avg {avg:.1f}%",
                  annotation_font=dict(color=T["muted"], size=9))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Rolling_Vol"],
        name="Volatility",
        line=dict(color=T["purple"], width=1.4),
        fill="tozeroy", fillcolor="rgba(163,113,247,0.06)",
    ))
    fig.update_yaxes(ticksuffix="%")
    return fig


def forecast_chart(df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """Historical + Monte Carlo forecast with confidence interval."""
    fig = _fig("AI Price Forecast · Monte Carlo", height=400)

    hist = df.tail(90)
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"],
        name="Historical",
        line=dict(color=T["blue"], width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
        y=pd.concat([forecast_df["Upper"], forecast_df["Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(227,179,65,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(color=T["amber"], width=2, dash="dot"),
    ))

    fig.update_yaxes(tickprefix="$")
    return fig
