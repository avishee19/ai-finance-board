"""
llm_insights.py — Hybrid AI/Heuristic stock analysis engine.

Priority chain for API key resolution:
  1. st.secrets["ANTHROPIC_API_KEY"]   (Streamlit Cloud deployment)
  2. ANTHROPIC_API_KEY environment var  (local .env / system)
  3. api_key= argument                  (sidebar user input)

If no key is found OR the API call fails for any reason,
get_heuristic_analysis() fires automatically and returns a
structured Markdown report built entirely from local KPIs.
The result dict always carries an 'engine' key so the UI
can show the correct status badge.
"""

import json
import os
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Key resolution ─────────────────────────────────────────────────────────────

def _resolve_key(api_key: Optional[str]) -> Optional[str]:
    """Return first valid key found across all sources. Never raises."""
    try:
        import streamlit as st
        secret = st.secrets.get("ANTHROPIC_API_KEY")
        if secret and secret.startswith("sk-"):
            return secret
    except Exception:
        pass
    env = os.getenv("ANTHROPIC_API_KEY", "")
    if env and env.startswith("sk-"):
        return env
    if api_key and api_key.strip().startswith("sk-"):
        return api_key.strip()
    return None


# ── Snapshot builder ───────────────────────────────────────────────────────────

def _build_snapshot(df, stats, ticker, short, long):
    close = df["Close"]
    snapshot = {
        "ticker": ticker,
        "period_days": len(df),
        "date_range": {"start": str(df["Date"].iloc[0].date()), "end": str(df["Date"].iloc[-1].date())},
        "price": {
            "current":   round(float(close.iloc[-1]), 2),
            "week_ago":  round(float(close.iloc[-5]),  2) if len(close) >= 5  else None,
            "month_ago": round(float(close.iloc[-21]), 2) if len(close) >= 21 else None,
            "high_52w":  round(float(close.tail(252).max()), 2),
            "low_52w":   round(float(close.tail(252).min()), 2),
        },
        "returns": {
            "total_pct":        round(stats.get("total_return",   0), 2),
            "ann_volatility":   round(stats.get("ann_volatility", 0), 2),
            "sharpe_ratio":     round(stats.get("sharpe_ratio",   0), 3),
            "max_drawdown_pct": round(stats.get("max_drawdown",   0), 2),
        },
        "moving_averages": {}, "oscillators": {}, "signals": {}, "recent_price_action": [],
    }
    short_col, long_col = f"SMA_{short}", f"SMA_{long}"
    if short_col in df.columns and long_col in df.columns:
        sma_s = float(df[short_col].dropna().iloc[-1])
        sma_l = float(df[long_col].dropna().iloc[-1])
        snapshot["moving_averages"] = {
            f"SMA_{short}": round(sma_s, 2), f"SMA_{long}": round(sma_l, 2),
            "trend": "bullish" if sma_s > sma_l else "bearish",
            "gap_pct": round((sma_s - sma_l) / sma_l * 100, 2),
        }
    if "RSI" in df.columns:
        rsi = float(df["RSI"].dropna().iloc[-1])
        snapshot["oscillators"]["RSI_14"] = {
            "value": round(rsi, 1),
            "zone": "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral"),
        }
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd = float(df["MACD"].dropna().iloc[-1])
        sig  = float(df["MACD_Signal"].dropna().iloc[-1])
        hist = float(df["MACD_Hist"].dropna().iloc[-1]) if "MACD_Hist" in df.columns else macd - sig
        snapshot["oscillators"]["MACD"] = {
            "macd": round(macd, 4), "signal": round(sig, 4), "histogram": round(hist, 4),
            "bias": "bullish" if macd > sig else "bearish",
        }
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        price = float(close.iloc[-1])
        upper = float(df["BB_Upper"].dropna().iloc[-1])
        lower = float(df["BB_Lower"].dropna().iloc[-1])
        mid   = float(df["BB_Mid"].dropna().iloc[-1]) if "BB_Mid" in df.columns else (upper + lower) / 2
        bb_pct = (price - lower) / (upper - lower) * 100 if upper != lower else 50
        snapshot["oscillators"]["Bollinger"] = {
            "upper": round(upper, 2), "middle": round(mid, 2), "lower": round(lower, 2),
            "pct_b": round(bb_pct, 1),
            "position": "near upper band" if bb_pct > 80 else ("near lower band" if bb_pct < 20 else "middle zone"),
        }
    if "Rolling_Vol" in df.columns:
        vol_s = df["Rolling_Vol"].dropna()
        c_vol, a_vol = float(vol_s.iloc[-1]), float(vol_s.mean())
        snapshot["returns"].update({
            "current_rolling_vol_ann": round(c_vol, 2),
            "avg_rolling_vol_ann": round(a_vol, 2),
            "vol_regime": "high" if c_vol > a_vol * 1.4 else ("low" if c_vol < a_vol * 0.7 else "normal"),
        })
    if "Position" in df.columns:
        recent = df[df["Position"].isin([2, -2])].tail(3)
        snapshot["signals"]["recent_crossovers"] = [
            {"date": str(r["Date"].date()),
             "type": "golden_cross" if r["Position"] == 2 else "death_cross",
             "price": round(float(r["Close"]), 2)}
            for _, r in recent.iterrows()
        ]
        last_sig = int(df["Signal"].iloc[-1])
        snapshot["signals"]["current_position"] = (
            "long" if last_sig == 1 else ("short" if last_sig == -1 else "flat")
        )
    snapshot["recent_price_action"] = [
        {"date": str(r["Date"].date()), "close": round(float(r["Close"]), 2)}
        for _, r in df.tail(5).iterrows()
    ]
    return snapshot


# ── Heuristic engine ───────────────────────────────────────────────────────────

def get_heuristic_analysis(snapshot: dict, analysis_type: str = "full") -> str:
    """Build a structured Markdown report from KPIs with zero API calls."""
    ticker  = snapshot.get("ticker", "N/A")
    price   = snapshot.get("price", {})
    ret     = snapshot.get("returns", {})
    ma      = snapshot.get("moving_averages", {})
    osc     = snapshot.get("oscillators", {})
    signals = snapshot.get("signals", {})
    dates   = snapshot.get("date_range", {})

    current   = price.get("current", 0)
    total_ret = ret.get("total_pct", 0)
    sharpe    = ret.get("sharpe_ratio", 0)
    ann_vol   = ret.get("ann_volatility", 0)
    max_dd    = ret.get("max_drawdown_pct", 0)
    vol_reg   = ret.get("vol_regime", "normal")
    avg_vol   = ret.get("avg_rolling_vol_ann", ann_vol)

    trend     = ma.get("trend", "unknown")
    gap_pct   = ma.get("gap_pct", 0)
    position  = signals.get("current_position", "flat")
    crossovers = signals.get("recent_crossovers", [])

    rsi_val  = osc.get("RSI_14", {}).get("value", 50)
    rsi_zone = osc.get("RSI_14", {}).get("zone", "neutral")
    macd_bias = osc.get("MACD", {}).get("bias", "neutral")
    macd_hist = osc.get("MACD", {}).get("histogram", 0)
    bb_pos    = osc.get("Bollinger", {}).get("position", "middle zone")
    bb_pct_b  = osc.get("Bollinger", {}).get("pct_b", 50)

    # Ratings
    sharpe_label = ("exceptional" if sharpe > 2 else "strong" if sharpe > 1
                    else "moderate" if sharpe > 0.5 else "weak" if sharpe > 0 else "negative")
    vol_label    = ("very high" if ann_vol > 40 else "elevated" if ann_vol > 25
                    else "moderate" if ann_vol > 15 else "low")
    dd_severity  = ("severe" if max_dd < -40 else "significant" if max_dd < -20
                    else "moderate" if max_dd < -10 else "mild")

    # Icons
    trend_icon  = "📈" if trend == "bullish" else "📉"
    ret_icon    = "🟢" if total_ret > 0 else "🔴"
    sharpe_icon = "✅" if sharpe > 1 else ("⚠️" if sharpe > 0 else "🔴")
    vol_icon    = "🔴" if vol_reg == "high" else ("🟢" if vol_reg == "low" else "🟡")
    rsi_icon    = "🔴" if rsi_zone == "overbought" else ("🟢" if rsi_zone == "oversold" else "🟡")
    macd_icon   = "📈" if macd_bias == "bullish" else "📉"

    # Momentum score
    score = 5.0
    score += 1.5 if trend == "bullish" else -1.5
    score += 1.0 if rsi_zone == "oversold" else (-0.5 if rsi_zone == "overbought" else 0)
    score += 1.0 if macd_bias == "bullish" else -0.5
    score += 0.5 if macd_hist > 0 else 0
    score += -0.5 if bb_pct_b > 80 else (0.5 if bb_pct_b < 20 else 0)
    score = round(max(1.0, min(10.0, score)), 1)
    mbar  = "█" * int(score) + "░" * (10 - int(score))

    # Crossover events text
    if crossovers:
        cx_lines = "\n".join(
            f"  - **{c['type'].replace('_',' ').title()}** · {c['date']} · ${c['price']:,.2f}"
            for c in crossovers
        )
        cx_text = f"\n**Recent crossover events:**\n{cx_lines}"
    else:
        cx_text = "\n*No crossover events detected in this period.*"

    # Risk flags
    risk_items = []
    if rsi_zone == "overbought":
        risk_items.append(f"RSI {rsi_val:.1f} — overbought, momentum exhaustion possible")
    if vol_reg == "high":
        risk_items.append(f"Volatility elevated at {ann_vol:.1f}% vs average {avg_vol:.1f}%")
    if max_dd < -20:
        risk_items.append(f"Max drawdown {max_dd:.1f}% — {dd_severity} historical risk")
    if sharpe < 0:
        risk_items.append(f"Negative Sharpe ({sharpe:.2f}) — returns not compensating for risk")
    if not risk_items:
        risk_items.append("No major risk flags in current data")
    risk_block = "\n".join(f"  - {r}" for r in risk_items)

    exec_summary = (
        f"{ticker} is trading at **${current:,.2f}** "
        f"({dates.get('start','?')} – {dates.get('end','?')}) "
        f"with a total return of **{total_ret:+.1f}%**. "
        f"The MA crossover system is **{trend}** with the strategy **{position}**; "
        f"short MA sits {abs(gap_pct):.2f}% {'above' if gap_pct > 0 else 'below'} the long MA."
    )

    if analysis_type == "quick":
        return f"""## {ticker} · Quick Summary

{exec_summary}

**Key signals:** RSI {rsi_icon} {rsi_val:.1f} ({rsi_zone}) · MACD {macd_icon} {macd_bias} · BB {bb_pos}

**Outlook:** {'Bullish MA structure favours upside continuation; monitor RSI for exhaustion.' if trend == 'bullish' else 'Bearish MA structure warrants caution. Watch for a golden cross as the re-entry trigger.'}

---
*Heuristic report · {dates.get('end','today')} · No API key used*"""

    if analysis_type == "momentum":
        return f"""## {ticker} · Momentum Analysis

**RSI (14):** {rsi_icon} {rsi_val:.1f} — {rsi_zone.upper()}
{'Overbought — momentum may be nearing exhaustion; watch for divergence.' if rsi_zone == 'overbought' else 'Oversold — potential contrarian entry zone; look for confirmation.' if rsi_zone == 'oversold' else 'Neutral zone — no extreme momentum reading.'}

**MACD:** {macd_icon} {macd_bias.upper()} · Histogram {macd_hist:+.4f}
{'Positive histogram confirms accelerating bullish momentum.' if macd_hist > 0 else 'Negative histogram signals weakening or bearish momentum.'}

**Bollinger Bands:** {bb_pos.title()} (%%B = {bb_pct_b:.1f})
{'Price pressing the upper band — trend is strong but mean reversion risk rises.' if bb_pct_b > 80 else 'Price at the lower band — potential bounce zone.' if bb_pct_b < 20 else 'Price in the mid-zone — no band extreme signal.'}

**MA Crossover:** {trend_icon} {trend.upper()} · Gap {abs(gap_pct):.2f}%{cx_text}

**Momentum Score: {score}/10** `[{mbar}]`

---
*Heuristic report · {dates.get('end','today')} · No API key used*"""

    if analysis_type == "risk":
        return f"""## {ticker} · Risk Assessment

**Volatility:** {vol_icon} {ann_vol:.1f}% annualised ({vol_label})
Regime is **{vol_reg}** relative to historical average ({avg_vol:.1f}%). {'Position sizing discipline is critical at this volatility level.' if vol_reg == 'high' else 'Conditions are relatively benign.' if vol_reg == 'low' else 'Volatility is within normal range.'}

**Max Drawdown:** {max_dd:.1f}% — {dd_severity.title()} peak-to-trough decline observed over the analysis period.

**Risk-Adjusted Return — Sharpe {sharpe:.2f}:** {sharpe_icon} {sharpe_label.title()}
{'Returns have well exceeded the risk profile.' if sharpe > 1 else 'Returns are not adequately compensating for volatility taken on.' if sharpe < 0.5 else 'Marginal risk-adjusted return.'}

**Risk Flags:**
{risk_block}

**Current position signal:** {position.upper()} — {'risk-on posture.' if position == 'long' else 'defensive / out of market.'}

---
*Heuristic report · {dates.get('end','today')} · Not financial advice*"""

    # Default: full
    return f"""## {ticker} · Full Market Analysis

### Executive Summary
{exec_summary}

---

### Trend Analysis {trend_icon}
The MA system is **{trend}** with a {abs(gap_pct):.2f}% gap between the short and long averages, {'indicating meaningful trend conviction' if abs(gap_pct) > 1 else 'a narrow spread suggesting the trend may be weakening or consolidating'}.
{cx_text}

---

### Momentum & Oscillators

| Indicator | Value | Reading |
|-----------|-------|---------|
| RSI (14) | {rsi_val:.1f} | {rsi_icon} {rsi_zone.title()} |
| MACD | {osc.get('MACD',{}).get('macd',0):+.4f} | {macd_icon} {macd_bias.title()} |
| MACD Histogram | {macd_hist:+.4f} | {'Expanding' if abs(macd_hist) > 0.01 else 'Flat'} |
| Bollinger %B | {bb_pct_b:.1f} | {bb_pos.title()} |
| Momentum Score | {score}/10 | `[{mbar}]` |

---

### Volatility Assessment {vol_icon}
Annualised volatility is **{ann_vol:.1f}%** ({vol_label}). The current regime is **{vol_reg}** vs a historical average of **{avg_vol:.1f}%**. {'⚠️ Elevated volatility demands tighter risk controls.' if vol_reg == 'high' else '✅ Volatility is subdued — conditions are stable.' if vol_reg == 'low' else 'Volatility within normal historical bounds.'}

---

### Risk Profile

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Return | {ret_icon} {total_ret:+.1f}% | — |
| Sharpe Ratio | {sharpe_icon} {sharpe:.2f} | {sharpe_label.title()} |
| Max Drawdown | {max_dd:.1f}% | {dd_severity.title()} |
| Ann. Volatility | {ann_vol:.1f}% | {vol_label.title()} |

**Risk Flags:**
{risk_block}

---

### Outlook
{'The preponderance of signals is bullish: MA trend positive, MACD ' + macd_bias + ', strategy ' + position + '. Primary risk to watch: ' + ('RSI overbought — watch for momentum divergence as an early exit signal.' if rsi_zone == 'overbought' else 'a narrowing MA gap that could precede a crossover reversal.') if trend == 'bullish' else 'The bearish MA structure favours defensive positioning. MACD is ' + macd_bias + ' and the strategy is ' + position + '. Primary re-entry trigger: golden cross confirmation above the long-term average.'}

---
*Heuristic analysis from statistical indicators only. Not financial advice.*
*Period: {dates.get('start','?')} → {dates.get('end','?')}*"""


# ── Claude prompt ──────────────────────────────────────────────────────────────

def _build_prompt(snapshot: dict, analysis_type: str) -> str:
    focus_map = {
        "full":     ("Comprehensive report: (1) exec summary 2 sentences, (2) trend + MA context, "
                     "(3) momentum & oscillators, (4) volatility, (5) key risks, (6) outlook. ~300 words."),
        "quick":    "3-paragraph summary: trend, key signal, one-sentence outlook. ~120 words.",
        "risk":     "Risk focus only: volatility regime, drawdown, overbought/oversold, what could go wrong.",
        "momentum": "Momentum focus: RSI, MACD, MA crossover, Bollinger. Give momentum score 1-10 at end.",
    }
    return f"""You are a professional quantitative analyst. Analyze this data and write market commentary.

STOCK DATA:
{json.dumps(snapshot, indent=2)}

TASK: {focus_map.get(analysis_type, focus_map['full'])}

RULES: Reference specific numbers. No buy/sell recommendations. One disclaimer at end only. Write in paragraphs, not bullets. Bloomberg brief tone."""


# ── Main public function ───────────────────────────────────────────────────────

def generate_llm_analysis(
    df: pd.DataFrame,
    stats: dict,
    ticker: str,
    short: int = 20,
    long: int = 50,
    analysis_type: str = "full",
    api_key: Optional[str] = None,
) -> dict:
    """
    Generate stock analysis. Tries Claude first; auto-falls-back to heuristic.
    Result always has: analysis, snapshot, tokens, error, engine.
    """
    result = {"analysis": None, "snapshot": None, "tokens": None, "error": None, "engine": "heuristic"}

    if df is None or df.empty or len(df) < 30:
        result["error"] = "Need at least 30 data points."
        return result

    try:
        snapshot = _build_snapshot(df, stats, ticker, short, long)
        result["snapshot"] = snapshot
    except Exception as e:
        result["error"] = f"Data extraction failed: {e}"
        return result

    key = _resolve_key(api_key)

    if not key:
        result["analysis"] = get_heuristic_analysis(snapshot, analysis_type)
        result["engine"]   = "heuristic"
        return result

    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="You are a concise data-driven financial analyst. Never fabricate data not in the input.",
            messages=[{"role": "user", "content": _build_prompt(snapshot, analysis_type)}],
        )
        result["analysis"] = message.content[0].text
        result["engine"]   = "claude"
        result["tokens"]   = {
            "input":  message.usage.input_tokens,
            "output": message.usage.output_tokens,
            "total":  message.usage.input_tokens + message.usage.output_tokens,
        }
    except Exception as e:
        logger.warning(f"Claude API failed ({type(e).__name__}) — falling back to heuristic")
        result["analysis"] = get_heuristic_analysis(snapshot, analysis_type)
        result["engine"]   = "heuristic"
        result["error"]    = f"Claude unavailable ({type(e).__name__}) — showing heuristic report"

    return result
