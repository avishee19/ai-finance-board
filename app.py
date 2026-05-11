"""
app.py — AI Finance Dashboard Pro  v2.0
Pure UI entry point. All business logic lives in src/.
Architecture: secrets-first API key, cached session state, demo-mode fallback.
"""

import streamlit as st
import pandas as pd
import time, logging

from src.data_loader   import fetch_live, load_csv
from src.processing    import (add_moving_averages, add_macd, add_rsi,
                               add_bollinger_bands, add_volatility,
                               add_signals, run_backtest, compute_summary_stats)
from src.signals       import detect_crossovers, generate_insights
from src.llm_insights  import generate_llm_analysis
from src.visualization import (price_chart, equity_chart, rsi_chart,
                               macd_chart, forecast_chart, volatility_chart)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Finance Board",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
    --bg:#0A0A0A; --surf:#111111; --surf2:#171717; --surf3:#1C1C1C;
    --border:#2A2A2A; --border2:#333333;
    --text:#E8E3D8; --muted:#6B6456;
    --gold:#C9A84C; --gold2:#E8C97A; --white:#F0EBE0;
    --red:#C0392B; --green:#27AE60; --blue:#2980B9;
}
html,body,[class*="css"],.main,.block-container {
    background:var(--bg) !important; color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}
.block-container{padding:0 !important;max-width:100% !important;}
.main>div{padding:0 !important;}
header[data-testid="stHeader"]{display:none !important;}
#MainMenu{display:none !important;}
footer{display:none !important;}
#root>div:first-child{padding-top:0 !important;}
[data-testid="stSidebar"]{background:#0D0D0D !important;border-right:1px solid var(--border) !important;}
[data-testid="stSidebar"] *{color:var(--text) !important;}
[data-testid="stSidebar"] section{padding:0.5rem 1rem !important;}
.sidebar-label{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.25em;
    text-transform:uppercase;color:var(--muted) !important;padding:16px 0 6px;display:block;
    border-top:1px solid var(--border);margin-top:4px;}
.stTextInput input,input[type="text"],input[type="password"]{
    background:var(--surf2) !important;border:1px solid var(--border2) !important;
    color:var(--text) !important;border-radius:4px !important;
    font-family:'JetBrains Mono',monospace !important;font-size:12px !important;padding:8px 10px !important;}
.stTextInput input:focus{border-color:var(--gold) !important;box-shadow:0 0 0 2px rgba(201,168,76,0.15) !important;}
.stTextInput input::placeholder{color:var(--muted) !important;}
.stSelectbox>div>div{background:var(--surf2) !important;border:1px solid var(--border2) !important;
    color:var(--text) !important;border-radius:4px !important;font-size:12px !important;}
.stButton>button{background:transparent !important;border:1px solid var(--border2) !important;
    color:var(--muted) !important;font-family:'JetBrains Mono',monospace !important;
    font-size:9px !important;letter-spacing:0.12em !important;text-transform:uppercase !important;
    border-radius:3px !important;padding:9px 0 !important;width:100% !important;transition:all 0.12s !important;}
.stButton>button:hover{border-color:var(--gold) !important;color:var(--gold2) !important;
    background:rgba(201,168,76,0.05) !important;}
[data-testid="stTabs"]{border-bottom:1px solid var(--border) !important;padding:0 28px !important;}
[data-testid="stTabs"] button{font-family:'JetBrains Mono',monospace !important;font-size:9px !important;
    letter-spacing:0.14em !important;text-transform:uppercase !important;color:var(--muted) !important;
    padding:12px 18px !important;background:transparent !important;
    border-bottom:2px solid transparent !important;transition:all 0.12s !important;}
[data-testid="stTabs"] button:hover{color:var(--text) !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--gold2) !important;border-bottom-color:var(--gold) !important;}
[data-testid="stTabsContent"]{padding:24px 28px !important;}
[data-testid="stMetric"]{background:var(--surf) !important;border:1px solid var(--border) !important;
    border-radius:6px !important;padding:18px 20px !important;transition:border-color 0.15s !important;
    position:relative !important;overflow:hidden !important;}
[data-testid="stMetric"]:hover{border-color:var(--gold) !important;}
[data-testid="stMetricLabel"]{font-family:'JetBrains Mono',monospace !important;font-size:8px !important;
    letter-spacing:0.2em !important;text-transform:uppercase !important;color:var(--muted) !important;}
[data-testid="stMetricValue"]{font-family:'Inter',sans-serif !important;font-size:24px !important;
    font-weight:700 !important;color:var(--white) !important;letter-spacing:-0.02em !important;}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',monospace !important;font-size:11px !important;}
.section-hdr{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.25em;
    text-transform:uppercase;color:var(--muted);padding-bottom:10px;
    border-bottom:1px solid var(--border);margin-bottom:16px;}
.ticker-pill{display:inline-flex;align-items:center;gap:7px;
    background:rgba(201,168,76,0.08);border:1px solid rgba(201,168,76,0.3);
    border-radius:20px;padding:4px 12px 4px 8px;
    font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:500;
    color:var(--gold2);letter-spacing:0.06em;}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.dot-live{background:var(--green);box-shadow:0 0 6px var(--green);animation:pdot 2s infinite;}
.dot-static{background:var(--muted);}
@keyframes pdot{0%,100%{opacity:1}50%{opacity:0.3}}
.insight{background:var(--surf);border:1px solid var(--border);border-left:2px solid;
    border-radius:4px;padding:12px 14px;margin-bottom:8px;
    font-family:'Inter',sans-serif;font-size:12px;line-height:1.65;transition:border-color 0.12s;}
.insight:hover{border-color:var(--border2);}
.insight.pos{border-left-color:var(--green);}
.insight.neg{border-left-color:var(--red);}
.insight.neu{border-left-color:var(--muted);}
.insight-ttl{font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:500;
    letter-spacing:0.18em;text-transform:uppercase;color:var(--muted);margin-bottom:5px;}
.badge{display:inline-flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;
    font-size:8px;letter-spacing:0.1em;text-transform:uppercase;
    padding:3px 9px;border-radius:3px;border:1px solid;margin-right:5px;}
.badge-ai{color:var(--gold2);border-color:rgba(201,168,76,0.3);background:rgba(201,168,76,0.06);}
.badge-heur{color:var(--green);border-color:rgba(39,174,96,0.3);background:rgba(39,174,96,0.06);}
.badge-mode{color:var(--muted);border-color:var(--border2);background:var(--surf2);}
.badge-demo{color:var(--gold);border-color:rgba(201,168,76,0.3);background:rgba(201,168,76,0.06);}
.analysis-wrap{background:var(--surf);border:1px solid var(--border);border-radius:6px;
    padding:24px 28px;font-family:'Inter',sans-serif;font-size:13px;line-height:1.8;}
.analysis-wrap h2{font-family:'Inter',sans-serif;font-size:15px;font-weight:600;color:var(--white);
    border-bottom:1px solid var(--border);padding-bottom:10px;margin:0 0 16px;}
.analysis-wrap h3{font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.18em;
    text-transform:uppercase;color:var(--muted);margin:20px 0 8px;}
.analysis-wrap table{width:100%;border-collapse:collapse;font-size:12px;margin:10px 0;}
.analysis-wrap th{text-align:left;padding:6px 12px;background:var(--surf2);color:var(--muted);
    font-size:8px;letter-spacing:0.14em;text-transform:uppercase;border-bottom:1px solid var(--border);}
.analysis-wrap td{padding:8px 12px;border-bottom:1px solid var(--border);color:var(--text);}
.analysis-wrap tr:hover td{background:rgba(201,168,76,0.03);}
.analysis-meta{display:flex;justify-content:space-between;padding-top:12px;margin-top:10px;
    border-top:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);}
.cx-row{display:flex;justify-content:space-between;align-items:center;
    padding:8px 0;border-bottom:1px solid var(--border);
    font-family:'JetBrains Mono',monospace;font-size:11px;}
.cx-golden{color:var(--gold2);}
.cx-death{color:var(--red);}
.disclaimer{background:rgba(201,168,76,0.04);border:1px solid rgba(201,168,76,0.15);
    border-radius:4px;padding:9px 13px;font-family:'JetBrains Mono',monospace;
    font-size:10px;color:rgba(201,168,76,0.7);margin:10px 0;}
.empty-state{display:flex;flex-direction:column;align-items:center;
    justify-content:center;min-height:55vh;color:var(--muted);}
.empty-icon{font-size:40px;margin-bottom:16px;opacity:0.2;}
.empty-text{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.2em;text-transform:uppercase;}
hr{border-color:var(--border) !important;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
[data-testid="stAlert"]{font-family:'Inter',sans-serif !important;font-size:12px !important;border-radius:4px !important;}
[data-testid="stDataFrame"]{border:1px solid var(--border) !important;border-radius:6px !important;}
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = dict(df=None, ticker_label="", stats=None,
                 forecast_df=None, forecast_meta=None,
                 llm_result=None, llm_analysis_key=None,
                 demo_mode=False)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Secrets-first API key resolution ──────────────────────────────────────────
def _get_secret_key() -> str:
    """Return API key from st.secrets if available — empty string otherwise."""
    try:
        k = st.secrets.get("ANTHROPIC_API_KEY", "")
        return k if k and k.startswith("sk-") else ""
    except Exception:
        return ""

_HAS_SECRET = bool(_get_secret_key())

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:16px 28px 14px;border-bottom:1px solid #2A2A2A;background:#0D0D0D;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div>
      <div style="font-family:'Inter',sans-serif;font-size:18px;font-weight:700;
      color:#F0EBE0;letter-spacing:-0.02em;">
        AI Finance <span style="color:#C9A84C;">Board</span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.2em;
      text-transform:uppercase;color:#6B6456;margin-top:5px;">
        Analysis &middot; Backtesting &middot; Hybrid AI Intelligence
      </div>
    </div>
    <div style="display:flex;gap:6px;">
      <span style="background:rgba(39,174,96,0.08);border:1px solid rgba(39,174,96,0.25);
      border-radius:3px;padding:3px 10px;font-family:'JetBrains Mono',monospace;
      font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:#27AE60;">Live</span>
      <span style="background:rgba(201,168,76,0.08);border:1px solid rgba(201,168,76,0.25);
      border-radius:3px;padding:3px 10px;font-family:'JetBrains Mono',monospace;
      font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:#C9A84C;">Hybrid AI</span>
      <span style="background:rgba(41,128,185,0.08);border:1px solid rgba(41,128,185,0.25);
      border-radius:3px;padding:3px 10px;font-family:'JetBrains Mono',monospace;
      font-size:8px;letter-spacing:0.12em;text-transform:uppercase;color:#2980B9;">Monte Carlo</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Data source
    st.markdown('<span class="sidebar-label">Data Source</span>',
                unsafe_allow_html=True)
    data_source = st.radio("", ["Live (yfinance)", "Upload CSV"],
                           horizontal=True, label_visibility="collapsed")

    if data_source == "Live (yfinance)":
        ticker   = st.text_input("", "AAPL", placeholder="Ticker…",
                                 label_visibility="collapsed").strip().upper()
        period   = st.selectbox("", ["6mo","1y","2y","5y"], index=1,
                                label_visibility="collapsed")
        fetch_btn = st.button("↓ Fetch Live Data")
        csv_file  = None
    else:
        ticker   = "CSV"
        period   = "—"
        csv_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        fetch_btn = bool(csv_file)

    # Demo mode — visible only when live fetch is selected
    if data_source == "Live (yfinance)":
        demo_mode = st.checkbox("Demo mode (use cached CSV on failure)",
                                value=st.session_state.demo_mode)
        st.session_state.demo_mode = demo_mode
    else:
        demo_mode = False

    # Indicators
    st.markdown('<span class="sidebar-label">Indicators</span>',
                unsafe_allow_html=True)
    short_ma = st.slider("Short MA", 5,  50,  20)
    long_ma  = st.slider("Long MA",  20, 200, 50)
    c1,c2,c3 = st.columns(3)
    show_bb   = c1.checkbox("BB",   True,  help="Bollinger Bands")
    show_rsi  = c2.checkbox("RSI",  True,  help="RSI 14")
    show_macd = c3.checkbox("MACD", True)

    # Forecast
    st.markdown('<span class="sidebar-label">AI Forecast</span>',
                unsafe_allow_html=True)
    forecast_days    = st.slider("Horizon", 10, 90, 30)
    run_forecast_btn = st.button("↻ Run Forecast")

    # Claude Analysis
    st.markdown('<span class="sidebar-label">Claude Analysis</span>',
                unsafe_allow_html=True)

    # Only show key input if no secret is configured
    if _HAS_SECRET:
        st.caption("🔒 API key loaded from secrets")
        sidebar_key = ""
    else:
        sidebar_key = st.text_input("API Key", type="password",
                                    placeholder="sk-ant-…  (optional)",
                                    help="console.anthropic.com → API Keys",
                                    label_visibility="collapsed")
        st.caption("No key? Free heuristic report auto-generates.")

    analysis_type = st.selectbox("", ["full","quick","momentum","risk"],
        format_func=lambda x: {"full":"Full Report","quick":"Quick Summary",
                               "momentum":"Momentum","risk":"Risk Assessment"}[x],
        label_visibility="collapsed")
    run_analysis_btn = st.button("→ Generate Analysis")

    # Live mode
    st.markdown('<span class="sidebar-label">Live Mode</span>',
                unsafe_allow_html=True)
    live_mode = st.checkbox("Auto-refresh (30s)")
    if live_mode and data_source != "Live (yfinance)":
        live_mode = False

# ── Processing pipeline ────────────────────────────────────────────────────────
def _run_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = add_moving_averages(raw_df, short_ma, long_ma)
    raw_df = add_macd(raw_df)
    raw_df = add_rsi(raw_df)
    if show_bb:
        raw_df = add_bollinger_bands(raw_df)
    raw_df = add_volatility(raw_df)
    raw_df = add_signals(raw_df, short_ma, long_ma)
    raw_df = run_backtest(raw_df)
    return raw_df

# ── Data loading ───────────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner("Loading…"):
        try:
            if data_source == "Live (yfinance)":
                raw = fetch_live(ticker, period)
                st.session_state.ticker_label = ticker
            else:
                raw = load_csv(csv_file)
                st.session_state.ticker_label = "CSV"

            st.session_state.df           = _run_pipeline(raw)
            st.session_state.stats        = compute_summary_stats(st.session_state.df)
            st.session_state.forecast_df  = None
            st.session_state.llm_result   = None
            st.session_state.llm_analysis_key = None
            st.success(f"Loaded {len(st.session_state.df):,} rows · {st.session_state.ticker_label}")

        except (ValueError, ConnectionError) as e:
            if demo_mode:
                try:
                    raw = load_csv("stock_data.csv")
                    st.session_state.df           = _run_pipeline(raw)
                    st.session_state.stats        = compute_summary_stats(st.session_state.df)
                    st.session_state.ticker_label = "DEMO"
                    st.session_state.llm_result   = None
                    st.warning(f"Live fetch failed — loaded demo data. ({e})")
                except Exception as e2:
                    st.error(f"Demo fallback also failed: {e2}")
                    st.stop()
            else:
                st.error(str(e))
                st.stop()
        except Exception as e:
            logger.exception("Load error")
            st.error(f"Unexpected error: {e}")
            st.stop()

# ── Live mode ──────────────────────────────────────────────────────────────────
if live_mode and st.session_state.df is not None:
    time.sleep(30)
    st.rerun()

# ── Guard ──────────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">📈</div>
      <div class="empty-text">Select a data source and fetch data to begin</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

df    = st.session_state.df
stats = st.session_state.stats or {}
label = st.session_state.ticker_label

# ── Forecast ───────────────────────────────────────────────────────────────────
if run_forecast_btn:
    with st.spinner("Running Monte Carlo simulation…"):
        try:
            from src.prediction import train_and_forecast
            fdf, r2, da = train_and_forecast(df, horizon=forecast_days)
            st.session_state.forecast_df   = fdf
            st.session_state.forecast_meta = {"r2": r2, "dir_acc": da}
        except Exception as e:
            st.error(f"Forecast failed: {e}")

# ── Analysis — cached by (ticker, analysis_type, engine) ──────────────────────
if run_analysis_btn:
    # Build a cache key — skip API call if same params already computed
    eff_key    = _get_secret_key() or sidebar_key or ""
    cache_key  = f"{label}|{analysis_type}|{bool(eff_key)}"
    if st.session_state.llm_analysis_key != cache_key:
        with st.spinner("Generating analysis…"):
            result = generate_llm_analysis(
                df=df, stats=stats, ticker=label,
                short=short_ma, long=long_ma,
                analysis_type=analysis_type,
                api_key=sidebar_key or None,
            )
            st.session_state.llm_result       = result
            st.session_state.llm_analysis_key = cache_key
    else:
        st.toast("Using cached analysis — same ticker & mode", icon="💾")

# ── Ticker info bar ────────────────────────────────────────────────────────────
d_from  = df["Date"].iloc[0].strftime("%b %d %Y")
d_to    = df["Date"].iloc[-1].strftime("%b %d %Y")
is_live = data_source == "Live (yfinance)" and label != "DEMO"
dot_cls = "dot-live" if is_live else "dot-static"
is_demo = label == "DEMO"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
  <span class="ticker-pill">
    <span class="dot {dot_cls}"></span>{label}
  </span>
  <span style="font-size:10px;color:var(--muted)">
    {len(df):,} days &nbsp;·&nbsp; {d_from} → {d_to}
  </span>
  {'<span class="badge badge-demo">Demo Data</span>' if is_demo else ""}
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_sum,tab_chart,tab_ind,tab_strat,tab_ai,tab_llm,tab_data = st.tabs([
    "Summary","Price Chart","Indicators","Strategy",
    "AI Forecast","Claude Analysis","Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_sum:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Latest Price",    f"${stats.get('latest_price',0):,.2f}")
    c2.metric("Total Return",    f"{stats.get('total_return',0):+.2f}%")
    c3.metric("Ann. Volatility", f"{stats.get('ann_volatility',0):.2f}%")
    c4.metric("Sharpe Ratio",    f"{stats.get('sharpe_ratio',0):.2f}")
    c5.metric("Max Drawdown",    f"{stats.get('max_drawdown',0):.2f}%")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])

    with col_l:
        st.markdown('<div class="section-hdr">Automated Insights</div>',
                    unsafe_allow_html=True)
        insights = generate_insights(df, stats, short_ma, long_ma)
        if insights:
            for ins in insights:
                cls = {"positive":"pos","negative":"neg","neutral":"neu"}.get(ins["sentiment"],"neu")
                st.markdown(f"""
                <div class="insight {cls}">
                  <div class="insight-ttl">{ins['title']}</div>
                  <div>{ins['body']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:var(--muted);font-size:12px;'>"
                        "No significant signals detected.</div>",
                        unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-hdr">Crossover Events</div>',
                    unsafe_allow_html=True)
        crossovers = detect_crossovers(df, short_ma, long_ma)
        if crossovers:
            for c in crossovers:
                is_g  = "Golden" in c["type"]
                cls   = "cx-golden" if is_g else "cx-death"
                arrow = "↑" if is_g else "↓"
                st.markdown(f"""
                <div class="cx-row">
                  <span class="{cls}">{arrow} {c['type']}</span>
                  <span style="color:var(--muted)">{c['date']}</span>
                  <span>{c['price']}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:var(--muted);font-size:11px;"
                        "padding:12px 0'>No crossovers in period</div>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRICE CHART
# ══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.plotly_chart(price_chart(df, short_ma, long_ma),
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ind:
    if show_rsi:
        st.plotly_chart(rsi_chart(df), use_container_width=True)
    if show_macd:
        st.plotly_chart(macd_chart(df), use_container_width=True)
    st.plotly_chart(volatility_chart(df), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
with tab_strat:
    st.markdown('<div class="section-hdr">MA Crossover Backtest</div>',
                unsafe_allow_html=True)
    initial = 10_000
    final_s  = df["Equity_Strategy"].dropna().iloc[-1] if "Equity_Strategy" in df.columns else initial
    final_b  = df["Equity_BuyHold"].dropna().iloc[-1]  if "Equity_BuyHold"  in df.columns else initial
    n_trades = int((df["Position"].abs() == 2).sum())  if "Position" in df.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Strategy Value", f"${final_s:,.2f}", f"${final_s-initial:+,.0f}")
    c2.metric("Buy & Hold",     f"${final_b:,.2f}", f"${final_b-initial:+,.0f}")
    c3.metric("Alpha",          f"${final_s-final_b:+,.2f}")
    c4.metric("Trades",         str(n_trades))

    st.plotly_chart(equity_chart(df), use_container_width=True)
    st.markdown("""<div class="disclaimer">
    ⚠ Simulation only · No transaction costs or taxes modelled ·
    Past performance does not indicate future results
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AI FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    fdf = st.session_state.forecast_df
    if fdf is None:
        st.info("Click **↻ Run Forecast** in the sidebar.")
    else:
        meta = st.session_state.forecast_meta or {}
        c1,c2,c3 = st.columns(3)
        c1.metric("Horizon",      f"{forecast_days} days")
        c2.metric("Model R²",     f"{meta.get('r2',0):.3f}")
        c3.metric("Dir. Acc.",    f"{meta.get('dir_acc',0):.1%}")

        st.markdown("""<div class="disclaimer">
        ⚠ Monte Carlo simulation · 80% confidence interval · Not a prediction
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(forecast_chart(df, fdf), use_container_width=True)

        cur   = stats.get("latest_price",0)
        end   = fdf["Forecast"].iloc[-1]
        chg   = (end/cur-1)*100 if cur else 0
        arrow = "↑" if chg > 0 else "↓"
        c     = "var(--green)" if chg > 0 else "var(--red)"
        st.markdown(f"""
        <div style="font-size:11px;color:var(--muted);padding:6px 0">
          {forecast_days}d target&nbsp;
          <span style="color:var(--text)">${end:.2f}</span>&nbsp;
          <span style="color:{c}">{arrow} {chg:+.1f}%</span>&nbsp;·&nbsp;
          80% CI: ${fdf['Lower'].iloc[-1]:.2f} – ${fdf['Upper'].iloc[-1]:.2f}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_llm:
    st.markdown('<div class="section-hdr">Hybrid AI / Heuristic Analysis</div>',
                unsafe_allow_html=True)
    llm = st.session_state.llm_result

    if llm is None:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size:11px;color:var(--muted);line-height:1.9">
            Click <strong style="color:var(--text)">→ Generate Analysis</strong>
            in the sidebar.<br><br>
            <strong style="color:var(--text)">With API key</strong> →
            Claude writes a natural language analyst brief.<br>
            <strong style="color:var(--text)">Without API key</strong> →
            Heuristic engine generates a full Markdown report at $0.00.<br><br>
            Results are <strong style="color:var(--text)">cached</strong> —
            switching tabs won't trigger a new call.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        text       = llm.get("analysis","")
        tokens     = llm.get("tokens")
        snapshot   = llm.get("snapshot",{})
        engine     = llm.get("engine","heuristic")
        soft_err   = llm.get("error")

        type_labels = {"full":"Full Report","quick":"Quick Summary",
                       "momentum":"Momentum","risk":"Risk Assessment"}

        if engine == "claude":
            eb = '<span class="badge badge-ai">🤖 AI-Generated</span>'
        else:
            eb = '<span class="badge badge-heur">⚙ Heuristic · $0.00</span>'
        mb = f'<span class="badge badge-mode">{type_labels.get(analysis_type,"")}</span>'
        st.markdown(f"<div style='margin-bottom:12px'>{eb}{mb}</div>",
                    unsafe_allow_html=True)

        if soft_err:
            st.warning(soft_err)

        # Render Markdown inside glass card
        try:
            import markdown as _md
            body_html = _md.markdown(text, extensions=["tables","nl2br"])
        except Exception:
            body_html = text.replace("\n","<br>")

        tok_str = (f"claude-sonnet-4-6 · {tokens['input']:,} in · {tokens['output']:,} out"
                   if tokens else "local engine · 0 tokens · $0.00")
        date_str = snapshot.get("date_range",{}).get("end","")

        st.markdown(f"""
        <div class="analysis-wrap">
          {body_html}
          <div class="analysis-meta">
            <span>{tok_str}</span><span>{date_str}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        c1,c2 = st.columns([1,5])
        with c1:
            st.download_button("↓ .md",
                data=text,
                file_name=f"{label}_{analysis_type}_{engine}.md",
                mime="text/markdown")
        with st.expander("Data snapshot sent to engine"):
            st.json(snapshot)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown(f'<div class="section-hdr">{len(df):,} rows · last 100 shown</div>',
                unsafe_allow_html=True)
    cols = [c for c in ["Date","Open","High","Low","Close","Volume",
                         f"SMA_{short_ma}",f"SMA_{long_ma}","RSI",
                         "MACD","BB_Upper","BB_Lower","Rolling_Vol","Signal"]
            if c in df.columns]
    st.dataframe(df[cols].tail(100).reset_index(drop=True),
                 use_container_width=True, height=460)
    st.download_button("↓ Download CSV",
        data=df[cols].to_csv(index=False),
        file_name=f"{label}_processed.csv", mime="text/csv")
