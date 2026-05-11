# AI Finance Dashboard Pro

**Full-stack financial analysis platform** built with Python and Streamlit.
Features a cost-aware hybrid AI architecture, production-grade module structure,
and live deployment on Streamlit Cloud.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Claude API](https://img.shields.io/badge/Claude-Sonnet_4.6-8A2BE2)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22863a)](LICENSE)

---

## Features

| | Feature | Details |
|--|---------|---------|
| 📊 | **Live Data** | Any ticker via yfinance · 6mo / 1y / 2y / 5y history |
| 📁 | **CSV Upload** | Yahoo Finance export format with auto-cleaning |
| 📉 | **Technical Indicators** | SMA · EMA · MACD · RSI(14) · Bollinger Bands · Ann. Volatility |
| 🔀 | **Signal Detection** | MA crossover · Golden Cross / Death Cross events |
| 🎯 | **Strategy Backtest** | Equity curve · Buy-and-hold alpha · Trade count |
| 🤖 | **AI Forecast** | Monte Carlo (500 paths) · Ridge regression · 80% CI band |
| 🧠 | **Hybrid Analysis** | Claude API narrative **or** free local heuristic report |
| 💾 | **Session Caching** | Analysis results cached — no redundant API calls |
| 🔄 | **Demo Mode** | Auto-fallback to cached CSV if live fetch fails |

---

## Cost-Aware AI Architecture

The core engineering decision is the **Hybrid Intelligence Engine** in `src/llm_insights.py`.

```
API Key Resolution  (priority order)
────────────────────────────────────
1. st.secrets["ANTHROPIC_API_KEY"]   ← Streamlit Cloud (production)
2. ANTHROPIC_API_KEY env variable     ← local development
3. Sidebar input field                ← user-supplied at runtime

If no key found  OR  API call fails for any reason:
  → get_heuristic_analysis() activates automatically
  → Full structured Markdown report from local KPIs
  → $0.00 cost · zero latency · zero external dependency
```

**Cost per Claude API call:** ~$0.001 (≈500 input + 300 output tokens via Sonnet)
**Cost with no API key:** $0.00 — the app is fully functional without one.

This design pattern — graceful AI degradation with a capable local fallback —
is the difference between a demo and a production system.

---

## Project Architecture

```
ai-finance-dashboard/
├── app.py                    # Streamlit UI — pure layout, zero business logic
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # Native dark theme tokens
│   └── secrets.toml          # API key (gitignored)
└── src/
    ├── data_loader.py        # yfinance + CSV ingestion, validation, cleaning
    ├── processing.py         # SMA·EMA·MACD·RSI·Bollinger·backtest·Sharpe
    ├── signals.py            # Crossover detection · rule-based insight cards
    ├── visualization.py      # Plotly chart factories (token-aligned dark theme)
    ├── prediction.py         # Monte Carlo · Ridge regression · TimeSeriesSplit
    └── llm_insights.py       # Hybrid AI/Heuristic analysis engine
```

**Enforced architecture rules:**
- `app.py` is UI-only — no business logic, no data manipulation
- No Streamlit imports inside `src/` — every module is independently testable
- All functions are pure: DataFrame in → DataFrame / dict out
- Session state caches expensive operations — no redundant recomputation

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOURUSERNAME/ai-finance-dashboard.git
cd ai-finance-dashboard

# 2. Virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open `http://localhost:8501` — works immediately, no API key required.

---

## Streamlit Cloud Deployment

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo · branch `main` · file `app.py`
4. **Advanced settings → Secrets:**
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Deploy — live in ~60 seconds

Without the secret, the app runs on the heuristic engine at zero cost.

---

## Tech Stack

`Python 3.10+` · `Streamlit` · `Pandas` · `Plotly` · `yfinance` ·
`scikit-learn` · `Anthropic Claude API` · `IBM Plex Mono` · `Syne`

Built using the **Antigravity IDE + Claude Code** integrated development stack.

---

## Roadmap

- [ ] Multi-stock comparison with correlation heatmap
- [ ] PDF report export
- [ ] Portfolio-level Sharpe across N tickers
- [ ] LLM-powered crossover alerts (webhook / email)
- [ ] Docker deployment config

---

*For educational and research purposes only. Not financial advice.*
