# Stock Analyzer Pro

Real-time stock analysis app with technical indicators, multi-timeframe trading signals, and ML-based predictions. Supports US and European markets.

## Features

- **Multi-market support** — US (NYSE/NASDAQ), Germany (XETRA), France (Euronext Paris), Netherlands, UK (LSE), Italy, Spain, Switzerland
- **12+ technical indicators** — RSI, MACD, Bollinger Bands, SMA/EMA, Stochastic, ADX, ATR, OBV, Fibonacci, volume analysis, trend strength
- **4 timeframe signals** — Intraday, Short Term (1-2 weeks), Mid Term (1-3 months), Long Term (6-12 months)
- **ML predictions** — Gradient Boosting model trained on each stock's historical data
- **Real-time updates** — WebSocket-powered live price feeds
- **Dark dashboard UI** — Price charts with overlays, indicator cards, signal breakdowns, news feed

## Quick Start

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
python run.py
```

Open **http://localhost:8899** in your browser.

## Architecture

```
app/
├── main.py                        # FastAPI entry point
├── config.py                      # Market definitions & settings
├── models.py                      # Pydantic schemas
├── api/
│   └── routes.py                  # REST + WebSocket endpoints
├── services/
│   ├── data_fetcher.py            # Yahoo Finance data with TTL cache
│   ├── technical_analysis.py      # 12 technical indicator calculators
│   └── signal_engine.py           # Signal scoring + ML prediction engine
└── templates/
    └── index.html                 # Single-page dashboard (Tailwind + Chart.js)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/markets` | List available markets |
| `POST` | `/api/markets/{id}` | Switch active market |
| `GET` | `/api/analyze/{ticker}` | Full analysis for a ticker |
| `GET` | `/api/search?q=...` | Search stocks |
| `GET` | `/api/watchlist` | Get current watchlist |
| `POST` | `/api/watchlist/{ticker}` | Add to watchlist |
| `DELETE` | `/api/watchlist/{ticker}` | Remove from watchlist |
| `WS` | `/api/ws` | WebSocket for real-time updates |

## Supported Markets

| Market | Exchange | Currency | Example Tickers |
|--------|----------|----------|-----------------|
| US | NYSE / NASDAQ | USD | AAPL, MSFT, GOOGL |
| DE | XETRA | EUR | SAP.DE, SIE.DE, BMW.DE |
| FR | Euronext Paris | EUR | MC.PA, OR.PA, AIR.PA |
| NL | Euronext Amsterdam | EUR | ASML.AS, PHIA.AS |
| UK | LSE | GBP | SHEL.L, AZN.L, BP.L |
| IT | Borsa Italiana | EUR | ENI.MI, RACE.MI |
| ES | BME | EUR | SAN.MC, ITX.MC |
| CH | SIX | CHF | NESN.SW, ROG.SW |

## Tech Stack

- **Backend**: Python, FastAPI, uvicorn
- **Data**: yfinance, pandas, numpy, ta (Technical Analysis)
- **ML**: scikit-learn (Gradient Boosting Classifier)
- **Frontend**: Tailwind CSS, Chart.js, vanilla JS, WebSocket
