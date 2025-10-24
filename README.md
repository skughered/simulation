# Portfolio Simulator (riskboot)

Block-bootstrap simulator for client-friendly portfolio risk analytics.  
Compares **static** vs **per-asset trend-following** portfolios using **joint (synchronous) stationary bootstrap** across equities, bonds, and cash.

## Quick start

1. Edit `riskboot/config.py` to point to your CSVs:
   - `equity_bond_data.csv` (cols: `date, stocks, bonds`)
   - `money_market.csv` (col: `rf_1m`)

2. Install & run:
```bash
python -m venv .venv
. .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
