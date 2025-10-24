import os
from pathlib import Path

# Detect environment (default = local)
RUN_ENV = os.getenv("RUN_ENV", "local").lower()  # 'local' or 'cloud'

# --- Path configuration ---
if RUN_ENV == "local":
    # Use your real data in the package (simple, portable)
    DATA_DIR = Path(__file__).parent / "data"
else:
    # Fallback demo data (optional — could be the same folder if public)
    DATA_DIR = Path(__file__).parent / "demo_data"

# --- Filenames ---
EQB_FILENAME = "equity_bond_data.csv"
RF_FILENAME  = "money_market.csv"

# --- Simulation defaults ---
DEFAULT_YEARS   = 30
DEFAULT_MONTHS  = DEFAULT_YEARS * 12
DEFAULT_SCENS   = 2000
SEED_SIM        = 2025
BLOCK_RANGE     = (6, 12)
