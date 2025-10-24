import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from riskboot.simulate import simulate_portfolios

# Test with small numbers
weights = {"stocks": 0.6, "bonds": 0.3, "rf_1m": 0.1}
out = simulate_portfolios(weights=weights, months=12, n_scenarios=10, seed=42, lookback=6, block_range=(6,12))

print("Static metrics keys:", out["static"]["metrics"].keys())
print("Static AnnReturn sample:", out["static"]["metrics"]["AnnReturn"][:5])
print("Static MaxDD sample:", out["static"]["metrics"]["MaxDD"][:5])
print("Any NaN in AnnReturn:", np.isnan(out["static"]["metrics"]["AnnReturn"]).any())
print("Any inf in AnnReturn:", np.isinf(out["static"]["metrics"]["AnnReturn"]).any())
print("Any NaN in MaxDD:", np.isnan(out["static"]["metrics"]["MaxDD"]).any())
print("Any inf in MaxDD:", np.isinf(out["static"]["metrics"]["MaxDD"]).any())
print("Any NaN in AnnVol:", np.isnan(out["static"]["metrics"]["AnnVol"]).any())
print("Any inf in AnnVol:", np.isinf(out["static"]["metrics"]["AnnVol"]).any())

# Check returns
static_returns = out["static"]["returns"]
print("Static returns shape:", static_returns.shape)
print("Any NaN in static returns:", np.isnan(static_returns).any())
print("Any inf in static returns:", np.isinf(static_returns).any())
