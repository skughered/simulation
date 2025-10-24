from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .config import DATA_DIR, EQB_FILENAME, RF_FILENAME, DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE
from .data import load_eqb, load_rf
from .bootstrap import joint_stationary_bootstrap
from .trend import apply_trend_filter
from .portfolio import combine_static, combine_trend
from .metrics import summarise_sims, wealth_paths, percentile_bands

def simulate_markets_joint(
    months: int = DEFAULT_MONTHS,
    n_scenarios: int = DEFAULT_SCENS,
    seed: int = SEED_SIM,
    block_range: Tuple[int, int] = BLOCK_RANGE
) -> Dict[str, np.ndarray]:
    """
    Joint bootstrap for ['stocks','bonds','rf_1m'].
    Returns dict of arrays (S, M).
    """
    eqb = load_eqb(DATA_DIR, EQB_FILENAME)
    rf  = load_rf(DATA_DIR, RF_FILENAME)
    df = pd.concat([eqb, rf], axis=1)  # ['stocks','bonds','rf_1m']
    df = df.dropna().reset_index(drop=True)
    paths = joint_stationary_bootstrap(df_hist=df, months=months, n_scenarios=n_scenarios,
                                       avg_block_range=block_range, seed=seed)
    cols = list(df.columns)
    idx = {c: i for i, c in enumerate(cols)}
    return {
        "stocks": paths[:, :, idx["stocks"]],
        "bonds":  paths[:, :, idx["bonds"]],
        "rf_1m":  paths[:, :, idx["rf_1m"]],
    }

def simulate_portfolios(
    weights: Dict[str, float],
    months: int = DEFAULT_MONTHS,
    n_scenarios: int = DEFAULT_SCENS,
    seed: int = SEED_SIM,
    lookback: int = 6,
    block_range: Tuple[int, int] = BLOCK_RANGE
) -> Dict[str, dict]:
    """
    Runs joint bootstrap, builds static & trend portfolios, and summarises both.
    Returns dict with metrics and bands for UI.
    """
    mkt = simulate_markets_joint(months=months, n_scenarios=n_scenarios, seed=seed, block_range=block_range)
    S, M = mkt["stocks"].shape

    # Trend per-asset
    stocks_tf = apply_trend_filter(mkt["stocks"], mkt["rf_1m"], lookback=lookback)
    bonds_tf  = apply_trend_filter(mkt["bonds"],  mkt["rf_1m"], lookback=lookback)

    # Portfolios
    static = combine_static(mkt["stocks"], mkt["bonds"], mkt["rf_1m"], weights)
    trend  = combine_trend(stocks_tf, bonds_tf, mkt["rf_1m"], weights)

    # Metrics
    static_metrics = summarise_sims(static)
    trend_metrics  = summarise_sims(trend)
    print("DEBUG static returns range:", np.nanmin(static), np.nanmax(static))

    # Wealth bands (for fan charts) — compute on a *thinned* subset to keep memory light if needed
    static_w = wealth_paths(static)
    trend_w  = wealth_paths(trend)
    static_bands = percentile_bands(static_w)
    trend_bands  = percentile_bands(trend_w)

    return {
        "static": {
            "returns": static, "metrics": static_metrics, "bands": static_bands
        },
        "trend": {
            "returns": trend, "metrics": trend_metrics, "bands": trend_bands
        }
    }
