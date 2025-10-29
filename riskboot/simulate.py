import numpy as np
import pandas as pd
from typing import Dict, Tuple

from riskboot.config import DATA_DIR, ALL_ASSETS_FILENAME, DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE
from riskboot.data import parse_meta_csv, DataPaths
from riskboot.bootstrap import joint_stationary_bootstrap
from riskboot.trend import apply_trend_filter
from riskboot.portfolio import combine_static, combine_trend
from riskboot.metrics import summarise_sims, wealth_paths, percentile_bands


def simulate_markets_joint(
        months: int = DEFAULT_MONTHS,
        n_scenarios: int = DEFAULT_SCENS,
        avg_block_range: Tuple[int, int] = BLOCK_RANGE,
        seed: int = SEED_SIM
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load all assets from the unified CSV and perform joint stationary bootstrap.
    Returns bootstrapped returns (n_scenarios, months, n_assets) and meta_df for asset info.
    """
    # Load data using the new parser
    data_paths = DataPaths(data_dir=DATA_DIR, data_filename=ALL_ASSETS_FILENAME)
    df_hist, meta_df = parse_meta_csv(
        data_dir=data_paths.data_dir,
        filename=data_paths.data_filename,
        meta_rows=data_paths.meta_rows,
        public_row=data_paths.public_row,
        name_row=data_paths.name_row
    )

    # Joint bootstrap on all assets (public and internal)
    bootstrapped = joint_stationary_bootstrap(
        df_hist=df_hist,
        months=months,
        n_scenarios=n_scenarios,
        avg_block_range=avg_block_range,
        seed=seed
    )
    return bootstrapped, meta_df

def simulate_portfolios(
    weights: Dict[str, float],
    months: int = DEFAULT_MONTHS,
    n_scenarios: int = DEFAULT_SCENS,
    seed: int = SEED_SIM,
    lookback: int = 6,
    block_range: Tuple[int, int] = BLOCK_RANGE,
    vol_increase: float | None = None
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

    # Optional volatility scaling for static portfolio
    if vol_increase is not None and vol_increase > 0:
        # Compute historical portfolio returns and volatility
        eqb = load_eqb(DATA_DIR, EQB_FILENAME)
        rf = load_rf(DATA_DIR, RF_FILENAME)
        df_hist = pd.concat([eqb, rf], axis=1).dropna().reset_index(drop=True)
        hist_port = (df_hist["stocks"] * weights["stocks"] +
                     df_hist["bonds"] * weights["bonds"] +
                     df_hist["rf_1m"] * weights["rf_1m"])
        hist_vol = np.std(hist_port) * np.sqrt(12)
        mu_hist = np.mean(hist_port)

        # Apply scaling
        scale = 1 + vol_increase / hist_vol
        static = mu_hist + scale * (static - mu_hist)
        static = np.clip(static, -0.95, None)

    # Metrics
    static_metrics = summarise_sims(static)
    trend_metrics  = summarise_sims(trend)

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
