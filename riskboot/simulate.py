import numpy as np
import pandas as pd
from typing import Dict, Tuple

from riskboot.config import DATA_DIR, ALL_ASSETS_FILENAME, TREND_WEIGHTS_FILENAME, DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE
from riskboot.data import parse_meta_csv, DataPaths, load_trend_weights
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
    df_hist, meta_df = parse_meta_csv(DATA_DIR, ALL_ASSETS_FILENAME)

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
    vol_increase: float | None = None,
    trend_portfolio: str = "TWP2",
    benchmark_portfolio: str | None = None
) -> Dict[str, dict]:
    """
    Runs joint bootstrap, builds static & trend portfolios, and optionally benchmark.
    Static uses public assets with user weights.
    Trend uses selected TWP portfolio with trend filter applied to the portfolio return.
    Benchmark uses selected BM portfolio with static weights.
    """
    # Get bootstrapped data and metadata
    bootstrapped, meta_df = simulate_markets_joint(
        months=months,
        n_scenarios=n_scenarios,
        avg_block_range=block_range,
        seed=seed
    )
    S, M, K = bootstrapped.shape

    # Load historical data for volatility scaling
    df_hist, _ = parse_meta_csv(DATA_DIR, ALL_ASSETS_FILENAME)

    # Load trend weights
    trend_weights_df = load_trend_weights(DATA_DIR, TREND_WEIGHTS_FILENAME)
    if trend_portfolio != "None":
        if trend_portfolio not in trend_weights_df.columns:
            raise ValueError(f"Trend portfolio '{trend_portfolio}' not found in weights CSV.")
        twp_weights = trend_weights_df[trend_portfolio].dropna()

    # Get public assets
    public_assets = meta_df[meta_df['public']].index.tolist()

    # Static: Combine public assets with user weights (normalize)
    total_w = sum(weights.values())
    if total_w == 0:
        static = np.zeros((S, M))  # All cash if no weights
    else:
        norm_weights = {k: v / total_w for k, v in weights.items()}
        static = np.zeros((S, M))
        for asset, w in norm_weights.items():
            if asset in public_assets:
                idx = meta_df.index.get_loc(asset)
                static += w * bootstrapped[:, :, idx]
            # If asset not public, skip

    # Trend: Apply trend filter per asset, then weight and sum
    trend = np.zeros((S, M))
    if trend_portfolio != "None":
        cash_ticker = meta_df[meta_df['name'].str.contains('Cash \\(3m\\)', case=False)].index[0]
        print(f"Debug: Cash ticker for trend: {cash_ticker}")
        trend = np.zeros((S, M))
        for ticker, w in twp_weights.items():
            if ticker in meta_df.index:
                asset_returns = bootstrapped[:, :, meta_df.index.get_loc(ticker)]
                cash_returns = bootstrapped[:, :, meta_df.index.get_loc(cash_ticker)]
                asset_tf = apply_trend_filter(asset_returns, cash_returns, lookback)
                trend += w * asset_tf
                print(f"Debug: Asset {ticker}, weight {w}, sample tf returns: {asset_tf[0, :5]}")
        print(f"Debug: Trend portfolio weights: {twp_weights.to_dict()}")
        print(f"Debug: Sample trend returns (first scenario, first 5 months): {trend[0, :5]}")

    # Benchmark: Compute BM portfolio return (static, no trend)
    output = {
        "static": {"returns": static, "metrics": summarise_sims(static), "bands": percentile_bands(wealth_paths(static))}
    }
    if trend_portfolio != "None":
        output["trend"] = {"returns": trend, "metrics": summarise_sims(trend), "bands": percentile_bands(wealth_paths(trend))}
    if benchmark_portfolio != "None":
        if benchmark_portfolio not in trend_weights_df.columns:
            raise ValueError(f"Benchmark portfolio '{benchmark_portfolio}' not found in weights CSV.")
        bm_weights = trend_weights_df[benchmark_portfolio].dropna()
        bm_return = np.zeros((S, M))
        for ticker, w in bm_weights.items():
            if ticker in meta_df.index:
                idx = meta_df.index.get_loc(ticker)
                bm_return += w * bootstrapped[:, :, idx]
            else:
                raise ValueError(f"Ticker '{ticker}' in BM not found in data.")
        output["benchmark"] = {"returns": bm_return, "metrics": summarise_sims(bm_return), "bands": percentile_bands(wealth_paths(bm_return))}

    # Optional volatility scaling for static portfolio
    if vol_increase is not None and vol_increase > 0:
        # Compute historical portfolio returns using df_hist
        hist_port = np.zeros(len(df_hist))
        for asset, w in norm_weights.items():
            if asset in public_assets and asset in df_hist.columns:
                hist_port += w * df_hist[asset].values
        hist_vol = np.std(hist_port) * np.sqrt(12)
        mu_hist = np.mean(hist_port)

        # Apply scaling
        scale = 1 + vol_increase / hist_vol
        static = mu_hist + scale * (static - mu_hist)
        static = np.clip(static, -0.95, None)

    return output
