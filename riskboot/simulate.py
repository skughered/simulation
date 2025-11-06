import numpy as np
import pandas as pd
from typing import Dict, Tuple

from riskboot.config import DATA_DIR, ALL_ASSETS_FILENAME, TREND_WEIGHTS_FILENAME, DEFAULT_MONTHS, DEFAULT_SCENS, SEED_SIM, BLOCK_RANGE
from riskboot.data import parse_meta_csv, DataPaths, load_trend_weights
from riskboot.bootstrap import joint_stationary_bootstrap
from riskboot.trend import apply_trend_filter
from riskboot.portfolio import combine_static, combine_trend
from riskboot.metrics import summarise_sims, wealth_paths, percentile_bands, compute_windowed_maxdd_percentile


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
    # Generate longer data to allow for trend lookback warm-up
    months_sim = months + lookback

    # Get bootstrapped data and metadata
    bootstrapped, meta_df = simulate_markets_joint(
        months=months_sim,
        n_scenarios=n_scenarios,
        avg_block_range=block_range,
        seed=seed
    )
    S, M_sim, K = bootstrapped.shape

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

    # Static: Combine public assets with user weights (normalize) only if weights are provided
    total_w = sum(weights.values())
    output = {}
    if total_w > 0:
        norm_weights = {k: v / total_w for k, v in weights.items()}
        static = np.zeros((S, M_sim))
        for asset, w in norm_weights.items():
            if asset in public_assets:
                idx = meta_df.index.get_loc(asset)
                static += w * bootstrapped[:, :, idx]
        # Slice to drop warm-up
        static = static[:, lookback:lookback + months]
        output["static"] = {"returns": static, "metrics": summarise_sims(static), "bands": percentile_bands(wealth_paths(static))}

    # Trend: Apply trend filter per asset, then weight and sum
    trend = np.zeros((S, M_sim))
    if trend_portfolio != "None":
        cash_ticker = meta_df[meta_df['name'].str.contains('Cash \\(3m\\)', case=False)].index[0]
        # print(f"Debug: Cash ticker for trend: {cash_ticker}")
        trend = np.zeros((S, M_sim))
        for ticker, w in twp_weights.items():
            if w > 0 and ticker in meta_df.index:  # Skip if weight is zero
                asset_returns = bootstrapped[:, :, meta_df.index.get_loc(ticker)]
                cash_returns = bootstrapped[:, :, meta_df.index.get_loc(cash_ticker)]
                asset_tf = apply_trend_filter(asset_returns, cash_returns, lookback)
                trend += w * asset_tf
                # print(f"Debug: Asset {ticker}, weight {w}, sample tf returns: {asset_tf[0, :5]}")
        # print(f"Debug: Trend portfolio weights: {twp_weights.to_dict()}")
        # print(f"Debug: Sample trend returns (first scenario, first 5 months): {trend[0, :5]}")
        # Slice to drop warm-up
        trend = trend[:, lookback:lookback + months]
        output["trend"] = {"returns": trend, "metrics": summarise_sims(trend), "bands": percentile_bands(wealth_paths(trend))}

    # Benchmark: Compute BM portfolio return (static, no trend)
    if benchmark_portfolio != "None":
        if benchmark_portfolio not in trend_weights_df.columns:
            raise ValueError(f"Benchmark portfolio '{benchmark_portfolio}' not found in weights CSV.")
        bm_weights = trend_weights_df[benchmark_portfolio].dropna()
        bm_return = np.zeros((S, M_sim))
        for ticker, w in bm_weights.items():
            if ticker in meta_df.index:
                idx = meta_df.index.get_loc(ticker)
                bm_return += w * bootstrapped[:, :, idx]
            else:
                raise ValueError(f"Ticker '{ticker}' in BM not found in data.")
        # Slice to drop warm-up
        bm_return = bm_return[:, lookback:lookback + months]
        output["benchmark"] = {"returns": bm_return, "metrics": summarise_sims(bm_return), "bands": percentile_bands(wealth_paths(bm_return))}

    # Optional volatility scaling for static portfolio (only if static exists)
    if vol_increase is not None and vol_increase > 0 and "static" in output:
        # Compute historical portfolio returns using df_hist
        hist_port = np.zeros(len(df_hist))
        for asset, w in norm_weights.items():
            if asset in public_assets and asset in df_hist.columns:
                hist_port += w * df_hist[asset].values
        hist_vol = np.std(hist_port) * np.sqrt(12)
        mu_hist = np.mean(hist_port)

        # Apply scaling
        scale = 1 + vol_increase / hist_vol
        output["static"]["returns"] = mu_hist + scale * (output["static"]["returns"] - mu_hist)
        output["static"]["returns"] = np.clip(output["static"]["returns"], -0.95, None)
        # Recompute metrics and bands after scaling
        output["static"]["metrics"] = summarise_sims(output["static"]["returns"])
        output["static"]["bands"] = percentile_bands(wealth_paths(output["static"]["returns"]))

    # Compute windowed MaxDD for all portfolios in output
    # window_years = 10
    # for key in output:
    #     output[key]["metrics"]["WindowedMaxDD5th"] = compute_windowed_maxdd_percentile(output[key]["returns"], window_years)
    window_years = 10
    for key in output:
        # call helper with explicit percentile and coerce to float so the UI always gets a scalar
        w5 = compute_windowed_maxdd_percentile(output[key]["returns"], window_years, percentile=5.0)
        w25 = compute_windowed_maxdd_percentile(output[key]["returns"], window_years, percentile=25.0)
        output[key]["metrics"]["WindowedMaxDD5th"] = float(w5) if not np.isnan(w5) else np.nan
        output[key]["metrics"]["WindowedMaxDD25th"] = float(w25) if not np.isnan(w25) else np.nan


    return output

def apply_withdrawals(returns: np.ndarray, start_value: float, annual_withdrawals: list, inflation_pct: float) -> dict:
    """
    Simulate withdrawals with inflation.
    returns: (S, M) monthly returns
    start_value: initial portfolio value
    annual_withdrawals: list of nominal annual withdrawals (length = years)
    inflation_pct: annual inflation rate
    Returns dict with survival_rate, wealth (S, M), bands, ruin_month (S,)
    """
    S, M = returns.shape
    years = len(annual_withdrawals)
    if M != years * 12:
        raise ValueError(f"Mismatch: {M} months but {years} years")

    # Compute monthly inflated withdrawals
    monthly_withdrawals = []
    for y in range(years):
        nominal = annual_withdrawals[y]
        for m in range(12):
            months_elapsed = y * 12 + m
            inflated_annual = nominal * (1 + inflation_pct) ** (months_elapsed / 12.0)
            monthly_withdrawals.append(inflated_annual / 12)

    # Simulate wealth paths
    wealth = np.zeros((S, M))
    ruin_month = np.full(S, np.nan, dtype=float)
    for s in range(S):
        w = start_value
        for t in range(M):
            if t > 0:
                w *= (1 + returns[s, t-1])
            withdrawal = min(monthly_withdrawals[t], w)
            w -= withdrawal
            wealth[s, t] = w
            if w <= 0 and np.isnan(ruin_month[s]):
                ruin_month[s] = t

    survival_rate = np.mean(np.all(wealth > 0, axis=1))
    bands = percentile_bands(wealth)

    return {
        'survival_rate': survival_rate,
        'wealth': wealth,
        'bands': bands,
        'ruin_month': ruin_month
    }
