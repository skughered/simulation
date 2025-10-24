import numpy as np

def max_drawdown(monthly_returns: np.ndarray) -> float:
    """
    monthly_returns: (M,) vector
    returns min peak-to-trough (negative number)
    """
    wealth = np.cumprod(1.0 + monthly_returns)
    running_max = np.maximum.accumulate(wealth)
    dd = wealth / running_max - 1.0
    return float(dd.min())

def annualised_return(series: np.ndarray) -> float:
    m = np.mean(series)
    return float(m * 12.0)

def annualised_vol(series: np.ndarray) -> float:
    v = np.std(series, ddof=0)
    return float(v * np.sqrt(12.0))

def summarise_sims(returns: np.ndarray) -> dict:
    """
    returns: (S, M)
    outputs per-scenario metrics arrays
    """
    S = returns.shape[0]
    ann_r = np.empty(S)
    ann_v = np.empty(S)
    dd    = np.empty(S)
    for i in range(S):
        r = returns[i]
        ann_r[i] = annualised_return(r)
        ann_v[i] = annualised_vol(r)
        dd[i]    = max_drawdown(r)
        # Handle inf or nan
        if np.isinf(ann_r[i]) or np.isinf(ann_v[i]) or np.isnan(dd[i]):
            ann_r[i] = np.nan
            ann_v[i] = np.nan
            dd[i] = np.nan
    return {"AnnReturn": ann_r, "AnnVol": ann_v, "MaxDD": dd}

def wealth_paths(returns: np.ndarray, start_value: float = 1.0) -> np.ndarray:
    """
    returns: (S, M)
    Wealth index paths (S, M)
    """
    wealth = start_value * np.cumprod(1.0 + returns, axis=1)
    # Clip to prevent inf
    wealth = np.clip(wealth, 0, 1e10)
    return wealth

def percentile_bands(wealth: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """
    wealth: (S, M)
    Returns dict with zero-padded keys: q05, q25, q50, q75, q95
    """
    q = np.quantile(wealth, qs, axis=0)
    bands = {}

    for i, p in enumerate(qs):
        key = f"q{int(p*100):02d}"  # zero-pad: 0.05 → 'q05'
        bands[key] = q[i]

    return bands


if __name__ == "__main__":
    import numpy as np
    wealth = np.random.normal(0.01, 0.05, size=(500, 240)).cumsum(axis=1)
    bands = percentile_bands(wealth)
    print("bands keys:", list(bands.keys()))