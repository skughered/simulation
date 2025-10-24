import numpy as np

def apply_trend_filter(returns: np.ndarray, cash: np.ndarray, lookback: int = 6) -> np.ndarray:
    """
    Per-asset simple momentum filter on monthly returns.
    If trailing 'lookback' total return > 0 => invest next month; else use cash.
    returns: (S, M) asset returns
    cash:    (S, M) cash returns
    """
    S, M = returns.shape
    out = np.zeros_like(returns)
    # Precompute rolling trailing returns via log space for numerical stability
    # But since horizons are small and returns are monthly, direct prod is fine.
    for s in range(S):
        r = returns[s]
        c = cash[s]
        # first 'lookback' months default to cash (no signal yet)
        out[s, :lookback] = c[:lookback]
        for t in range(lookback, M):
            trailing = np.prod(1.0 + r[t - lookback:t]) - 1.0
            out[s, t] = r[t] if trailing > 0.0 else c[t]
    return out
