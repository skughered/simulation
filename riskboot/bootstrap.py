import numpy as np
import pandas as pd
from typing import Tuple

def stationary_bootstrap_indices(T: int, length: int, avg_block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Stationary bootstrap (Politis–Romano). Returns index path of length 'length'.
    """
    p = 1.0 / max(1, int(avg_block))
    i = int(rng.integers(0, T))
    idx = np.empty(length, dtype=int)
    for t in range(length):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, T))
        else:
            i = (i + 1) % T
    return idx

def joint_stationary_bootstrap(
    df_hist: pd.DataFrame,
    months: int,
    n_scenarios: int,
    avg_block_range: Tuple[int, int],
    seed: int
) -> np.ndarray:
    """
    Jointly resample all columns (synchronous indices).
    Returns shape: (n_scenarios, months, n_assets)
    """
    if df_hist.isna().any().any():
        raise ValueError("Historical data contains NaNs; please clean first.")
    rng = np.random.default_rng(seed)
    T, K = df_hist.shape
    X = df_hist.to_numpy()
    out = np.empty((n_scenarios, months, K), dtype=float)
    low, high = avg_block_range
    for s in range(n_scenarios):
        avg_b = int(rng.integers(low, high + 1))
        idx = stationary_bootstrap_indices(T=T, length=months, avg_block=avg_b, rng=rng)
        out[s, :, :] = X[idx, :]
    return out
