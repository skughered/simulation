import numpy as np
from typing import Dict

def combine_static(stocks: np.ndarray, bonds: np.ndarray, cash: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    """
    Combine per-month returns for static allocation.
    Arrays shape (S, M). weights keys: 'stocks','bonds','rf_1m'
    """
    w_s = weights.get("stocks", 0.0)
    w_b = weights.get("bonds", 0.0)
    w_c = weights.get("rf_1m", 0.0)
    return w_s * stocks + w_b * bonds + w_c * cash

def combine_trend(stocks_tf: np.ndarray, bonds_tf: np.ndarray, cash: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
    """
    Combine per-month returns for trend-filtered sleeves + cash sleeve.
    """
    w_s = weights.get("stocks", 0.0)
    w_b = weights.get("bonds", 0.0)
    w_c = weights.get("rf_1m", 0.0)
    return w_s * stocks_tf + w_b * bonds_tf + w_c * cash
