from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataPaths:
    data_dir: Path
    eqb_filename: str
    rf_filename: str

def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]

def load_eqb(data_dir: Path, filename: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns ['stocks','bonds'] aligned, no NaNs.
    """
    df = pd.read_csv(data_dir / filename)
    df = drop_unnamed(df)
    if "date" in df.columns:
        df = df.sort_values("date").drop(columns=["date"])
    keep = [c for c in ["stocks", "bonds"] if c in df.columns]
    if not keep:
        raise ValueError("equity_bond_data.csv must contain 'stocks' and 'bonds'")
    out = df[keep].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return out

def load_rf(data_dir: Path, filename: str) -> pd.Series:
    """
    Returns Series 'rf_1m' (monthly cash proxy).
    """
    df = pd.read_csv(data_dir / filename)
    col = "rf_1m" if "rf_1m" in df.columns else df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna().reset_index(drop=True)
    s.name = "rf_1m"
    return s
