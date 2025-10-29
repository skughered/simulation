from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import pandas as pd

@dataclass
class DataPaths:
    data_dir: Path
    data_filename: str = "app_data_Oct_25.csv"
    meta_rows: int = 2
    public_row: int = 0
    name_row: int = 1

    def __post_init__(self):
        # ensure data_dir is a Path even if caller passed a str
        self.data_dir = Path(self.data_dir)

def _make_unique(names: List[str]) -> List[str]:
    seen = {}
    out = []
    for n in names:
        base = str(n).strip()
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            out.append(base)
    return out

def parse_meta_csv(data_dir: Path, filename: str, meta_rows: int = 2,
                   public_row: int = 0, name_row: int = 1,
                   dayfirst: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read CSV whose first `meta_rows` rows are column metadata.
    Returns (data_df, meta_df):
      - data_df: numeric DataFrame indexed by datetime with friendly unique column names
      - meta_df: DataFrame indexed by those column names with columns ['name','public']
    """
    path = Path(data_dir) / filename
    header_levels = list(range(meta_rows)) if meta_rows > 0 else None
    df = pd.read_csv(path, header=header_levels, index_col=0)

    # Ensure datetime index and drop unparsable rows
    df.index = pd.to_datetime(df.index, dayfirst=dayfirst, errors="coerce")
    df = df[~df.index.isna()]

    # Extract meta values from MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        try:
            public_vals = pd.Series(df.columns.get_level_values(public_row).astype(str))
        except Exception:
            public_vals = pd.Series([""] * len(df.columns))
        try:
            name_vals = pd.Series(df.columns.get_level_values(name_row).astype(str))
        except Exception:
            name_vals = pd.Series([str(c) for c in df.columns])
    else:
        public_vals = pd.Series(["yes"] * len(df.columns))
        name_vals = pd.Series([str(c) for c in df.columns])

    public_bool = public_vals.str.strip().str.lower().isin({"yes", "y", "true", "1"})
    friendly = _make_unique(name_vals.tolist())

    data_df = df.copy()
    data_df.columns = friendly
    data_df = data_df.apply(pd.to_numeric, errors="coerce")
    # drop any rows which are all NaN after conversion
    data_df = data_df.dropna(how="all")
    meta_df = pd.DataFrame({"name": friendly, "public": public_bool.values}, index=friendly)
    return data_df, meta_df

def load_trend_weights(data_dir: Path, filename: str = "trend_port_weights.csv") -> pd.DataFrame:
    """
    Load trend portfolio weights CSV.
    Returns DataFrame with asset as index and weight columns (w1 to w8).
    """
    path = data_dir / filename
    df = pd.read_csv(path, header=None, names=["asset", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8"])
    df.set_index("asset", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df
