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

def parse_meta_csv(data_dir: Path, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the CSV with specific meta rows.
    Row 0: Public (Yes/No)
    Row 1: Name
    Row 2: Ticker (used as column headers)
    Rows 3-8: Ignored meta
    Row 9+: Data, first column date, others returns
    Returns (data_df, meta_df)
    - data_df: DataFrame with datetime index, ticker columns, numeric data
    - meta_df: DataFrame with ticker index, columns ['name', 'public']
    """
    path = data_dir / filename
    df = pd.read_csv(path, header=None)

    # Extract meta
    public_row = df.iloc[0, 1:]  # Skip first column (date)
    name_row = df.iloc[1, 1:]
    ticker_row = df.iloc[2, 1:]

    # Make ticker_row unique
    ticker_row = _make_unique(ticker_row)

    # Data starts at row 9
    data_df = df.iloc[9:, [0] + list(range(1, len(ticker_row)+1))].copy()
    data_df.columns = ['date'] + ticker_row
    data_df['date'] = pd.to_datetime(data_df['date'], dayfirst=True, errors='coerce')
    data_df = data_df.dropna(subset=['date']).set_index('date')

    # Convert to numeric, drop rows with all NaN, then drop any remaining NaNs
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    data_df = data_df.dropna(how='all')
    data_df = data_df.dropna()  # Drop rows with any NaN

    # Meta df
    public_bool = pd.Series(public_row.values).astype(str).str.strip().str.lower().isin(['yes', 'y'])
    meta_df = pd.DataFrame({
        'name': name_row.values,
        'public': public_bool.values
    }, index=ticker_row)

    return data_df, meta_df

def load_trend_weights(data_dir: Path, filename: str = "trend_port_weights.csv") -> pd.DataFrame:
    """
    Load trend portfolio weights CSV.
    Returns DataFrame with asset as index and portfolio columns (TWP2, TWP3, ..., BM1, etc.).
    """
    path = data_dir / filename
    df = pd.read_csv(path)
    df.set_index("Ticker", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df
