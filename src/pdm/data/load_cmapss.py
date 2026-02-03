from __future__ import annotations

from pathlib import Path
import pandas as pd

# C-MAPSS: each row = one cycle for one engine
# Columns: engine_id, cycle, 3 settings, 21 sensors
CMAPSS_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

def load_cmapss_train(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some versions include an extra empty column at the end due to spacing
    if df.shape[1] > len(CMAPSS_COLUMNS):
        df = df.iloc[:, : len(CMAPSS_COLUMNS)]
    df.columns = CMAPSS_COLUMNS
    return df

def load_cmapss_rul(path: str | Path) -> pd.Series:
    path = Path(path)
    rul_df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # RUL file usually has a single integer column (sometimes + empty)
    rul = rul_df.iloc[:, 0].astype(int)
    return rul

def add_rul_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each engine, compute RUL = max_cycle - cycle
    This is the standard label for the training set.
    """
    max_cycle = train_df.groupby("engine_id")["cycle"].max()
    df = train_df.copy()
    df["max_cycle"] = df["engine_id"].map(max_cycle)
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df
