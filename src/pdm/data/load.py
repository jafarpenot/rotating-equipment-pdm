from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_ims_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a single IMS bearing file into a DataFrame.

    Many IMS files are plain text with whitespace-separated numeric columns.
    This loader keeps it flexible: it loads all numeric columns as signals and
    adds an index column 't' (sample index).
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df.columns = [f"s{i}" for i in range(df.shape[1])]
    df.insert(0, "t", range(len(df)))
    return df


def list_data_files(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    files = sorted([p for p in root_dir.rglob("*") if p.is_file()])
    # filter out obvious non-data files if needed
    files = [p for p in files if p.suffix.lower() in (".txt", "")]
    return files
