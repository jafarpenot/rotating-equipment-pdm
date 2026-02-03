from __future__ import annotations
import pandas as pd

def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 20,
) -> pd.DataFrame:
    """
    Adds rolling mean/std and rolling deltas per engine over cycles.
    Assumes df has columns: engine_id, cycle, sensor columns.
    """
    out = df.sort_values(["engine_id", "cycle"]).copy()

    g = out.groupby("engine_id", group_keys=False)

    for c in sensor_cols:
        out[f"{c}_rm{window}"] = g[c].transform(lambda s: s.rolling(window, min_periods=5).mean())
        out[f"{c}_rs{window}"] = g[c].transform(lambda s: s.rolling(window, min_periods=5).std())
        out[f"{c}_diff1"] = g[c].transform(lambda s: s.diff())

    # Fill NaNs created by rolling/diff
    feat_cols = [col for col in out.columns if "_rm" in col or "_rs" in col or col.endswith("_diff1")]
    out[feat_cols] = out[feat_cols].fillna(method="bfill").fillna(0.0)

    return out, feat_cols
