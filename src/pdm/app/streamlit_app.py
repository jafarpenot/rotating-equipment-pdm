from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from pdm.data.load_cmapss import load_cmapss_train, add_rul_labels

from pdm.features.rolling import add_rolling_features
from pdm.models.anomaly import fit_isolation_forest, score_anomalies



st.set_page_config(page_title="Rotating Equipment PdM", layout="wide")
st.title("Predictive Maintenance (C-MAPSS)")
st.caption("Step 1: Load real multivariate time-series data and visualize sensors + RUL.")

root = st.sidebar.text_input("Dataset folder", value="data/raw/CMAPSSData")
dataset = st.sidebar.selectbox("Dataset", ["FD001", "FD002", "FD003", "FD004"], index=0)

root_path = Path(root)
train_path = root_path / f"train_{dataset}.txt"

if not train_path.exists():
    st.error(f"Could not find: {train_path.resolve()}")
    st.stop()

df = load_cmapss_train(train_path)
df = add_rul_labels(df)

engine_ids = sorted(df["engine_id"].unique().tolist())
engine_id = st.sidebar.selectbox("Engine ID", engine_ids, index=0)

engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

sensor_cols = [c for c in engine_df.columns if c.startswith("s")]
sensor = st.sidebar.selectbox("Sensor", sensor_cols, index=0)

col1, col2 = st.columns(2)

# Prepare features on FULL training set so anomaly model learns broad normal behavior
sensor_cols = [c for c in df.columns if c.startswith("s")]
df_feat, feat_cols = add_rolling_features(df, sensor_cols=sensor_cols, window=20)

# Fit anomaly model on early-life data only (more "normal")
early = df_feat[df_feat["rul"] > 50]  # simple heuristic, tweak later
model, scaler = fit_isolation_forest(early, feat_cols)

# Score selected engine
engine_feat = df_feat[df_feat["engine_id"] == engine_id].sort_values("cycle").copy()
engine_feat["anomaly_score"] = score_anomalies(engine_feat, feat_cols, model, scaler)


with col1:
    st.subheader(f"Sensor {sensor} over cycles (Engine {engine_id})")
    st.line_chart(engine_df.set_index("cycle")[sensor])

with col2:
    st.subheader("RUL over cycles (training label)")
    st.line_chart(engine_df.set_index("cycle")["rul"])

st.subheader("Preview")
st.dataframe(engine_df[["engine_id", "cycle", sensor, "rul"]].head(25))

st.subheader("Basic stats")
st.json({
    "rows": int(len(engine_df)),
    "cycles_min": int(engine_df["cycle"].min()),
    "cycles_max": int(engine_df["cycle"].max()),
    "sensor_mean": float(engine_df[sensor].mean()),
    "sensor_std": float(engine_df[sensor].std()),
    "rul_start": int(engine_df["rul"].iloc[0]),
    "rul_end": int(engine_df["rul"].iloc[-1]),
})

st.subheader("Anomaly score over cycles")
st.line_chart(engine_feat.set_index("cycle")["anomaly_score"])

