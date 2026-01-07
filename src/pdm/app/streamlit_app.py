from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from data.load import list_data_files, load_ims_file


st.set_page_config(page_title="Rotating Equipment PdM", layout="wide")
st.title("Rotating Equipment Predictive Maintenance")
st.caption("Step 1: load real vibration/sensor data and visualize it.")

root = st.sidebar.text_input("Dataset folder", value="data/raw/ims_bearings")

root_path = Path(root)
if not root_path.exists():
    st.warning(f"Folder not found: {root_path.resolve()}")
    st.stop()

files = list_data_files(root_path)
if not files:
    st.warning("No data files found in that folder.")
    st.stop()

file_strs = [str(p.relative_to(root_path)) for p in files]
choice = st.sidebar.selectbox("Select a data file", file_strs, index=0)

df = load_ims_file(root_path / choice)

st.subheader("Preview")
st.dataframe(df.head(20))

# Pick a signal column to plot
signal_cols = [c for c in df.columns if c.startswith("s")]
sig = st.sidebar.selectbox("Signal", signal_cols, index=0)

st.subheader(f"Signal plot: {sig}")
st.line_chart(df.set_index("t")[sig])

# Simple stats
st.subheader("Basic stats")
stats = {
    "samples": len(df),
    "mean": float(df[sig].mean()),
    "std": float(df[sig].std()),
    "min": float(df[sig].min()),
    "max": float(df[sig].max()),
    "rms": float((df[sig] ** 2).mean() ** 0.5),
}
st.json(stats)
