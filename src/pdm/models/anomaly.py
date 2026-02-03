from __future__ import annotations
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def fit_isolation_forest(df: pd.DataFrame, feature_cols: list[str], random_state: int = 42):
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(Xs)
    return model, scaler

def score_anomalies(df: pd.DataFrame, feature_cols: list[str], model, scaler) -> pd.Series:
    X = df[feature_cols].values
    Xs = scaler.transform(X)
    # IsolationForest: higher = more normal; we invert so higher = more anomalous
    normal_score = model.score_samples(Xs)
    anomaly_score = -normal_score
    return pd.Series(anomaly_score, index=df.index, name="anomaly_score")
