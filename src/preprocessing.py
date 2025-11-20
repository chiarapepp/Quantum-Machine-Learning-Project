import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

FEATURES_12 = [
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "pkt_size_avg",
    "fwd_pkt_len_max",
    "bwd_pkt_len_max",
    "flow_byts_s",
    "flow_pkts_s",
    "fwd_iat_mean",
    "bwd_iat_mean",
]

def load_raw_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()
    X = df[FEATURES_12].astype(float).values
    y = df["Label"].astype(int).values
    return X, y

def preprocess(X, y, scaler_path="data/processed/scaler.pkl"):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # angle encoding: 0..pi
    X_theta = X_scaled * np.pi

    joblib.dump(scaler, scaler_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_theta, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
