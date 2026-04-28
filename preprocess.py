import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CSV_FILES = [
    'Trojan.csv', 'trojan.csv', 'Trojan_Detection.csv',
    'trojan_detection.csv', 'network.csv'
]
LEAKAGE_COLUMNS = [
    'Unnamed: 0', 'Unnamed: 0.1', 'index',
    'Flow ID', 'Source IP', 'Destination IP',
    'Src IP', 'Dst IP', 'Timestamp'
]
COMMON_TROJAN_PORTS = [4444, 1234, 6666, 7777, 8888, 9999, 31337]


def find_csv():
    for filename in CSV_FILES:
        if os.path.exists(filename):
            return filename
    raise FileNotFoundError(
        'Dataset CSV not found. Place Trojan.csv in the project folder.')


def load_data(nrows=100000):
    csv_path = find_csv()
    df = pd.read_csv(csv_path, nrows=nrows)
    df.columns = df.columns.str.strip()
    df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns],
            errors='ignore', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if 'Class' not in df.columns:
        raise KeyError('Target column "Class" is missing.')

    if not pd.api.types.is_numeric_dtype(df['Class'].dtype):
        df['Class'] = LabelEncoder().fit_transform(df['Class'])

    return df.reset_index(drop=True)


def engineer_features(df):
    if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
        df['Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        df['Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']

    if 'Flow Bytes/s' in df.columns and 'Flow Packets/s' in df.columns:
        df['Avg_Bytes_per_Packet'] = df['Flow Bytes/s'] / (df['Flow Packets/s'] + 1)

    if 'Flow Bytes/s' in df.columns and 'Flow Duration' in df.columns:
        df['Flow_Intensity'] = df['Flow Bytes/s'] / (df['Flow Duration'] + 1)

    flag_cols = [
        'SYN Flag Count', 'ACK Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'URG Flag Count'
    ]
    found_flags = [c for c in flag_cols if c in df.columns]
    if found_flags:
        df['Flag_Score'] = df[found_flags].sum(axis=1)

    if 'Source Port' in df.columns:
        df['Is_Well_Known_Port'] = (df['Source Port'] < 1024).astype(int)

    if 'Destination Port' in df.columns:
        df['Is_Common_Trojan_Port'] = df['Destination Port'].isin(
            COMMON_TROJAN_PORTS).astype(int)

    if 'Protocol' in df.columns:
        df['Is_TCP'] = (df['Protocol'] == 6).astype(int)
        df['Is_UDP'] = (df['Protocol'] == 17).astype(int)
        df['Is_ICMP'] = (df['Protocol'] == 1).astype(int)

    return df


def preprocess_data(df, scaler=None, fit_scaler=True):
    df = engineer_features(df.copy())
    if 'Class' not in df.columns:
        raise KeyError('Target column "Class" is missing after feature engineering.')

    target = df['Class'].astype(int)
    features = df.drop(columns=['Class']).select_dtypes(include=['number']).copy()
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(features)
    else:
        X = scaler.transform(features)

    return X, target.values, scaler, list(features.columns)


def save_pickle(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)
