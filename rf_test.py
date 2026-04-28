import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

fn = next((name for name in ['Trojan.csv','trojan.csv','Trojan_Detection.csv','trojan_detection.csv'] if os.path.exists(name)), None)
if fn is None:
    raise FileNotFoundError('CSV not found')

df = pd.read_csv(fn, nrows=100000)
df.columns = df.columns.str.strip()
LEAKAGE = ['Unnamed: 0','Unnamed: 0.1','index','Flow ID','Source IP','Destination IP','Src IP','Dst IP','Timestamp']
df.drop(columns=[c for c in LEAKAGE if c in df.columns], errors='ignore', inplace=True)
df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
df.dropna(inplace=True)
if 'Class' not in df.columns:
    raise KeyError('Class column missing')
if df['Class'].dtype == object:
    df['Class'] = LabelEncoder().fit_transform(df['Class'])

class_col = df['Class']
df = df.select_dtypes(include=['number'])
df['Class'] = class_col

if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
    df['Packet_Ratio'] = df['Total Fwd Packets']/(df['Total Backward Packets']+1)
if 'Flow Bytes/s' in df.columns and 'Flow Packets/s' in df.columns:
    df['Avg_Bytes_per_Packet'] = df['Flow Bytes/s']/(df['Flow Packets/s']+1)
if 'Flow Bytes/s' in df.columns and 'Flow Duration' in df.columns:
    df['Flow_Intensity'] = df['Flow Bytes/s']/(df['Flow Duration']+1)
flag_cols = ['SYN Flag Count','ACK Flag Count','RST Flag Count','PSH Flag Count','URG Flag Count']
found_flags = [c for c in flag_cols if c in df.columns]
if found_flags:
    df['Flag_Score'] = df[found_flags].sum(axis=1)
if 'Source Port' in df.columns:
    df['Is_Well_Known_Port'] = (df['Source Port'] < 1024).astype(int)
if 'Destination Port' in df.columns:
    df['Is_Common_Trojan_Port'] = df['Destination Port'].isin([4444,1234,6666,7777,8888,9999,31337]).astype(int)

X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler(); Xs = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.1, random_state=42, stratify=y)
print('features', X.shape[1], 'rows', X.shape[0])
for params in [
    {'n_estimators':500,'max_depth':None,'max_features':'sqrt','min_samples_leaf':1},
    {'n_estimators':800,'max_depth':20,'max_features':0.7,'min_samples_leaf':2},
    {'n_estimators':1000,'max_depth':30,'max_features':0.5,'min_samples_leaf':2},
    {'n_estimators':1200,'max_depth':40,'max_features':'sqrt','min_samples_leaf':3},
]:
    clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, **params)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(params, 'accuracy', accuracy_score(y_test, pred))
