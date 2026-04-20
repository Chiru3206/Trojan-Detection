# ===============================
# Trojan Detection - predict.py
# ===============================

import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Load Saved Files ──────────────────────────────────────────────────────────
model    = pickle.load(open("model.pkl",    "rb"))
scaler   = pickle.load(open("scaler.pkl",   "rb"))
features = pickle.load(open("features.pkl", "rb"))
metrics  = pickle.load(open("metrics.pkl",  "rb"))

print("=" * 50)
print("     TROJAN DETECTION - PREDICTION TEST")
print("=" * 50)
print(f"\n✅ Model loaded successfully")
print(f"\n📈 Model Scores (from training):")
print(f"   Accuracy  : {metrics['accuracy']}%")
print(f"   Precision : {metrics['precision']}%")
print(f"   Recall    : {metrics['recall']}%")
print(f"   F1 Score  : {metrics['f1']}%")

# ── Load Dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("Trojan.csv")

# Show key columns from the dataset
important_cols = ['Source IP', 'Source Port', 'Destination IP',
                  'Destination Port', 'Protocol', 'Timestamp',
                  'Flow Duration', 'Total Fwd Packets']
show_cols = [c for c in important_cols if c in df.columns]
print(f"\n📋 Testing on this sample record:")
print(df[show_cols].iloc[0].to_string())

# ── Preprocessing ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

drop_cols = ['Flow ID', 'Source IP', 'Destination IP',
             'Src IP', 'Dst IP', 'Timestamp']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

if 'Class' in df.columns and df['Class'].dtype == object:
    df['Class'] = LabelEncoder().fit_transform(df['Class'])

# Feature Engineering
if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
    df['Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)

if 'Flow Bytes/s' in df.columns and 'Flow Packets/s' in df.columns:
    df['Avg_Bytes_per_Packet'] = df['Flow Bytes/s'] / (df['Flow Packets/s'] + 1)

if 'Flow Bytes/s' in df.columns and 'Flow Duration' in df.columns:
    df['Flow_Intensity'] = df['Flow Bytes/s'] / (df['Flow Duration'] + 1)

flag_cols = ['SYN Flag Count', 'ACK Flag Count', 'RST Flag Count',
             'PSH Flag Count', 'URG Flag Count']
found_flags = [c for c in flag_cols if c in df.columns]
if found_flags:
    df['Flag_Score'] = df[found_flags].sum(axis=1)

if 'Source Port' in df.columns:
    df['Is_Well_Known_Port'] = (df['Source Port'] < 1024).astype(int)

if 'Destination Port' in df.columns:
    df['Is_Common_Trojan_Port'] = df['Destination Port'].isin(
        [4444, 1234, 6666, 7777, 8888, 9999, 31337]
    ).astype(int)

df = df.select_dtypes(include=['number'])
df.replace([float('inf'), float('-inf')], 0, inplace=True)
df.dropna(inplace=True)

if 'Class' in df.columns:
    X = df.drop("Class", axis=1)
else:
    X = df.copy()

X = X.reindex(columns=features, fill_value=0)
X_scaled = scaler.transform(X)

# ── Predict ───────────────────────────────────────────────────────────────────
sample = X_scaled[0].reshape(1, -1)
prediction = model.predict(sample)
confidence = round(max(model.predict_proba(sample)[0]) * 100, 2)

print(f"\n{'='*50}")
if prediction[0] == 1:
    print("  🚨 RESULT  : TROJAN Traffic Detected")
else:
    print("  ✅ RESULT  : SAFE / Benign Traffic")
print(f"  📊 Confidence : {confidence}%")
print(f"{'='*50}")