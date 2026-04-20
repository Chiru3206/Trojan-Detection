# ============================================================
# Trojan Detection System — main.py  (FIXED: data leakage removed)
# ============================================================

import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# ─────────────────────────────────────────────────────────────
# GRAPH STYLE
# ─────────────────────────────────────────────────────────────
BG   = '#0b0b0f'
BG2  = '#111116'
ACC  = '#7c5cfc'
RED  = '#e8374a'
GRN  = '#22c55e'
AMB  = '#f59e0b'
BLU  = '#38bdf8'
DIM  = '#2a2a36'
TXT  = '#f1f1f8'
T2   = '#a0a0b8'

def style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   BG2,
        'axes.edgecolor':   DIM,
        'axes.labelcolor':  T2,
        'axes.titlecolor':  TXT,
        'text.color':       TXT,
        'xtick.color':      T2,
        'ytick.color':      T2,
        'grid.color':       DIM,
        'grid.linestyle':   '--',
        'grid.alpha':       0.4,
        'font.family':      'monospace',
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })

style()

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
print("="*60)
print("  TROJAN DETECTION SYSTEM — TRAINING")
print("="*60)

df = pd.read_csv("Trojan.csv")
df.columns = df.columns.str.strip()
print(f"[+] Loaded  → {df.shape[0]:,} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# 2. *** CRITICAL FIX — DROP ALL INDEX / LEAKAGE COLUMNS ***
#    Must happen BEFORE encode and BEFORE any feature work.
#    'Unnamed: 0' is the CSV row index written by pandas when
#    the dataset was saved without index=False.  If it is left
#    in, the model learns row-number → class (data leakage)
#    and reports a fake ~100 % accuracy.
# ─────────────────────────────────────────────────────────────
LEAKAGE_COLS = [
    'Unnamed: 0',          # pandas default row-index column  ← main culprit
    'Unnamed: 0.1',        # double-saved index
    'index',               # explicit index column
    'Flow ID',             # session identifier, not a feature
    'Source IP',  'Src IP',
    'Destination IP', 'Dst IP',
    'Timestamp',           # time ordering can encode class order
]

dropped = [c for c in LEAKAGE_COLS if c in df.columns]
if dropped:
    df.drop(columns=dropped, inplace=True)
    print(f"[!] DATA LEAKAGE COLUMNS REMOVED: {dropped}")
else:
    print("[+] No leakage columns found — dataset is clean")

print(f"[+] Shape after leakage removal → {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────
# 3. ENCODE TARGET
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
vc = df['Class'].value_counts()
print(f"[+] Classes → {list(le.classes_)}  Benign:{vc.get(0,0):,}  Trojan:{vc.get(1,0):,}")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def gc(*names):
    for n in names:
        for c in df.columns:
            if c.strip().lower() == n.lower(): return c
    return None

FWD = gc('Total Fwd Packets')
BWD = gc('Total Backward Packets')
BPS = gc('Flow Bytes/s')
PPS = gc('Flow Packets/s')
DUR = gc('Flow Duration')
SPT = gc('Source Port')
DPT = gc('Destination Port')

if FWD and BWD: df['Packet_Ratio']         = df[FWD] / (df[BWD] + 1)
if BPS and PPS: df['Avg_Bytes_per_Packet']  = df[BPS] / (df[PPS] + 1)
if BPS and DUR: df['Flow_Intensity']        = df[BPS] / (df[DUR] + 1)

flags = ['SYN Flag Count','ACK Flag Count','RST Flag Count','PSH Flag Count','URG Flag Count']
found = [c for c in flags if c in df.columns]
if found: df['Flag_Score'] = df[found].sum(axis=1)

if SPT: df['Is_Well_Known_Port']    = (df[SPT] < 1024).astype(int)
if DPT: df['Is_Common_Trojan_Port'] = df[DPT].isin(
    [4444,1234,6666,7777,8888,9999,31337]).astype(int)

print(f"[+] Feature engineering done → {df.shape[1]} total cols")

# ─────────────────────────────────────────────────────────────
# 5. NUMERIC ONLY + CLEAN
# ─────────────────────────────────────────────────────────────
df = df.select_dtypes(include=['number'])
df.replace([float('inf'), float('-inf')], 0, inplace=True)
df.dropna(inplace=True)
print(f"[+] Clean shape → {df.shape[0]:,} rows × {df.shape[1]} cols")

# Final sanity check — make sure Unnamed:0 is truly gone
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
    print("[!] WARNING: Unnamed:0 survived to numeric step — force-dropped")

# ─────────────────────────────────────────────────────────────
# 6. FEATURES & TARGET
# ─────────────────────────────────────────────────────────────
X = df.drop('Class', axis=1)
y = df['Class']
print(f"[+] Features used: {X.shape[1]}")
print(f"    Columns: {list(X.columns[:8])} ...")

# ─────────────────────────────────────────────────────────────
# 7. SCALE
# ─────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────
# 8. TRAIN / TEST SPLIT  (stratified — balanced classes)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[+] Train: {len(X_train):,}   Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────
# 9. TRAIN  (tuned to prevent overfitting)
# ─────────────────────────────────────────────────────────────
print("\n[*] Training Random Forest ...")
model = RandomForestClassifier(
    n_estimators    = 200,
    max_depth       = 20,      # capped — prevents memorising
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features    = 'sqrt',  # uses ~9 features per split, not all
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1,
)
model.fit(X_train, y_train)
print("[+] Training done")

# ─────────────────────────────────────────────────────────────
# 10. EVALUATE
# ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score (y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score   (y_test, y_pred, zero_division=0)
f1   = f1_score       (y_test, y_pred, zero_division=0)

# 5-fold cross-validation on the full scaled set
print("\n[*] Running 5-fold cross-validation ...")
cv = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)

print("\n" + "="*60)
print("  MODEL PERFORMANCE  (data-leakage-free)")
print("="*60)
print(f"  Accuracy      : {acc*100:.2f}%")
print(f"  Precision     : {prec*100:.2f}%")
print(f"  Recall        : {rec*100:.2f}%")
print(f"  F1 Score      : {f1*100:.2f}%")
print(f"  ROC-AUC       : {roc_auc*100:.2f}%")
print(f"  CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Benign','Trojan']))

# ─────────────────────────────────────────────────────────────
# 11. GRAPHS
# ─────────────────────────────────────────────────────────────
style()

# ── G1: Confusion Matrix ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
cmap = LinearSegmentedColormap.from_list('v', [BG2, ACC+'66', ACC])
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG)
ax.set_facecolor(BG2)
im = ax.imshow(cm, cmap=cmap)
for i in range(2):
    for j in range(2):
        v = cm[i,j]
        ax.text(j, i, f'{v:,}', ha='center', va='center',
                fontsize=20, fontweight='bold',
                color='#000' if v > cm.max()*.6 else TXT)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Benign','Trojan'], fontsize=12, color=BLU)
ax.set_yticklabels(['Benign','Trojan'], fontsize=12, color=BLU)
ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('Actual', fontsize=11)
ax.set_title('Confusion Matrix', fontsize=13, pad=16)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
cb.outline.set_edgecolor(DIM)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=T2)
fig.text(.5,.01,f'Accuracy:{acc*100:.2f}%  Precision:{prec*100:.2f}%  Recall:{rec*100:.2f}%',
         ha='center', fontsize=9, color=T2)
plt.tight_layout(rect=[0,.04,1,1])
plt.savefig('graph_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_confusion_matrix.png")

# ── G2: ROC Curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
ax.fill_between(fpr, tpr, alpha=.1, color=ACC)
for lw, al in [(5,.06),(3,.12),(1.5,1.)]:
    ax.plot(fpr, tpr, color=ACC, lw=lw, alpha=al)
ax.plot([0,1],[0,1],'--',color=RED,lw=1.2,alpha=.5,label='Random (0.50)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve  —  AUC = {roc_auc:.4f}', fontsize=13, pad=16)
ax.grid(alpha=.25); ax.legend(facecolor=BG, edgecolor=DIM, labelcolor=T2, fontsize=9)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_roc_curve.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_roc_curve.png")

# ── G3: Metrics Bar ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
names   = ['ACCURACY','PRECISION','RECALL','F1-SCORE','ROC-AUC','CV ACC']
vals    = [acc, prec, rec, f1, roc_auc, cv.mean()]
colors  = [ACC, BLU, GRN, AMB, '#c084fc', '#fb923c']
bars = ax.bar(names, [v*100 for v in vals], color=colors, width=.55, zorder=3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
            f'{v*100:.2f}%', ha='center', fontsize=10, fontweight='bold',
            color='#fff')
ax.set_ylim(0, 115); ax.set_ylabel('Score %')
ax.set_title('Model Performance Metrics  (no data leakage)', fontsize=13, pad=16)
ax.tick_params(axis='x', labelsize=8.5); ax.grid(axis='y', alpha=.25, zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_metrics_bar.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_metrics_bar.png")

# ── G4: Feature Importance  (top 15 — EXCLUDES Unnamed:0) ────
importances = model.feature_importances_
feat_names  = X.columns
top_n       = 15
idx         = np.argsort(importances)[::-1][:top_n]

# Safety assertion — make sure Unnamed:0 is not in features
assert 'Unnamed: 0' not in feat_names, \
    "LEAKAGE: Unnamed:0 still present in features — check drop logic"

fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
imp_vals = importances[idx]
bar_colors = [ACC if v > imp_vals.mean() else ACC+'99' for v in imp_vals]
ax.barh(range(top_n), imp_vals[::-1], color=bar_colors[::-1], height=.65)
for i, (bar, v) in enumerate(zip(ax.patches, imp_vals[::-1])):
    ax.text(v+.001, bar.get_y()+bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=8, color=BLU)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feat_names[i] for i in idx][::-1], fontsize=9, color=TXT)
ax.set_xlabel('Importance Score'); ax.set_xlim(0, imp_vals.max()*1.18)
ax.set_title(f'Top {top_n} Feature Importances  (legitimate features only)',
             fontsize=13, pad=16)
ax.grid(axis='x', alpha=.2)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_feature_importance.png")

# ── G5: Class Distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
cnt = [vc.get(0,0), vc.get(1,0)]
bars2 = ax.bar(['BENIGN','TROJAN'], cnt, color=[BLU, RED], width=.45, zorder=3)
for bar, c in zip(bars2, cnt):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+600,
            f'{c:,}', ha='center', fontsize=13, fontweight='bold', color=TXT)
ax.set_ylabel('Record Count'); ax.set_ylim(0, max(cnt)*1.15)
ax.set_title('Class Distribution', fontsize=13, pad=16)
ax.grid(axis='y', alpha=.2, zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_class_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_class_distribution.png")

# ── G6: Precision-Recall ─────────────────────────────────────
pr_p, pr_r, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(pr_r, pr_p)
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
ax.fill_between(pr_r, pr_p, alpha=.1, color=AMB)
for lw, al in [(4,.06),(2,.12),(1.5,1.)]:
    ax.plot(pr_r, pr_p, color=AMB, lw=lw, alpha=al)
ax.axhline(y=sum(y_test)/len(y_test), color=RED, ls='--', lw=1.2, alpha=.5)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall  —  AUC = {pr_auc:.4f}', fontsize=13, pad=16)
ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.grid(alpha=.2)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_precision_recall.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("[+] graph_precision_recall.png")

# ─────────────────────────────────────────────────────────────
# 12. SAVE
# ─────────────────────────────────────────────────────────────
pickle.dump(model,              open("model.pkl",    "wb"))
pickle.dump(scaler,             open("scaler.pkl",   "wb"))
pickle.dump(X.columns.tolist(), open("features.pkl", "wb"))
pickle.dump({
    'accuracy'   : round(acc*100,  2),
    'precision'  : round(prec*100, 2),
    'recall'     : round(rec*100,  2),
    'f1'         : round(f1*100,   2),
    'roc_auc'    : round(roc_auc*100, 2),
    'cv_accuracy': round(cv.mean()*100, 2),
}, open("metrics.pkl", "wb"))

print("\n[+] Saved: model.pkl  scaler.pkl  features.pkl  metrics.pkl")
print("[+] 6 graphs generated")
print("\n   Run next:  python pie_charts.py")
print("   Then:      python app.py   →   http://127.0.0.1:5000\n")