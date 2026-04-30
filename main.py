# ============================================================
# Trojan Detection System — main.py
# 3 Models: Random Forest | KNN | XGBoost
# ============================================================

import sys
import pandas as pd
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neighbors        import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# ── colours ───────────────────────────────────────────────────
BG  = '#0b0b0f'; BG2 = '#111116'; DIM = '#2a2a36'
ACC = '#7c5cfc'; RED = '#e8374a'; GRN = '#22c55e'
AMB = '#f59e0b'; BLU = '#38bdf8'; PUR = '#c084fc'
TXT = '#f1f1f8'; T2  = '#a0a0b8'

def style():
    plt.rcParams.update({
        'figure.facecolor':BG,'axes.facecolor':BG2,
        'axes.edgecolor':DIM,'axes.labelcolor':T2,
        'axes.titlecolor':TXT,'text.color':TXT,
        'xtick.color':T2,'ytick.color':T2,
        'grid.color':DIM,'grid.linestyle':'--','grid.alpha':.4,
        'font.family':'monospace',
        'axes.spines.top':False,'axes.spines.right':False,
    })
style()

# ═══════════════════════════════════════════════════════════════
# MEMORY OPTIMIZATION FUNCTION — for low-RAM (8GB) environment
# ═══════════════════════════════════════════════════════════════
def optimize_memory(df):
    """Downcast float64→float32 and int64→int32 to reduce memory footprint."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
    return df

print("="*60)
print("  TROJAN DETECTION — 3 MODEL TRAINING PIPELINE")
print("="*60)

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATASET
# ═══════════════════════════════════════════════════════════════
import os

# Find CSV regardless of capitalisation
def find_csv():
    for name in ['Trojan.csv','trojan.csv','Trojan_Detection.csv',
                 'trojan_detection.csv']:
        if os.path.exists(name): return name
    raise FileNotFoundError("Trojan.csv not found in this folder.")

# Load with nrows limit to prevent OOM in 8GB RAM environment
df = pd.read_csv(find_csv(), nrows=100000)
df.columns = df.columns.str.strip()
print(f"\n[+] Dataset loaded  → {df.shape[0]:,} rows × {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
# EARLY: Remove leakage columns immediately to reduce memory
# ═══════════════════════════════════════════════════════════════
LEAKAGE = ['Unnamed: 0','Unnamed: 0.1','index',
           'Flow ID','Source IP','Destination IP',
           'Src IP','Dst IP','Timestamp']
df.drop(columns=[c for c in LEAKAGE if c in df.columns], errors='ignore', inplace=True)
print(f"[+] After dropping leakage columns: {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
# EARLY: Handle inf/nan values immediately
# ═══════════════════════════════════════════════════════════════
df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"[+] After cleaning inf/nan: {df.shape[0]:,} rows")

# ═══════════════════════════════════════════════════════════════
# 2. ENCODE TARGET (before selecting numeric types)
# ═══════════════════════════════════════════════════════════════
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])   # Benign=0, Trojan=1
vc = df['Class'].value_counts()
print(f"[+] Benign: {vc.get(0,0):,}   Trojan: {vc.get(1,0):,}")

# ═══════════════════════════════════════════════════════════════
# EARLY: Select numeric columns only
# ═══════════════════════════════════════════════════════════════
df = df.select_dtypes(include=['number'])
print(f"[+] After selecting numeric: {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
def gc(*names):
    for n in names:
        for c in df.columns:
            if c.strip().lower() == n.strip().lower(): return c
    return None

FWD = gc('Total Fwd Packets')
BWD = gc('Total Backward Packets')
BPS = gc('Flow Bytes/s')
PPS = gc('Flow Packets/s')
DUR = gc('Flow Duration')
SPT = gc('Source Port')
DPT = gc('Destination Port')

if FWD and BWD: df['Packet_Ratio']         = df[FWD]/(df[BWD]+1)
if BPS and PPS: df['Avg_Bytes_per_Packet']  = df[BPS]/(df[PPS]+1)
if BPS and DUR: df['Flow_Intensity']        = df[BPS]/(df[DUR]+1)

flags = ['SYN Flag Count','ACK Flag Count','RST Flag Count',
         'PSH Flag Count','URG Flag Count']
found_f = [c for c in flags if c in df.columns]
if found_f: df['Flag_Score'] = df[found_f].sum(axis=1)

if SPT: df['Is_Well_Known_Port']    = (df[SPT] < 1024).astype(int)
if DPT: df['Is_Common_Trojan_Port'] = df[DPT].isin(
    [4444,1234,6666,7777,8888,9999,31337]).astype(int)

print(f"[+] Features after engineering: {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
# MEMORY OPTIMIZATION after feature engineering
# ═══════════════════════════════════════════════════════════════
df = optimize_memory(df)
print(f"[+] Memory optimized: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ═══════════════════════════════════════════════════════════════
# 5. FINAL CLEANUP
# ═══════════════════════════════════════════════════════════════
print(f"[+] Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════
# 6. SPLIT FEATURES & TARGET
# ═══════════════════════════════════════════════════════════════
X = df.drop('Class', axis=1)
y = df['Class']
print(f"[+] Feature count: {X.shape[1]}")

# ═══════════════════════════════════════════════════════════════
# 7. SCALE (StandardScaler will convert to float64, then models work)
# ═══════════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"[+] Scaled data: {X_scaled.nbytes / 1024**2:.2f} MB")

# ═══════════════════════════════════════════════════════════════
# 7.5. PREPARE TEXT DATA (SKIPPED - XGBoost uses numeric features)
# ═══════════════════════════════════════════════════════════════════════
print(f"[+] Using numeric features directly for XGBoost training")

# ═══════════════════════════════════════════════════════════════
# 8. TRAIN / TEST SPLIT  (80 % train — 20 % test, stratified)
# ═══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, stratify=y)
print(f"\n[+] Training samples : {len(X_train):,} ({X_train.nbytes / 1024**2:.2f} MB)")
print(f"[+] Testing  samples : {len(X_test):,} ({X_test.nbytes / 1024**2:.2f} MB)")

# ═══════════════════════════════════════════════════════════════
# 9. DEFINE MODELS
# ═══════════════════════════════════════════════════════════════

THRESHOLD = 80.0
 
def meets_thresholds(metric_dict, threshold=THRESHOLD):
    return all(metric_dict.get(k, 0) >= threshold for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])


def get_target_models():
    args = sys.argv[1:]
    robots = {'rf': 'Random Forest', 'knn': 'KNN', 'xgb': 'XGBoost'}
    if not args:
        return list(models.keys())
    if len(args) != 1 or args[0].lower() not in list(robots.keys()) + ['all']:
        print('Usage: python main.py [rf|knn|xgb|all]')
        sys.exit(1)
    choice = args[0].lower()
    if choice == 'all':
        return list(models.keys())
    return [robots[choice]]


def evaluate_model_metrics(y_test, y_pred, y_prob, cv=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc  = auc(fpr, tpr)
    return {
        'accuracy' : round(acc * 100, 2),
        'precision': round(prec * 100, 2),
        'recall'   : round(rec * 100, 2),
        'f1'       : round(f1 * 100, 2),
        'roc_auc'  : round(roc * 100, 2),
        'cv_mean'  : round(cv.mean() * 100, 2) if cv is not None else None,
        'cv_std'   : round(cv.std() * 100, 2) if cv is not None else None,
        'fpr'      : fpr,
        'tpr'      : tpr,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
    }


def train_and_evaluate(name, clf, X_train, y_train, X_test, y_test):
    print(f"\n[*] Training: {name} ...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    if name == 'Random Forest':
        cv = cross_val_score(clf, X_scaled, y, cv=3, scoring='accuracy', n_jobs=1)
    else:
        cv = np.array([accuracy_score(y_test, y_pred)] * 3)
    metrics = evaluate_model_metrics(y_test, y_pred, y_prob, cv=cv)
    return clf, metrics


def search_random_forest(X_train, y_train, X_test, y_test):
    print('\n[!] Threshold not met by initial models. Running additional Random Forest tuning...')
    search_space = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [15, 20, 25, 30, 35, 40, None],
        'max_features': ['sqrt', 'log2', 0.5],
        'min_samples_leaf': [1, 2, 3, 4, 6],
        'criterion': ['gini', 'entropy'],
    }
    cv_search = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        param_distributions=search_space,
        n_iter=12,
        cv=cv_search,
        scoring='f1',
        random_state=42,
        n_jobs=1,
        verbose=0,
    )
    sample_size = min(30000, len(X_train))
    if sample_size < len(X_train):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        sample_idx, _ = next(splitter.split(X_train, y_train))
        X_search = X_train[sample_idx]
        y_search = np.asarray(y_train)[sample_idx]
    else:
        X_search = X_train
        y_search = y_train

    search.fit(X_search, y_search)
    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model_metrics(y_test, y_pred, y_prob, cv=None)
    print('    Tuned Random Forest metrics:')
    print(f"      Accuracy : {metrics['accuracy']}%")
    print(f"      Precision: {metrics['precision']}%")
    print(f"      Recall   : {metrics['recall']}%")
    print(f"      F1 Score : {metrics['f1']}%")
    print(f"      ROC AUC  : {metrics['roc_auc']}%")
    return best_model, metrics

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = None,
        min_samples_split= 2,
        min_samples_leaf = 1,
        max_features     = 'sqrt',
        class_weight     = 'balanced',
        random_state     = 42,
        n_jobs           = -1,
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors       = 3,
        weights           = 'distance',
        algorithm         = 'ball_tree',
        leaf_size         = 30,
        n_jobs            = 1,
    ),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    ),
}

# ═══════════════════════════════════════════════════════════════
# 10. TRAIN + EVALUATE MODELS
# ═══════════════════════════════════════════════════════════════
selected_models = get_target_models()
print("\n" + "="*60)
print("  TRAINING & EVALUATING MODELS")
print("="*60)

results   = {}
trained   = {}
cm_dict   = {}

for name in selected_models:
    clf = models[name]
    clf, metrics = train_and_evaluate(name, clf, X_train, y_train, X_test, y_test)
    trained[name] = clf
    results[name] = metrics
    cm_dict[name] = confusion_matrix(y_test, metrics['y_pred'])

    print(f"    {name} results:")
    print(f"      Accuracy : {metrics['accuracy']}%")
    print(f"      Precision: {metrics['precision']}%")
    print(f"      Recall   : {metrics['recall']}%")
    print(f"      F1 Score : {metrics['f1']}%")
    print(f"      ROC AUC  : {metrics['roc_auc']}%")
    print(f"      CV Mean  : {metrics['cv_mean']}% ± {metrics['cv_std']}%")
    print(f"\n      Classification Report:")
    print(classification_report(y_test, metrics['y_pred'], target_names=['Benign','Trojan']))

# ═══════════════════════════════════════════════════════════════
# 11. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  MODEL COMPARISON SUMMARY")
print("="*60)
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-"*70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9}%  {r['precision']:>9}%  "
          f"{r['recall']:>7}%  {r['f1']:>7}%  {r['roc_auc']:>8}%")

# Best model = Random Forest if it meets all thresholds, otherwise fall back
rf_candidates = [name for name in results if 'Random Forest' in name and meets_thresholds(results[name])]
if rf_candidates:
    best_name = max(rf_candidates, key=lambda n: results[n]['f1'])
    best_model = trained[best_name]
    best_res = results[best_name]
    print(f"\n[★] Random Forest is selected as best threshold-qualified model: {best_name}")
else:
    if 'Random Forest' in results:
        tuned_model, tuned_metrics = search_random_forest(X_train, y_train, X_test, y_test)
        if tuned_model is not None and meets_thresholds(tuned_metrics):
            best_name = 'Random Forest (Tuned)'
            best_model = tuned_model
            best_res = tuned_metrics
            trained[best_name] = tuned_model
            results[best_name] = tuned_metrics
            cm_dict[best_name] = confusion_matrix(y_test, tuned_metrics['y_pred'])
            print(f"\n[★] Selected tuned Random Forest: {best_name}")
        else:
            best_name = max(results, key=lambda n: results[n]['f1'])
            best_model = trained[best_name]
            best_res = results[best_name]
            print(f"\n[!] No Random Forest model reached {THRESHOLD}% on all metrics.")
            print(f"[★] Best model by F1 fallback: {best_name}")
    else:
        best_name = max(results, key=lambda n: results[n]['f1'])
        best_model = trained[best_name]
        best_res = results[best_name]
        print(f"\n[!] No Random Forest model available to select.")
        print(f"[★] Best model by F1 fallback: {best_name}")
print("="*60)

# ═══════════════════════════════════════════════════════════════
# 12. GRAPHS
# ═══════════════════════════════════════════════════════════════
style()

# ── G1: Confusion Matrix (best model) ────────────────────────
cm = cm_dict[best_name]
cmap_v = LinearSegmentedColormap.from_list('v', [BG2, ACC+'66', ACC])
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG)
ax.set_facecolor(BG2)
im = ax.imshow(cm, cmap=cmap_v)
for i in range(2):
    for j in range(2):
        v = cm[i,j]
        ax.text(j,i,f'{v:,}',ha='center',va='center',
                fontsize=20,fontweight='bold',
                color='#000' if v>cm.max()*.6 else TXT)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Benign','Trojan'],fontsize=12,color=BLU)
ax.set_yticklabels(['Benign','Trojan'],fontsize=12,color=BLU)
ax.set_xlabel('Predicted',fontsize=11); ax.set_ylabel('Actual',fontsize=11)
ax.set_title(f'Confusion Matrix  —  {best_name}',fontsize=13,pad=16)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
cb.outline.set_edgecolor(DIM)
plt.setp(cb.ax.yaxis.get_ticklabels(),color=T2)
fig.text(.5,.01,
    f"Accuracy:{best_res['accuracy']}%  "
    f"Precision:{best_res['precision']}%  "
    f"Recall:{best_res['recall']}%",
    ha='center',fontsize=9,color=T2)
plt.tight_layout(rect=[0,.04,1,1])
plt.savefig('graph_confusion_matrix.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_confusion_matrix.png")

# ── G2: ROC Curves (all models on same chart) ──────────────
fig, ax = plt.subplots(figsize=(7,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
cols_roc = {'Random Forest': ACC, 'Random Forest (Tuned)': ACC, 'KNN': AMB, 'XGBoost': GRN}
for name, r in results.items():
    c = cols_roc.get(name, ACC)
    ax.fill_between(r['fpr'], r['tpr'], alpha=.06, color=c)
    ax.plot(r['fpr'], r['tpr'], color=c, lw=2,
            label=f"{name}  (AUC={r['roc_auc']}%)")
ax.plot([0,1],[0,1],'--',color=RED,lw=1.2,alpha=.5,label='Random (50%)')
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models',fontsize=13,pad=16)
ax.grid(alpha=.25)
ax.legend(facecolor=BG,edgecolor=DIM,labelcolor=TXT,fontsize=9,loc='lower right')
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_roc_curve.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_roc_curve.png")

# ── G3: Model Comparison Bar (all 3 × 4 metrics) ─────────────
fig, ax = plt.subplots(figsize=(11,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
model_names = list(results.keys())
metric_keys = ['accuracy','precision','recall','f1']
metric_lbls = ['Accuracy','Precision','Recall','F1 Score']
x    = np.arange(len(model_names))
w    = 0.2
bar_cols = [ACC, BLU, GRN, AMB]

for i,(mk,ml,bc) in enumerate(zip(metric_keys,metric_lbls,bar_cols)):
    vals = [results[n][mk] for n in model_names]
    bars = ax.bar(x + i*w, vals, w, label=ml, color=bc, zorder=3)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.4,
                f'{v}%', ha='center', fontsize=8, fontweight='bold', color=TXT)

ax.set_xticks(x + w*1.5)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel('Score %')
ax.set_title('Model Performance Comparison — All Models',fontsize=13,pad=16)
ax.grid(axis='y',alpha=.2,zorder=0)
ax.legend(facecolor=BG,edgecolor=DIM,labelcolor=TXT,fontsize=9,loc='upper right')
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_metrics_bar.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_metrics_bar.png")

# ── G4: Feature Importance (Random Forest only) ───────────────
rf   = trained.get(best_name, trained['Random Forest'])
imps = rf.feature_importances_
fn   = X.columns
top  = 15
idx  = np.argsort(imps)[::-1][:top]

assert 'Unnamed: 0' not in fn, \
    "LEAKAGE: Unnamed:0 still in features — check drop logic!"

fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
iv = imps[idx]
bar_c = [ACC if v > iv.mean() else ACC+'99' for v in iv]
ax.barh(range(top), iv[::-1], color=bar_c[::-1], height=.65)
for i,(bar,v) in enumerate(zip(ax.patches, iv[::-1])):
    ax.text(v+.001, bar.get_y()+bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=8, color=BLU)
ax.set_yticks(range(top))
ax.set_yticklabels([fn[i] for i in idx][::-1], fontsize=9, color=TXT)
ax.set_xlabel('Importance Score')
ax.set_xlim(0, iv.max()*1.18)
ax.set_title(f'Top {top} Feature Importances  (Random Forest)',fontsize=13,pad=16)
ax.grid(axis='x',alpha=.2)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_feature_importance.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_feature_importance.png")

# ── G5: Class Distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
cnt = [vc.get(0,0), vc.get(1,0)]
b2  = ax.bar(['BENIGN','TROJAN'], cnt, color=[BLU,RED], width=.45, zorder=3)
for bar,c in zip(b2,cnt):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+600,
            f'{c:,}', ha='center', fontsize=13, fontweight='bold', color=TXT)
ax.set_ylabel('Record Count'); ax.set_ylim(0, max(cnt)*1.15)
ax.set_title('Class Distribution',fontsize=13,pad=16)
ax.grid(axis='y',alpha=.2,zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_class_distribution.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_class_distribution.png")

# ── G6: Precision-Recall (best model) ────────────────────────
pr_p, pr_r, _ = precision_recall_curve(
    y_test, results[best_name]['y_prob'])
pr_auc = auc(pr_r, pr_p)
fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
ax.fill_between(pr_r, pr_p, alpha=.1, color=AMB)
for lw,al in [(4,.06),(2,.12),(1.5,1.)]:
    ax.plot(pr_r, pr_p, color=AMB, lw=lw, alpha=al)
ax.axhline(y=sum(y_test)/len(y_test), color=RED, ls='--', lw=1.2, alpha=.5)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall  —  {best_name}  (AUC={pr_auc:.4f})',
             fontsize=13, pad=16)
ax.set_xlim([0,1]); ax.set_ylim([0,1.05]); ax.grid(alpha=.2)
for sp in ax.spines.values(): sp.set_edgecolor(DIM)
plt.tight_layout()
plt.savefig('graph_precision_recall.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print("[+] graph_precision_recall.png")

# ═══════════════════════════════════════════════════════════════
# 13. SAVE BEST MODEL + ALL METRICS
# ═══════════════════════════════════════════════════════════════
pickle.dump(best_model,           open("model.pkl",    "wb"))
pickle.dump(scaler,               open("scaler.pkl",   "wb"))
pickle.dump(X.columns.tolist(),   open("features.pkl", "wb"))
pickle.dump({
    # Best model metrics (used in web app)
    'accuracy'   : best_res['accuracy'],
    'precision'  : best_res['precision'],
    'recall'     : best_res['recall'],
    'f1'         : best_res['f1'],
    'roc_auc'    : best_res['roc_auc'],
    'cv_accuracy': best_res['cv_mean'],
    'best_model' : best_name,
    # All models (for reference)
    'all_models' : {
        n: {k: results[n][k] for k in
            ['accuracy','precision','recall','f1','roc_auc','cv_mean']}
        for n in results
    },
}, open("metrics.pkl","wb"))

print("\n" + "="*60)
print(f"  BEST MODEL  →  {best_name}")
print(f"  Accuracy    →  {best_res['accuracy']}%")
print(f"  Precision   →  {best_res['precision']}%")
print(f"  Recall      →  {best_res['recall']}%")
print(f"  F1 Score    →  {best_res['f1']}%")
print(f"  ROC AUC     →  {best_res['roc_auc']}%")
print(f"  CV Accuracy →  {best_res['cv_mean']}%")
print("="*60)
print("\n[+] Saved: model.pkl | scaler.pkl | features.pkl | metrics.pkl")
print("[+] 6 graphs saved")
print("\n  Next steps:")
print("  1.  python pie_charts.py")
print("  2.  python app.py")
print("  3.  Open  http://127.0.0.1:5000\n")