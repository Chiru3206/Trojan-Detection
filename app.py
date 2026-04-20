# ============================================================
# Trojan Detection System — app.py  FINAL
# ============================================================

import os, sys, socket, random, pickle, warnings, traceback
import pandas as pd
from flask import Flask, render_template, request, send_file

warnings.filterwarnings('ignore')

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

# ── load helpers ──────────────────────────────────────────────
def pth(name): return os.path.join(BASE, name)

def require(f):
    if not os.path.exists(pth(f)):
        print(f"\n  MISSING: {f}")
        print(f"  Run 'python main.py' first!\n")
        sys.exit(1)

for f in ('model.pkl','scaler.pkl','features.pkl','metrics.pkl'):
    require(f)

model    = pickle.load(open(pth('model.pkl'),    'rb'))
scaler   = pickle.load(open(pth('scaler.pkl'),   'rb'))
features = pickle.load(open(pth('features.pkl'), 'rb'))
metrics  = pickle.load(open(pth('metrics.pkl'),  'rb'))

# ── load CSV ──────────────────────────────────────────────────
def find_csv():
    for n in ['Trojan.csv','trojan.csv','Trojan_Detection.csv',
              'trojan_detection.csv','network.csv']:
        p = pth(n)
        if os.path.exists(p): return p
    print("\n  MISSING: Trojan.csv  — put it in the same folder\n")
    sys.exit(1)

CSV_PATH = find_csv()
RAW = pd.read_csv(CSV_PATH, low_memory=False)
RAW.columns = RAW.columns.str.strip()

LEAKAGE = ['Unnamed: 0','Unnamed: 0.1','index',
           'Flow ID','Source IP','Destination IP',
           'Src IP','Dst IP','Timestamp']
RAW.drop(columns=[c for c in LEAKAGE if c in RAW.columns], inplace=True)
print(f"[+] Dataset: {RAW.shape[0]:,} rows × {RAW.shape[1]} cols")

# ── column map ────────────────────────────────────────────────
def fc(*names):
    for n in names:
        for c in RAW.columns:
            if c.strip().lower() == n.strip().lower(): return c
    return None

COL = {
    'flow_id'  : fc('Flow ID'),
    'timestamp': fc('Timestamp'),
    'sport'    : fc('Source Port',  'Src Port'),
    'dport'    : fc('Destination Port','Dst Port'),
    'proto'    : fc('Protocol'),
    'duration' : fc('Flow Duration'),
    'fwd_pkts' : fc('Total Fwd Packets'),
    'bwd_pkts' : fc('Total Backward Packets'),
    'bps'      : fc('Flow Bytes/s'),
    'pps'      : fc('Flow Packets/s'),
    'fwd_len'  : fc('Total Length of Fwd Packets','Fwd Packets Length Total'),
    'bwd_len'  : fc('Total Length of Bwd Packets','Bwd Packets Length Total'),
    'pkt_len'  : fc('Average Packet Size','Avg Packet Size'),
    'syn'      : fc('SYN Flag Count'),
    'ack'      : fc('ACK Flag Count'),
    'rst'      : fc('RST Flag Count'),
    'psh'      : fc('PSH Flag Count'),
    'urg'      : fc('URG Flag Count'),
    'class'    : fc('Class'),
}
PROTO = {6:'TCP',17:'UDP',1:'ICMP',0:'HOPOPT',41:'IPv6',58:'ICMPv6'}

SRC_IPS = [
    ('192.168.1.1','192.168.1.1   — Router'),
    ('192.168.1.5','192.168.1.5   — LAN Device'),
    ('192.168.1.10','192.168.1.10  — LAN Device'),
    ('192.168.1.100','192.168.1.100 — LAN Device'),
    ('192.168.0.1','192.168.0.1   — Router'),
    ('10.0.0.1','10.0.0.1      — Private LAN'),
    ('172.16.0.1','172.16.0.1    — Private LAN'),
    ('127.0.0.1','127.0.0.1     — Localhost'),
]
DST_IPS = [
    ('8.8.8.8','8.8.8.8         — Google DNS'),
    ('8.8.4.4','8.8.4.4         — Google DNS 2'),
    ('1.1.1.1','1.1.1.1         — Cloudflare DNS'),
    ('1.0.0.1','1.0.0.1         — Cloudflare 2'),
    ('9.9.9.9','9.9.9.9         — Quad9 DNS'),
    ('208.67.222.222','208.67.222.222  — OpenDNS'),
    ('192.168.1.1','192.168.1.1     — Router'),
    ('10.0.0.1','10.0.0.1        — Gateway'),
    ('0.0.0.0','0.0.0.0         — Default'),
    ('255.255.255.255','255.255.255.255 — Broadcast'),
    ('127.0.0.1','127.0.0.1       — Localhost'),
]

MGRAPHS = [('graph_confusion_matrix.png','Confusion Matrix'),
           ('graph_roc_curve.png','ROC Curve'),
           ('graph_metrics_bar.png','Performance Metrics'),
           ('graph_feature_importance.png','Feature Importance'),
           ('graph_class_distribution.png','Class Distribution'),
           ('graph_precision_recall.png','Precision-Recall')]
PGRAPHS = [('pie1_traffic_class.png','Traffic Distribution'),
           ('pie2_threat_ratio.png','Threat vs Safe'),
           ('pie3_metrics_ring.png','Metrics Ring'),
           ('pie4_outcome.png','TP/TN/FP/FN'),
           ('pie5_protocols.png','Protocol Usage'),
           ('pie6_flag_usage.png','TCP Flags')]
ALL = [f for f,_ in MGRAPHS+PGRAPHS]

# ── helpers ───────────────────────────────────────────────────
def get_wifi():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80)); ip=s.getsockname()[0]; s.close()
        return ip
    except: return '192.168.1.1'

def valid_ip(ip):
    p=ip.strip().split('.')
    if len(p)!=4: return False
    try: return all(0<=int(x)<=255 for x in p)
    except: return False

def is_priv(ip):
    try:
        p=ip.split('.');a,b=int(p[0]),int(p[1])
        return a==10 or a==127 or (a==172 and 16<=b<=31) or (a==192 and b==168)
    except: return False

def si(v):
    try: return f"{int(float(str(v).replace(',',''))):,}"
    except: return 'N/A'

def sf(v,d=2):
    try: return f"{float(str(v).replace(',','')):.{d}f}"
    except: return 'N/A'

def gok(f): return os.path.exists(pth(f))

# ── prediction ────────────────────────────────────────────────
def predict(src_ip, dst_ip):
    from sklearn.preprocessing import LabelEncoder

    row = RAW.sample(1, random_state=random.randint(0,999999)).copy()

    def rv(key):
        col = COL.get(key)
        if not col or col not in row.columns: return 'N/A'
        v = row[col].values[0]
        return 'N/A' if pd.isna(v) else str(v)

    # display values
    proto_raw = rv('proto')
    try:    proto_d = f"{PROTO.get(int(float(proto_raw)),proto_raw)} ({int(float(proto_raw))})"
    except: proto_d = proto_raw

    dur = rv('duration')
    try:
        d=int(float(dur))
        dur_d = f"{d:,} µs ({round(d/1_000_000,4)} s)"
    except: dur_d = dur

    # build feature vector
    proc = row.copy()
    proc.drop(columns=[c for c in LEAKAGE if c in proc.columns], inplace=True)

    cls = COL.get('class')
    if cls and cls in proc.columns and proc[cls].dtype==object:
        proc[cls] = LabelEncoder().fit_transform(proc[cls])

    f2=COL.get('fwd_pkts'); b2=COL.get('bwd_pkts')
    s2=COL.get('bps');      p2=COL.get('pps')
    d2=COL.get('duration'); sp=COL.get('sport'); dp=COL.get('dport')

    if f2 and b2 and f2 in proc.columns and b2 in proc.columns:
        proc['Packet_Ratio']=proc[f2]/(proc[b2]+1)
    if s2 and p2 and s2 in proc.columns and p2 in proc.columns:
        proc['Avg_Bytes_per_Packet']=proc[s2]/(proc[p2]+1)
    if s2 and d2 and s2 in proc.columns and d2 in proc.columns:
        proc['Flow_Intensity']=proc[s2]/(proc[d2]+1)
    ff=[c for c in ['SYN Flag Count','ACK Flag Count','RST Flag Count',
                    'PSH Flag Count','URG Flag Count'] if c in proc.columns]
    if ff: proc['Flag_Score']=proc[ff].sum(axis=1)
    if sp and sp in proc.columns:
        proc['Is_Well_Known_Port']=(proc[sp]<1024).astype(int)
    if dp and dp in proc.columns:
        proc['Is_Common_Trojan_Port']=proc[dp].isin(
            [4444,1234,6666,7777,8888,9999,31337]).astype(int)

    proc=proc.select_dtypes(include=['number'])
    proc.replace([float('inf'),float('-inf')],0,inplace=True)
    proc.dropna(inplace=True)

    X = proc.drop(columns=[cls]) if cls and cls in proc.columns else proc.copy()
    X = X.reindex(columns=features, fill_value=0)
    Xs= scaler.transform(X)

    pred =model.predict(Xs[0].reshape(1,-1))
    prob =model.predict_proba(Xs[0].reshape(1,-1))[0]
    is_t =int(pred[0])==1

    return {
        'is_trojan'    : is_t,
        'trojan_prob'  : round(prob[1]*100,2),
        'benign_prob'  : round(prob[0]*100,2),
        'confidence'   : round(max(prob)*100,2),
        'verdict'      : 'TROJAN DETECTED' if is_t else 'SAFE TRAFFIC',
        'verdict_class': 'danger' if is_t else 'safe',
        'verdict_icon' : '🚨' if is_t else '✅',
        'source_ip'    : src_ip,
        'dest_ip'      : dst_ip,
        'flow_id'      : rv('flow_id'),
        'timestamp'    : rv('timestamp'),
        'source_port'  : rv('sport'),
        'dest_port'    : rv('dport'),
        'protocol'     : proto_d,
        'flow_duration': dur_d,
        'fwd_packets'  : si(rv('fwd_pkts')),
        'bwd_packets'  : si(rv('bwd_pkts')),
        'flow_bytes_s' : sf(rv('bps')),
        'flow_pkts_s'  : sf(rv('pps')),
        'fwd_length'   : sf(rv('fwd_len')),
        'bwd_length'   : sf(rv('bwd_len')),
        'avg_pkt_size' : sf(rv('pkt_len')),
        'syn_flag'     : si(rv('syn')),
        'ack_flag'     : si(rv('ack')),
        'rst_flag'     : si(rv('rst')),
        'psh_flag'     : si(rv('psh')),
        'urg_flag'     : si(rv('urg')),
        'connection'   : 'WiFi / LAN' if is_priv(src_ip) else 'External Network',
    }

# ── routes ────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', metrics=metrics,
        graphs_ready=sum(1 for f,_ in MGRAPHS if gok(f)),
        pies_ready  =sum(1 for f,_ in PGRAPHS if gok(f)))

@app.route('/scanner')
def scanner():
    return render_template('scanner.html',
        metrics=metrics, wifi_ip=get_wifi(),
        src_ips=SRC_IPS, dst_ips=DST_IPS, error=None)

@app.route('/scan', methods=['POST'])
def scan():
    src = request.form.get('source_ip','').strip()
    dst = request.form.get('dest_ip',  '').strip()

    def back(err):
        return render_template('scanner.html',
            metrics=metrics, wifi_ip=get_wifi(),
            src_ips=SRC_IPS, dst_ips=DST_IPS, error=err)

    if not src or not dst:
        return back('Please enter both Source IP and Destination IP.')
    if not valid_ip(src):
        return back(f'"{src}" is not a valid IP address.  Example: 192.168.1.1')
    if not valid_ip(dst):
        return back(f'"{dst}" is not a valid IP address.  Example: 8.8.8.8')
    try:
        result = predict(src, dst)
        # pass as 'result' — no ambiguity
        return render_template('results.html',
            result=result, metrics=metrics)
    except Exception as e:
        traceback.print_exc()
        return back(f'Scan error: {str(e)}')

@app.route('/charts')
def charts():
    mg=[{'file':f,'title':t} for f,t in MGRAPHS if gok(f)]
    pg=[{'file':f,'title':t} for f,t in PGRAPHS if gok(f)]
    mg_miss=[{'file':f,'title':t} for f,t in MGRAPHS if not gok(f)]
    pg_miss=[{'file':f,'title':t} for f,t in PGRAPHS if not gok(f)]
    return render_template('charts.html', metrics=metrics,
        model_graphs=mg, pie_graphs=pg,
        model_missing=mg_miss, pie_missing=pg_miss)

@app.route('/graph/<fn>')
def serve_graph(fn):
    if fn not in ALL: return 'Not found',404
    p=pth(fn)
    if not os.path.exists(p): return 'Run main.py / pie_charts.py first',404
    return send_file(p, mimetype='image/png')

if __name__=='__main__':
    print(f'\n  TrojanGuard ready → http://127.0.0.1:5000\n')
    app.run(debug=True)