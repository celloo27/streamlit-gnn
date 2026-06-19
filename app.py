import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import re
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import community as community_louvain
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Dashboard — Immoderma Skin Clinic",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #444 !important; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; }

    .metric-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-box .val {
        font-size: 2rem; font-weight: 700; color: #2c3e50; line-height: 1.1;
    }
    .metric-box .lbl {
        font-size: 0.75rem; color: #888; text-transform: uppercase;
        letter-spacing: 0.06em; margin-top: 4px;
    }
    .metric-box .sub {
        font-size: 0.7rem; color: #aaa; margin-top: 2px;
    }

    .best-badge {
        background: #d4edda; color: #155724;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.72rem; font-weight: 600;
    }
    .tier1-pill { background:#fde8e8; color:#c0392b; border-radius:10px; padding:2px 9px; font-size:0.72rem; font-weight:600; }
    .tier2-pill { background:#fef3cd; color:#856404; border-radius:10px; padding:2px 9px; font-size:0.72rem; font-weight:600; }
    .inf-pill   { background:#d4edda; color:#155724; border-radius:10px; padding:2px 9px; font-size:0.72rem; font-weight:600; }

    .section-header {
        border-left: 4px solid #764ba2;
        padding-left: 10px;
        margin: 1rem 0 0.5rem 0;
        font-size: 1rem; font-weight: 600; color: #2c3e50;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stMetric"] { background: #f8f9fa; border-radius: 10px; padding: 10px 14px; }
</style>
""", unsafe_allow_html=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL FUNCTIONS (cached)
# ══════════════════════════════════════════════════════════════════════════════

def is_valid(u):
    if pd.isna(u): return False
    u = str(u).strip().lower()
    if u in ['immoderma_purwokerto', 'suka', '']: return False
    if re.fullmatch(r'\d+', u): return False
    if u.startswith('#'): return False
    return True

def parse_likes(v):
    m = re.match(r'^(\d+)\s*suka', str(v).strip().lower()) if not pd.isna(v) else None
    return int(m.group(1)) if m else 0

def parse_replies(v):
    m = re.search(r'\((\d+)\)', str(v)) if not pd.isna(v) else None
    return int(m.group(1)) if m else 0


@st.cache_data(show_spinner=False)
def load_and_process(uploaded_file):
    df = pd.read_csv(uploaded_file)
    col_map = {
        'x1i10hfl': 'akun_utama', '_ap3a': 'caption_post',
        'x1i10hfl 7': 'username_commenter', '_ap3a 2': 'teks_komentar',
        'x193iq5w': 'reaksi_komentar', '_a9yi': 'balasan_komentar',
        'x1i10hfl 8': 'username_reply_1',
    }
    df = df.rename(columns=col_map)
    df_c = df[df['username_commenter'].notna() & df['teks_komentar'].notna()].copy()
    df_c = df_c[df_c['username_commenter'].apply(is_valid)].copy()
    df_c = df_c.drop_duplicates(subset=['username_commenter', 'teks_komentar']).reset_index(drop=True)
    df_c['comment_likes'] = df_c['reaksi_komentar'].apply(parse_likes)
    df_c['reply_count']   = df_c['balasan_komentar'].apply(parse_replies)
    return df_c


@st.cache_data(show_spinner=False)
def build_graph(_df_c):
    KLINIK = 'immoderma_purwokerto'
    G = nx.DiGraph()
    G.add_node(KLINIK)
    for _, r in _df_c.iterrows():
        u = r['username_commenter']
        if not G.has_node(u): G.add_node(u)
        if G.has_edge(u, KLINIK): G[u][KLINIK]['weight'] += 1
        else: G.add_edge(u, KLINIK, weight=1)
        if r['reply_count'] > 0:
            if G.has_edge(KLINIK, u): G[KLINIK][u]['weight'] += 2
            else: G.add_edge(KLINIK, u, weight=2)
        r1 = r['username_reply_1']
        if is_valid(r1) and r1 != u:
            if not G.has_node(r1): G.add_node(r1)
            if G.has_edge(u, r1): G[u][r1]['weight'] += 3
            else: G.add_edge(u, r1, weight=3)
    return G


@st.cache_data(show_spinner=False)
def compute_sna(_G, _df_c):
    KLINIK = 'immoderma_purwokerto'
    customers = [n for n in _G.nodes() if n != KLINIK]
    deg_c  = nx.in_degree_centrality(_G)
    btwn   = nx.betweenness_centrality(_G, normalized=True, k=min(400, _G.number_of_nodes()))
    pr     = nx.pagerank(_G, alpha=0.85)
    lcc    = max(nx.weakly_connected_components(_G), key=len)
    clos   = nx.closeness_centrality(_G.subgraph(lcc))
    in_deg = dict(_G.in_degree())
    out_deg = dict(_G.out_degree())
    in_wdeg  = {n: sum(d.get('weight', 1) for _, _, d in _G.in_edges(n, data=True)) for n in _G.nodes()}
    out_wdeg = {n: sum(d.get('weight', 1) for _, _, d in _G.out_edges(n, data=True)) for n in _G.nodes()}

    sna = pd.DataFrame({'username': customers})
    for col, d in [('in_degree', in_deg), ('out_degree', out_deg),
                   ('in_wdeg', in_wdeg), ('out_wdeg', out_wdeg),
                   ('degree_c', deg_c), ('betweenness', btwn),
                   ('pagerank', pr), ('closeness', clos)]:
        sna[col] = sna['username'].map(d).fillna(0)

    us = _df_c.groupby('username_commenter').agg(
        n_comments   = ('teks_komentar', 'count'),
        total_likes  = ('comment_likes', 'sum'),
        total_replies= ('reply_count', 'sum'),
    ).reset_index().rename(columns={'username_commenter': 'username'})
    sna = sna.merge(us, on='username', how='left').fillna(0)
    sna['engagement_score']  = sna['n_comments'] * 1 + sna['total_likes'] * 2 + sna['total_replies'] * 3
    sna['reply_per_comment'] = np.where(sna['n_comments'] > 0, sna['total_replies'] / sna['n_comments'], 0)

    sna['sna_composite'] = (
        sna['degree_c'].rank(pct=True) * 0.30 +
        sna['betweenness'].rank(pct=True) * 0.30 +
        sna['pagerank'].rank(pct=True) * 0.30 +
        sna['engagement_score'].rank(pct=True) * 0.10
    )
    threshold = sna['sna_composite'].quantile(0.95)
    sna['label'] = (sna['sna_composite'] >= threshold).astype(int)
    return sna


@st.cache_data(show_spinner=False)
def compute_community(_G):
    G_und = _G.to_undirected()
    partition  = community_louvain.best_partition(G_und, random_state=SEED)
    modularity = community_louvain.modularity(partition, G_und)
    return partition, modularity


# ── GNN MODELS ────────────────────────────────────────────────────────────────
class ImprovedGCN(nn.Module):
    def __init__(self, in_c, hid, out_c, drop=0.4):
        super().__init__()
        self.c1=GCNConv(in_c,hid); self.c2=GCNConv(hid,hid); self.c3=GCNConv(hid,hid)
        self.bn1=nn.BatchNorm1d(hid); self.bn2=nn.BatchNorm1d(hid); self.bn3=nn.BatchNorm1d(hid)
        self.skip=nn.Linear(in_c,hid)
        self.head=nn.Sequential(nn.Linear(hid,hid//2),nn.ReLU(),nn.Dropout(drop),nn.Linear(hid//2,out_c))
        self.drop=drop
    def forward(self, x, ei):
        sk=self.skip(x)
        h=F.dropout(F.relu(self.bn1(self.c1(x,ei))),p=self.drop,training=self.training)
        h=F.dropout(F.relu(self.bn2(self.c2(h,ei))),p=self.drop,training=self.training)
        return self.head(F.relu(self.bn3(self.c3(h,ei)))+sk)

class ImprovedGAT(nn.Module):
    def __init__(self, in_c, hid, out_c, heads=4, drop=0.4):
        super().__init__()
        self.c1=GATConv(in_c,hid,heads=heads,dropout=drop,concat=True)
        self.c2=GATConv(hid*heads,hid,heads=heads,dropout=drop,concat=True)
        self.c3=GATConv(hid*heads,hid,heads=1,dropout=drop,concat=False)
        self.bn1=nn.BatchNorm1d(hid*heads); self.bn2=nn.BatchNorm1d(hid*heads); self.bn3=nn.BatchNorm1d(hid)
        self.skip=nn.Linear(in_c,hid)
        self.head=nn.Sequential(nn.Linear(hid,hid//2),nn.ReLU(),nn.Dropout(drop),nn.Linear(hid//2,out_c))
        self.drop=drop
    def forward(self, x, ei):
        sk=self.skip(x)
        h=F.dropout(F.elu(self.bn1(self.c1(x,ei))),p=self.drop,training=self.training)
        h=F.dropout(F.elu(self.bn2(self.c2(h,ei))),p=self.drop,training=self.training)
        return self.head(F.elu(self.bn3(self.c3(h,ei)))+sk)

class ImprovedGIN(nn.Module):
    def __init__(self, in_c, hid, out_c, drop=0.4):
        super().__init__()
        def mlp(a,b): return nn.Sequential(nn.Linear(a,b),nn.BatchNorm1d(b),nn.ReLU(),nn.Dropout(drop),nn.Linear(b,b),nn.BatchNorm1d(b))
        self.c1=GINConv(mlp(in_c,hid),train_eps=True)
        self.c2=GINConv(mlp(hid,hid),train_eps=True)
        self.c3=GINConv(mlp(hid,hid),train_eps=True)
        self.skip=nn.Linear(in_c,hid)
        self.head=nn.Sequential(nn.Linear(hid,hid//2),nn.ReLU(),nn.Dropout(drop),nn.Linear(hid//2,out_c))
        self.drop=drop
    def forward(self, x, ei):
        sk=self.skip(x)
        h=F.dropout(F.relu(self.c1(x,ei)),p=self.drop,training=self.training)
        h=F.dropout(F.relu(self.c2(h,ei)),p=self.drop,training=self.training)
        return self.head(F.relu(self.c3(h,ei))+sk)


@st.cache_data(show_spinner=False)
def train_all_models(_sna, _G):
    FEAT_COLS = ['in_degree','out_degree','in_wdeg','out_wdeg','degree_c',
                 'betweenness','pagerank','closeness','n_comments',
                 'engagement_score','reply_per_comment']
    X = StandardScaler().fit_transform(_sna[FEAT_COLS].values.astype(float))
    y = _sna['label'].values
    N = len(y)

    node_list = _sna['username'].tolist()
    n2i = {n: i for i, n in enumerate(node_list)}
    src_l, tgt_l = [], []
    for u, v in _G.edges():
        if u in n2i and v in n2i:
            src_l.append(n2i[u]); tgt_l.append(n2i[v])
    edge_index = torch.tensor([src_l, tgt_l], dtype=torch.long)
    x_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    idx = np.arange(N)
    tr_idx, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    vl_idx, te_idx = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    tr_m = torch.zeros(N, dtype=torch.bool); tr_m[tr_idx] = True
    vl_m = torch.zeros(N, dtype=torch.bool); vl_m[vl_idx] = True
    te_m = torch.zeros(N, dtype=torch.bool); te_m[te_idx] = True

    cw = compute_class_weight('balanced', classes=np.unique(y[tr_idx]), y=y[tr_idx])
    cw_t = torch.tensor(cw, dtype=torch.float)

    IN, HID, OUT = len(FEAT_COLS), 128, 2
    results = {}

    def run(ModelCls, kwargs, name, epochs=300, lr=5e-4, wd=1e-4, pat=50):
        m = ModelCls(**kwargs)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=20, factor=0.5)
        crit = nn.CrossEntropyLoss(weight=cw_t)
        best_vf1, best_state, no_imp = 0, None, 0
        loss_hist, vf1_hist = [], []
        for ep in range(1, epochs + 1):
            m.train(); opt.zero_grad()
            out = m(x_t, edge_index); loss = crit(out[tr_m], y_t[tr_m])
            loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                out2 = m(x_t, edge_index)
                vf1 = f1_score(y_t[vl_m].numpy(), out2[vl_m].argmax(1).numpy(), zero_division=0)
            sched.step(vf1)
            loss_hist.append(round(loss.item(), 4)); vf1_hist.append(round(vf1, 4))
            if vf1 > best_vf1:
                best_vf1 = vf1
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= pat: break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            out = m(x_t, edge_index)
            yp = out[te_m].argmax(1).numpy()
            yprob = F.softmax(out[te_m], dim=1)[:, 1].numpy()
        yt = y_t[te_m].numpy()
        try: auc_v = round(roc_auc_score(yt, yprob), 4)
        except: auc_v = 0.0
        return m, {
            'name': name,
            'accuracy':  round(accuracy_score(yt, yp), 4),
            'precision': round(precision_score(yt, yp, zero_division=0), 4),
            'recall':    round(recall_score(yt, yp, zero_division=0), 4),
            'f1':        round(f1_score(yt, yp, zero_division=0), 4),
            'auc':       auc_v,
            'yt': yt, 'yp': yp, 'yprob': yprob,
            'loss_hist': loss_hist, 'vf1_hist': vf1_hist,
        }

    gcn_m, gcn_r = run(ImprovedGCN, dict(in_c=IN, hid=HID, out_c=OUT), 'GCN')
    gat_m, gat_r = run(ImprovedGAT, dict(in_c=IN, hid=HID, out_c=OUT, heads=4), 'GAT')
    gin_m, gin_r = run(ImprovedGIN, dict(in_c=IN, hid=HID, out_c=OUT), 'GIN')

    all_models = {'GCN': (gcn_m, gcn_r), 'GAT': (gat_m, gat_r), 'GIN': (gin_m, gin_r)}
    best_name  = max(['GCN','GAT','GIN'], key=lambda k: all_models[k][1]['f1'])
    best_m     = all_models[best_name][0]

    best_m.eval()
    with torch.no_grad():
        out_all   = best_m(x_t, edge_index)
        probs_all = F.softmax(out_all, dim=1)[:, 1].numpy()
    _sna = _sna.copy()
    _sna['prob_inf'] = probs_all

    te_sna = _sna.iloc[te_idx]
    yt_te  = y[te_idx]
    def bl(col):
        th = np.percentile(te_sna[col].values, 100*(1-yt_te.mean()))
        yp = (te_sna[col].values >= th).astype(int)
        try: auc_v = round(roc_auc_score(yt_te, te_sna[col].values), 4)
        except: auc_v = 0.0
        return {'accuracy': round(accuracy_score(yt_te, yp), 4),
                'precision': round(precision_score(yt_te, yp, zero_division=0), 4),
                'recall':    round(recall_score(yt_te, yp, zero_division=0), 4),
                'f1':        round(f1_score(yt_te, yp, zero_division=0), 4),
                'auc':       auc_v,
                'yt': yt_te, 'yp': yp, 'yprob': te_sna[col].values}

    return {
        'models': {'GCN': gcn_r, 'GAT': gat_r, 'GIN': gin_r,
                   'Degree Centrality': bl('degree_c'),
                   'Betweenness Centrality': bl('betweenness'),
                   'PageRank': bl('pagerank')},
        'best': best_name,
        'sna_with_prob': _sna,
        'edge_index': edge_index,
        'n2i': n2i,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 GNN Dashboard")
    st.markdown("**Immoderma Skin Clinic Purwokerto**")
    st.markdown("Prince Bayu Saputra · 2211103077")
    st.divider()

    uploaded = st.file_uploader("📂 Upload instagram.csv", type=["csv"])

    st.divider()
    st.markdown("### ⚙️ Konfigurasi Model")
    tier1_thresh = st.slider("Threshold Tier 1 (prob ≥)", 0.50, 0.95, 0.70, 0.05)
    tier2_thresh = st.slider("Threshold Tier 2 (prob ≥)", 0.30, 0.70, 0.50, 0.05)

    st.divider()
    st.markdown("### 📋 Navigasi")
    page = st.radio("Halaman", [
        "🏠 Overview",
        "📊 Performa Model",
        "🌐 Struktur Jaringan",
        "⭐ Influencer & Target",
        "📁 Export Data",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🧠 GNN Social Network Dashboard")
st.markdown("**Analisis Jaringan Sosial Instagram @immodermaskinclinic · Model: GCN · GAT · GIN**")
st.divider()

if uploaded is None:
    st.info("👈 Upload file **instagram.csv** di sidebar untuk memulai analisis.")
    st.markdown("""
    **Cara penggunaan:**
    1. Upload file `instagram.csv` (hasil scraping Instagram)
    2. Dashboard otomatis memproses data, membangun graf, dan melatih model GNN
    3. Jelajahi hasil melalui menu navigasi di sidebar

    **Pipeline:**
    `CSV` → `Preprocessing` → `Graf` → `SNA` → `GNN Training` → `Identifikasi Influencer`
    """)
    st.stop()

# ── PROCESS DATA ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Memuat dan memproses data..."):
    df_c = load_and_process(uploaded)

with st.spinner("🕸️ Membangun graf jaringan sosial..."):
    G = build_graph(df_c)

with st.spinner("📐 Menghitung metrik SNA..."):
    sna = compute_sna(G, df_c)
    partition, modularity = compute_community(G)

with st.spinner("🤖 Melatih model GNN (GCN, GAT, GIN)... mungkin 1–2 menit"):
    train_out = train_all_models(sna, G)

models_res  = train_out['models']
best_name   = train_out['best']
sna_prob    = train_out['sna_with_prob']
KLINIK      = 'immoderma_purwokerto'
customers   = [n for n in G.nodes() if n != KLINIK]
comps       = list(nx.weakly_connected_components(G))

# Tier classification
tier1 = sna_prob[sna_prob['prob_inf'] >= tier1_thresh].sort_values('prob_inf', ascending=False)
tier2 = sna_prob[(sna_prob['prob_inf'] >= tier2_thresh) & (sna_prob['prob_inf'] < tier1_thresh)].sort_values('prob_inf', ascending=False)
inf_set = set(tier1['username']) | set(tier2['username'])

influencee_list = []
n2i = train_out['n2i']
for nd in G.nodes():
    if nd == KLINIK or nd in inf_set or nd not in n2i: continue
    nb = (set(G.predecessors(nd)) | set(G.successors(nd))) & inf_set
    if nb:
        influencee_list.append({'username': nd, 'n_inf_neighbors': len(nb),
                                'via_influencer': ', '.join(list(nb)[:2]),
                                'prob_inf': round(float(sna_prob.loc[sna_prob['username']==nd,'prob_inf'].values[0]) if nd in sna_prob['username'].values else 0, 4)})
inf_df = pd.DataFrame(influencee_list).sort_values('n_inf_neighbors', ascending=False).reset_index(drop=True)

gin_r   = models_res['GIN']
best_r  = models_res[best_name]

# ── COLOR PALETTE ─────────────────────────────────────────────────────────────
C = {'gcn':'#3498db','gat':'#e67e22','gin':'#e74c3c',
     'dc':'#95a5a6','bc':'#7f8c8d','pr':'#bdc3c7',
     'tier1':'#e74c3c','tier2':'#e67e22','inf':'#27ae60','klinik':'#2c3e50'}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl, sub in [
        (c1, f"{G.number_of_nodes():,}", "Total Node", "incl. klinik"),
        (c2, f"{G.number_of_edges():,}", "Total Edge", "directed weighted"),
        (c3, f"{len(customers):,}", "Pelanggan", "unique user"),
        (c4, f"{len(partition):,}", "Komunitas", f"modularity {modularity:.3f}"),
        (c5, f"{len(tier1):,}", "Influencer T1", f"prob ≥ {tier1_thresh}"),
        (c6, f"{len(inf_df):,}", "Target Promo", "influencee 1-hop"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🥧 Distribusi Tier Pelanggan")
        pie_data = {
            'Influencer Tier 1': len(tier1),
            'Influencer Tier 2': len(tier2),
            'Influencee (Target)': len(inf_df),
            'Pelanggan Reguler': len(customers) - len(tier1) - len(tier2) - len(inf_df),
        }
        fig_pie = go.Figure(go.Pie(
            labels=list(pie_data.keys()), values=list(pie_data.values()),
            hole=0.55, marker_colors=[C['tier1'], C['tier2'], C['inf'], '#bdc3c7'],
            textinfo='label+percent', textfont_size=11,
        ))
        fig_pie.update_layout(margin=dict(t=10,b=10,l=0,r=0), height=300,
                              showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### 🕸️ Statistik Graf")
        stat_df = pd.DataFrame({
            'Parameter': ['Jenis Graf','Total Node','Total Edge','Graph Density',
                          'Komponen','Modularity','Label Influencer','Imbalance'],
            'Nilai': ['Directed Weighted', f"{G.number_of_nodes():,}", f"{G.number_of_edges():,}",
                      f"{nx.density(G):.5f}", str(len(comps)),
                      f"{modularity:.4f}", f"{sna['label'].sum()} (5%)",
                      f"1 : {(sna['label']==0).sum() // max(sna['label'].sum(),1)}"]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True, height=280)

    st.markdown("---")
    st.markdown("#### 🏆 Perbandingan Cepat Model")
    comp_rows = []
    for nm in ['Degree Centrality','Betweenness Centrality','PageRank','GCN','GAT','GIN']:
        mr = models_res[nm]
        comp_rows.append({'Model': nm + (' ★' if nm == best_name else ''),
                          'Accuracy': mr['accuracy'], 'Precision': mr['precision'],
                          'Recall': mr['recall'], 'F1-Score': mr['f1'], 'AUC-ROC': mr['auc']})
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df.style.highlight_max(
        subset=['Accuracy','Precision','Recall','F1-Score','AUC-ROC'],
        color='#d4edda'), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PERFORMA MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Performa Model":

    st.markdown("### 📈 Learning Curves")
    tab1, tab2 = st.tabs(["Validation F1-Score", "Training Loss"])

    with tab1:
        fig_lc = go.Figure()
        for nm, col in [('GCN', C['gcn']), ('GAT', C['gat']), ('GIN', C['gin'])]:
            vf1 = models_res[nm]['vf1_hist']
            fig_lc.add_trace(go.Scatter(
                y=vf1, mode='lines', name=nm, line=dict(color=col, width=2),
                fill='tozeroy' if nm == best_name else None,
                fillcolor=col.replace(')', ',0.07)').replace('rgb','rgba') if nm == best_name else None,
            ))
        fig_lc.add_hline(y=0.8, line_dash='dash', line_color='gray', opacity=0.5, annotation_text="F1 = 0.8")
        fig_lc.update_layout(xaxis_title='Epoch', yaxis_title='Val F1-Score',
                             yaxis=dict(range=[0, 1.05]), height=320,
                             legend=dict(x=0.01, y=0.99), margin=dict(t=10,b=30))
        st.plotly_chart(fig_lc, use_container_width=True)

    with tab2:
        fig_loss = go.Figure()
        for nm, col in [('GCN', C['gcn']), ('GAT', C['gat']), ('GIN', C['gin'])]:
            fig_loss.add_trace(go.Scatter(
                y=models_res[nm]['loss_hist'], mode='lines', name=nm,
                line=dict(color=col, width=2)))
        fig_loss.update_layout(xaxis_title='Epoch', yaxis_title='Cross-Entropy Loss',
                               height=320, margin=dict(t=10,b=30),
                               legend=dict(x=0.7, y=0.99))
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📊 Bar — F1-Score & AUC-ROC")
        bar_models = ['Betweenness\nCentrality','PageRank','Degree\nCentrality','GCN','GAT','GIN']
        bar_keys   = ['Betweenness Centrality','PageRank','Degree Centrality','GCN','GAT','GIN']
        bar_colors = ['#95a5a6','#bdc3c7','#7f8c8d', C['gcn'], C['gat'], C['gin']]
        fig_bar = make_subplots(rows=1, cols=2, subplot_titles=('F1-Score','AUC-ROC'))
        for i, metric in enumerate(['f1','auc'], 1):
            vals = [models_res[k][metric] for k in bar_keys]
            fig_bar.add_trace(go.Bar(
                y=bar_models, x=vals, orientation='h',
                marker_color=bar_colors, showlegend=False,
                text=[f"{v:.4f}" for v in vals], textposition='outside',
                textfont=dict(size=10)
            ), row=1, col=i)
        fig_bar.update_layout(height=300, margin=dict(t=30,b=10,l=120,r=60))
        fig_bar.update_xaxes(range=[0, 1.08])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.markdown(f"### 🔢 Confusion Matrix — {best_name} (Best)")
        br = best_r
        cm = confusion_matrix(br['yt'], br['yp'])
        fig_cm = go.Figure(go.Heatmap(
            z=[[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]],
            x=['Prediksi: Influencer','Prediksi: Non-Inf'],
            y=['Aktual: Influencer','Aktual: Non-Inf'],
            text=[[str(cm[1,1]),str(cm[1,0])],[str(cm[0,1]),str(cm[0,0])]],
            texttemplate='<b>%{text}</b>',
            colorscale=[[0,'#f8f9fa'],[0.5,'#85b7eb'],[1,'#185fa5']],
            showscale=False, textfont=dict(size=22),
        ))
        fig_cm.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📉 ROC Curve")
    fig_roc = go.Figure()
    for nm, col in [('GCN', C['gcn']), ('GAT', C['gat']), ('GIN', C['gin'])]:
        r = models_res[nm]
        fpr, tpr, _ = roc_curve(r['yt'], r['yprob'])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{nm} (AUC={r['auc']:.4f})",
                                     line=dict(color=col, width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                 line=dict(color='gray', dash='dash', width=1)))
    fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          height=350, legend=dict(x=0.55, y=0.1), margin=dict(t=10,b=30))
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🕸️ Radar — Profil Performa Semua Metrik")
    cats = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
    fig_radar = go.Figure()
    for nm, col in [('GCN', C['gcn']), ('GAT', C['gat']), ('GIN', C['gin'])]:
        r = models_res[nm]
        vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['auc']]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], name=nm, fill='toself',
            line_color=col, opacity=0.75 if nm == best_name else 0.4,
            fillcolor=col
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0.5, 1.05])),
                            height=360, margin=dict(t=20,b=20))
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STRUKTUR JARINGAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Struktur Jaringan":

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Node", f"{G.number_of_nodes():,}")
    c2.metric("Total Edge", f"{G.number_of_edges():,}")
    c3.metric("Modularity", f"{modularity:.4f}")
    c4.metric("Density", f"{nx.density(G):.5f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 📊 Distribusi In-Degree (Top 30)")
        top_deg = sorted([(n, G.in_degree(n)) for n in customers], key=lambda x:-x[1])[:30]
        fig_deg = go.Figure(go.Bar(
            x=[x[0][:15] for x in top_deg],
            y=[x[1] for x in top_deg],
            marker_color=[C['tier1'] if x[0] in inf_set else '#85b7eb' for x in top_deg],
        ))
        fig_deg.update_layout(xaxis_tickangle=-40, height=300,
                              margin=dict(t=10,b=80,l=0,r=0),
                              yaxis_title='In-Degree')
        st.plotly_chart(fig_deg, use_container_width=True)

    with col_r:
        st.markdown("#### 🗺️ Distribusi PageRank (Top 30)")
        top_pr = sna.sort_values('pagerank', ascending=False).head(30)
        fig_pr = go.Figure(go.Bar(
            x=top_pr['username'].str[:15],
            y=top_pr['pagerank'],
            marker_color=[C['tier1'] if u in inf_set else C['gat'] for u in top_pr['username']],
        ))
        fig_pr.update_layout(xaxis_tickangle=-40, height=300,
                             margin=dict(t=10,b=80,l=0,r=0),
                             yaxis_title='PageRank')
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔗 Scatter — Betweenness vs PageRank")
    fig_scat = go.Figure()
    sna_plot = sna[sna['pagerank'] > sna['pagerank'].median()].copy()
    sna_plot['tier'] = sna_plot['username'].apply(
        lambda u: 'Tier 1' if u in set(tier1['username'])
        else ('Tier 2' if u in set(tier2['username']) else 'Reguler')
    )
    for tier_name, col in [('Tier 1', C['tier1']), ('Tier 2', C['tier2']), ('Reguler', '#bdc3c7')]:
        sub = sna_plot[sna_plot['tier'] == tier_name]
        fig_scat.add_trace(go.Scatter(
            x=sub['betweenness'], y=sub['pagerank'],
            mode='markers', name=tier_name,
            marker=dict(color=col, size=7 if tier_name != 'Reguler' else 4, opacity=0.8),
            text=sub['username'], hovertemplate='%{text}<br>BW: %{x:.5f}<br>PR: %{y:.5f}'
        ))
    fig_scat.update_layout(xaxis_title='Betweenness Centrality', yaxis_title='PageRank',
                           height=350, margin=dict(t=10,b=30))
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌐 Visualisasi Jaringan (Force-Directed Subgraph)")
    important_nodes = {KLINIK} | set(tier1['username'].head(20)) | set(tier2['username'].head(10)) | set(inf_df['username'].head(20))
    extra = set()
    for nd in important_nodes:
        if nd in G:
            extra.update(list(G.predecessors(nd))[:2])
            extra.update(list(G.successors(nd))[:2])
    lcc_nodes = max(nx.weakly_connected_components(G), key=len)
    G_plot = G.subgraph((important_nodes | extra) & lcc_nodes).copy()

    pos = nx.spring_layout(G_plot, seed=SEED, k=1.8)
    node_x, node_y, node_col, node_size, node_text = [], [], [], [], []
    for nd in G_plot.nodes():
        x, y = pos[nd]
        node_x.append(x); node_y.append(y)
        node_text.append(f"@{nd}")
        if nd == KLINIK:      node_col.append(C['klinik']); node_size.append(22)
        elif nd in set(tier1['username']): node_col.append(C['tier1']); node_size.append(14)
        elif nd in set(tier2['username']): node_col.append(C['tier2']); node_size.append(11)
        elif nd in set(inf_df['username']): node_col.append(C['inf']); node_size.append(8)
        else:                  node_col.append('#bdc3c7'); node_size.append(5)

    edge_x, edge_y = [], []
    for u, v in G_plot.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                  line=dict(width=0.5, color='#cccccc'), hoverinfo='none'))
    fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                  marker=dict(size=node_size, color=node_col, line=dict(width=0.5, color='white')),
                                  text=['@'+nd[:10] if nd in important_nodes else '' for nd in G_plot.nodes()],
                                  textposition='top center', textfont=dict(size=8),
                                  hovertext=node_text, hoverinfo='text'))
    fig_net.update_layout(height=480, showlegend=False,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_net, use_container_width=True)

    st.caption(f"🔴 Tier 1 · 🟠 Tier 2 · 🟢 Influencee · ⚫ Klinik · ⚪ Reguler — menampilkan {G_plot.number_of_nodes()} node dari {G.number_of_nodes():,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INFLUENCER & TARGET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Influencer & Target":

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Influencer Tier 1", len(tier1), f"prob ≥ {tier1_thresh}")
    c2.metric("🟠 Influencer Tier 2", len(tier2), f"prob {tier2_thresh}–{tier1_thresh}")
    c3.metric("🟢 Target Promosi", len(inf_df), "influencee 1-hop")
    c4.metric("📊 Coverage", f"{100*(len(tier1)+len(tier2)+len(inf_df))/len(customers):.1f}%", "dari pelanggan")

    st.markdown("---")
    tab_t1, tab_t2, tab_inf, tab_chart = st.tabs(["🔴 Tier 1", "🟠 Tier 2", "🟢 Target Promosi", "📊 Visualisasi"])

    with tab_t1:
        st.markdown(f"**{len(tier1)} Influencer Tier 1** (probabilitas ≥ {tier1_thresh})")
        show_t1 = tier1[['username','prob_inf','in_degree','betweenness','pagerank','n_comments','total_replies','engagement_score']].copy()
        show_t1.columns = ['Username','Prob. Influencer','In-Degree','Betweenness','PageRank','Komentar','Balasan','Eng. Score']
        show_t1 = show_t1.reset_index(drop=True)
        show_t1.index += 1
        st.dataframe(
    show_t1,
    use_container_width=True,
    height=420
)

    with tab_t2:
        st.markdown(f"**{len(tier2)} Influencer Tier 2** (probabilitas {tier2_thresh}–{tier1_thresh})")
        show_t2 = tier2[['username','prob_inf','in_degree','betweenness','pagerank','n_comments','total_replies']].copy()
        show_t2.columns = ['Username','Prob. Influencer','In-Degree','Betweenness','PageRank','Komentar','Balasan']
        show_t2 = show_t2.reset_index(drop=True); show_t2.index += 1
        st.dataframe(show_t2.style.background_gradient(subset=['Prob. Influencer'], cmap='Oranges'),
                     use_container_width=True)

    with tab_inf:
        st.markdown(f"**{len(inf_df)} Influencee** — target promosi bundling (terhubung langsung ke influencer)")
        show_inf = inf_df[['username','n_inf_neighbors','via_influencer','prob_inf']].copy()
        show_inf.columns = ['Username','Jml. Influencer Terhubung','Via Influencer','Prob. Inf']
        show_inf.index += 1
        st.dataframe(show_inf.style.background_gradient(subset=['Prob. Inf'], cmap='Greens'),
                     use_container_width=True, height=420)

    with tab_chart:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Top 15 Influencer — Engagement Score")
            top15 = tier1.head(15).sort_values('engagement_score')
            fig_eng = go.Figure(go.Bar(
                y=top15['username'].str[:18],
                x=top15['engagement_score'],
                orientation='h',
                marker_color=[C['tier1'] if i == len(top15)-1 else C['gcn']
                              for i in range(len(top15))],
                text=top15['engagement_score'].astype(int),
                textposition='outside'
            ))
            fig_eng.update_layout(height=380, xaxis_title='Engagement Score',
                                  margin=dict(t=10,b=10,l=0,r=50))
            st.plotly_chart(fig_eng, use_container_width=True)

        with col_r:
            st.markdown("#### Distribusi Probabilitas Influencer")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=sna_prob[sna_prob['prob_inf'] < tier2_thresh]['prob_inf'],
                name='Reguler', marker_color='#bdc3c7', opacity=0.8, nbinsx=30))
            fig_hist.add_trace(go.Histogram(
                x=sna_prob[(sna_prob['prob_inf'] >= tier2_thresh) & (sna_prob['prob_inf'] < tier1_thresh)]['prob_inf'],
                name='Tier 2', marker_color=C['tier2'], opacity=0.85, nbinsx=10))
            fig_hist.add_trace(go.Histogram(
                x=sna_prob[sna_prob['prob_inf'] >= tier1_thresh]['prob_inf'],
                name='Tier 1', marker_color=C['tier1'], opacity=0.9, nbinsx=10))
            fig_hist.add_vline(x=tier1_thresh, line_dash='dash', line_color='red',
                               annotation_text=f"T1={tier1_thresh}")
            fig_hist.add_vline(x=tier2_thresh, line_dash='dash', line_color='orange',
                               annotation_text=f"T2={tier2_thresh}")
            fig_hist.update_layout(barmode='overlay', height=380,
                                   xaxis_title='Prob. Influencer',
                                   legend=dict(x=0.6, y=0.95),
                                   margin=dict(t=10,b=30))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Scatter — Probabilitas vs Engagement Score")
        fig_sc2 = go.Figure()
        for label, subset, col_name in [
            ('Tier 1', tier1.head(50), C['tier1']),
            ('Tier 2', tier2, C['tier2']),
        ]:
            fig_sc2.add_trace(go.Scatter(
                x=subset['engagement_score'], y=subset['prob_inf'],
                mode='markers+text', name=label,
                marker=dict(color=col_name, size=9, opacity=0.85),
                text=subset['username'].str[:12],
                textposition='top center', textfont=dict(size=8),
                hovertemplate='%{text}<br>Eng: %{x}<br>Prob: %{y:.4f}'
            ))
        fig_sc2.update_layout(xaxis_title='Engagement Score', yaxis_title='Probabilitas Influencer',
                              height=350, margin=dict(t=10,b=30))
        st.plotly_chart(fig_sc2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Export Data":

    st.markdown("### 💾 Download Hasil Analisis")
    st.info("Klik tombol di bawah untuk mengunduh masing-masing file CSV hasil analisis.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔴 Influencer Tier 1")
        csv_t1 = tier1[['username','prob_inf','in_degree','betweenness','pagerank',
                          'n_comments','total_replies','engagement_score']].to_csv(index=False)
        st.download_button("⬇️ Download Tier 1 CSV", csv_t1, "influencer_tier1.csv", "text/csv")
        st.caption(f"{len(tier1)} akun")

    with col2:
        st.markdown("#### 🟠 Influencer Tier 2")
        csv_t2 = tier2[['username','prob_inf','in_degree','betweenness','pagerank',
                          'n_comments','total_replies']].to_csv(index=False)
        st.download_button("⬇️ Download Tier 2 CSV", csv_t2, "influencer_tier2.csv", "text/csv")
        st.caption(f"{len(tier2)} akun")

    with col3:
        st.markdown("#### 🟢 Influencee Target")
        csv_inf = inf_df.to_csv(index=False)
        st.download_button("⬇️ Download Influencee CSV", csv_inf, "influencee_target.csv", "text/csv")
        st.caption(f"{len(inf_df)} akun")

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### 📊 Perbandingan Model")
        comp_rows = []
        for nm in ['Degree Centrality','Betweenness Centrality','PageRank','GCN','GAT','GIN']:
            mr = models_res[nm]
            comp_rows.append({'Model': nm, 'Accuracy': mr['accuracy'], 'Precision': mr['precision'],
                               'Recall': mr['recall'], 'F1-Score': mr['f1'], 'AUC-ROC': mr['auc']})
        csv_comp = pd.DataFrame(comp_rows).to_csv(index=False)
        st.download_button("⬇️ Download Perbandingan Model", csv_comp, "perbandingan_model.csv", "text/csv")

    with col5:
        st.markdown("#### 📋 Node Features Lengkap")
        csv_all = sna_prob[['username','prob_inf','in_degree','out_degree','degree_c',
                              'betweenness','pagerank','closeness','n_comments',
                              'total_replies','engagement_score','label']].to_csv(index=False)
        st.download_button("⬇️ Download Node Features", csv_all, "node_features_lengkap.csv", "text/csv")

    st.markdown("---")
    st.markdown("### 📋 Ringkasan Hasil")
    summary = {
        "Komentar bersih": f"{len(df_c):,}",
        "Total node graf": f"{G.number_of_nodes():,}",
        "Total edge graf": f"{G.number_of_edges():,}",
        "Komunitas (Louvain)": len(partition),
        "Modularity": f"{modularity:.4f}",
        "Model terbaik": best_name,
        f"F1-Score ({best_name})": models_res[best_name]['f1'],
        f"AUC-ROC ({best_name})": models_res[best_name]['auc'],
        "Influencer Tier 1": len(tier1),
        "Influencer Tier 2": len(tier2),
        "Influencee / Target": len(inf_df),
    }
    for k, v in summary.items():
        st.markdown(f"- **{k}:** {v}")
