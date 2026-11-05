"""Rodada 2: atributos crus, grafo cruzado, e piso p50."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
CRU = "neuralmind/bert-base-portuguese-cased"
K, DIM, DROP, LR, WD, EPOCAS, PAC = 5, 128, 0.5, 0.01, 5e-4, 300, 40
SEEDS = [42, 7, 2024, 1, 13, 99, 256, 777, 31337, 5]
torch.set_num_threads(12)

df = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
y = df["label"].values
fonte = df["fonte"].values
n = len(df)
ACASO = float(((np.bincount(y) / n) ** 2).sum())

emb_aj = np.load(BASE_DIR / "embeddings_base_humana.npy")

cache_cru = BASE_DIR / "emb_humana_cru.npy"
if cache_cru.exists():
    emb_cru = np.load(cache_cru)
else:
    print("gerando embeddings do BERTimbau CRU...")
    tok = AutoTokenizer.from_pretrained(CRU)
    mod = AutoModel.from_pretrained(CRU).eval()
    out = []
    for i in range(0, n, 32):
        e = tok(df["text"][i:i + 32].tolist(), return_tensors="pt",
                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            h = mod(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
    emb_cru = np.vstack(out)
    np.save(cache_cru, emb_cru)
print(f"embeddings: ajustado {emb_aj.shape}  cru {emb_cru.shape}\n")

idx_a = np.where(fonte == "base_A")[0]
idx_b = np.where(fonte == "base_B")[0]
idx_tr, idx_val = train_test_split(idx_a, test_size=100, stratify=y[idx_a], random_state=42)


def mk(ids):
    m = torch.zeros(n, dtype=torch.bool); m[ids] = True; return m


mask_tr, mask_val, mask_te = mk(idx_tr), mk(idx_val), mk(idx_b)
Y = torch.tensor(y, dtype=torch.long)
pesos = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]),
                                          y=y[idx_tr]), dtype=torch.float)


def grafo(emb, pct=0):
    nb = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(emb)
    dist, ind = nb.kneighbors(emb)
    p = {}
    for i, (viz, ds) in enumerate(zip(ind, dist)):
        for j, d in zip(viz, ds):
            if i != j:
                pr = (min(i, int(j)), max(i, int(j)))
                p[pr] = max(p.get(pr, 0.0), 1.0 - d)
    if pct:
        corte = np.percentile(list(p.values()), pct)
        p = {k_: v for k_, v in p.items() if v >= corte}
    s = [a for a, _ in p]; t = [b for _, b in p]
    ei = to_undirected(torch.tensor([s, t], dtype=torch.long), num_nodes=n)
    ss, tt = ei.numpy()
    lig = set(s) | set(t)
    return ei, {"arestas": len(p), "homofilia": float((y[ss] == y[tt]).mean()),
                "isolados": n - len(lig),
                "iso_teste": int(sum(1 for i in idx_b if i not in lig))}


class MLP(torch.nn.Module):
    def __init__(s, a, b, c):
        super().__init__(); s.l1, s.l2 = Linear(a, b), Linear(b, c)
    def forward(s, x, ei=None):
        return s.l2(F.dropout(F.relu(s.l1(x)), DROP, s.training))


class GCN(torch.nn.Module):
    def __init__(s, a, b, c):
        super().__init__(); s.c1, s.c2 = GCNConv(a, b), GCNConv(b, c)
    def forward(s, x, ei):
        return s.c2(F.dropout(F.relu(s.c1(x, ei)), DROP, s.training), ei)


class GAT(torch.nn.Module):
    def __init__(s, a, b, c, h=4):
        super().__init__()
        s.c1 = GATConv(a, b, heads=h, dropout=DROP)
        s.c2 = GATConv(b * h, c, heads=1, concat=False, dropout=DROP)
    def forward(s, x, ei):
        return s.c2(F.dropout(F.elu(s.c1(x, ei)), DROP, s.training), ei)


def treinar(classe, X, ei, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mo = classe(X.size(1), DIM, 2)
    ot = torch.optim.Adam(mo.parameters(), lr=LR, weight_decay=WD)
    melhor, est, esp = -1.0, None, 0
    for _ in range(EPOCAS):
        mo.train(); ot.zero_grad()
        F.cross_entropy(mo(X, ei)[mask_tr], Y[mask_tr], weight=pesos).backward()
        ot.step()
        mo.eval()
        with torch.no_grad():
            pv = mo(X, ei).argmax(1)
        f1v = f1_score(Y[mask_val], pv[mask_val], average="macro", zero_division=0)
        if f1v > melhor:
            melhor, est, esp = f1v, copy.deepcopy(mo.state_dict()), 0
        else:
            esp += 1
            if esp >= PAC: break
    mo.load_state_dict(est)
    return mo


ei_aj, info_aj = grafo(emb_aj)
ei_cru, info_cru = grafo(emb_cru)
ei_p50, info_p50 = grafo(emb_aj, 50)

CONFIGS = [
    ("cru feat + cru grafo", emb_cru, ei_cru, info_cru),
    ("cru feat + graf.ajust", emb_cru, ei_aj, info_aj),
    ("ajust feat + piso p50", emb_aj, ei_p50, info_p50),
]

saida = {}
for rot, emb, ei, info in CONFIGS:
    X = torch.tensor(emb, dtype=torch.float)
    print("=" * 72)
    print(f"{rot.upper()}")
    print(f"  grafo: {info['arestas']} arestas, homofilia {info['homofilia']:.4f}, "
          f"{info['isolados']} isolados ({info['iso_teste']} no teste)")
    print("=" * 72)
    print(f"{'modelo':<18} {'F1 macro':>18} {'F1 odio':>18}")
    por = {}
    for nome, classe in [("MLP (sem grafo)", MLP), ("GCN", GCN), ("GAT", GAT)]:
        f1s, f1o = [], []
        for sd in SEEDS:
            mo = treinar(classe, X, ei, sd)
            mo.eval()
            with torch.no_grad():
                p = mo(X, ei).argmax(1)
            f1s.append(f1_score(Y[mask_te], p[mask_te], average="macro", zero_division=0))
            f1o.append(f1_score(Y[mask_te], p[mask_te], pos_label=1, zero_division=0))
        por[nome] = np.array(f1s)
        print(f"{nome:<18} {np.mean(f1s):>10.4f} +-{np.std(f1s):.4f} "
              f"{np.mean(f1o):>10.4f} +-{np.std(f1o):.4f}")
    linhas = {}
    for arq in ["GCN", "GAT"]:
        d = por[arq] - por["MLP (sem grafo)"]
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        v = "GANHO" if lo > 0 else ("PERDA" if hi < 0 else "inconclusivo")
        print(f"  {arq} - MLP: {m:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"venceu {int((d>0).sum())}/10  -> {v}")
        linhas[arq] = {"dif": float(m), "ic": [float(lo), float(hi)],
                       "vitorias": int((d > 0).sum()), "veredito": v}
    saida[rot] = {"grafo": info,
                  "f1": {k_: float(v.mean()) for k_, v in por.items()},
                  "comparacao": linhas}
    print()

with open(RESULTS / "gnn_rodada2.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False, default=float)
print(f"Salvo: {RESULTS / 'gnn_rodada2.json'}")
