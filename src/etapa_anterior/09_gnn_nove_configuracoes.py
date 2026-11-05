"""GNN sobre 3 construcoes de grafo. Treino em base_A, TESTE nos 206 de base_B."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
K, DIM, DROP, LR, WD, EPOCAS, PAC = 5, 128, 0.5, 0.01, 5e-4, 300, 40
SEEDS = [42, 7, 2024, 1, 13, 99, 256, 777, 31337, 5]
torch.set_num_threads(12)

df = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
emb = np.load(BASE_DIR / "embeddings_base_humana.npy")
y = df["label"].values
fonte = df["fonte"].values
n = len(df)

idx_a = np.where(fonte == "base_A")[0]
idx_b = np.where(fonte == "base_B")[0]
idx_tr, idx_val = train_test_split(idx_a, test_size=100, stratify=y[idx_a], random_state=42)

print(f"treino {len(idx_tr)} (base_A)   val {len(idx_val)} (base_A)   "
      f"TESTE {len(idx_b)} (base_B, nunca visto)")


def mk(ids):
    m = torch.zeros(n, dtype=torch.bool); m[ids] = True; return m


mask_tr, mask_val, mask_te = mk(idx_tr), mk(idx_val), mk(idx_b)

# ------------------------------------------------------------ construcoes
nb = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(emb)
dist, ind = nb.kneighbors(emb)
pares = {}
for i, (viz, ds) in enumerate(zip(ind, dist)):
    for j, d in zip(viz, ds):
        if i != j:
            p = (min(i, int(j)), max(i, int(j)))
            pares[p] = max(pares.get(p, 0.0), 1.0 - d)

sims = np.array(list(pares.values()))
CONSTR = {}
for rot, pct in [("sem piso", 0), ("piso p10", 10), ("piso p25", 25)]:
    corte = np.percentile(sims, pct) if pct else 0.0
    f_ = {k_: v for k_, v in pares.items() if v >= corte}
    s = [a for a, _ in f_]; t = [b for _, b in f_]
    ei = to_undirected(torch.tensor([s, t], dtype=torch.long), num_nodes=n)
    ligados = len(set(s) | set(t))
    iso_te = int(sum(1 for i in idx_b if i not in set(s) | set(t)))
    ss, tt = ei.numpy()
    CONSTR[rot] = {"ei": ei, "arestas": len(f_), "isolados": n - ligados,
                   "iso_teste": iso_te,
                   "homofilia": float((y[ss] == y[tt]).mean())}
    print(f"{rot:<10} {len(f_):>5} arestas   {n-ligados:>3} isolados "
          f"({iso_te} deles no teste)   homofilia {CONSTR[rot]['homofilia']:.4f}")


# ----------------------------------------------------------------- modelos
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


pesos = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]),
                                          y=y[idx_tr]), dtype=torch.float)
X = torch.tensor(emb, dtype=torch.float)
Y = torch.tensor(y, dtype=torch.long)


def treinar(classe, ei, seed):
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


saida = {}
for rot, cfg in CONSTR.items():
    ei = cfg["ei"]
    print(f"\n{'='*70}\n{rot.upper()}  ({cfg['arestas']} arestas, homofilia "
          f"{cfg['homofilia']:.4f})\n{'='*70}")
    print(f"{'modelo':<18} {'F1 macro':>18} {'F1 odio':>18}")
    por = {}
    for nome, classe in [("MLP (sem grafo)", MLP), ("GCN", GCN), ("GAT", GAT)]:
        f1s, f1o = [], []
        for sd in SEEDS:
            mo = treinar(classe, ei, sd)
            mo.eval()
            with torch.no_grad():
                p = mo(X, ei).argmax(1)
            f1s.append(f1_score(Y[mask_te], p[mask_te], average="macro", zero_division=0))
            f1o.append(f1_score(Y[mask_te], p[mask_te], pos_label=1, zero_division=0))
        por[nome] = {"f1": np.array(f1s), "odio": np.array(f1o)}
        print(f"{nome:<18} {np.mean(f1s):>10.4f} +-{np.std(f1s):.4f} "
              f"{np.mean(f1o):>10.4f} +-{np.std(f1o):.4f}")

    for arq in ["GCN", "GAT"]:
        d = por[arq]["f1"] - por["MLP (sem grafo)"]["f1"]
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        veredito = "GANHO" if lo > 0 else ("PERDA" if hi < 0 else "inconclusivo")
        print(f"  {arq} - MLP: {m:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"venceu {int((d>0).sum())}/10  -> {veredito}")

    saida[rot] = {
        "arestas": cfg["arestas"], "homofilia": cfg["homofilia"],
        "isolados": cfg["isolados"], "iso_teste": cfg["iso_teste"],
        "modelos": {k_: {"f1_macro": float(v["f1"].mean()),
                         "std": float(v["f1"].std()),
                         "f1_odio": float(v["odio"].mean())}
                    for k_, v in por.items()},
    }

print(f"\n{'='*70}\nRESUMO — F1 macro nos 206 de base_B\n{'='*70}")
print(f"{'construcao':<12} {'homof.':>7} {'MLP':>9} {'GCN':>9} {'GAT':>9}")
for rot, r in saida.items():
    m = r["modelos"]
    print(f"{rot:<12} {r['homofilia']:>7.4f} {m['MLP (sem grafo)']['f1_macro']:>9.4f} "
          f"{m['GCN']['f1_macro']:>9.4f} {m['GAT']['f1_macro']:>9.4f}")

with open(RESULTS / "gnn_ablacao.json", "w", encoding="utf-8") as f:
    json.dump({"seeds": SEEDS, "teste": "base_B (206 nos)", "resultados": saida},
              f, indent=2, ensure_ascii=False, default=float)
print(f"\nSalvo: {RESULTS / 'gnn_ablacao.json'}")
