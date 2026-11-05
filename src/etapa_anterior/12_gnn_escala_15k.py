"""GNN semissupervisionada sobre 15.000 nos (1.167 rotulados). Teste: base_B."""

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
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_undirected
from transformers import AutoTokenizer, BertForSequenceClassification

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "modelo_1"
K, DIM, DROP, LR, WD, EPOCAS, PAC = 5, 128, 0.5, 0.01, 5e-4, 300, 40
SEEDS = [42, 7, 2024, 1, 13, 99, 256, 777, 31337, 5]
torch.set_num_threads(12)

df = pd.read_csv(BASE_DIR / "base_grafo_15k.csv", encoding="utf-8-sig")
n = len(df)
rot = df["origem"].values == "manual"
fonte = df["fonte"].values
print(f"{n} nos   rotulados {int(rot.sum())}   sem rotulo {int((~rot).sum())}")

# ------------------------------------------------------------ embeddings
CACHE = BASE_DIR / "emb_grafo_15k.npy"
if CACHE.exists():
    emb = np.load(CACHE)
    print(f"embeddings do cache: {emb.shape}")
else:
    print("gerando embeddings (modelo_1, mean pooling)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mod = BertForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    out, lote = [], 64
    for i in range(0, n, lote):
        e = tok(df["text"][i:i + lote].astype(str).tolist(), return_tensors="pt",
                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            h = mod.bert(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
        if (i // lote) % 30 == 0:
            print(f"  {i}/{n}")
    emb = np.vstack(out)
    np.save(CACHE, emb)
    print(f"embeddings: {emb.shape}")

# ----------------------------------------------------------------- grafo
print("\nconstruindo kNN...")
nb = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(emb)
dist, ind = nb.kneighbors(emb)

o, d, sims = [], [], []
for i, (viz, ds) in enumerate(zip(ind, dist)):
    for j, dd in zip(viz, ds):
        if i != j:
            o.append(i); d.append(int(j)); sims.append(1.0 - dd)
ei = to_undirected(torch.tensor([o, d], dtype=torch.long), num_nodes=n)
sims = np.array(sims)
print(f"arestas (nao dirigidas): {ei.size(1)//2}")

s, t = ei.numpy()
y_full = df["label"].fillna(-1).astype(int).values
ambos = (y_full[s] >= 0) & (y_full[t] >= 0)
print(f"\narestas entre dois nos ROTULADOS: {int(ambos.sum())} "
      f"({ambos.mean():.2%} do total)")
if ambos.sum():
    h = (y_full[s][ambos] == y_full[t][ambos]).mean()
    p = np.bincount(y_full[rot]) / int(rot.sum())
    ac = float((p ** 2).sum())
    print(f"homofilia nessas arestas: {h:.4f}   (acaso {ac:.4f}, excedente {h-ac:+.4f})")
    print(f"referencia com 1.167 nos: 0.7436")

print(f"\nsimilaridade das arestas: mediana {np.median(sims):.4f}  "
      f"p90 {np.percentile(sims, 90):.4f}")
print(f"  arestas com sim > 0,95 (quase-duplicatas): "
      f"{(sims > 0.95).mean():.2%}")
print(f"  arestas com sim > 0,90: {(sims > 0.90).mean():.2%}")

# ---------------------------------------------------------------- split
idx_a = np.where(fonte == "base_A")[0]
idx_b = np.where(fonte == "base_B")[0]
idx_tr, idx_val = train_test_split(idx_a, test_size=100,
                                   stratify=y_full[idx_a], random_state=42)


def mk(ids):
    m = torch.zeros(n, dtype=torch.bool); m[ids] = True; return m


mask_tr, mask_val, mask_te = mk(idx_tr), mk(idx_val), mk(idx_b)
X = torch.tensor(emb, dtype=torch.float)
Y = torch.tensor(np.where(y_full < 0, 0, y_full), dtype=torch.long)
pesos = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]),
                                          y=y_full[idx_tr]), dtype=torch.float)
print(f"\ntreino {len(idx_tr)}   val {len(idx_val)}   TESTE {len(idx_b)}")


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


def treinar(classe, seed):
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


print("\n" + "=" * 68)
print("RESULTADOS — teste nos 206 de base_B")
print("=" * 68)
print(f"{'modelo':<18} {'F1 macro':>18} {'F1 odio':>18}")
por, saida = {}, {}
for nome, classe in [("MLP (sem grafo)", MLP), ("GCN", GCN), ("GAT", GAT)]:
    f1s, f1o = [], []
    for sd in SEEDS:
        mo = treinar(classe, sd)
        mo.eval()
        with torch.no_grad():
            p = mo(X, ei).argmax(1)
        f1s.append(f1_score(Y[mask_te], p[mask_te], average="macro", zero_division=0))
        f1o.append(f1_score(Y[mask_te], p[mask_te], pos_label=1, zero_division=0))
    por[nome] = {"f1": np.array(f1s), "odio": np.array(f1o)}
    print(f"{nome:<18} {np.mean(f1s):>10.4f} +-{np.std(f1s):.4f} "
          f"{np.mean(f1o):>10.4f} +-{np.std(f1o):.4f}")

print()
for arq in ["GCN", "GAT"]:
    for met, rot_m in [("f1", "F1 macro"), ("odio", "F1 odio")]:
        dd = por[arq][met] - por["MLP (sem grafo)"][met]
        m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        v = "GANHO" if lo > 0 else ("PERDA" if hi < 0 else "inconclusivo")
        print(f"  {arq} - MLP ({rot_m}): {m:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"venceu {int((dd>0).sum())}/10  -> {v}")
    saida[arq] = {}

print("\nReferencia com 1.167 nos (mesmo teste):")
print("  MLP 0.7942   GCN 0.7592 (-0.0350)   GAT 0.7454 (-0.0488)")

with open(RESULTS / "gnn_15k.json", "w", encoding="utf-8") as f:
    json.dump({"n_nos": n, "rotulados": int(rot.sum()), "arestas": ei.size(1) // 2,
               "modelos": {k_: {"f1_macro": float(v["f1"].mean()),
                                "std": float(v["f1"].std()),
                                "f1_odio": float(v["odio"].mean())}
                           for k_, v in por.items()}},
              f, indent=2, ensure_ascii=False)
print(f"\nSalvo: {RESULTS / 'gnn_15k.json'}")
