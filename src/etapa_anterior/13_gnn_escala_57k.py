"""GNN semissupervisionada sobre 57.441 nos (1.167 rotulados). Teste: base_B."""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
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

t0 = time.time()
df = pd.read_csv(BASE_DIR / "base_grafo_57k.csv", encoding="utf-8-sig")
n = len(df)
fonte = df["fonte"].values
print(f"{n} nos   rotulados {int((df['origem']=='manual').sum())}", flush=True)

# ------------------------------------------------------------ embeddings
CACHE = BASE_DIR / "emb_grafo_57k.npy"
if CACHE.exists():
    emb = np.load(CACHE)
    print(f"embeddings do cache: {emb.shape}", flush=True)
else:
    print("gerando embeddings (modelo_1, mean pooling)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mod = BertForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    textos = [t if isinstance(t, str) and t.strip() else "vazio"
              for t in df["text"].tolist()]
    PARCIAL = BASE_DIR / "emb_grafo_57k_parcial.npy"
    out, lote = [], 64
    ini = 0
    if PARCIAL.exists():                       # retoma de onde parou
        anterior = np.load(PARCIAL)
        out.append(anterior)
        ini = len(anterior)
        print(f"  retomando de {ini}/{n}", flush=True)
    for i in range(ini, n, lote):
        e = tok(textos[i:i + lote], return_tensors="pt",
                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            h = mod.bert(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy().astype(np.float32))
        if (i // lote) % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{n}  ({el/60:.1f} min decorridos)", flush=True)
            np.save(PARCIAL, np.vstack(out))   # checkpoint
    emb = np.vstack(out)
    np.save(CACHE, emb)
    PARCIAL.unlink(missing_ok=True)
    print(f"embeddings: {emb.shape}  ({(time.time()-t0)/60:.1f} min)", flush=True)

# ------------------------------------------- kNN em blocos (seguro de memoria)
print("\nconstruindo kNN em blocos...", flush=True)
E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
E = E.astype(np.float32)

o, d, sims = [], [], []
BL = 2000
for ini in range(0, n, BL):
    fim = min(ini + BL, n)
    S = E[ini:fim] @ E.T                      # similaridade cosseno
    for r in range(fim - ini):
        S[r, ini + r] = -2.0                  # exclui o proprio no
    viz = np.argpartition(-S, K, axis=1)[:, :K]
    for r in range(fim - ini):
        i = ini + r
        for j in viz[r]:
            o.append(i); d.append(int(j)); sims.append(float(S[r, j]))
    if (ini // BL) % 8 == 0:
        print(f"  {ini}/{n}", flush=True)

ei = to_undirected(torch.tensor([o, d], dtype=torch.long), num_nodes=n)
sims = np.array(sims)
print(f"arestas nao dirigidas: {ei.size(1)//2}", flush=True)

s, t = ei.numpy()
y_full = df["label"].fillna(-1).astype(int).values
ambos = (y_full[s] >= 0) & (y_full[t] >= 0)
rot = df["origem"].values == "manual"
p = np.bincount(y_full[rot]) / int(rot.sum())
ACASO = float((p ** 2).sum())

print(f"\narestas entre dois nos ROTULADOS: {int(ambos.sum())} ({ambos.mean():.2%})")
if ambos.sum():
    h = (y_full[s][ambos] == y_full[t][ambos]).mean()
    print(f"homofilia nessas arestas: {h:.4f}  (acaso {ACASO:.4f}, "
          f"excedente {h-ACASO:+.4f})")
print(f"  1.167 nos: 0,7436   |   15.000 nos: 0,8058")
print(f"\nsimilaridade: mediana {np.median(sims):.4f}  "
      f"> 0,95: {(sims>0.95).mean():.2%}  > 0,90: {(sims>0.90).mean():.2%}",
      flush=True)

# ---------------------------------------------------------------- treino
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
print("=" * 68, flush=True)
print(f"{'modelo':<18} {'F1 macro':>18} {'F1 odio':>18}")
por = {}
for nome, classe in [("MLP (sem grafo)", MLP), ("GCN", GCN), ("GAT", GAT)]:
    f1s, f1o = [], []
    for sd in SEEDS:
        mo = treinar(classe, sd)
        mo.eval()
        with torch.no_grad():
            pr = mo(X, ei).argmax(1)
        f1s.append(f1_score(Y[mask_te], pr[mask_te], average="macro", zero_division=0))
        f1o.append(f1_score(Y[mask_te], pr[mask_te], pos_label=1, zero_division=0))
    por[nome] = {"f1": np.array(f1s), "odio": np.array(f1o)}
    print(f"{nome:<18} {np.mean(f1s):>10.4f} +-{np.std(f1s):.4f} "
          f"{np.mean(f1o):>10.4f} +-{np.std(f1o):.4f}", flush=True)

print()
for arq in ["GCN", "GAT"]:
    for met, rm in [("f1", "F1 macro"), ("odio", "F1 odio")]:
        dd = por[arq][met] - por["MLP (sem grafo)"][met]
        m, se = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        v = "GANHO" if lo > 0 else ("PERDA" if hi < 0 else "inconclusivo")
        print(f"  {arq} - MLP ({rm}): {m:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"venceu {int((dd>0).sum())}/10  -> {v}")

print("\nCURVA DE ESCALA (GAT - MLP, F1 macro):")
print("   1.167 nos: -0.0488")
print("  15.000 nos: -0.0160")
dd = por["GAT"]["f1"] - por["MLP (sem grafo)"]["f1"]
print(f"  57.441 nos: {dd.mean():+.4f}")

with open(RESULTS / "gnn_57k.json", "w", encoding="utf-8") as f:
    json.dump({"n_nos": n, "arestas": ei.size(1) // 2,
               "modelos": {k_: {"f1_macro": float(v["f1"].mean()),
                                "std": float(v["f1"].std()),
                                "f1_odio": float(v["odio"].mean())}
                           for k_, v in por.items()}},
              f, indent=2, ensure_ascii=False)
print(f"\nTempo total: {(time.time()-t0)/60:.1f} min")
print(f"Salvo: {RESULTS / 'gnn_57k.json'}")
