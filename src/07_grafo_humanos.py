"""Grafo construido SO com os 1.167 textos de rotulo humano. Zero circularidade."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "modelo_1"
CRU = "neuralmind/bert-base-portuguese-cased"

torch.set_num_threads(12)

df = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
y = df["label"].values
fonte = df["fonte"].values
n = len(df)

p = np.bincount(y) / n
ACASO = float((p ** 2).sum())

print(f"Base humana: {n} textos")
print(f"  base_A {int((fonte=='base_A').sum())}  (vistos no treino do modelo_1)")
print(f"  base_B {int((fonte=='base_B').sum())}  (NUNCA vistos)")
print(f"  nao_odio {int((y==0).sum())}  odio {int((y==1).sum())}")
print(f"  homofilia por acaso: {ACASO:.4f}\n")


def embeddings(qual):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR if qual == "ajustado" else CRU)
    if qual == "ajustado":
        m = BertForSequenceClassification.from_pretrained(MODEL_DIR).eval()
        enc_fn = m.bert
    else:
        m = AutoModel.from_pretrained(CRU).eval()
        enc_fn = m
    out = []
    for i in range(0, n, 32):
        e = tok(df["text"][i:i + 32].tolist(), return_tensors="pt",
                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            h = enc_fn(**e).last_hidden_state
        msk = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * msk).sum(1) / msk.sum(1)).numpy())
    return np.vstack(out)


def grafo(emb, k, idx=None):
    """kNN entre os nos de idx (ou todos). Devolve arestas em indices LOCAIS."""
    sub = emb if idx is None else emb[idx]
    m = len(sub)
    kk = min(k, m - 1)
    nb = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(sub)
    _, ind = nb.kneighbors(sub)
    o, d = [], []
    for i, viz in enumerate(ind):
        for j in viz:
            if i != j:
                o.append(i)
                d.append(int(j))
    return to_undirected(torch.tensor([o, d], dtype=torch.long), num_nodes=m).numpy()


resumo = {}

for qual in ["ajustado", "cru"]:
    print("=" * 68)
    print(f"EMBEDDINGS: BERTimbau {qual.upper()}")
    print("=" * 68)
    emb = embeddings(qual)
    if qual == "ajustado":
        np.save(BASE_DIR / "embeddings_base_humana.npy", emb)

    print(f"{'k':>3} {'arestas':>9} {'homofilia':>11} {'excedente':>11}")
    for k in [1, 3, 5, 10]:
        s, t = grafo(emb, k)
        h = (y[s] == y[t]).mean()
        print(f"{k:>3} {len(s):>9} {h:>11.4f} {h - ACASO:>+11.4f}")

    s, t = grafo(emb, 5)
    h5 = float((y[s] == y[t]).mean())

    print(f"\n--- k=5, por classe ---")
    for c, nm in [(0, "nao_odio"), (1, "odio")]:
        sai = y[s] == c
        print(f"  {nm:<9} {(y[t][sai] == c).mean():.4f}")

    print(f"\n--- k=5, por origem da aresta ---")
    a = fonte[s] == "base_A"
    b = fonte[t] == "base_A"
    for nm, msk in [("A-A", a & b), ("B-B", (~a) & (~b)), ("mistas", a ^ b)]:
        if msk.sum():
            print(f"  {nm:<7} {int(msk.sum()):>6} arestas   homofilia "
                  f"{(y[s] == y[t])[msk].mean():.4f}")

    # subgrafo SO com base_B: rotulo humano E nunca visto pelo modelo
    idx_b = np.where(fonte == "base_B")[0]
    sb, tb = grafo(emb, 5, idx_b)
    yb = y[idx_b]
    hb = float((yb[sb] == yb[tb]).mean())
    pb = np.bincount(yb) / len(yb)
    acaso_b = float((pb ** 2).sum())
    print(f"\n--- SUBGRAFO SO base_B ({len(idx_b)} nos, nunca vistos) ---")
    print(f"  arestas {len(sb)}   homofilia {hb:.4f}   acaso {acaso_b:.4f}   "
          f"excedente {hb - acaso_b:+.4f}")

    resumo[qual] = {"homofilia_k5": h5, "excedente": h5 - ACASO,
                    "base_B_homofilia": hb, "base_B_excedente": hb - acaso_b}

print("\n" + "=" * 68)
print("COMPARACAO FINAL")
print("=" * 68)
print(f"{'':<22} {'todos (1167)':>16} {'so base_B (206)':>18}")
for qual in ["cru", "ajustado"]:
    r = resumo[qual]
    print(f"BERTimbau {qual:<12} {r['homofilia_k5']:>8.4f} ({r['excedente']:+.4f}) "
          f"{r['base_B_homofilia']:>10.4f} ({r['base_B_excedente']:+.4f})")

print("\nReferencias anteriores:")
print("  base antiga, BERT cru      : 0.7207  (acaso 0.6797, excedente +0.0410)")
print("  base ampliada, so humanos  : 0.6791  (acaso 0.5067, excedente +0.1724)")

with open(RESULTS / "grafo_humanos.json", "w", encoding="utf-8") as f:
    json.dump({"n": n, "acaso": ACASO, "resultados": resumo}, f,
              indent=2, ensure_ascii=False, default=float)
print(f"\nSalvo: {RESULTS / 'grafo_humanos.json'}")
