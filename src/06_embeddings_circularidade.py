"""Semana 3: embeddings da base_ampliada a partir do modelo_1 ajustado."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected
from transformers import AutoTokenizer, BertForSequenceClassification

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "modelo_1"
BASE_CSV = BASE_DIR / "base_ampliada.csv"
SAIDA = BASE_DIR / "embeddings_base_ampliada.npy"

torch.set_num_threads(12)

df = pd.read_csv(BASE_CSV, encoding="utf-8-sig")
y = df["label"].values
n = len(df)

print(f"Base: {n} textos")
print(f"  manual {int((df['origem']=='manual').sum())}   "
      f"pseudo {int((df['origem']=='pseudo').sum())}")
print(f"  nao_odio {int((y==0).sum())}   odio {int((y==1).sum())} "
      f"({y.mean():.1%} de odio)")

# --------------------------------------------------------- embeddings
print("\nGerando embeddings (modelo_1 ajustado, mean pooling)...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
modelo = BertForSequenceClassification.from_pretrained(MODEL_DIR).eval()

saida, lote = [], 32
for i in range(0, n, lote):
    enc = tok(df["text"][i:i + lote].tolist(), return_tensors="pt",
              truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        h = modelo.bert(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    saida.append(((h * m).sum(1) / m.sum(1)).numpy())
    if (i // lote) % 20 == 0:
        print(f"  {i}/{n}")

emb = np.vstack(saida)
np.save(SAIDA, emb)
print(f"Embeddings: {emb.shape}  ->  {SAIDA.name}")

# procedencia, para nunca mais carregar cache velho por engano
h = hashlib.sha256()
with open(BASE_CSV, "rb") as f:
    for bloco in iter(lambda: f.read(65536), b""):
        h.update(bloco)

proc = {
    "arquivo": SAIDA.name,
    "gerado_de": BASE_CSV.name,
    "sha256_da_base": h.hexdigest(),
    "n_textos": int(n),
    "dimensao": int(emb.shape[1]),
    "modelo": "modelo_1 (BERTimbau ajustado, freeze_layers=4)",
    "pooling": "mean (com mascara de atencao)",
    "max_length": 128,
}
with open(BASE_DIR / "embeddings_base_ampliada.json", "w", encoding="utf-8") as f:
    json.dump(proc, f, indent=2, ensure_ascii=False)

# ------------------------------------------------- diagnostico de homofilia
print("\n" + "=" * 66)
print("DIAGNOSTICO: homofilia do grafo kNN")
print("=" * 66)

prop = pd.Series(y).value_counts(normalize=True)
ACASO = float((prop ** 2).sum())
print(f"Homofilia esperada por acaso: {ACASO:.4f}  "
      f"(base com {y.mean():.1%} de odio)\n")


def grafo(k):
    nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, ind = nb.kneighbors(emb)
    o, d = [], []
    for i, viz in enumerate(ind):
        for j in viz:
            if i != j:
                o.append(i)
                d.append(int(j))
    return to_undirected(torch.tensor([o, d], dtype=torch.long), num_nodes=n).numpy()


print(f"{'k':>3} {'arestas':>9} {'homofilia':>11} {'excedente':>11}")
for k in [1, 3, 5, 10]:
    s, t = grafo(k)
    hh = (y[s] == y[t]).mean()
    print(f"{k:>3} {len(s):>9} {hh:>11.4f} {hh - ACASO:>+11.4f}")

s, t = grafo(5)
h5 = (y[s] == y[t]).mean()

print("\n--- k=5, detalhado ---")
for c, nome in [(0, "nao_odio"), (1, "odio")]:
    saindo = y[s] == c
    print(f"  homofilia da classe {nome:<9} {(y[t][saindo] == c).mean():.4f}")

# o efeito da selecao: nos pseudo sao os que o modelo achou faceis
origem = df["origem"].values
manual = origem == "manual"
print()
print(f"  arestas manual-manual : {(manual[s] & manual[t]).sum():>6}  "
      f"homofilia {(y[s]==y[t])[manual[s] & manual[t]].mean():.4f}")
print(f"  arestas pseudo-pseudo : {((~manual[s]) & (~manual[t])).sum():>6}  "
      f"homofilia {(y[s]==y[t])[(~manual[s]) & (~manual[t])].mean():.4f}")
print(f"  arestas mistas        : {(manual[s] ^ manual[t]).sum():>6}  "
      f"homofilia {(y[s]==y[t])[manual[s] ^ manual[t]].mean():.4f}")

print("\nReferencia da Semana 1 (base antiga, BERT cru):  0.7207 (acaso 0.6797)")
print("Referencia da Semana 1 (base antiga, ajustado) :  0.7664")

with open(RESULTS / "embeddings_semana3.json", "w", encoding="utf-8") as f:
    json.dump({**proc, "homofilia_k5": float(h5), "acaso": ACASO,
               "excedente": float(h5 - ACASO)}, f, indent=2, ensure_ascii=False)
print(f"\nSalvo: {RESULTS / 'embeddings_semana3.json'}")
