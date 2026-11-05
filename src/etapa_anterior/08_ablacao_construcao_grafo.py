"""Ablacao de construcao de grafo: modelo do embedding x piso de similaridade x peso."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
K = 5
torch.set_num_threads(12)

df = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
y = df["label"].values
n = len(df)
ACASO = float(((np.bincount(y) / n) ** 2).sum())
print(f"n={n}   acaso={ACASO:.4f}\n")


def gerar(model_name, tag):
    cache = BASE_DIR / f"emb_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    print(f"baixando/gerando: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModel.from_pretrained(model_name).eval()
    out = []
    for i in range(0, n, 32):
        e = tok(df["text"][i:i + 32].tolist(), return_tensors="pt",
                truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            h = mod(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v, p=2, dim=1)   # ST usa vetor normalizado
        out.append(v.numpy())
    emb = np.vstack(out)
    np.save(cache, emb)
    print(f"  -> {emb.shape}")
    return emb


EMBEDDINGS = {
    "modelo_1 (BERTimbau ajustado)": np.load(BASE_DIR / "embeddings_base_humana.npy"),
}
for nome, mid, tag in [
    ("ST MiniLM multilingue", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "st_minilm"),
    ("ST mpnet multilingue", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "st_mpnet"),
]:
    try:
        EMBEDDINGS[nome] = gerar(mid, tag)
    except Exception as e:
        print(f"  FALHOU {nome}: {type(e).__name__} {e}")


def arestas_knn(emb, k=K):
    nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    dist, ind = nb.kneighbors(emb)
    pares = {}
    for i, (viz, ds) in enumerate(zip(ind, dist)):
        for j, d in zip(viz, ds):
            if i != j:
                p = (min(i, int(j)), max(i, int(j)))
                pares[p] = max(pares.get(p, 0.0), 1.0 - d)
    return pares


def homofilia(pares):
    if not pares:
        return float("nan")
    s = np.array([a for a, _ in pares]); t = np.array([b for _, b in pares])
    return float((y[s] == y[t]).mean())


def isolados(pares):
    tem = set()
    for i, j in pares:
        tem.add(i); tem.add(j)
    return n - len(tem)


resultados = {}

print("\n" + "=" * 74)
print("1) MODELO DO EMBEDDING  (kNN k=5, sem piso)")
print("=" * 74)
print(f"{'modelo':<32} {'arestas':>8} {'homofilia':>11} {'excedente':>11}")
base_pares = {}
for nome, emb in EMBEDDINGS.items():
    p = arestas_knn(emb)
    base_pares[nome] = p
    h = homofilia(p)
    resultados[nome] = {"knn5": {"arestas": len(p), "homofilia": h, "excedente": h - ACASO}}
    print(f"{nome:<32} {len(p):>8} {h:>11.4f} {h - ACASO:>+11.4f}")

print("\n" + "=" * 74)
print("2) PESO NAS ARESTAS  (homofilia ponderada pela similaridade)")
print("=" * 74)
for nome, p in base_pares.items():
    s = np.array([a for a, _ in p]); t = np.array([b for _, b in p])
    w = np.array(list(p.values()))
    mesma = (y[s] == y[t]).astype(float)
    hp = float((w * mesma).sum() / w.sum())
    hn = float(mesma.mean())
    print(f"{nome:<32} simples {hn:.4f}   ponderada {hp:.4f}   ganho {hp-hn:+.4f}")
    resultados[nome]["homofilia_ponderada"] = hp

print("\n" + "=" * 74)
print("3) PISO DE SIMILARIDADE  (corta as arestas mais fracas, por percentil)")
print("=" * 74)
for nome, p in base_pares.items():
    print(f"\n{nome}")
    print(f"  {'piso':>16} {'arestas':>8} {'isolados':>9} {'homofilia':>11} {'excedente':>11}")
    sims = np.array(list(p.values()))
    resultados[nome]["pisos"] = {}
    for pct in [0, 10, 25, 50]:
        corte = np.percentile(sims, pct) if pct else 0.0
        filtrado = {k_: v for k_, v in p.items() if v >= corte}
        h = homofilia(filtrado)
        iso = isolados(filtrado)
        rot = "sem piso" if pct == 0 else f"p{pct} ({corte:.3f})"
        print(f"  {rot:>16} {len(filtrado):>8} {iso:>9} {h:>11.4f} {h - ACASO:>+11.4f}")
        resultados[nome]["pisos"][f"p{pct}"] = {
            "corte": float(corte), "arestas": len(filtrado),
            "isolados": iso, "homofilia": h}

with open(RESULTS / "ablacao_grafo.json", "w", encoding="utf-8") as f:
    json.dump({"n": n, "acaso": ACASO, "k": K, "resultados": resultados},
              f, indent=2, ensure_ascii=False, default=float)
print(f"\nSalvo: {RESULTS / 'ablacao_grafo.json'}")
