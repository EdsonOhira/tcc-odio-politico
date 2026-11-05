"""Tabelas do grafo de 57 mil para a monografia, recalculadas do JSON com IC pela t."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR = Path(r"d:\TCC")
R = json.loads((BASE_DIR / "grafo_resultados (1).json").read_text(encoding="utf-8"))


def f1(k):
    return np.array(R[k]["f1"])


def par(a, b):
    d = f1(a) - f1(b)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_rel(f1(a), f1(b)).pvalue
    return {"dif": round(d.mean(), 4), "ic95": [round(d.mean() - t * se, 4),
                                                round(d.mean() + t * se, 4)],
            "vitorias": f"{int((d > 0).sum())}/{n}", "p": float(f"{p:.2g}")}


saida = {"configuracoes": {}, "vs_sem_grafo": {}, "hibrido_menos_semantico": {},
         "densidade_ganho": {}, "corte_095": {}}
for k in R:
    saida["configuracoes"][k] = {"media": round(R[k]["media"], 4),
                                 "desvio": round(R[k]["desvio"], 4)}

for regime in ["humanos", "denso", "denso@0.95"]:
    base = f"{regime}|A sem arestas|MLP"
    for est in ["B kNN semantico", "C hibrido"]:
        for mod in ["GCN", "GAT"]:
            saida["vs_sem_grafo"][f"{regime}|{est}|{mod}"] = par(f"{regime}|{est}|{mod}", base)
    for mod in ["GCN", "GAT"]:
        saida["hibrido_menos_semantico"][f"{regime}|{mod}"] = par(
            f"{regime}|C hibrido|{mod}", f"{regime}|B kNN semantico|{mod}")

for cfg in ["A sem arestas|MLP", "B kNN semantico|GCN", "B kNN semantico|GAT",
            "C hibrido|GCN", "C hibrido|GAT"]:
    saida["densidade_ganho"][cfg] = par(f"denso|{cfg}", f"humanos|{cfg}")
    saida["corte_095"][cfg] = par(f"denso@0.95|{cfg}", f"denso|{cfg}")

# sensibilidade do IC da melhor configuração ao nível de confiança
d = f1("denso|C hibrido|GAT") - f1("denso|A sem arestas|MLP")
se = d.std(ddof=1) / np.sqrt(len(d))
saida["melhor_niveis"] = {
    str(nv): [round(d.mean() - stats.t.ppf(1 - (1 - nv) / 2, 9) * se, 4),
              round(d.mean() + stats.t.ppf(1 - (1 - nv) / 2, 9) * se, 4)]
    for nv in [0.90, 0.95, 0.99]}
saida["melhor_t"] = {"t": round(d.mean() / se, 4), "se": round(se, 6)}

destino = BASE_DIR / "results" / "tabelas_monografia.json"
destino.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(saida, ensure_ascii=False, indent=1))
