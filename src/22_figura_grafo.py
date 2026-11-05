"""Figura dos experimentos de grafo, em dois paineis.

Esquerda: F1-macro absoluto, com a ablacao (sem arestas) como referencia.
Direita: diferenca contra a ablacao, com IC95 pareado - e o que responde a
pergunta do trabalho, porque o zero e a linha que separa ganho de perda.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = Path(r"d:\TCC")
res = json.loads((BASE / "grafo_resultados.json").read_text(encoding="utf-8"))

CONJ = [("humanos", "961 rótulos humanos"), ("denso", "~50 mil rótulos")]
GRAFOS = [("B kNN semantico", "semântico"), ("C hibrido", "semântico + interação")]
MODELOS = [("GCN", "#2b6cb0"), ("GAT", "#7c4dbe")]
CINZA, ZERO = "#8a94a3", "#c05621"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=160)

y, rotulos_y = [], []
pos = 0
for cj, cj_nome in CONJ:
    for g, g_nome in GRAFOS:
        for m, _ in MODELOS:
            if f"{cj}|{g}|{m}" in res:
                y.append((cj, g, m, pos))
                rotulos_y.append(f"{g_nome} · {m}")
                pos += 1
    pos += 0.8

alt = max(p for *_, p in y)

# ---------------------------------------------------- painel 1: absoluto
for cj, cj_nome in CONJ:
    base = res[f"{cj}|A sem arestas|MLP"]
    linhas = [p for c, _, _, p in y if c == cj]
    ax1.plot([base["media"]] * 2, [min(linhas) - 0.45, max(linhas) + 0.45],
             color=CINZA, ls="--", lw=1.4, zorder=1)
    ax1.text(base["media"] - 0.0008, min(linhas) - 0.25,
             f"sem arestas {base['media']:.4f}", color=CINZA, fontsize=7.5,
             ha="right", va="center", rotation=90)

for cj, g, m, p in y:
    r = res[f"{cj}|{g}|{m}"]
    cor = dict(MODELOS)[m]
    ax1.errorbar(r["media"], p, xerr=r["desvio"], fmt="o", ms=7, lw=2,
                 capsize=3.5, color=cor, zorder=3)
    ax1.text(r["media"], p - 0.3, f"{r['media']:.4f}", fontsize=7.5,
             color="#3c4656", ha="center")

ax1.set_yticks([p for *_, p in y])
ax1.set_yticklabels(rotulos_y, fontsize=8.5)
ax1.invert_yaxis()
ax1.set_xlabel("F1-macro no teste congelado (206 nós humanos)")
ax1.set_title("Desempenho absoluto", fontsize=11, pad=12)
ax1.grid(axis="x", alpha=0.25)
ax1.set_axisbelow(True)

# ---------------------------------------------------- painel 2: diferenca
ax2.axvline(0, color=ZERO, lw=1.6, zorder=1)
ax2.text(0, alt + 0.85, "sem efeito", color=ZERO, fontsize=8, ha="center")

for cj, g, m, p in y:
    base = np.array(res[f"{cj}|A sem arestas|MLP"]["f1"])
    d = np.array(res[f"{cj}|{g}|{m}"]["f1"]) - base
    se = d.std(ddof=1) / np.sqrt(len(d))
    tc = stats.t.ppf(0.975, len(d) - 1)
    cor = dict(MODELOS)[m]
    ganhou = d.mean() - tc * se > 0
    ax2.errorbar(d.mean(), p, xerr=tc * se, fmt="o", ms=7, lw=2, capsize=3.5,
                 color=cor, zorder=3,
                 markerfacecolor=cor if ganhou else "white")
    ax2.text(d.mean(), p - 0.3,
             f"{d.mean():+.4f}" + ("  ✓" if ganhou else ""),
             fontsize=7.5, color="#3c4656", ha="center")

ax2.set_yticks([])
ax2.set_xlabel("diferença contra a ablação sem arestas\n"
               "barra = intervalo de confiança de 95% sobre as 10 sementes pareadas")
ax2.set_title("O que as arestas acrescentam", fontsize=11, pad=12)
ax2.grid(axis="x", alpha=0.25)
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    ax.set_ylim(alt + 1.1, -1.4)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

# faixas de conjunto
for ax in (ax1, ax2):
    for cj, cj_nome in CONJ:
        linhas = [p for c, _, _, p in y if c == cj]
        ax.text(-0.012 if ax is ax2 else ax.get_xlim()[0],
                min(linhas) - 0.95, cj_nome, fontsize=9, fontweight="bold",
                color="#2d3648", ha="left")

fig.suptitle("Classificação em grafo · 10 sementes por configuração",
             fontsize=12.5, y=0.985)
fig.text(0.5, 0.945,
         "ponto preenchido = intervalo inteiro de um lado do zero (efeito confirmado)   ·   "
         "ponto vazado = intervalo cruza o zero (efeito não confirmado)",
         ha="center", fontsize=8.5, color="#5a6478")
plt.tight_layout(rect=[0, 0, 1, 0.925])
plt.savefig(BASE / "fig_grafo_resultados.png", dpi=160, bbox_inches="tight")
print("salvo: fig_grafo_resultados.png")
