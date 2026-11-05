"""Figura do artigo sem intervalos de confiança: diferença para o modelo sem grafo em cada semente."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(r"d:\TCC")
R = json.loads((BASE_DIR / "grafo_resultados (1).json").read_text(encoding="utf-8"))

COR = {"GCN": "#2a78d6", "GAT": "#eb6834"}
MARCA = {"GCN": "o", "GAT": "s"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e6e5e1"

linhas, y = [], 0
for regime in ["humanos", "denso"]:
    for est, nome in [("B kNN semantico", "semântico"), ("C hibrido", "híbrido")]:
        for mod in ["GCN", "GAT"]:
            linhas.append((y, regime, est, mod, f"{nome} · {mod}"))
            y += 1
    y += 0.8

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=300)
ax.axvline(0, color=TINTA2, lw=1)
rng = np.random.default_rng(0)
for y, regime, est, mod, _ in linhas:
    d = np.array(R[f"{regime}|{est}|{mod}"]["f1"]) - np.array(R[f"{regime}|A sem arestas|MLP"]["f1"])
    ax.scatter(d, y + rng.uniform(-0.18, 0.18, len(d)), s=10, color=COR[mod], alpha=0.45,
               linewidths=0, zorder=2)
    ax.plot(d.mean(), y, MARCA[mod], color=COR[mod], ms=8, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)
    venc = int((d > 0).sum())
    ax.text(0.034, y, f"{d.mean():+.4f}  ({venc}/10)".replace(".", ",").replace("-", "−"),
            va="center", fontsize=7.5, color=TINTA2)

ax.set_yticks([l[0] for l in linhas])
ax.set_yticklabels([l[4] for l in linhas], color=TINTA)
ax.text(-0.0545, -0.9, "961 rótulos humanos", fontweight="bold", color=TINTA, fontsize=8.5)
ax.text(-0.0545, 3.9, "~50 mil rótulos", fontweight="bold", color=TINTA, fontsize=8.5)
ax.text(0.034, -0.9, "média  (vitórias)", color=TINTA2, fontsize=7.5, fontweight="bold")
ax.set_ylim(8.4, -1.5)
ax.set_xlim(-0.055, 0.052)
ax.set_xlabel("diferença de F1-macro em relação ao modelo sem grafo\n"
              "(pontos pequenos = cada semente; ponto grande = média das 10)", color=TINTA2)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:+.2f}".replace(".", ",").replace("-", "−") if v else "0"))
ax.set_xticks([-0.04, -0.02, 0, 0.02])
ax.grid(axis="x", color=GRADE, lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(TINTA2)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=TINTA2)
h = [plt.Line2D([], [], color=COR[m], marker=MARCA[m], lw=0, ms=6, label=m) for m in COR]
ax.legend(handles=h, loc="lower left", frameon=False, fontsize=8)
fig.tight_layout()
destino = BASE_DIR / "fig_artigo_sementes.png"
fig.savefig(destino, facecolor="white")
print("salvo:", destino)
