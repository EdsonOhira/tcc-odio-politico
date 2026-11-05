"""Figura da monografia: o que as arestas acrescentam ao modelo sem grafo (57 mil nós)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(r"d:\TCC")
T = json.loads((BASE_DIR / "results" / "tabelas_monografia.json").read_text(encoding="utf-8"))
V = T["vs_sem_grafo"]

COR = {"GCN": "#2a78d6", "GAT": "#eb6834"}
MARCA = {"GCN": "o", "GAT": "s"}
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e6e5e1"

linhas = []  # (y, regime, estrutura, modelo)
y = 0
for regime, nome_regime in [("humanos", "961 rótulos humanos"), ("denso", "~50 mil rótulos")]:
    for est, nome_est in [("B kNN semantico", "semântico"), ("C hibrido", "híbrido")]:
        for mod in ["GCN", "GAT"]:
            linhas.append((y, regime, est, mod, f"{nome_est} · {mod}"))
            y += 1
    y += 0.8

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=300)
ax.axvline(0, color=TINTA2, lw=1)
for y, regime, est, mod, rot in linhas:
    r = V[f"{regime}|{est}|{mod}"]
    lo, hi = r["ic95"]
    ax.plot([lo, hi], [y, y], color=COR[mod], lw=2, solid_capstyle="round")
    ax.plot(r["dif"], y, MARCA[mod], color=COR[mod], ms=7,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.text(r["dif"], y - 0.3, f"{r['dif']:+.4f}".replace(".", ",").replace("-", "−"),
            ha="center", va="bottom", fontsize=7.5, color=TINTA2)

ax.set_yticks([l[0] for l in linhas])
ax.set_yticklabels([l[4] for l in linhas], color=TINTA)
ax.invert_yaxis()
ax.text(-0.0455, -0.9, "961 rótulos humanos", fontweight="bold", color=TINTA, fontsize=8.5)
ax.text(-0.0455, 3.9, "~50 mil rótulos", fontweight="bold", color=TINTA, fontsize=8.5)
ax.set_ylim(8.4, -1.5)
ax.set_xlim(-0.046, 0.026)
ax.set_xlabel("diferença de F1-macro em relação ao modelo sem grafo\n"
              "(ponto = média de 10 sementes; barra = IC de 95%)", color=TINTA2)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v:+.2f}".replace(".", ",") if v else "0"))
ax.grid(axis="x", color=GRADE, lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(TINTA2)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=TINTA2)

h = [plt.Line2D([], [], color=COR[m], marker=MARCA[m], lw=2, ms=6, label=m) for m in COR]
ax.legend(handles=h, loc="lower left", frameon=False, fontsize=8)

fig.tight_layout()
destino = BASE_DIR / "fig_monografia_grafo.png"
fig.savefig(destino, facecolor="white")
print("salvo:", destino)
