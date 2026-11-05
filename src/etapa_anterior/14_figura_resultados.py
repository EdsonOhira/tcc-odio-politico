"""Figuras de apresentacao: a rede e os resultados."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(r"d:\TCC")

# paleta validada (dataviz, modo claro)
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2df"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "medium",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})

# ====================================================== FIGURA 1: a rede
df = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
emb = np.load(BASE_DIR / "embeddings_base_humana.npy")
y = df["label"].values
n = len(df)

nb = NearestNeighbors(n_neighbors=6, metric="cosine").fit(emb)
_, ind = nb.kneighbors(emb)
G = nx.Graph()
G.add_nodes_from(range(n))
for i, viz in enumerate(ind):
    for j in viz:
        if i != j:
            G.add_edge(i, int(j))

s, t = np.array(G.edges()).T
homof = (y[s] == y[t]).mean()
print(f"rede: {n} nos, {G.number_of_edges()} arestas, homofilia {homof:.4f}")

pos = nx.spring_layout(G, k=0.15, iterations=60, seed=42)

fig, ax = plt.subplots(figsize=(13, 11))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID, width=0.4, alpha=0.7)
for c, cor, rot in [(0, AZUL, "não-ódio"), (1, LARANJA, "ódio")]:
    sel = [i for i in range(n) if y[i] == c]
    nx.draw_networkx_nodes(G, pos, nodelist=sel, node_color=cor, node_size=26,
                           alpha=0.9, ax=ax, linewidths=0.4, edgecolors=SURF)

ax.set_title("Grafo de similaridade semântica entre textos\n"
             "1.167 textos com rótulo humano · k=5 vizinhos mais próximos",
             pad=18)
ax.text(0.5, -0.04,
        f"homofilia {homof:.3f}  —  {homof:.0%} das ligações conectam textos da mesma classe "
        f"(o acaso daria 50%)",
        transform=ax.transAxes, ha="center", fontsize=12, color=INK2)
ax.legend(handles=[mpatches.Patch(color=LARANJA, label="ódio"),
                   mpatches.Patch(color=AZUL, label="não-ódio")],
          loc="upper right", frameon=False, fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.savefig(BASE_DIR / "fig_grafo.png", dpi=140, bbox_inches="tight", facecolor=SURF)
print("salvo: fig_grafo.png")
plt.close()

# ================================================ FIGURA 2: os resultados
nos = np.array([1167, 15000, 57440])
gcn = np.array([-0.0350, -0.0241, -0.0160])
gat = np.array([-0.0488, -0.0160, -0.0088])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5))

# --- painel A: curva de escala ---
a1.axhline(0, color=INK2, lw=1.4, ls=(0, (5, 4)), zorder=1)
a1.text(1200, 0.0018, "paridade com o modelo textual", fontsize=11, color=INK2)

for vals, cor, rot in [(gcn, AZUL, "GCN"), (gat, LARANJA, "GAT")]:
    a1.plot(nos, vals, color=cor, lw=2, marker="o", ms=9,
            mec=SURF, mew=1.4, label=rot, zorder=3)
    for x, v in zip(nos, vals):
        a1.annotate(f"{v:+.4f}", (x, v), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=11, color=INK2)

a1.set_xscale("log")
a1.set_xticks(nos)
a1.set_xticklabels(["1.167", "15.000", "57.440"])
a1.set_xlabel("número de nós no grafo (escala log)")
a1.set_ylabel("diferença de F1-macro em relação ao MLP")
a1.set_title("O déficit da GNN encolhe com a escala,\nmas não chega a cruzar o zero")
a1.set_ylim(-0.058, 0.010)
a1.grid(axis="y", alpha=0.7)
a1.legend(frameon=False, loc="lower right", fontsize=12)

# --- painel B: comparacao dos modelos (dot plot, nao barra) ---
mods = ["MLP\n(sem grafo)", "GAT", "GCN"]
vals = [0.8006, 0.7918, 0.7847]
errs = [0.0116, 0.0075, 0.0030]
cores = [AGUA, LARANJA, AZUL]
ypos = np.arange(len(mods))[::-1]

a2.errorbar(vals, ypos, xerr=errs, fmt="none", ecolor=GRID, elinewidth=3,
            capsize=0, zorder=2)
for v, yy, c in zip(vals, ypos, cores):
    a2.plot(v, yy, "o", ms=13, color=c, mec=SURF, mew=1.6, zorder=3)
    a2.annotate(f"{v:.4f}", (v, yy), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=12, color=INK)

a2.set_yticks(ypos)
a2.set_yticklabels(mods, fontsize=12)
a2.set_xlabel("F1-macro nos 206 textos de teste (rótulo humano)")
a2.set_title("Grafo de 57.440 nós\nmédia de 10 execuções, barras = desvio-padrão")
a2.set_xlim(0.775, 0.815)
a2.grid(axis="x", alpha=0.7)
a2.set_ylim(-0.6, 2.6)

plt.tight_layout()
plt.savefig(BASE_DIR / "fig_resultados.png", dpi=140, bbox_inches="tight",
            facecolor=SURF)
print("salvo: fig_resultados.png")
