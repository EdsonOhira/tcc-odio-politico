"""Rede em alta resolucao, sem lupa, preenchendo o quadro."""

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
AZUL, LARANJA = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd8"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK,
})

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

pos = nx.spring_layout(G, k=0.75, iterations=200, seed=42)
P = np.array([pos[i] for i in range(n)])

# 20 x 16 polegadas a 200 dpi = 4000 x 3200 px
fig = plt.figure(figsize=(20, 16))
ax = fig.add_axes([0.015, 0.045, 0.97, 0.895])

nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID, width=0.45, alpha=0.75)
for c, cor in [(0, AZUL), (1, LARANJA)]:
    sel = [i for i in range(n) if y[i] == c]
    nx.draw_networkx_nodes(G, pos, nodelist=sel, node_color=cor, node_size=48,
                           alpha=0.97, ax=ax, linewidths=0.9, edgecolors=SURF)

# recorta as margens vazias -> o grafo preenche o quadro
mx, my = np.ptp(P[:, 0]) * 0.025, np.ptp(P[:, 1]) * 0.025
ax.set_xlim(P[:, 0].min() - mx, P[:, 0].max() + mx)
ax.set_ylim(P[:, 1].min() - my, P[:, 1].max() + my)
ax.axis("off")

fig.text(0.5, 0.972, "Grafo de similaridade semântica entre textos",
         ha="center", fontsize=27)
fig.text(0.5, 0.949,
         "1.167 textos com rótulo humano  ·  4.929 ligações  ·  k = 5 vizinhos mais próximos",
         ha="center", fontsize=18, color=INK2)

ax.legend(handles=[mpatches.Patch(color=LARANJA, label="ódio  (576)"),
                   mpatches.Patch(color=AZUL, label="não-ódio  (591)")],
          loc="upper left", frameon=False, fontsize=21,
          handlelength=1.5, handleheight=1.1, borderpad=0.8)

fig.text(0.5, 0.016,
         f"homofilia {homof:.3f}  —  {homof:.0%} das ligações conectam textos da mesma "
         f"classe (o acaso daria 50%)",
         ha="center", fontsize=18, color=INK2)

plt.savefig(BASE_DIR / "fig_grafo_grande.png", dpi=200, facecolor=SURF)
print(f"salvo: fig_grafo_grande.png  ({n} nos, {G.number_of_edges()} arestas)")
