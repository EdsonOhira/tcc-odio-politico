"""Figura do artigo: F1-macro por iteração do ciclo de auto-rotulagem (teste e validação)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(r"d:\TCC")
R = json.loads((BASE_DIR / "iteracoes (2).json").read_text(encoding="utf-8"))
rodadas = sorted(int(k.split("_")[1]) for k in R if k.startswith("rodada_"))

teste = [[m["f1_macro"] for m in R[f"rodada_{r}"]["medidas"]] for r in rodadas]
val = [R[f"rodada_{r}"]["val_medio"] for r in rodadas]
media_t = [np.mean(t) for t in teste]

COR_T, COR_V = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=300)

ax.axhline(media_t[0], color=TINTA2, lw=0.8, ls="--")
ax.text(7.15, media_t[0] + 0.0015, "nível da rodada 0", ha="right", va="bottom", fontsize=7.5, color=TINTA2)
for r, t in zip(rodadas, teste):
    ax.scatter([r] * len(t), t, s=12, color=COR_T, alpha=0.45, linewidths=0, zorder=2)
ax.plot(rodadas, media_t, "-o", color=COR_T, lw=2, ms=6, markeredgecolor="white",
        markeredgewidth=1.2, zorder=3, label="teste (média de 2 sementes)")
ax.plot(rodadas, val, "-s", color=COR_V, lw=2, ms=6, markeredgecolor="white",
        markeredgewidth=1.2, zorder=3, label="validação (critério de escolha)")
melhor = int(R["_estado"]["melhor_rodada"])
ax.annotate("melhor na validação", (melhor, val[melhor]), (melhor + 0.25, val[melhor] - 0.018),
            fontsize=7.5, color=TINTA2, arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.6))

ax.set_xticks(rodadas)
ax.set_xticklabels(["0"] + [str(r) for r in rodadas[1:]])
ax.set_xlabel("iteração (rodada) do ciclo de auto-rotulagem", color=TINTA2)
ax.set_ylabel("F1-macro", color=TINTA2)
ax.set_xlim(-0.4, 7.2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
ax.grid(axis="y", color=GRADE, lw=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(TINTA2)
ax.tick_params(colors=TINTA2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
destino = BASE_DIR / "fig_artigo_iteracoes.png"
fig.savefig(destino, facecolor="white")
print("salvo:", destino, "| teste:", [round(x, 4) for x in media_t], "| val:", [round(x, 4) for x in val])
