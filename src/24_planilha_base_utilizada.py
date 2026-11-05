"""Planilha com os 1.167 textos usados nos experimentos e seus rótulos.

As colunas dos anotadores vêm da planilha de anotação e ficam vazias para os textos
que não estão nela (os vindos de Ódio.csv).
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"d:\TCC")
ANOT = ["Vinola", "Edin", "Carlin"]


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s*:?\s*", "", t)
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


hum = pd.read_csv(BASE_DIR / "base_humana.csv")
hum["k"] = hum["text"].map(chave)

kap = pd.read_excel(BASE_DIR / "_originais" / "KAPPA - Separado.xlsx",
                    sheet_name="Rotulaçao - TCC")
for c in ANOT:
    kap[c] = pd.to_numeric(kap[c], errors="coerce")
kap["k"] = kap["text"].map(chave)
kap = kap.drop_duplicates("k", keep="first")

base = hum.merge(kap[["k"] + ANOT], on="k", how="left")
base["conjunto"] = base["fonte"].map({"base_A": "treino", "base_B": "teste"})
base["origem"] = base[ANOT[0]].notna().map({True: "planilha de anotação", False: "Ódio.csv"})
base["rotulo"] = base["label_nome"].map({"odio": "ódio", "nao_odio": "não ódio"})
for c in ANOT:
    base[c] = base[c].astype("Int64")

cols = ["text", "rotulo", "label", "conjunto", "origem"] + ANOT
base = base[cols].rename(columns={"text": "texto"})

destino = BASE_DIR / "base_utilizada_rotulos.xlsx"
with pd.ExcelWriter(destino) as w:
    base.to_excel(w, sheet_name="todos", index=False)
    base[base.label == 1].to_excel(w, sheet_name="ódio", index=False)
    base[base.label == 0].to_excel(w, sheet_name="não ódio", index=False)

print(f"salvo: {destino}")
print(pd.crosstab([base.conjunto, base.origem], base.rotulo, margins=True))
