"""Monta base_grafo_57k (completa) e base_grafo_15k (subconjunto para ensaio)."""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"d:\TCC")
SEED = 42
N_15K = 15000


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s*:?\s*", "", t)
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


hum = pd.read_csv(BASE_DIR / "base_humana.csv", encoding="utf-8-sig")
hum["chave"] = hum["text"].map(chave)
print(f"base_humana: {len(hum)}  ({dict(hum['fonte'].value_counts())})")

corp = pd.read_excel(BASE_DIR / "Dataset-100000k.xlsx", sheet_name=0)
corp["text"] = corp["text"].astype(str).str.strip()
corp = corp[(corp["text"] != "") & (corp["text"].str.lower() != "nan")]
corp["chave"] = corp["text"].map(chave)
corp = corp[corp["chave"] != ""]
print(f"corpus bruto: {len(corp)} linhas nao vazias")

n0 = len(corp)
corp = corp.drop_duplicates("chave", keep="first")
print(f"  apos deduplicar: {len(corp)}  (-{n0 - len(corp)})")

n0 = len(corp)
corp = corp[~corp["chave"].isin(set(hum["chave"]))]
print(f"  apos remover os ja rotulados: {len(corp)}  (-{n0 - len(corp)})")

# ------------------------------------------------------------------ montar
rot = pd.DataFrame({
    "text": hum["text"], "label": hum["label"].astype("Int64"),
    "origem": "manual", "fonte": hum["fonte"],
})
nao = pd.DataFrame({
    "text": corp["text"], "label": pd.array([pd.NA] * len(corp), dtype="Int64"),
    "origem": "sem_rotulo", "fonte": "corpus",
})

full = pd.concat([rot, nao], ignore_index=True)
full.to_csv(BASE_DIR / "base_grafo_57k.csv", index=False, encoding="utf-8-sig")

# ensaio: todos os rotulados + amostra dos nao rotulados
rng = np.random.default_rng(SEED)
n_amostra = N_15K - len(rot)
sel = rng.choice(len(nao), size=n_amostra, replace=False)
peq = pd.concat([rot, nao.iloc[np.sort(sel)]], ignore_index=True)
peq.to_csv(BASE_DIR / "base_grafo_15k.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
for nome, d in [("base_grafo_57k.csv", full), ("base_grafo_15k.csv", peq)]:
    r = d[d["origem"] == "manual"]
    print(f"\n{nome}: {len(d)} nos")
    print(f"  rotulados   {len(r):>6}  "
          f"(base_A {int((r['fonte']=='base_A').sum())}, "
          f"base_B {int((r['fonte']=='base_B').sum())})")
    print(f"  sem rotulo  {len(d) - len(r):>6}")
    print(f"  proporcao rotulada: {len(r)/len(d):.2%}")

print(f"\nO 15k e subconjunto do 57k (mesmos 1167 rotulados, seed {SEED}).")
