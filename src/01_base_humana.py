"""Junta Base A + Base B numa unica base humana."""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

BASE_DIR = Path(r"d:\TCC")
NOME = {0: "nao_odio", 1: "odio"}


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s*:?\s*", "", t)
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def carregar(caminho):
    wb = load_workbook(caminho, data_only=True, read_only=True)
    ws = wb["Sheet1"]
    linhas = []
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=2, values_only=True):
        if r[0] is None or str(r[0]).strip() == "" or r[1] is None:
            continue
        linhas.append((str(r[0]).strip(), int(str(r[1]).strip())))
    wb.close()
    return pd.DataFrame(linhas, columns=["text", "label"])


A = carregar(BASE_DIR / "Base A.xlsx")
B = carregar(BASE_DIR / "Base B.xlsx")
A["chave"] = A["text"].map(chave)
B["chave"] = B["text"].map(chave)

print(f"Base A: {len(A)}   Base B: {len(B)}")

nA = len(A); A = A.drop_duplicates("chave", keep="first")
nB = len(B); B = B.drop_duplicates("chave", keep="first")
print(f"apos duplicatas internas -> A: {len(A)} (-{nA-len(A)})   B: {len(B)} (-{nB-len(B)})")

n = len(B)
conflito = B[B["chave"].isin(set(A["chave"]))].merge(
    A[["chave", "label"]], on="chave", suffixes=("_B", "_A"))
dif = conflito[conflito["label_B"] != conflito["label_A"]]
print(f"B que ja estavam em A: {n - len(B[~B['chave'].isin(set(A['chave']))])}")
print(f"  destes, com rotulo DIFERENTE entre as bases: {len(dif)}")
for _, r in dif.iterrows():
    print(f"    A={NOME[r['label_A']]}  B={NOME[r['label_B']]}  | {r['text'][:75]}")

B = B[~B["chave"].isin(set(A["chave"]))]

A["fonte"] = "base_A"
B["fonte"] = "base_B"

hum = pd.concat([A, B], ignore_index=True)
hum["label_nome"] = hum["label"].map(NOME)
hum = hum[["text", "label", "label_nome", "fonte"]]

hum.to_csv(BASE_DIR / "base_humana.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 58)
print("BASE HUMANA")
print("=" * 58)
print(f"Total: {len(hum)} textos\n")
print(f"{'fonte':<10} {'nao_odio':>9} {'odio':>7} {'total':>7}")
for f in ["base_A", "base_B"]:
    s = hum[hum["fonte"] == f]
    print(f"{f:<10} {int((s['label']==0).sum()):>9} "
          f"{int((s['label']==1).sum()):>7} {len(s):>7}")
print(f"{'TOTAL':<10} {int((hum['label']==0).sum()):>9} "
      f"{int((hum['label']==1).sum()):>7} {len(hum):>7}")
print(f"\nProporcao de odio: {hum['label'].mean():.1%}")
print(f"\nSalvo: {BASE_DIR / 'base_humana.csv'}")
