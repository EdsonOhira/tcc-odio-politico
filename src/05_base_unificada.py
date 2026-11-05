"""Base unica: Base A + Base B (humanas) + pseudo-rotulos aceitos."""

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


def carregar_xlsx(caminho):
    wb = load_workbook(caminho, data_only=True, read_only=True)
    ws = wb["Sheet1"]
    linhas = []
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=2, values_only=True):
        if r[0] is None or str(r[0]).strip() == "" or r[1] is None:
            continue
        linhas.append((str(r[0]).strip(), int(str(r[1]).strip())))
    wb.close()
    return pd.DataFrame(linhas, columns=["text", "label"])


A = carregar_xlsx(BASE_DIR / "Base A.xlsx")
B = carregar_xlsx(BASE_DIR / "Base B.xlsx")
ace = pd.read_csv(BASE_DIR / "aceites_automaticos.csv", encoding="utf-8-sig")

for d in (A, B, ace):
    d["chave"] = d["text"].map(chave)

print(f"Base A: {len(A)}   Base B: {len(B)}   aceitos: {len(ace)}")

# 1) duplicatas internas
A = A.drop_duplicates("chave", keep="first")
B = B.drop_duplicates("chave", keep="first")
print(f"\napos duplicatas internas -> A: {len(A)}   B: {len(B)}")

# 2) B nao repete o que ja esta em A (o rotulo humano de A prevalece)
n = len(B)
B = B[~B["chave"].isin(set(A["chave"]))]
print(f"B apos remover os que ja estao em A: {len(B)}  (-{n - len(B)})")

# 3) pseudo nao repete nada humano
humanos = set(A["chave"]) | set(B["chave"])
n = len(ace)
ace = ace[~ace["chave"].isin(humanos)]
print(f"pseudo apos remover os que ja tem rotulo humano: {len(ace)}  (-{n - len(ace)})")


def bloco(df, origem, fonte, conf=None):
    return pd.DataFrame({
        "text": df["text"], "label": df["label"],
        "label_nome": df["label"].map(NOME),
        "origem": origem, "fonte": fonte,
        "confianca": conf if conf is not None else pd.NA,
    })


nova = pd.concat([
    bloco(A, "manual", "base_A"),
    bloco(B, "manual", "base_B"),
    bloco(ace, "pseudo", "pseudo", ace["confianca"]),
], ignore_index=True)

nova["chave"] = nova["text"].map(chave)
assert nova["chave"].nunique() == len(nova), "sobrou duplicata"
nova = nova.drop(columns=["chave"])

nova.to_csv(BASE_DIR / "base_unificada.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 64)
print("BASE UNIFICADA")
print("=" * 64)
print(f"Total: {len(nova)} textos\n")
print(f"{'fonte':<10} {'origem':<8} {'nao_odio':>9} {'odio':>7} {'total':>7} {'%':>7}")
for f, o in [("base_A", "manual"), ("base_B", "manual"), ("pseudo", "pseudo")]:
    s = nova[nova["fonte"] == f]
    print(f"{f:<10} {o:<8} {int((s['label']==0).sum()):>9} "
          f"{int((s['label']==1).sum()):>7} {len(s):>7} {len(s)/len(nova):>6.1%}")

hum = nova[nova["origem"] == "manual"]
print(f"\n{'HUMANOS':<19} {int((hum['label']==0).sum()):>9} "
      f"{int((hum['label']==1).sum()):>7} {len(hum):>7} {len(hum)/len(nova):>6.1%}")
print(f"{'TOTAL':<19} {int((nova['label']==0).sum()):>9} "
      f"{int((nova['label']==1).sum()):>7} {len(nova):>7}")
print(f"\nProporcao de odio: {nova['label'].mean():.1%}")
print(f"\nSalvo: {BASE_DIR / 'base_unificada.csv'}")
