"""Rotula as 10.500 frases com o modelo_1 e separa pelo corte 0,95."""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "modelo_1"
CORTE = 0.95
BATCH = 64
NOME = {0: "nao_odio", 1: "odio"}

torch.set_num_threads(12)


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


# ------------------------------------------------------------------ dados
df = pd.read_csv(BASE_DIR / "10000 frases.csv", encoding="utf-8-sig",
                 engine="python", on_bad_lines="skip")
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)
df["chave"] = df["text"].map(chave)

A = carregar_xlsx(BASE_DIR / "Base A.xlsx")
B = carregar_xlsx(BASE_DIR / "Base B.xlsx")
A["chave"] = A["text"].map(chave)
B["chave"] = B["text"].map(chave)

set_a = set(A["chave"])
gab_b = dict(zip(B["chave"], B["label"]))

df["visto_no_treino"] = df["chave"].isin(set_a).astype(int)
df["duplicado"] = df.duplicated("chave", keep="first").astype(int)
df["gabarito_B"] = df["chave"].map(gab_b)

print("=" * 68)
print("DADOS")
print("=" * 68)
print(f"Linhas: {len(df)}")
print(f"Textos distintos: {df['chave'].nunique()}  "
      f"({int(df['duplicado'].sum())} repeticoes)")
print(f"Ja vistos no treino (Base A): {int(df['visto_no_treino'].sum())}")
print(f"Com gabarito humano (Base B): {int(df['gabarito_B'].notna().sum())}")

# ------------------------------------------------------------- inferencia
print("\n" + "=" * 68)
print("ROTULACAO com modelo_1")
print("=" * 68)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
modelo = BertForSequenceClassification.from_pretrained(MODEL_DIR).eval()


class DS(Dataset):
    def __init__(self, textos):
        self.enc = tok(list(textos), max_length=128, padding="max_length",
                       truncation=True, return_tensors="pt")

    def __len__(self):
        return self.enc["input_ids"].size(0)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i]}


ds = DS(df["text"])
logits = []
with torch.no_grad():
    for k, b in enumerate(DataLoader(ds, batch_size=BATCH)):
        logits.append(modelo(input_ids=b["input_ids"],
                             attention_mask=b["attention_mask"]).logits)
        if k % 25 == 0:
            print(f"  {k * BATCH}/{len(df)}")
logits = torch.cat(logits)

probs = torch.softmax(logits, dim=1).numpy()
df["label"] = probs.argmax(axis=1)
df["confianca"] = probs.max(axis=1).round(6)
df["label_nome"] = df["label"].map(NOME)
df["aceito"] = (df["confianca"] >= CORTE).astype(int)

# ------------------------------------------------------------- resultados
print("\n" + "=" * 68)
print("RESULTADOS")
print("=" * 68)

print("Distribuicao prevista em TODAS as 10.500:")
for c in [0, 1]:
    n = int((df["label"] == c).sum())
    print(f"  {NOME[c]:<9} {n:>6}  ({n / len(df):.1%})")

aceitos = df[df["aceito"] == 1]
revisao = df[df["aceito"] == 0]

print(f"\nCorte {CORTE}:")
print(f"  aceite automatico: {len(aceitos):>6}  ({len(aceitos)/len(df):.1%})")
print(f"  revisao manual:    {len(revisao):>6}  ({len(revisao)/len(df):.1%})")
print(f"  (na Base B a cobertura foi 21,2% -- comparar)")

print("\nDistribuicao entre os ACEITOS:")
for c in [0, 1]:
    n = int((aceitos["label"] == c).sum())
    print(f"  {NOME[c]:<9} {n:>6}  ({n / max(len(aceitos), 1):.1%})")

print("\nHistograma de confianca (todas as 10.500):")
faixas = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90),
          (0.90, 0.95), (0.95, 0.96), (0.96, 0.97), (0.97, 1.01)]
for lo, hi in faixas:
    n = int(((df["confianca"] >= lo) & (df["confianca"] < hi)).sum())
    barra = "#" * int(60 * n / len(df))
    print(f"  {lo:.2f}-{hi:.2f}  {n:>6} ({n/len(df):>5.1%}) {barra}")

# validacao real, onde houver gabarito humano
com_gab = df[df["gabarito_B"].notna()]
if len(com_gab) > 0:
    print("\n" + "=" * 68)
    print(f"VALIDACAO REAL nos {len(com_gab)} textos que tem gabarito da Base B")
    print("=" * 68)
    ok = (com_gab["label"] == com_gab["gabarito_B"]).mean()
    print(f"  acuracia geral: {ok:.4f}")
    alto = com_gab[com_gab["confianca"] >= CORTE]
    if len(alto) > 0:
        ok_alto = (alto["label"] == alto["gabarito_B"]).mean()
        print(f"  no corte {CORTE}: {int((alto['label'] == alto['gabarito_B']).sum())}/"
              f"{len(alto)} = {ok_alto:.4f}")

# --------------------------------------------------------------- arquivos
COLS = ["text", "label", "label_nome", "confianca", "visto_no_treino", "duplicado"]
aceitos[COLS].to_csv(BASE_DIR / "aceites_automaticos.csv", index=False, encoding="utf-8-sig")
revisao[COLS].to_csv(BASE_DIR / "revisao_manual.csv", index=False, encoding="utf-8-sig")
df[COLS].to_csv(BASE_DIR / "corpus_rotulado.csv", index=False, encoding="utf-8-sig")

registro = {
    "corte": CORTE,
    "n_total": len(df),
    "n_distintos": int(df["chave"].nunique()),
    "duplicados": int(df["duplicado"].sum()),
    "vistos_no_treino": int(df["visto_no_treino"].sum()),
    "distribuicao_total": {NOME[c]: int((df["label"] == c).sum()) for c in [0, 1]},
    "aceitos": len(aceitos),
    "revisao": len(revisao),
    "cobertura": len(aceitos) / len(df),
    "distribuicao_aceitos": {NOME[c]: int((aceitos["label"] == c).sum()) for c in [0, 1]},
}
with open(RESULTS / "rotulagem_10k.json", "w", encoding="utf-8") as f:
    json.dump(registro, f, indent=2, ensure_ascii=False)

print("\nSalvos:")
for nome, n in [("aceites_automaticos.csv", len(aceitos)),
                ("revisao_manual.csv", len(revisao)),
                ("corpus_rotulado.csv", len(df))]:
    print(f"  {BASE_DIR / nome}  ({n} linhas)")
print(f"  {RESULTS / 'rotulagem_10k.json'}")
