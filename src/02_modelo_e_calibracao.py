"""Passos 1-3: treina o Modelo 1 na Base A, rotula a Base B, varre os cortes."""

import copy
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from openpyxl import load_workbook
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)

BASE_DIR = Path(r"d:\TCC")
RESULTS = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "modelo_1"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LEN, EPOCAS, LR, BATCH, BATCH_INF = 128, 5, 2e-5, 16, 64
WEIGHT_DECAY, WARMUP, PACIENCIA, FREEZE = 0.01, 0.1, 2, 4
SEED = 42

torch.set_num_threads(12)
torch.manual_seed(SEED)
np.random.seed(SEED)


def carregar(caminho):
    wb = load_workbook(caminho, data_only=True, read_only=True)
    ws = wb["Sheet1"]
    linhas = []
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=2, values_only=True):
        t, y = r[0], r[1]
        if t is None or str(t).strip() == "" or y is None:
            continue
        linhas.append((str(t).strip(), int(str(y).strip())))
    wb.close()
    return pd.DataFrame(linhas, columns=["text", "label"])


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s*:?\s*", "", t)
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


A = carregar(BASE_DIR / "Base A.xlsx")
B = carregar(BASE_DIR / "Base B.xlsx")

A["chave"] = A["text"].map(chave)
B["chave"] = B["text"].map(chave)

n_b_antes = len(B)
B = B[~B["chave"].isin(set(A["chave"]))].reset_index(drop=True)
removidos = n_b_antes - len(B)

print("=" * 70)
print("PASSO 0 - dados")
print("=" * 70)
print(f"Base A: {len(A)} linhas   {dict(A['label'].value_counts().sort_index())}")
print(f"Base B: {n_b_antes} -> {len(B)} linhas ({removidos} removidos por estarem em A)")
print(f"        {dict(B['label'].value_counts().sort_index())}")
dup_b = len(B) - B["chave"].nunique()
print(f"        duplicatas internas remanescentes em B: {dup_b}")

# ------------------------------------------------------- PASSO 1: treino
print("\n" + "=" * 70)
print("PASSO 1 - treinar o Modelo 1 (so na Base A)")
print("=" * 70)

tr, val = train_test_split(A, test_size=100, stratify=A["label"], random_state=SEED)
print(f"treino {len(tr)}  {dict(tr['label'].value_counts().sort_index())}")
print(f"val    {len(val)}  {dict(val['label'].value_counts().sort_index())}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


class DS(Dataset):
    def __init__(self, textos, rot):
        self.enc = tok(list(textos), max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        self.y = torch.tensor(list(rot), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.y[i]}


ds_tr, ds_val, ds_b = DS(tr["text"], tr["label"]), DS(val["text"], val["label"]), \
    DS(B["text"], B["label"])

modelo = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "nao_odio", 1: "odio"}, label2id={"nao_odio": 0, "odio": 1})

total_c = len(modelo.bert.encoder.layer)
for p in modelo.parameters():
    p.requires_grad = False
for camada in modelo.bert.encoder.layer[total_c - FREEZE:]:
    for p in camada.parameters():
        p.requires_grad = True
for p in modelo.bert.pooler.parameters():
    p.requires_grad = True
for p in modelo.classifier.parameters():
    p.requires_grad = True

trein = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
todos = sum(p.numel() for p in modelo.parameters())
print(f"freeze_layers={FREEZE}: {trein:,}/{todos:,} treinaveis ({100*trein/todos:.1f}%)")

loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
otim = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad],
                         lr=LR, weight_decay=WEIGHT_DECAY)
passos = EPOCAS * len(loader)
sched = get_linear_schedule_with_warmup(otim, int(WARMUP * passos), passos)
perda_fn = nn.CrossEntropyLoss()   # base balanceada: sem pesos


def prever(m, ds):
    m.eval()
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=BATCH_INF):
            out.append(m(input_ids=b["input_ids"],
                         attention_mask=b["attention_mask"]).logits)
    return torch.cat(out)


melhor, estado, espera, historico = -1.0, None, 0, []
for ep in range(1, EPOCAS + 1):
    modelo.train()
    soma = 0.0
    for b in loader:
        otim.zero_grad()
        out = modelo(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        perda = perda_fn(out.logits, b["labels"])
        perda.backward()
        otim.step()
        sched.step()
        soma += perda.item()

    pv = prever(modelo, ds_val).argmax(1).numpy()
    f1v = f1_score(val["label"], pv, average="macro", zero_division=0)
    historico.append({"epoca": ep, "perda": soma / len(loader), "val_f1_macro": f1v})
    print(f"  epoca {ep}/{EPOCAS}  perda={soma/len(loader):.4f}  val_F1={f1v:.4f}")

    if f1v > melhor + 1e-4:
        melhor, estado, espera = f1v, copy.deepcopy(modelo.state_dict()), 0
    else:
        espera += 1
        if espera >= PACIENCIA:
            print("  early stopping")
            break

modelo.load_state_dict(estado)
print(f"  melhor val F1-macro: {melhor:.4f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
modelo.save_pretrained(MODEL_DIR)
tok.save_pretrained(MODEL_DIR)
print(f"  modelo salvo em {MODEL_DIR}")

# --------------------------------------------------- PASSO 2: rotular B
print("\n" + "=" * 70)
print("PASSO 2 - Modelo 1 rotula a Base B")
print("=" * 70)

logits = prever(modelo, ds_b)
probs = torch.softmax(logits, dim=1).numpy()
pred = probs.argmax(axis=1)
conf = probs.max(axis=1)
true = B["label"].values

B_out = pd.DataFrame({
    "text": B["text"], "label_humano": true, "label_modelo": pred,
    "confianca": conf.round(6), "acertou": (pred == true).astype(int),
})
B_out.to_csv(BASE_DIR / "base_B_rotulada.csv", index=False, encoding="utf-8-sig")

print(f"Acuracia sem corte: {(pred == true).mean():.4f}  "
      f"({int((pred == true).sum())}/{len(true)})")
print(f"O modelo previu: {dict(pd.Series(pred).value_counts().sort_index())}")
print(f"Gabarito humano: {dict(pd.Series(true).value_counts().sort_index())}")
print()
print(classification_report(true, pred, target_names=["nao_odio", "odio"], digits=4))
print("Matriz de confusao (linha=humano, coluna=modelo):")
print(confusion_matrix(true, pred))

# ------------------------------------------------ PASSO 3: varrer cortes
print("\n" + "=" * 70)
print("PASSO 3 - varredura dos cortes de confianca")
print("=" * 70)

GRADE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
         0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def wilson_inf(acertos, n, z=1.96):
    if n == 0:
        return float("nan")
    p = acertos / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h)


def varrer(mask_classe=None, nome="GLOBAL"):
    linhas = []
    total = len(true) if mask_classe is None else int((pred == mask_classe).sum())
    for t in GRADE:
        m = conf >= t
        if mask_classe is not None:
            m = m & (pred == mask_classe)
        n = int(m.sum())
        if n == 0:
            linhas.append({"corte": t, "n": 0, "acertos": 0, "precisao": None,
                           "cobertura": 0.0, "wilson_inf": None, "regra_de_tres": None})
            continue
        ac = int((pred[m] == true[m]).sum())
        p = ac / n
        linhas.append({
            "corte": t, "n": n, "acertos": ac, "precisao": p,
            "cobertura": n / max(total, 1),
            "wilson_inf": wilson_inf(ac, n),
            "regra_de_tres": (3.0 / n) if ac == n else None,
        })
    return linhas


def imprimir(linhas, nome):
    print(f"\n--- {nome} ---")
    print(f"{'corte':>6} {'n':>5} {'acertos':>8} {'precisao':>9} {'cobertura':>10} "
          f"{'IC95 inf':>9} {'lim.erro':>9}")
    for r in linhas:
        if r["n"] == 0:
            print(f"{r['corte']:>6.3f} {0:>5}        -         -          -         -         -")
            continue
        le = f"{r['regra_de_tres']:.1%}" if r["regra_de_tres"] is not None else "-"
        print(f"{r['corte']:>6.3f} {r['n']:>5} {r['acertos']:>8} {r['precisao']:>9.4f} "
              f"{r['cobertura']:>10.1%} {r['wilson_inf']:>9.4f} {le:>9}")


tabelas = {"global": varrer(None)}
imprimir(tabelas["global"], "GLOBAL")
for cid, nome in [(0, "nao_odio"), (1, "odio")]:
    tabelas[nome] = varrer(cid)
    imprimir(tabelas[nome], f"CLASSE {nome} (entre os que o modelo previu como {nome})")


def menor_100(linhas):
    for r in linhas:
        if r["n"] > 0 and r["precisao"] == 1.0:
            return r
    return None


print("\n" + "=" * 70)
print("MENOR CORTE COM 100% DE PRECISAO")
print("=" * 70)
resumo = {}
for nome, linhas in tabelas.items():
    r = menor_100(linhas)
    resumo[nome] = r
    if r is None:
        print(f"  {nome:<10} nunca atinge 100% na grade testada")
    else:
        print(f"  {nome:<10} corte {r['corte']:.3f}  |  {r['acertos']}/{r['n']} acertos  |  "
              f"cobertura {r['cobertura']:.1%}  |  erro real pode chegar a "
              f"{r['regra_de_tres']:.1%}")

with open(RESULTS / "calibracao_corte.json", "w", encoding="utf-8") as f:
    json.dump({
        "base_A": {"n": len(A), "distribuicao": {str(k): int(v) for k, v in A["label"].value_counts().items()}},
        "base_B": {"n_original": n_b_antes, "n_usado": len(B), "removidos_por_overlap": removidos,
                   "distribuicao": {str(k): int(v) for k, v in B["label"].value_counts().items()}},
        "treino": {"n_treino": len(tr), "n_val": len(val), "freeze_layers": FREEZE,
                   "seed": SEED, "historico": historico, "melhor_val_f1_macro": melhor},
        "acuracia_B_sem_corte": float((pred == true).mean()),
        "grade": GRADE, "tabelas": tabelas,
        "menor_corte_100pct": {k: v for k, v in resumo.items()},
    }, f, indent=2, ensure_ascii=False, default=float)

print(f"\nSalvos:")
print(f"  {BASE_DIR / 'base_B_rotulada.csv'}")
print(f"  {RESULTS / 'calibracao_corte.json'}")
print(f"  {MODEL_DIR}")
print("\nPARADO NO PASSO 3 - passo 4 aguarda seu ok.")
