"""Extrai o corpus bruto do CIBERDEM (zip de 2,85 GB) para um CSV enxuto.

Os 7 arquivos trazem todos os campos da API do Twitter - entities, context_annotations,
public_metrics - que nao servem para classificar texto e pesam 95% do volume. Aqui fica
so o necessario, processado em fluxo porque a maquina tem ~3 GB de RAM livre.

Passo A (por arquivo, retomavel): le em blocos direto de dentro do zip e grava uma parte
        enxuta em results/partes_08jan/. Mantem so lang=pt.
Passo B: junta as partes, remove duplicatas por texto normalizado e separa retweets de
        tweets originais, gravando corpus_08jan.csv (originais, para rotular) e
        contando quantas copias cada texto teve.

Retweets sao separados porque a API os entrega TRUNCADOS (terminam em "…"), entao o
texto esta incompleto - rotular texto cortado gera rotulo ruim. A contagem de copias
fica registrada para quem quiser usar popularidade depois.
"""

import re
import time
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

ZIP = Path(r"C:\Users\e50054811\Downloads\08Jan 1.zip")
BASE = Path(r"d:\TCC")
PARTES = BASE / "results" / "partes_08jan"
COLS = ["created_at", "author_id", "text", "conversation_id", "lang"]
BLOCO = 100_000

PARTES.mkdir(parents=True, exist_ok=True)


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s+@\w+\s*:\s*", "", t)          # prefixo de retweet
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


# ---------------------------------------------------------------- passo A
t0 = time.time()
with zipfile.ZipFile(ZIP) as z:
    nomes = sorted(n for n in z.namelist() if n.lower().endswith(".csv"))
    print(f"{len(nomes)} arquivos no zip\n")
    for nome in nomes:
        rotulo = nome.split("/")[0]
        saida = PARTES / f"{rotulo}.csv"
        if saida.exists():
            print(f"[pulando] {rotulo} - parte ja existe")
            continue
        tam = z.getinfo(nome).file_size / 1e6
        print(f"--- {rotulo}  ({tam:.0f} MB)", flush=True)
        n_lidas = n_pt = 0
        primeiro = True
        idiomas = {}
        with z.open(nome) as f:
            leitor = pd.read_csv(f, usecols=COLS, chunksize=BLOCO, dtype=str,
                                 on_bad_lines="skip", low_memory=False)
            for bloco in leitor:
                n_lidas += len(bloco)
                for k, v in bloco["lang"].value_counts().items():
                    idiomas[k] = idiomas.get(k, 0) + int(v)
                bloco = bloco[bloco["lang"] == "pt"]
                bloco = bloco[bloco["text"].notna()]
                if bloco.empty:
                    continue
                bloco = bloco.assign(
                    text=bloco["text"].astype(str).str.strip(),
                    eh_rt=bloco["text"].astype(str).str.match(r"^RT\s+@"))
                bloco = bloco[bloco["text"].str.len() >= 10]
                n_pt += len(bloco)
                bloco.to_csv(saida, mode="w" if primeiro else "a",
                             header=primeiro, index=False, encoding="utf-8-sig")
                primeiro = False
                print(f"    {n_lidas:>9,} linhas lidas, {n_pt:>9,} em pt "
                      f"({(time.time()-t0)/60:.0f} min)", flush=True)
        top = sorted(idiomas.items(), key=lambda x: -x[1])[:4]
        print(f"  {rotulo}: {n_lidas:,} linhas -> {n_pt:,} em pt")
        print(f"  idiomas: {', '.join(f'{k}={v:,}' for k, v in top)}\n", flush=True)

# ---------------------------------------------------------------- passo B
print("=" * 70)
print("PASSO B - juntar partes, remover duplicatas, separar retweets")
print("=" * 70, flush=True)

vistos = {}          # hash do texto normalizado -> numero de copias
linhas = []
n_total = n_rt = 0
for parte in sorted(PARTES.glob("*.csv")):
    for bloco in pd.read_csv(parte, chunksize=BLOCO, dtype=str,
                             encoding="utf-8-sig", on_bad_lines="skip"):
        n_total += len(bloco)
        bloco["eh_rt"] = bloco["eh_rt"].astype(str).str.lower() == "true"
        n_rt += int(bloco["eh_rt"].sum())
        for r in bloco.itertuples(index=False):
            k = hash(chave(r.text))
            if k in vistos:
                vistos[k] += 1
                continue
            vistos[k] = 1
            linhas.append((k, r.text, r.eh_rt, r.created_at,
                           r.author_id, r.conversation_id))
    print(f"  {parte.name}: acumulado {n_total:,} linhas, "
          f"{len(linhas):,} textos distintos ({(time.time()-t0)/60:.0f} min)",
          flush=True)

df = pd.DataFrame(linhas, columns=["k", "text", "eh_rt", "created_at",
                                   "author_id", "conversation_id"])
df["n_copias"] = df["k"].map(vistos)
df = df.drop(columns="k")
del linhas, vistos

orig = df[~df["eh_rt"]].drop(columns="eh_rt").reset_index(drop=True)
rts = df[df["eh_rt"]].drop(columns="eh_rt").reset_index(drop=True)
orig.to_csv(BASE / "corpus_08jan.csv", index=False, encoding="utf-8-sig")
rts.to_csv(BASE / "corpus_08jan_retweets.csv", index=False, encoding="utf-8-sig")

print(f"\n{'linhas em pt':<28} {n_total:>10,}")
print(f"{'retweets':<28} {n_rt:>10,}  ({n_rt/max(n_total,1):.1%})")
print(f"{'textos distintos':<28} {len(df):>10,}")
print(f"{'  originais (para rotular)':<28} {len(orig):>10,}")
print(f"{'  retweets (truncados)':<28} {len(rts):>10,}")
print(f"\ncorpus_08jan.csv          {len(orig):,} textos")
print(f"corpus_08jan_retweets.csv {len(rts):,} textos")
print(f"tempo total: {(time.time()-t0)/60:.0f} min")
