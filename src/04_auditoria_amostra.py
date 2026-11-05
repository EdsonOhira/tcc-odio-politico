"""Sorteia 100 pseudo-rotulos para auditoria manual cega."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"d:\TCC")
SEED = 42
POR_CLASSE = 50

df = pd.read_csv(BASE_DIR / "base_ampliada.csv", encoding="utf-8-sig")
pseudo = df[df["origem"] == "pseudo"].reset_index(drop=True)

print(f"Pool de pseudo-rotulos: {len(pseudo)}")
print(f"  nao_odio {int((pseudo['label']==0).sum())}  "
      f"odio {int((pseudo['label']==1).sum())}")

# amostra estratificada: 50 de cada classe prevista
partes = []
for c in [0, 1]:
    sub = pseudo[pseudo["label"] == c]
    partes.append(sub.sample(n=min(POR_CLASSE, len(sub)), random_state=SEED))

amostra = pd.concat(partes)
# embaralha para nao dar pista da ordem
amostra = amostra.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
amostra.insert(0, "id", range(1, len(amostra) + 1))

# planilha CEGA - sem o rotulo do modelo
cega = pd.DataFrame({
    "id": amostra["id"],
    "text": amostra["text"],
    "seu_rotulo": "",
})
cega.to_csv(BASE_DIR / "auditoria_100.csv", index=False, encoding="utf-8-sig")

# gabarito - so abrir DEPOIS
gab = pd.DataFrame({
    "id": amostra["id"],
    "label_modelo": amostra["label"],
    "label_modelo_nome": amostra["label_nome"],
    "confianca": amostra["confianca"],
})
gab.to_csv(BASE_DIR / "auditoria_100_gabarito.csv", index=False, encoding="utf-8-sig")

print(f"\nAmostra: {len(amostra)} textos")
print(f"  o modelo previu: nao_odio {int((amostra['label']==0).sum())}  "
      f"odio {int((amostra['label']==1).sum())}")
print(f"  confianca: min {amostra['confianca'].min():.4f}  "
      f"media {amostra['confianca'].mean():.4f}  max {amostra['confianca'].max():.4f}")

print(f"\nSalvos:")
print(f"  {BASE_DIR / 'auditoria_100.csv'}          <- preencher esta")
print(f"  {BASE_DIR / 'auditoria_100_gabarito.csv'} <- NAO abrir antes")

print("\nPrimeiras 3 linhas da planilha cega:")
for i in range(3):
    print(f"  {cega['id'].iloc[i]:>3} | {cega['text'].iloc[i][:95]}")
