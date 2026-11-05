"""Concordância entre anotadores restrita aos textos que entraram na base usada.

O kappa_metricas.json foi calculado sobre as 1.396 linhas da planilha de anotação.
Este script recalcula as métricas considerando os textos da planilha que entraram
na base usada nos experimentos (base_humana.csv, 1.167 textos).
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"d:\TCC")
ANOT = ["Vinola", "Edin", "Carlin"]
HUMANOS = ["Vinola", "Carlin"]


def chave(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = re.sub(r"^rt\s*:?\s*", "", t)
    t = re.sub(r"@\w+|#\w+|http\S+", " ", t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fleiss(x):
    n, m = x.shape
    um = x.sum(axis=1)
    cont = np.stack([m - um, um], axis=1)
    p_i = ((cont ** 2).sum(axis=1) - m) / (m * (m - 1))
    p_bar = p_i.mean()
    p_j = cont.sum(axis=0) / (n * m)
    p_e = (p_j ** 2).sum()
    return p_bar, (p_bar - p_e) / (1 - p_e)


def krippendorff_nominal(x):
    n, m = x.shape
    um = x.sum(axis=1)
    cont = np.stack([m - um, um], axis=1).astype(float)
    o = np.zeros((2, 2))
    for c in cont:
        for a in range(2):
            for b in range(2):
                o[a, b] += c[a] * (c[b] - (a == b)) / (m - 1)
    n_c = o.sum(axis=1)
    tot = n_c.sum()
    d_o = o[0, 1] + o[1, 0]
    d_e = 2 * n_c[0] * n_c[1] / (tot - 1)
    return 1 - d_o / d_e


def gwet_ac1(x):
    n, m = x.shape
    um = x.sum(axis=1)
    cont = np.stack([m - um, um], axis=1)
    p_a = (((cont * (cont - 1)).sum(axis=1)) / (m * (m - 1))).mean()
    pi = (cont / m).mean(axis=0)
    p_e = (pi * (1 - pi)).sum()  # (1/(Q-1)) com Q = 2
    return (p_a - p_e) / (1 - p_e)


def cohen(a, b):
    p_o = (a == b).mean()
    p_e = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    if p_e == 1:  # os dois deram sempre a mesma classe: kappa indefinido
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


def metricas(df):
    x = df[ANOT].to_numpy(int)
    votos = x.sum(axis=1)
    p_bar, kf = fleiss(x)
    xh = df[HUMANOS].to_numpy(int)
    return {
        "n_textos": int(len(df)),
        "concordancia_bruta": round(float(p_bar), 4),
        "fleiss_kappa": round(float(kf), 4),
        "krippendorff_alpha": round(float(krippendorff_nominal(x)), 4),
        "gwet_ac1": round(float(gwet_ac1(x)), 4),
        "cohen_pares": {
            f"{a} x {b}": round(float(cohen(df[a].to_numpy(int), df[b].to_numpy(int))), 4)
            for i, a in enumerate(ANOT) for b in ANOT[i + 1:]
        },
        "so_humanos_vinola_x_carlin": {
            "concordancia_bruta": round(float((xh[:, 0] == xh[:, 1]).mean()), 4),
            "cohen_kappa": round(float(cohen(xh[:, 0], xh[:, 1])), 4),
            "gwet_ac1": round(float(gwet_ac1(xh)), 4),
        },
        "unanimes": int(((votos == 0) | (votos == 3)).sum()),
        "divididos": int(((votos == 1) | (votos == 2)).sum()),
        "votos_odio_por_texto": {str(v): int((votos == v).sum()) for v in range(4)},
        "taxa_odio_por_anotador": {a: round(float(df[a].mean()), 4) for a in ANOT},
        "rotulo_usado_x_anotador": {
            "concorda_com_maioria": int(((votos >= 2).astype(int) == df["label"]).sum()),
            **{f"concorda_com_{a}": int((df[a] == df["label"]).sum()) for a in ANOT},
            **{f"cohen_rotulo_x_{a}": round(float(cohen(df["label"].to_numpy(int),
                                                         df[a].to_numpy(int))), 4)
               for a in ANOT},
        },
    }


kap = pd.read_excel(BASE_DIR / "_originais" / "KAPPA - Separado.xlsx",
                    sheet_name="Rotulaçao - TCC")
for c in ANOT:
    kap[c] = pd.to_numeric(kap[c], errors="coerce")
kap = kap.dropna(subset=ANOT)

# conferência: a planilha inteira precisa reproduzir o kappa_metricas.json
ref = json.loads((BASE_DIR / "results" / "kappa_metricas.json").read_text(encoding="utf-8"))
p_bar, kf = fleiss(kap[ANOT].to_numpy(int))
assert abs(kf - ref["fleiss_kappa"]) < 1e-9, (kf, ref["fleiss_kappa"])
assert abs(p_bar - ref["concordancia_bruta"]) < 1e-9
assert abs(krippendorff_nominal(kap[ANOT].to_numpy(int)) - ref["krippendorff_alpha"]) < 1e-9
assert abs(gwet_ac1(kap[ANOT].to_numpy(int)) - ref["gwet_ac1"]) < 1e-9
print(f"planilha inteira reproduzida: Fleiss {kf:.4f}, n = {len(kap)}")

kap["k"] = kap["text"].map(chave)
kap = kap.drop_duplicates("k", keep="first")

hum = pd.read_csv(BASE_DIR / "base_humana.csv")
hum["k"] = hum["text"].map(chave)
usada = hum.merge(kap[["k"] + ANOT], on="k", how="inner")

saida = {
    "descricao": ("Concordância restrita aos textos da planilha de anotação que entraram "
                  "em base_humana.csv."),
    "procedencia_das_colunas": ref["procedencia_das_colunas"],
    "base_humana_total": int(len(hum)),
    "base_humana_na_planilha": int(len(usada)),
    "base_humana_fora_da_planilha": int(len(hum) - len(usada)),
    "planilha_inteira_referencia": {k: ref[k] for k in
                                    ["n_linhas", "concordancia_bruta", "fleiss_kappa",
                                     "krippendorff_alpha", "gwet_ac1"]},
    "base_usada": metricas(usada),
    "treino_base_A": metricas(usada[usada.fonte == "base_A"]),
    "teste_base_B": metricas(usada[usada.fonte == "base_B"]),
}
destino = BASE_DIR / "results" / "kappa_base_utilizada.json"


def sem_nan(o):
    if isinstance(o, dict):
        return {k: sem_nan(v) for k, v in o.items()}
    if isinstance(o, float) and o != o:
        return None
    return o


texto = json.dumps(sem_nan(saida), ensure_ascii=False, indent=2)
destino.write_text(texto, encoding="utf-8")
print(texto)
