# Código — TCC 2

Scripts que produzem os resultados do `Registro_Tecnico_TCC2.pdf`. Numerados na
ordem em que devem ser executados: cada um consome a saída do anterior.

Ambiente: `d:\TCC\.venv` (Python 3.12, torch CPU, torch-geometric, transformers).

```
d:\TCC\.venv\Scripts\python.exe codigo\02_modelo_e_calibracao.py
```

---

## Pipeline

| # | Script | Entrada | Saída | Seção do documento |
|---|---|---|---|---|
| 01 | `base_humana.py` | `Base A.xlsx`, `Base B.xlsx` | `base_humana.csv` (1.167) | 1 |
| 02 | `modelo_e_calibracao.py` | `Base A.xlsx`, `Base B.xlsx` | `modelo_1/`, `results/calibracao_corte.json` | 2 e 3 |
| 03 | `rotulagem_corpus.py` | corpus bruto, `modelo_1/` | `aceites_automaticos.csv` | 3 |
| 04 | `auditoria_amostra.py` | pseudo-rótulos | `auditoria_100.csv` + gabarito | 3 |
| 05 | `base_unificada.py` | humanas + pseudo | `base_unificada.csv` (2.602) | — |
| 06 | `embeddings_circularidade.py` | `base_unificada.csv`, `modelo_1/` | embeddings, homofilia por origem | 4 |
| 07 | `grafo_humanos.py` | `base_humana.csv` | `results/grafo_humanos.json` | 5 |
| 08 | `ablacao_construcao_grafo.py` | `base_humana.csv` | `results/ablacao_grafo.json` | 5 |
| 09 | `gnn_nove_configuracoes.py` | `base_humana.csv` + embeddings | `results/gnn_ablacao.json` | 6 |
| 10 | `gnn_experimento_cruzado.py` | idem | `results/gnn_rodada2.json` | 6 |
| 11 | `bases_de_escala.py` | `base_humana.csv`, corpus bruto | `base_grafo_15k.csv`, `base_grafo_57k.csv` | 7 |
| 12 | `gnn_escala_15k.py` | `base_grafo_15k.csv` | `results/gnn_15k.json` | 7 |
| 13 | `gnn_escala_57k.py` | `base_grafo_57k.csv` | `results/gnn_57k.json` | 7 |
| 14 | `figura_resultados.py` | os JSON de resultados | `fig_resultados.png` | — |
| 15 | `figura_grafo.py` | `base_humana.csv` + embeddings | `fig_grafo_grande.png` | — |

---

## Decisões de projeto embutidas no código

**`freeze_layers = 4`** (script 02). Congela as 8 camadas inferiores do BERTimbau
e treina apenas as 4 superiores, o pooler e o classificador — 26,6% dos
parâmetros. Segue o pipeline de referência da orientação.

**Mean pooling, não `[CLS]`** (scripts 06, 07, 12, 13). O vetor `[CLS]` de um BERT
não ajustado não é treinado para representar a frase. A média dos tokens,
mascarada pelo `attention_mask`, mede melhor similaridade.

**Embeddings do modelo ajustado** (scripts 06 em diante). Grafos construídos sobre
o BERTimbau cru têm homofilia 0,6397; sobre o ajustado, 0,7436. Medido nos 206
textos nunca vistos pelo modelo, para descartar memorização.

**`to_undirected` obrigatório** (todos os scripts de grafo). O kNN produz arestas
dirigidas; sem simetrizar, metade da vizinhança fica invisível à passagem de
mensagem.

**Avaliação restrita a nós humanos** (scripts 09 em diante). Após detectar
homofilia 1,0000 entre nós pseudo-rotulados — circularidade, pois rótulos e
embeddings vêm do mesmo modelo — toda métrica passou a usar apenas os 206 textos
da Base B.

**MLP de controle** (scripts 09, 10, 12, 13). Rede que recebe os mesmos vetores
sem acesso às arestas. É ele que separa o mérito do grafo do mérito do embedding.

**10 sementes e intervalo pareado** (scripts 09 em diante). Cada resultado é média
de 10 execuções; a comparação com o MLP usa diferença pareada por semente e
intervalo de confiança de 95%.

---

## Reprodutibilidade

Os embeddings ficam em cache (`.npy`) e são reaproveitados entre execuções. Para
regerar do zero, apague os arquivos correspondentes.

Os scripts 12 e 13 levam, respectivamente, cerca de 25 minutos e 2h40 em CPU — a
maior parte em gerar embeddings. O script 13 grava checkpoints parciais e retoma
de onde parou se interrompido.

Sementes fixas em 42 para divisão de dados e `[42, 7, 2024, 1, 13, 99, 256, 777,
31337, 5]` para as execuções dos modelos.

---

## Notebooks anteriores

`Notebook_TCC.ipynb`, `LLM_SemiRotulado.ipynb` e `Limpeza Dados.ipynb` são da
primeira etapa do trabalho e foram mantidos como registro histórico.

`Iteracao_1_GNN.ipynb`, `Semana1_Kappa_Consolidacao.ipynb` e
`Semana1_Treino_BERTimbau.ipynb` foram escritos antes da reconstrução da base e
**não reproduzem os resultados atuais** — referenciam arquivos que já não existem
e usam decisões posteriormente substituídas (pooling `[CLS]`, embeddings do modelo
cru). Não devem ser usados como referência do estado do trabalho.
