# Código

Scripts de preparação dos dados e de geração das tabelas e figuras do artigo,
numerados na ordem de execução.

Ambiente: Python 3.12, pandas, numpy, matplotlib, PyTorch, Hugging Face
Transformers e PyTorch Geometric. O treinamento dos modelos foi executado no
Google Colab, com GPU NVIDIA T4.

Os scripts usam um diretório-base fixo (`BASE_DIR` / `BASE`, definido no início
de cada arquivo). Ajuste essa variável para o local dos dados antes de executar.

## Scripts

| # | Script | Entrada | Saída |
|---|---|---|---|
| 21 | `extrair_corpus_08jan.py` | corpus bruto (CIBERDEM) | `corpus_08jan.csv` (textos originais), `results/extracao_log.txt` |
| 01 | `base_humana.py` | planilhas de anotação | `base_humana.csv` (1.167 textos) |
| 23 | `kappa_base_utilizada.py` | planilha de anotação, `results/kappa_metricas.json` | conferência da concordância entre anotadores |
| 25 | `tabelas_monografia.py` | resultados do grafo (JSON) | `results/tabelas_monografia.json` |
| 22 | `figura_grafo.py` | resultados do grafo (JSON) | `fig_grafo_resultados.png` |
| 26 | `figura_monografia.py` | `results/tabelas_monografia.json` | `fig_monografia_grafo.png` |
| 27 | `figura_artigo_sementes.py` | resultados do grafo (JSON) | `fig_artigo_sementes.png` |
| 28 | `figura_iteracoes.py` | resultados das iterações (JSON) | `fig_artigo_iteracoes.png` |

Os hiperparâmetros dos modelos estão descritos no [README principal](../README.md#configuração-dos-experimentos).

## Sementes

Divisão dos dados com semente 42; execuções dos modelos com as sementes
42, 7, 2024, 1, 13, 99, 256, 777, 31337 e 5.

## `etapa_anterior/`

Scripts de uma fase exploratória anterior do trabalho (rotulagem com corte
único, grafos apenas com os textos de rótulo humano). Foram mantidos como
registro do desenvolvimento e **não** reproduzem os resultados apresentados
no artigo.
