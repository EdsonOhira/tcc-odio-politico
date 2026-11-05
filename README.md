# Detecção de Discurso de Ódio Político em Redes Sociais por meio de Redes Neurais em Grafos

Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação
Faculdade de Computação e Informática (FCI) — Universidade Presbiteriana Mackenzie — 2026

**Autores:** Carlos Eduardo Hayashi, Edson Luiz Cardoso Ohira e Vinicius Oliveira Piccazzio
**Orientador:** Prof. Ivan Carlos Alcantara de Oliveira

## Sobre o projeto

O trabalho investiga se a estrutura de vizinhança entre textos, representada como grafo, melhora a detecção de discurso de ódio político em português. O corpus reúne publicações do X (antigo Twitter) sobre a invasão das sedes dos Três Poderes em 8 de janeiro de 2023.

Etapas:

1. **Corpus:** 1.389.772 publicações; após deduplicação, 130.974 textos originais.
2. **Anotação:** amostra aleatória rotulada por três avaliadores (ódio político / não ódio), com rótulo final por maioria, seguindo o [manual de anotação](docs/Manual_Anotacao_Odio_Politico.pdf). Base final de 1.167 textos (811 treino, 150 validação, 206 teste congelado).
3. **Classificador de referência:** BERTimbau (`neuralmind/bert-base-portuguese-cased`) com as quatro camadas superiores ajustadas.
4. **Auto-rotulagem:** ciclo iterativo de pseudo-rotulagem com limiares calibrados na validação.
5. **Grafos:** GCN e GAT sobre grafos de 57.440 nós (semântico kNN e híbrido com conversa e autoria), comparadas a um MLP sem grafo com as mesmas representações, em 10 sementes.

Principal resultado: com cerca de 50 mil nós rotulados, a GAT sobre o grafo híbrido alcançou F1-macro de 0,8164, contra 0,8026 do modelo sem grafo, com resultado superior nas 10 sementes avaliadas.

## Estrutura do repositório

```
docs/        manual de anotação (PDF)
notebooks/   notebooks dos experimentos (Google Colab)
src/         scripts do pipeline, numerados na ordem de execução (ver src/README.md)
configs/     configurações e hiperparâmetros
data/        dados (não versionados — corpus confidencial)
```

## Configuração dos experimentos

| Componente | Configuração |
|---|---|
| BERTimbau | 4 camadas superiores ajustadas; textos de até 128 tokens; lotes de 16; taxa de aprendizado 2 × 10⁻⁵; decaimento de peso 0,01; aquecimento de 10%; AdamW; parada antecipada (paciência 2) pelo F1-macro na validação |
| Auto-rotulagem | até 6 rodadas, 2 sementes por rodada; grade de limiares {0,90; 0,95; 0,97; 0,99; 0,995}; precisão mínima de 90% na validação; seleção balanceada, até 40.000 textos por classe |
| Grafo semântico | kNN com k = 5 pelo cosseno da média dos vetores dos tokens |
| Arestas de interação | mesma conversa ou mesmo autor, até 3 vizinhos anteriores e 3 posteriores em ordem cronológica |
| GCN / GAT / MLP | 2 camadas de 128 unidades; dropout 0,5; Adam (taxa 0,01, decaimento 5 × 10⁻⁴); 200 épocas; pesos por classe; GAT com 4 cabeças |
| Sementes | 42, 7, 2024, 1, 13, 99, 256, 777, 31337 e 5 |

Ambiente: Python, PyTorch, Hugging Face Transformers e PyTorch Geometric, no Google Colab com GPU NVIDIA T4.

## Dados e ética

O corpus é confidencial e **não** está neste repositório. Nenhum texto real é reproduzido aqui ou no artigo; os exemplos do manual de anotação são sintéticos.

## Como citar

HAYASHI, Carlos Eduardo; OHIRA, Edson Luiz Cardoso; PICCAZZIO, Vinicius Oliveira. **Detecção de Discurso de Ódio Político em Redes Sociais por meio de Redes Neurais em Grafos**. Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação) — Faculdade de Computação e Informática, Universidade Presbiteriana Mackenzie, São Paulo, 2026.
