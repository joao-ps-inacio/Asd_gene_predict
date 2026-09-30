# CLAUDE.md — contexto do projeto

Este ficheiro é o ponto de partida para qualquer sessão de trabalho neste repo. Está sempre atualizado com o objetivo, as decisões tomadas e os próximos passos.

## Objetivo

Reescrever de forma limpa e reprodutível o código da tese de mestrado do João, *"Predicting Autism Risk Genes Using Graph and Sequence Embeddings"* (FCUL / INSA Ricardo Jorge, 2025). A longo prazo, o projeto pode vir a ter uma **app** para consultar o score de risco de cada gene.

- Repo original (só leitura, conta de estudante sem acesso): https://github.com/a59490/Tese_ASD_Gene_Pred
- Repo novo: https://github.com/joao-ps-inacio/Asd_gene_predict

## Como trabalhamos

- `main` está sempre estável. Cada tarefa vai num branch (`feat/...`, `fix/...`, `setup/...`) com um PR para o João rever.
- Descrição do PR curta: 2 a 4 linhas sobre o que muda e, se houver, uma checklist "Antes do merge". Nada de tabelas, detalhes de implementação ou perguntas; os detalhes ficam no código, nos commits e neste ficheiro.
- No fim de cada bloco de trabalho: fazer push e atualizar a secção **Estado atual** deste ficheiro.
- Documentação em português; código, nomes de funções e docstrings em inglês.
- Dados nunca vão para o Git. Tabelas intermédias guardadas em **Parquet**, não em CSV com vetores em texto.
- Caminhos sempre via `asd_gene_predict.paths`, nunca relativos ao diretório atual.
- Parâmetros em `configs/*.yaml`, não hardcoded nos scripts.

## Mapa do repo antigo → repo novo

| Antigo | Novo |
|---|---|
| `01_Data_preparation/` (notebooks SFARI, Krishnan, NCBI→Ensembl) | `src/asd_gene_predict/data/` + comando `asd labels` |
| `02_Embedding_creation/sequence/` (`dna_embeddings.py`, `prot_emb.py`) | `src/asd_gene_predict/embeddings/` + `asd embed dna\|protein` |
| `03_ML/graph/Create_embs.py` (GRAPE, ~25 métodos) | `src/asd_gene_predict/embeddings/` + `asd embed graph` |
| `03_ML/*/gene_lists/dataset_creator*.py` | `src/asd_gene_predict/data/` |
| `03_ML/*/fold_model_compare.py` (3 cópias quase iguais) | `src/asd_gene_predict/models/` (uma só implementação) + `asd train` |
| `03_ML/sequence_vs_graph/` (combinação de embeddings) | `models/`, com a fonte de features configurável |
| `04_Validation/01_Ranked_list/Gene_list_pred.py` | `src/asd_gene_predict/validation/` + `asd rank` |
| `04_Validation/02_Enrichment_analysis/` (g:Profiler, `.gmt`) | `validation/` |
| `04_Validation/03_Network/` | `validation/` |
| `05_Results/` (CSVs e notebooks de comparação) | `reports/` + `notebooks/` |

## Decisões de desenho herdadas da tese (manter)

- **Positivos**: SFARI Gene, testadas várias combinações (`cat_1`, `cat_1_2`, `cat_1_2_3`, com ou sem genes sindrómicos `_sd`).
- **Negativos**: lista de Krishnan et al. (2016), a mesma para todas as combinações.
- **Avaliação**: CV estratificada de 5 folds definida sobre `cat_1`. O teste é sempre feito nos genes `cat_1` do fold e os datasets maiores só acrescentam genes ao treino. Assim, todas as combinações são comparadas no mesmo conjunto de teste.
- **Afinação de hiperparâmetros**: `GridSearchCV` dentro de cada fold de treino (CV aninhada, sem fuga de dados para o teste).

## Problemas encontrados no código antigo (corrigir na reescrita)

1. **AUC e Average Precision calculados sobre previsões binárias** (`roc_auc_score(y_true, y_pred)`) em vez de probabilidades. Os valores de AUC da tese estão provavelmente subestimados. Corrigir com `predict_proba` / `decision_function`.
2. `XGBClassifier(class_weight="balanced")` não é um parâmetro válido do XGBoost e era ignorado. Usar `scale_pos_weight`.
3. Embeddings guardados como strings dentro de CSV e reconvertidos com `str.replace`. É lento e frágil.
4. `Requirements.txt` vazio, por isso o ambiente não era reprodutível.
5. Dados e scripts duplicados em várias pastas, com caminhos relativos ao diretório de execução.
6. Três genes excluídos à mão sem explicação (`ENSG00000142599`, `ENSG00000135636`, `ENSG00000285508`). **Perguntar ao João porquê.**
7. Grelha do SVM com `sigmoid` repetido e `degree` testado em todos os kernels (só afeta o `poly`), o que multiplica o tempo de treino sem ganho.
8. DNABERT-2: sequências acima de 10 000 bp eram descartadas em silêncio, dentro de um `try/except` genérico. Registar quais genes ficam de fora.
9. Não havia seed fixa, pelo que os resultados não eram reprodutíveis.
10. Grafo STRING usado com todas as arestas, incluindo as de baixa confiança (sem filtro de `combined_score`).

## Roadmap

Plano detalhado, com fases e ordem dos PRs: [`docs/PLANO.md`](docs/PLANO.md).

- [x] Estrutura base do repo, `pyproject.toml`, config, CLI esqueleto, teste de fumo, CI (GitHub Actions)
- [x] Registo de fontes com checksums (`asd fetch`) e tabela de IDs (`asd gene-map`)
- [ ] **Etapa 1 — Labels**: download/leitura do SFARI e de Krishnan, conversão NCBI→Ensembl, geração dos conjuntos de positivos
- [ ] **Etapa 2 — Embeddings de proteína** (ProtT5), a primeira fonte a portar
- [ ] **Etapa 3 — Treino/avaliação** unificado, com métricas corrigidas e resultados em Parquet/CSV
- [ ] Etapa 2b — Embeddings de DNA (DNABERT-2) e de grafo (GRAPE, só os 5 métodos principais)
- [ ] Etapa 4 — Ranking de todos os genes
- [ ] Etapa 5 — Enriquecimento por decis e análise de rede
- [ ] Reproduzir os resultados principais da tese e comparar com as métricas corrigidas
- [ ] (Futuro) Atualizar para uma release recente do SFARI
- [ ] (Futuro) App para consultar o score de cada gene

## Estado atual

**2026-09-29**
- PR #1 (estrutura base, CI, plano): integrado em `main`.
- PR #2 (`feat/fetch-gene-map`): `configs/sources.yaml` + `asd fetch` (download com `.part`, SHA-256 gravado em `configs/sources.lock.yaml`) e `asd gene-map` → `data/processed/gene_map.parquet` (HGNC como base, MANE Select, STRING v12). Função `map_ids()` para converter IDs (vai servir o Krishnan NCBI→Ensembl no PR 3). Testado com fixtures offline.
- Ainda não foi feito um `asd fetch` real (a sessão cloud não chega a esses servidores). O João deve correr `asd fetch --stage gene_map && asd gene-map` localmente e fazer commit do `configs/sources.lock.yaml`.
- Próximo passo: PR 3 (`asd labels`: SFARI + Krishnan + `configs/exclusions.yaml`). Precisa dos ficheiros originais do SFARI (release 16/01/2024) e de Krishnan.

Para abrir PRs a partir da sessão: API do GitHub via `curl` (o proxy da sessão trata da autenticação), com o header `Content-Type: application/json`.

## Questões em aberto

- Porque foram excluídos os três genes acima?
- Onde estão os dados grandes originais (embeddings, FASTA, STRING)? Voltar a gerá-los ou há cópia?
- Há acesso a GPU para voltar a gerar os embeddings de ProtT5 e DNABERT-2?
- O João ainda tem o CSV do SFARI de 16/01/2024 e a tabela de negativos de Krishnan usados na tese? (necessários para o PR 3)
- A tese usou o transcrito canónico do Ensembl; o `gene_map` usa MANE Select. Na grande maioria dos genes codificantes coincidem, mas confirmar se há diferenças relevantes ao reproduzir os resultados.
