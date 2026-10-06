# CLAUDE.md — contexto do projeto

Este ficheiro é o ponto de partida para qualquer sessão de trabalho neste repo. Está sempre atualizado com o objetivo, as decisões tomadas e os próximos passos.

## Objetivo

Reescrever de forma limpa e reprodutível o código da tese de mestrado do João, *"Predicting Autism Risk Genes Using Graph and Sequence Embeddings"* (FCUL / INSA Ricardo Jorge, 2025). A longo prazo, o projeto pode vir a ter uma **app** para consultar o score de risco de cada gene.

- Repo original (só leitura, conta de estudante sem acesso): https://github.com/a59490/Tese_ASD_Gene_Pred
- Repo novo: https://github.com/joao-ps-inacio/Asd_gene_predict

## Como trabalhamos

- `main` está sempre estável. Cada tarefa vai num branch (`feat/...`, `fix/...`, `setup/...`) com um PR para o João rever.
- Cada PR abre sempre contra `main`, nunca contra o branch de outro PR (PRs empilhados não chegam a `main` quando são integrados).
- Descrição do PR curta: 2 a 4 linhas sobre o que muda e, se houver, uma checklist "Antes do merge". Nada de tabelas, detalhes de implementação ou perguntas; os detalhes ficam no código, nos commits e neste ficheiro.
- No fim de cada bloco de trabalho: fazer push e atualizar a secção **Estado atual** deste ficheiro.
- Documentação em português; código, nomes de funções e docstrings em inglês. Exceção: tudo o que um visitante do repo lê é em inglês: `README.md`, notebook de demonstração, `reports/` e `docs/changes_from_thesis.md` (decisão do João, 2026-10-03). O `CLAUDE.md` e o `docs/PLANO.md` ficam em português.
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

## Problemas encontrados no código antigo

A lista completa (em inglês, pensada para quem visita o repo) está em [`docs/changes_from_thesis.md`](docs/changes_from_thesis.md). Ao corrigir um problema novo, acrescentá-lo lá com o estado (Fixed / Planned). Ainda por fazer: DNABERT-2 com janelas para sequências longas e filtro de `combined_score` no STRING.

## Roadmap

Plano detalhado, com fases e ordem dos PRs: [`docs/PLANO.md`](docs/PLANO.md).

- [x] Estrutura base do repo, `pyproject.toml`, config, CLI esqueleto, teste de fumo, CI (GitHub Actions)
- [x] Registo de fontes com checksums (`asd fetch`) e tabela de IDs (`asd gene-map`)
- [x] **Etapa 1 — Labels** (`asd labels`): iguais aos da tese nos 6 conjuntos
- [ ] **Etapa 2 — Embeddings de proteína** (ProtT5), a primeira fonte a portar
- [x] **Etapa 3 — Treino/avaliação** (`asd train`): uma só implementação, métricas sobre probabilidades (as da tese ficam como `*_legacy`), resultados em `reports/results/*.parquet`
- [x] Comparação justa proteína vs grafo com os embeddings da tese (`asd compare`)
- [ ] Etapa 2b — Embeddings de DNA (DNABERT-2) e de grafo (GRAPE, só os 5 métodos principais)
- [x] Validação temporal com o SFARI 2026 Q2 (`asd validate-temporal`, [`reports/temporal_validation.md`](reports/temporal_validation.md))
- [ ] Etapa 4 — Ranking de todos os genes
- [ ] Etapa 5 — Enriquecimento por decis e análise de rede
- [x] Reproduzir os resultados da tese com ProtT5 e comparar com as métricas corrigidas ([`reports/prott5_reproduction.md`](reports/prott5_reproduction.md))
- [ ] (Futuro) Atualizar para uma release recente do SFARI
- [ ] (Futuro) App para consultar o score de cada gene

## Estado atual

**2026-10-04**
- Tudo em `main` (PR #8 levou para `main` o trabalho dos PRs empilhados #4–#7). Não voltar a empilhar PRs: abrir sempre contra `main`.
- Reprodução: a pipeline reproduz a tese (AUC da LR à maneira da tese 0,830 vs. 0,829). Corrigido: AUC ~0,91, AUPRC ~0,76.
- Validação temporal: dos genes acrescentados ao SFARI entre jan/2024 e jul/2026, 55% estavam no top 10% do ranking de proteína da tese (AUC 0,84). O comprimento da proteína sozinho dá AUC 0,79; a proteína ganha-lhe (+0,05, IC95 [0,01; 0,10]). O grafo não ganha de forma significativa.
- O SFARI 2026 é fonte manual. Nesta sessão foi transcrito do ficheiro do projeto para `data/raw/sfari_2026q2_min.csv` (4 colunas), validado contra o SFARI de 2024 e as listas da tese.
- Página com os resultados (privada, partilhável pelo João): https://claude.ai/artifact/JLJaWvVfvq8tW3GuSctM5Y
- README em inglês pensado para recrutadores: resumo "What / How / Result / Why / Run", grelha de 4 figuras (geradas por `scripts/readme_figures.py`), tabela de engenharia. `notebooks/demo.ipynb` executado. Quando o João importar o repo da tese para a conta nova (`joao-ps-inacio/Tese_ASD_Gene_Pred`), trocar o link em "Related repositories".
- Comparação justa proteína vs grafo (`asd compare`, [`reports/embedding_comparison.md`](reports/embedding_comparison.md)): mesmos 1915 genes, mesmos folds 5×5, LR/SVM/KNN, teste t corrigido de Nadeau-Bengio. Nos genes conhecidos o grafo ganha (AUC 0,96 vs 0,91, p < 0,001); de cat. 1 para cat. 2/3 também (0,86 vs 0,71). Na validação temporal a proteína fica à frente (0,84 vs 0,80, rankings da tese com classificadores diferentes). Hipótese: o grafo capta o quão estudado o gene já está; testar com baseline de grau no STRING.
- Próximo: baseline de grau STRING e LOEUF (precisam do PC do João) (o gnomAD não é acessível a partir da sessão cloud), CV repetida, refazer o ranking com a pipeline nova.

Para abrir PRs a partir da sessão: API do GitHub via `curl` (o proxy da sessão trata da autenticação), com o header `Content-Type: application/json`.

## Questões em aberto

- Porque foram excluídos os três genes acima?
- Os embeddings de DNA da tese não estão no repo antigo (só os de proteína e alguns de grafo). Há cópia noutro sítio?
- Há acesso a GPU para gerar embeddings novos (ESM-2, DNABERT-2)? Para reproduzir o ProtT5 já não é preciso.
- A tese usou o transcrito canónico do Ensembl; o `gene_map` usa MANE Select. Na grande maioria dos genes codificantes coincidem, mas confirmar se há diferenças relevantes ao reproduzir os resultados.
