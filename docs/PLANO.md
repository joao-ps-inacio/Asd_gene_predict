# Plano de melhoria do repo

Objetivo: transformar o código da tese num projeto **reprodutível, testado e extensível**, que sirva de base a uma app. O plano está dividido em fases. Cada fase fecha com um ou mais PRs e deixa `main` funcional.

```
 Fase 0        Fase 1        Fase 2          Fase 3          Fase 4         Fase 5         Fase 6
Fundações ─► Dados/labels ─► Embeddings ─► Modelação ─► Validação ─► Orquestração ─► App + publicação
 (feito)                     (GPU)          (núcleo)      científica    Snakemake/CI
```

---

## Fase 0 — Fundações ✅ (PR #1)

- Estrutura `src/`, `pyproject.toml`, CLI `asd`, config YAML, teste de fumo
- **CI no GitHub Actions**: lint (ruff) + testes em cada PR
- `CLAUDE.md` com o contexto e os problemas do código antigo

**Por fazer:** `pre-commit` com ruff, para o lint correr antes de cada commit local.

---

## Fase 1 — Dados e labels

**Objetivo:** um único comando (`asd labels`) que, a partir de ficheiros brutos com versão fixa, produz uma tabela `labels.parquet` com o esquema `ensembl_gene_id | symbol | label | sfari_score | syndromic | source`.

| Tarefa | Detalhe |
|---|---|
| Registo de fontes | `configs/sources.yaml` com URL, versão e checksum de cada ficheiro. Download com `asd fetch` e verificação do SHA-256 |
| SFARI | Ler o export, normalizar colunas, gerar os conjuntos `cat_1`, `cat_1_2`, `cat_1_2_3` e as variantes `_sd` |
| Krishnan et al. 2016 | Negativos: converter IDs NCBI→Ensembl com uma versão fixa do Ensembl (p. ex. 112) e remover sobreposições com o SFARI |
| Mapeamento de IDs | Uma só tabela `gene_map.parquet` (Ensembl gene ↔ símbolo HGNC ↔ Entrez ↔ proteína STRING ↔ transcrito canónico), usada por todas as fases |
| Exclusões | Os 3 genes excluídos à mão passam a ficar em `configs/exclusions.yaml`, com o motivo |
| Validação de esquema | `pandera` (ou asserts simples): sem duplicados, labels ∈ {0,1}, nenhum gene positivo e negativo ao mesmo tempo |
| Testes | Fixtures pequenas (~20 genes) em `tests/fixtures/` que exercitam todo o fluxo sem internet |

**Nota:** o SFARI tem termos de utilização. Confirmar se os ficheiros derivados podem ser redistribuídos; se não puderem, a app publica só os scores.

---

## Fase 2 — Embeddings

**Objetivo:** `asd embed {protein,dna,graph}` → `data/processed/emb_<fonte>_<modelo>.parquet` (uma linha por gene, colunas `f0..fN`) com metadados: modelo, versão, pooling e data.

### 2a. Proteína — ProtT5 (primeira a portar)
- Sequência do transcrito **canónico do Ensembl** (a tese já fazia isto). Considerar o **MANE Select**, mais padronizado.
- Mean pooling, processamento em batches, `torch.no_grad()`, fp16 na GPU
- Proteínas longas: dividir em janelas e fazer a média, em vez de as descartar

### 2b. DNA — DNABERT-2
- Hoje as sequências acima de 10 kb são descartadas em silêncio. Passam a ser divididas em janelas com a média dos embeddings, e fica registado o número de janelas por gene.
- Log de todos os genes que falham, com o motivo

### 2c. Grafo — GRAPE sobre STRING
- **Filtrar arestas por confiança** (`combined_score ≥ 700`, "high confidence"). A tese usou todas as arestas, incluindo as de baixa confiança.
- Reduzir de ~25 para os 5 métodos com melhor desempenho na tese
- Fixar a seed (os embeddings de random walk variam entre execuções)

### Onde correr
- Sem GPU local: Kaggle/Colab (GPU grátis) ou Azure ML. Scripts em `scripts/` que correm `asd embed` e fazem upload do Parquet.
- Os embeddings são o passo mais caro. Gerar uma vez, guardar com versão (DVC, ou um bucket com hash) e nunca voltar a gerar sem necessidade.

---

## Fase 3 — Modelação (núcleo)

**Objetivo:** uma só implementação (`asd train`) que substitui as 9 cópias de `fold_model_compare.py`.

### Correções obrigatórias
1. **Métricas sobre probabilidades** (`predict_proba` / `decision_function`) para o AUC-ROC e o AUPRC. Isto muda os números da tese.
2. `scale_pos_weight` no XGBoost em vez de `class_weight`
3. Seeds fixas em todo o lado
4. Grelhas de hiperparâmetros limpas: `degree` só para o kernel `poly`; `RandomizedSearchCV` ou Optuna em vez de grelhas enormes

### Melhorias metodológicas
- **Métrica principal: AUPRC.** As classes estão desequilibradas e o que interessa é o topo do ranking. Reportar também AUC-ROC, MCC e precision@k.
- **CV repetida** (5 folds × 5 repetições) com **intervalos de confiança**, para saber se as diferenças entre modelos são reais
- **Baselines** para comparar:
  - modelo com apenas o **grau do gene no grafo PPI**. Os genes do SFARI são muito estudados e têm mais interações descritas (viés de estudo), por isso é preciso mostrar que os embeddings ganham a este baseline;
  - modelo com **métricas de restrição genética** (pLI / LOEUF do gnomAD), um preditor forte e conhecido de genes de doença
- **Fusão de fontes**: concatenar os embeddings (early fusion) vs. combinar as probabilidades de modelos separados (late fusion / stacking)
- **Calibração** das probabilidades (Platt/isotónica), necessária para a app mostrar um "score" interpretável
- **Pipeline sklearn** (`StandardScaler` → modelo) dentro da CV, para não haver fuga de dados no scaling

### Registo de experiências
- **MLflow** (já o usas no Databricks): parâmetros, métricas por fold e artefactos. Uma experiência por combinação fonte × dataset × modelo.
- Resultados finais em `reports/results.parquet` + tabela markdown gerada automaticamente

---

## Fase 4 — Validação científica

| Tarefa | Porquê |
|---|---|
| Ranking de todos os genes (`asd rank`) | É o produto final e a base da app |
| **Validação temporal** | Treinar com o SFARI de 01/2024 e testar nos genes **adicionados ao SFARI depois dessa data**. É a validação mais convincente: o modelo previu genes que só mais tarde foram associados ao autismo |
| Enriquecimento por decis | Portar a análise g:Profiler e `.gmt` (VariCarta, MSSNG, Krishnan/Duda) para uma função reprodutível com testes estatísticos e correção FDR |
| Rede do decil de topo | Comunidades + enriquecimento por comunidade, com figuras geradas por código |
| Interpretabilidade | SHAP (modelos de árvores) ou análise de vizinhos no espaço de embeddings: "este gene está perto de X e Y do SFARI" |

---

## Fase 5 — Orquestração da pipeline

**Objetivo:** um único comando que corre tudo e só volta a correr o que mudou.

**Recomendação: Snakemake**, a norma em bioinformática:
- cada regra chama um comando da CLI (`asd labels`, `asd embed protein`, …)
- só repete os passos cujos inputs mudaram
- corre localmente, em cluster SLURM (como o do INSA) ou na cloud sem mudar o código

```
fetch ─► labels ─┬─► embed_protein ─┐
                 ├─► embed_dna ─────┼─► train ─► rank ─► enrichment ─► report
fetch ─► string ─┴─► embed_graph ───┘
```

Alternativa mais leve: **DVC** (`dvc.yaml` + versionamento dos dados num bucket). Útil se quisermos também versionar os embeddings grandes.

**CI (GitHub Actions)**:
- em cada PR: lint + testes unitários + **pipeline completa sobre as fixtures pequenas** (teste end-to-end em ~1 min)
- em cada release: build do pacote e publicação dos scores como artefacto

---

## Fase 6 — App e publicação

### App
- **Backend FastAPI**: `GET /genes/{symbol}` devolve o score, o percentil, o label SFARI (se existir), os vizinhos mais próximos e as contribuições por fonte
- Os scores são **pré-calculados** (tabela Parquet/SQLite), não é preciso GPU em produção
- **Frontend**: começar com Streamlit (rápido); mais tarde, um frontend próprio
- Visualizações: posição do gene na distribuição de scores, subrede PPI do gene, comparação entre fontes
- Deploy: container em Azure Container Apps / Hugging Face Spaces

### Publicação
- Releases com DOI no **Zenodo** (scores + modelo)
- **Model card**: dados, limitações (viés de estudo, rótulos incompletos do SFARI) e uso pretendido. A app é uma ferramenta de investigação, não de diagnóstico.
- Considerar um preprint com os resultados corrigidos e a validação temporal

---

## Ordem de trabalho proposta (PRs)

| # | PR | Depende de |
|---|---|---|
| 1 | Estrutura base + CI | — |
| 2 | `asd fetch` + registo de fontes + `gene_map` | 1 |
| 3 | `asd labels` + testes com fixtures | 2 |
| 4 | Módulo de treino/avaliação com métricas corrigidas (testado com embeddings aleatórios) | 3 |
| 5 | `asd embed protein` (ProtT5) + script para GPU | 2 |
| 6 | Reprodução dos resultados da tese com ProtT5 e comparação das métricas antigas vs. corrigidas | 4, 5 |
| 7 | Baselines (grau PPI, pLI/LOEUF) | 4 |
| 8 | Embeddings DNA e grafo | 5 |
| 9 | Fusão de fontes + calibração | 6, 8 |
| 10 | `asd rank` + validação temporal | 9 |
| 11 | Enriquecimento + rede | 10 |
| 12 | Snakemake + teste end-to-end na CI | 3–11 |
| 13 | App (FastAPI + Streamlit) | 10 |

O PR 4 pode avançar em paralelo com o 5: o módulo de treino é testado com embeddings sintéticos e não precisa de GPU.
