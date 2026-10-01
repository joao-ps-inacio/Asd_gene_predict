# Reprodução da tese: embeddings ProtT5

Objetivo: correr a pipeline nova sobre **os mesmos embeddings ProtT5 e os mesmos labels da tese** e verificar (1) se reproduz os números publicados e (2) quanto mudam quando as métricas são calculadas corretamente.

![ProtT5, tese vs corrigido](figures/prott5_tese_vs_corrigido.png)

## Resultado principal (teste nos genes `cat_1`, treino com `cat_1`)

<!-- TABELA_CAT1 -->
| Modelo | AUC tese | AUC (cálculo da tese, pipeline nova) | **AUC corrigido** | AUPRC tese | **AUPRC corrigido** |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.829 | 0.830 | **0.910** ± 0.016 | 0.533 | **0.765** ± 0.058 |
| SVM | 0.815 | 0.824 | **0.904** ± 0.019 | 0.564 | **0.760** ± 0.056 |
| KNN | 0.791 | 0.800 | **0.901** ± 0.036 | 0.537 | **0.744** ± 0.075 |
| Naive Bayes | 0.805 | 0.815 | **0.840** ± 0.030 | 0.477 | **0.523** ± 0.047 |
<!-- /TABELA_CAT1 -->

- **A pipeline nova reproduz a tese.** Com o AUC calculado como na tese (sobre previsões 0/1), a Logistic Regression dá 0,830; a tese publicou 0,829.
- **A tese subestimava o desempenho.** Calculado sobre probabilidades, o AUC sobe para ~0,90–0,91 e o AUPRC de ~0,53 para ~0,76. A referência aleatória do AUPRC é a proporção de positivos no teste: 0,23.
- **A diferença entre modelos é pequena.** LR, SVM e KNN ficam dentro de um desvio-padrão uns dos outros. O sinal está nos embeddings, não no classificador. O Naive Bayes é claramente pior.

## Acrescentar positivos das categorias 2 e 3 ao treino não ajuda

AUPRC no teste `cat_1` consoante os positivos usados no treino:

<!-- TABELA_SETS -->
| Positivos no treino | Logistic Regression | SVM | KNN | Naive Bayes |
|---|---:|---:|---:|---:|
| `cat_1` | 0.765 | 0.760 | 0.744 | 0.523 |
| `cat_1_sd` | 0.763 | 0.772 | 0.757 | 0.521 |
| `cat_1_2` | 0.735 | 0.743 | 0.714 | 0.449 |
| `cat_1_2_sd` | 0.743 | 0.748 | 0.700 | 0.455 |
| `cat_1_2_3` | 0.731 | 0.740 | 0.711 | 0.441 |
| `complete` | 0.737 | 0.743 | 0.725 | 0.451 |
<!-- /TABELA_SETS -->

Juntar os genes sindrómicos sem score (`_sd`) é neutro. Juntar as categorias 2 e 3 piora ligeiramente, o que é consistente com serem genes com evidência mais fraca.

## Como reproduzir

```bash
asd fetch --stage labels && asd fetch legacy_prott5
asd labels
asd embed protein --legacy
asd train --features protein_prott5 --model lr --model svm --model knn --model nb
asd train --features protein_prott5 --model rf --model lgbm --model xgb --set cat_1
python scripts/report_reproduction.py
```

## Diferenças em relação à tese

A comparação da coluna "cálculo da tese" com os números publicados não é exata, porque mudaram também:
- a métrica de afinação dos hiperparâmetros (AUPRC em vez de F1);
- as grelhas de hiperparâmetros, que ficaram mais pequenas;
- a divisão dos folds, que passa a ser aleatória com seed fixa em vez de seguir a ordem do ficheiro.

Mesmo assim, as diferenças são de ~0,01 e mostram que a mudança nos resultados vem da correção das métricas.

## Limitações e próximos passos

- Random Forest, LightGBM e XGBoost ainda não correram: com as grelhas atuais demoram horas em 2 CPUs. Na tese, nenhum deles superou a LR no `cat_1`. O segundo comando de "Como reproduzir" corre-os numa máquina com mais CPUs; depois basta voltar a correr o script do relatório.
- Com 5 folds, o desvio-padrão do AUPRC é ~0,06. Para comparar modelos com confiança, é preciso CV repetida (`asd train --repeats 5`).
- Os genes do SFARI são mais estudados do que a média. Falta um baseline que use apenas o grau do gene na rede PPI, para confirmar que os embeddings captam mais do que esse viés.
- Falta o mesmo exercício para os embeddings de DNA (DNABERT-2) e de grafo (GRAPE).
