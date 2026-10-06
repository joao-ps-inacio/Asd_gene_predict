# Which embedding works best? Protein vs graph

The thesis compared protein (ProtT5), DNA (DNABERT-2) and graph (DeepWalk on the STRING network) embeddings, but with the AUC computed on 0/1 predictions and without a paired test. This report redoes the protein vs graph comparison under strict conditions:

- **same genes**: the 1,915 genes with both embeddings (1,007 in the `cat_1` test set, 228 positives);
- **same folds**: 5-fold cross-validation repeated 5 times, identical splits for every source;
- **same models and tuning**: Logistic Regression, SVM and KNN, hyperparameters tuned inside each training fold;
- **paired test**: differences tested fold by fold with the corrected resampled t-test (Nadeau & Bengio, 2003), which accounts for the overlap between training sets in repeated cross-validation.

![Protein vs graph embeddings](figures/embedding_comparison.png)

## On known SFARI genes, graph embeddings win clearly

Cross-validation on the `cat_1` genes (mean ± sd over 25 folds):

<!-- CV -->
| Model | Features | ROC-AUC | AUPRC |
|---|---|---:|---:|
| Logistic Regression | Protein (ProtT5) | 0.906 ± 0.025 | 0.770 ± 0.056 |
| Logistic Regression | Graph (DeepWalk) | 0.962 ± 0.013 | 0.915 ± 0.025 |
| Logistic Regression | Protein + graph | 0.959 ± 0.013 | 0.908 ± 0.029 |
| SVM | Protein (ProtT5) | 0.896 ± 0.034 | 0.750 ± 0.066 |
| SVM | Graph (DeepWalk) | 0.960 ± 0.014 | 0.913 ± 0.027 |
| SVM | Protein + graph | 0.957 ± 0.012 | 0.901 ± 0.026 |
| KNN | Protein (ProtT5) | 0.896 ± 0.027 | 0.750 ± 0.074 |
| KNN | Graph (DeepWalk) | 0.955 ± 0.018 | 0.906 ± 0.034 |
| KNN | Protein + graph | 0.924 ± 0.024 | 0.822 ± 0.048 |
<!-- /CV -->

Paired differences:

<!-- TESTS -->
| Model | Comparison | Metric | Difference | 95% CI | p |
|---|---|---|---:|---|---:|
| KNN | Graph (DeepWalk) − Protein (ProtT5) | ROC-AUC | +0.059 | +0.035 to +0.083 | < 0.001 |
| KNN | Graph (DeepWalk) − Protein (ProtT5) | AUPRC | +0.157 | +0.087 to +0.226 | < 0.001 |
| Logistic Regression | Graph (DeepWalk) − Protein (ProtT5) | ROC-AUC | +0.056 | +0.032 to +0.079 | < 0.001 |
| Logistic Regression | Graph (DeepWalk) − Protein (ProtT5) | AUPRC | +0.145 | +0.093 to +0.198 | < 0.001 |
| SVM | Graph (DeepWalk) − Protein (ProtT5) | ROC-AUC | +0.064 | +0.032 to +0.096 | < 0.001 |
| SVM | Graph (DeepWalk) − Protein (ProtT5) | AUPRC | +0.162 | +0.096 to +0.229 | < 0.001 |
| KNN | Protein + graph − Graph (DeepWalk) | ROC-AUC | -0.031 | -0.050 to -0.011 | 0.004 |
| KNN | Protein + graph − Graph (DeepWalk) | AUPRC | -0.084 | -0.130 to -0.038 | < 0.001 |
| Logistic Regression | Protein + graph − Graph (DeepWalk) | ROC-AUC | -0.002 | -0.014 to +0.009 | 0.690 |
| Logistic Regression | Protein + graph − Graph (DeepWalk) | AUPRC | -0.008 | -0.030 to +0.014 | 0.472 |
| SVM | Protein + graph − Graph (DeepWalk) | ROC-AUC | -0.003 | -0.013 to +0.007 | 0.580 |
| SVM | Protein + graph − Graph (DeepWalk) | AUPRC | -0.012 | -0.037 to +0.013 | 0.341 |
<!-- /TESTS -->

- Graph embeddings beat protein embeddings with every model: about +0.06 ROC-AUC and +0.15 AUPRC, p < 0.001 after correction.
- Concatenating both sources adds nothing over graph alone with Logistic Regression or SVM, and makes KNN worse (its distances are dominated by the 1,024 protein dimensions).
- The thesis reached the same ranking (graph above protein) with its original metric.

The advantage also holds when the model is trained on category 1 genes only and tested on SFARI category 2 and 3 genes, which it never saw: ROC-AUC 0.86 for graph against 0.71 for protein (Logistic Regression, 5 × 5 splits of the negatives).

## On genes discovered later, the picture reverses

The [temporal validation](temporal_validation.md) asks a different question: did the rankings built in 2024 anticipate the genes SFARI added by 2026?

| | Protein (ProtT5) | Graph (DeepWalk) |
|---|---:|---:|
| ROC-AUC on genes added after 2024 | **0.84** | 0.80 |
| New genes in the top 10% | **55%** | 49% |
| Gain over a protein-length baseline | **+0.05** (95% CI 0.01 to 0.10) | +0.01 (95% CI −0.05 to 0.07) |

Here protein embeddings do at least as well as graph embeddings, and only protein beats the gene-length baseline. The protein − graph difference itself is +0.04 (95% CI −0.01 to 0.10), so it is not conclusive on its own.

## Interpretation

Graph embeddings describe where a gene sits in the known interaction network. Genes already in SFARI by 2024, whatever their category, are well studied and densely connected to other autism genes in STRING, which also uses co-mentions in the literature. For those genes the graph signal is strong. Genes discovered after 2024 have fewer recorded interactions, so their graph position says less about them, while their protein sequence is as informative as for any other gene.

The practical reading:

- to **recover genes the field already knows**, graph embeddings are the better feature;
- to **find genes the field does not know yet**, which is the point of a ranking, protein embeddings generalise better, and averaging the two rankings was slightly better than either (temporal AUC 0.855).

## Caveats and next checks

- The temporal comparison uses the two rankings produced in the thesis, which used different classifiers (XGBoost for protein, Logistic Regression for graph). A same-classifier temporal test needs embeddings for every gene, not only the training genes.
- The explanation above is a hypothesis. The direct test is a network-degree baseline (number of STRING interactions per gene): if graph scores track degree, the graph advantage is largely study bias. It needs the STRING file, which `asd fetch` downloads on a machine with open network access.
- The graph embeddings are the 500-dimensional DeepWalk vectors from the thesis. Other graph methods and a confidence filter on STRING edges are not yet tested.

## How to reproduce

```bash
asd fetch --stage labels && asd fetch legacy_prott5 legacy_graph_deepwalk
asd labels
asd embed protein --legacy && asd embed graph --legacy
asd compare --features protein_prott5 --features graph_deepwalk --model lr --repeats 5 --cross-category
asd compare --features protein_prott5 --features graph_deepwalk --model svm --repeats 5
asd compare --features protein_prott5 --features graph_deepwalk --model knn --repeats 5
python scripts/report_comparison.py   # needs asd validate-temporal for the right-hand panel
```
