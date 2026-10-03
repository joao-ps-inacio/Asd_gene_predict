# Changes from the original thesis code

The original code is at [a59490/Tese_ASD_Gene_Pred](https://github.com/a59490/Tese_ASD_Gene_Pred). Rewriting it surfaced the issues below. Each one is either fixed in this repository or scheduled for the stage of the pipeline it affects.

## Issues that change the results

| # | Issue in the thesis code | Fix | Effect | Status |
|---|---|---|---|---|
| 1 | ROC-AUC and average precision computed on hard 0/1 predictions (`roc_auc_score(y_true, y_pred)`), which keeps one point of the ROC curve | Metrics computed on `predict_proba` / `decision_function`. The thesis-style values are still reported as `roc_auc_legacy` and `ap_legacy`. | ProtT5 + Logistic Regression: ROC-AUC 0.83 → 0.91, average precision 0.53 → 0.77 ([report](../reports/prott5_reproduction.md)) | Fixed |
| 2 | `XGBClassifier(class_weight="balanced")` is not an XGBoost parameter and was silently ignored, so XGBoost trained without class balancing | `scale_pos_weight` computed from each training fold | XGBoost results change | Fixed |
| 3 | No fixed random seeds | Seeds set for folds, models and searches | Runs are reproducible | Fixed |
| 4 | Scaler fitted on the whole training fold before the hyperparameter search, so it leaked into the inner validation folds | `StandardScaler` inside a scikit-learn `Pipeline`, refitted in every inner fold | Small; removes optimistic bias in model selection | Fixed |
| 5 | Test genes selected by gene symbol, while one symbol (PDE8B) maps to two Ensembl IDs, so both rows moved between folds together | Genes identified by Ensembl ID everywhere | Negligible; each gene is now in exactly one test fold | Fixed |
| 6 | DNABERT-2: sequences over 10,000 bp dropped silently inside a broad `try/except` | Long sequences split into windows and averaged; skipped genes logged | Fewer missing genes | Planned (DNA stage) |
| 7 | STRING graph used with every edge, including low-confidence ones | Filter on `combined_score` (e.g. ≥ 700, high confidence) | Less noisy graph embeddings | Planned (graph stage) |

## Reproducibility and engineering

| Issue in the thesis code | Change |
|---|---|
| Empty `Requirements.txt` | `pyproject.toml` with pinned minimum versions and optional extras |
| Scripts and data copied across folders, paths relative to the working directory | One package (`src/asd_gene_predict`) and one CLI (`asd`); paths defined in one module |
| Three near-identical copies of the training script | One implementation, with the feature source as a parameter |
| Embeddings stored as text inside CSV cells and parsed with `str.replace` | Parquet files with float32 columns and metadata |
| Inputs edited by hand, without versions | Sources declared in `configs/sources.yaml`, pinned to a commit or release, verified by SHA-256 |
| Folds followed file order (no shuffle) | Stratified folds shuffled with a fixed seed |
| SVM grid repeated the `sigmoid` kernel and tested `degree` for every kernel | Grid split by kernel; `degree` only for `poly` |
| No tests | Unit tests on small offline fixtures, lint and tests on every pull request |

## Data decisions made explicit

These were implicit in the thesis notebooks. They are kept, to reproduce the thesis, and now documented in code and configuration.

- The SFARI export used in the thesis had nine genes without an Ensembl ID (e.g. MET, AR, ADA). They were filled in by hand, and one gene with no ID (RPS10P2-AS1) was removed.
- 35 genes appear both in SFARI and in the Krishnan et al. negative list. They are treated as positives.
- Three genes were excluded by hand during training (RERE, a SFARI category 1 gene, and two negatives). The reason was not recorded. They are listed in `configs/exclusions.yaml` so the choice is visible and easy to revisit.
