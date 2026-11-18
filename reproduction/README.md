# Reproduction data

The files in this directory contain sample IDs, labels, token lengths, split membership, fix-pair metadata, and scores. Obtain function bodies and original datasets from their upstream releases.

| Dataset | Upstream | Partition used in the paper | Train / validation / test fix pairs |
| --- | --- | --- | --- |
| PrimeVul | [PrimeVul repository](https://github.com/DLVulDet/PrimeVul#-primevul-dataset) | Official original-release split | 4,354 / 562 / 564 |
| BigVul | [MSR 2020 dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) | Fixing-commit split with whitespace normalization and fixed versions added | 8,229 / 1,199 / 1,222 |
| ReVeal | [ReVeal repository](https://github.com/VulDetProject/ReVeal) | Fixing-commit split after removing duplicates and label conflicts | 1,369 / 197 / 195 |
| MegaVul | [MegaVul repository](https://github.com/Icyrockton/MegaVul) | April 2024 C/C++ release, fixing-commit split after whitespace normalization and deduplication | 12,768 / 1,500 / 1,712 |

## Dataset records

- Each `datasets/<dataset>/splits.parquet` records the exact `idx`, `target`, and `split` for every function used in the paper. For the three commit-grouped datasets, `commit_group_sha256` is the SHA-256 digest of the fixing-commit identifier when available.
- `input-digests.json` records the SHA-256 of each original PrimeVul split file or of each prepared split file for the other datasets. IDs refer to the authors' dataset records. PrimeVul uses upstream `idx`. BigVul uses the original CSV row identifier, prefixed with `f` for added fixed functions. ReVeal uses the original per-file row index, prefixed with `v` or `n` for the vulnerable and non-vulnerable source files. MegaVul IDs refer to the source record position and the retained version.
- Each `<dataset>_pairs.parquet` records `vuln_idx` and `fixed_idx`, the matching split, original token lengths, and the window annotations. `positive_windows` and `hard_negative_windows` contain zero-based window indices on the vulnerable and fixed sides, respectively. The PrimeVul table is the exact training input and is accompanied by its original manifest and coverage report.
- `datasets/bigvul/training-adjustments.parquet` marks which original training IDs are retained in each of the three reported BigVul training interventions. Each intervention uses the same original validation and test sets.
- `datasets/primevul/test-fix-positions.parquet` gives the first fix token offset and whether the entire fix lies in the prefix. Negative offsets denote unavailable positions.

## Frozen scores

Each `scores/<dataset>-<split>.parquet` contains one record per validation or test function. The shared columns are `idx`, `target`, `n_tokens`, `stratum`, and `log_length`. The other columns store one model output each.

- `codebert` is the PrimeVul baseline. `cfa` and `cfa_raw` are the length-standardised and actual-length scores from the same CFVul checkpoint. Columns beginning with `cfa_` and the dataset-specific baseline pairs are used in the cross-model and cross-dataset comparisons.
- PrimeVul's `has_cpg` marks the graph-available subset. Scores are present only on this subset for `taccs_dfa`, `vullmgnn`, and `cfa_vullmgnn`. The subset contains 23,948 validation functions and 24,788 test functions. Both functions have a graph in 433 test fix pairs. The TaCCS-DFA score column uses the three-epoch checkpoint reported in the paper.
- The ablation columns include `wo_a`, `wo_b`, `cfa_raw` (without inference length standardisation), `a_plus_b`, `only_a`, `only_b`, `no_aux`, `own_lengths`, `pair_weight005`, and `pair_weight05`.
- Scores retain their original scales. The paper's precision, recall, and F1 use the threshold maximising validation F1 for each output. The separate deployment thresholds in `protocol/primevul-cfvul-fpr-thresholds.json` are validation-negative quantiles on the length-standardised score.

## Code transformations

`transformations/cohort.csv` identifies the 133 analysed pairs, including 74 validation and 59 test pairs. The `pair` column in the Parquet files is the vulnerable PrimeVul ID. `selection-flags.parquet` records the exact-length and visibility criteria for each condition used to select the cohort. A candidate must pass the criteria for all five detectors.

`transformations/scores.parquet` contains the stored scores `s_v` and `s_f` for the two versions in each reported condition. `cond` is the original pair, padding of the vulnerable side, or both sides padded. `filler` identifies sampled real comments (`corpus`), generated comments (`llm`), unused variable declarations (`decl`), or hyphen-only comments (`dash`). `k` is the padding dose as a multiple of the fix-induced token gap. The recomputation script verifies the selected cohort and compares the bootstrap summaries with `reported-results.json`.

## Run settings

`protocol/primevul-checkpoints.json` records the selected runs' optimisation steps, validation selection scores, model revision, and CFVul's 32 reference log lengths. The repository provides training configurations for CodeBERT and CFVul on PrimeVul. Model weights, graph files, and transformed source text are outside this data bundle.
