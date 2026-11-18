# Reproduction data

This directory contains frozen validation and test scores for the four datasets, test fix pairs for ranking, and the full PrimeVul fix-pair table for training. Obtain function bodies and original datasets from their upstream releases.

| Dataset | Upstream | Partition used in the paper | Test fix pairs in this bundle |
| --- | --- | --- | --- |
| PrimeVul | [PrimeVul repository](https://github.com/DLVulDet/PrimeVul#-primevul-dataset) | Official original-release split | 564 |
| BigVul | [MSR 2020 dataset](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) | Fixing-commit split with whitespace normalization and fixed versions added | 1,222 |
| ReVeal | [ReVeal repository](https://github.com/VulDetProject/ReVeal) | Fixing-commit split after removing duplicates and label conflicts | 195 |
| MegaVul | [MegaVul repository](https://github.com/Icyrockton/MegaVul) | April 2024 C/C++ release, fixing-commit split after whitespace normalization and deduplication | 1,712 |

## Fix pairs and PrimeVul training inputs

- The BigVul, ReVeal, and MegaVul pair tables contain test pairs only. Each row records `split`, `vuln_idx`, `fixed_idx`, `n_tokens`, and `fixed_n_tokens` for joining test scores and measuring pairwise changes in length.
- IDs follow the dataset records. PrimeVul uses upstream `idx`. BigVul uses the original CSV row identifier, prefixed with `f` for added fixed functions. ReVeal uses the original per-file row index, prefixed with `v` or `n`. MegaVul IDs refer to the source record position and the retained version.
- `datasets/primevul/primevul_pairs.parquet` retains all 4,354 training, 562 validation, and 564 test pairs with their window annotations. `positive_windows` and `hard_negative_windows` contain zero-based window indices on the vulnerable and fixed sides, respectively. The original manifest records its settings and digest, and `coverage_by_stratum.csv` records the annotation coverage. `input-digests.json` records the SHA-256 of the original PrimeVul split files.
- `datasets/primevul/test-fix-positions.parquet` gives the first fix token offset and whether the entire fix lies in the prefix. Negative offsets denote unavailable positions.

## Frozen scores

Each `scores/<dataset>-<split>.parquet` contains one record per validation or test function. The shared columns are `idx`, `target`, `n_tokens`, `stratum`, and `log_length`. The other columns store one model output each.

- `codebert` is the PrimeVul baseline. `cfa` and `cfa_raw` are the length-standardised and actual-length scores from the same CFVul checkpoint. Columns beginning with `cfa_` and the dataset-specific baseline pairs are used in the cross-model and cross-dataset comparisons.
- PrimeVul's `has_cpg` marks the graph-available subset. Scores are present only on this subset for `taccs_dfa`, `vullmgnn`, and `cfa_vullmgnn`. The subset contains 23,948 validation functions and 24,788 test functions. Both functions have a graph in 433 test fix pairs. The TaCCS-DFA score column uses the three-epoch checkpoint reported in the paper.
- The ablation columns include `wo_a`, `wo_b`, `cfa_raw` (without inference length standardisation), `a_plus_b`, `only_a`, `only_b`, `no_aux`, `own_lengths`, `pair_weight005`, and `pair_weight05`.
- Scores retain their original scales. The paper's precision, recall, and F1 use the threshold maximising validation F1 for each output. The separate deployment thresholds in `protocol/primevul-cfvul-fpr-thresholds.json` are validation-negative quantiles on the length-standardised score.

## Code transformations

`transformations/cohort.csv` identifies the 133 analysed pairs, including 74 validation and 59 test pairs. The `pair` column in the Parquet files is the vulnerable PrimeVul ID. `selection-flags.parquet` records the exact-length and visibility criteria for each condition used to select the cohort. A candidate must pass the criteria for all five detectors.

`transformations/scores.parquet` contains the stored scores `s_v` and `s_f` for the original, padded, and both-padded pairs reported in the paper. It includes the sampled-comment dose curve and four-filler controls for the four baseline detectors, plus CFVul's equal-length contrasts. `filler` identifies sampled real comments (`corpus`), generated comments (`llm`), unused variable declarations (`decl`), or hyphen-only comments (`dash`). `k` gives the padding dose as a multiple of the fix-induced token gap. The recomputation script verifies the cohort and checks the bootstrap summaries in `reported-results.json`, where `dose` is keyed by multiplier.

## Run settings

`protocol/primevul-checkpoints.json` records the selected runs' optimisation steps, validation selection scores, model revision, and CFVul's 32 reference log lengths. The repository provides training configurations for CodeBERT and CFVul on PrimeVul. Model weights, graph files, and transformed source text are outside this data bundle.
