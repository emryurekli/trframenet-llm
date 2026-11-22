# Sensitivity to accepted audit revisions

Supplement for **Turkish FrameNet 2.0: Synset-Level Frame Lexicon Expansion with Model Assistance**.

Repository: https://github.com/emryurekli/trframenet-llm

## Reproduce

Place this folder at `sensitivity/` in the repository root. The original benchmark data, parsed predictions, audit records, and evaluation functions already belong to that repository. Use Python 3.10 or later and run from the repository root:

```bash
python -m pip install -r sensitivity/requirements.txt
python sensitivity/code/run_label_sensitivity.py --repo . --crosswalk sensitivity/data/audit_400_crosswalk.csv --output-dir ../trframenet-sensitivity-rerun --n-boot 5000 --seed 42
```

A GitHub Download ZIP copy also works. The output directory must be outside the source repository. The runner reads archived predictions; it does not train a model or call an API.

## Scope

The audit contains 400 entries: 330 training and 70 test entries. Of the 101 expert-accepted reannotation decisions, 86 concern training and 15 concern test entries. Only the 15 accepted test decisions are applied in a separate scoring scenario. One accepted test label is already the frozen label, so 14 test labels change. Training labels, predictions, and other benchmark labels remain fixed.

The crosswalk retains the audit reference, original benchmark, and accepted labels as separate fields. Only accepted test decisions govern replacement. Accuracy uses exact-string equality. Bootstrap calculations use 5,000 resamples and seed 42; pairwise tests use the repository's exact McNemar implementation. Pairwise p-values are exploratory and unadjusted for multiple comparisons. The 70 audited test entries are diagnostically selected, not representative of the whole test distribution.

## Included files

| File | Purpose |
|---|---|
| `code/run_label_sensitivity.py` | Rescoring code using the released evaluation functions. |
| `requirements.txt` | Python dependencies. |
| `data/audit_400_crosswalk.csv` | Audit-to-benchmark IDs, labels, outcomes, and train/test alignment. |
| `data/audit_101_decisions.csv` | The 101 expert-accepted decisions promised in the manuscript. |
| `results/sensitivity_metrics.csv` | Metrics for 13 systems across three scopes. |
| `results/model_pairwise_tests.csv` | Paired comparisons for original and revised labels. |
| `results/run_manifest.json` | Actual run source, input hashes, parameters, and correction counts. |

The runner also generates item-level and gold-comparison diagnostic outputs when executed. Those optional output files are not included in this minimal distribution. The separate case-insensitive check mentioned in the manuscript is not rerun by this script.

## Verification and source

The results were reproduced on 22 November 2026 from a fresh Git checkout of the repository above at benchmark commit `aa6ad459dd3c706006baf8bdf881e54e06456108`. This baseline commit predates addition of the supplement. The archived numerical outputs matched the prior checked results byte-for-byte. The manifest records the actual origin and commit. The minimal package retains the same runner, data, and table outputs.

Verified environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and scikit-learn 1.8.0.

The existing 400-item workbook and protocol remain under `data/resource_quality/` in the main repository. Archived parsed predictions are sufficient for this analysis; complete raw API responses are not available for every system. Refer to the repository's `LICENSE` and `DATA_LICENSE.md` for applicable licences.
