# Submission-package cleanup

The uploaded working repository was reduced to a review-safe, paper-aligned package.

Removed:

- `.git` history and hooks;
- `__pycache__`, `.pytest_cache`, and notebook checkpoint artefacts;
- notebooks containing personal Colab metadata and hard-coded Drive paths;
- duplicate copies of train/test CSV files;
- sentence-context and translation-ablation working data no longer reported as main experiments;
- duplicate or superseded raw model runs;
- temporary annotation workbooks and judge-input spreadsheets;
- cached encoder embedding matrices duplicated elsewhere;
- incomplete tokenizer/model folders without LoRA adapter weights;
- legacy EMNLP title, Setup A/B/C documentation, and stale table references.

Added:

- a single clean end-to-end reproduction notebook;
- deterministic orchestration and release-validation scripts;
- blind-reannotation and independent expert-adjudication evaluation code;
- anonymised CSV annotation sheets without workbook metadata;
- lightweight and model-specific requirement files;
- explicit data/licence and reproducibility documentation;
- checksums and a machine-verifiable reproduction status.

## Resource-quality audit update

The previous two-annotator files and their IAA-derived outputs were removed. They are replaced by one anonymised 400-item blind reannotation record followed by independent expert adjudication. The package now reproduces outcome counts, stratum-level results, and grouped adjudication-reason tables from `data/resource_quality/audit_records.csv`.
