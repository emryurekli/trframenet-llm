# Verification report

The anonymous release package was tested from its repository root on 4 August 2026.

## Deterministic paper-result reproduction

Commands:

```bash
python scripts/reproduce_paper_results.py --quick
python scripts/evaluate_resource_quality.py \
  --audit-records data/resource_quality/audit_records.csv \
  --output-dir reproduced_results/resource_quality \
  --n-boot 5000
```

Status: **PASS**

Verified point estimates:

| Result | Reproduced value |
|---|---:|
| Unconstrained LoRA accuracy | 0.392708 |
| Candidate-conditioned LoRA accuracy | 0.367603 |
| mMPNET + LinearSVC accuracy | 0.334728 |
| Gemini 2.5 Flash few-shot accuracy | 0.345487 |
| GPT-4o-mini expansion soft agreement | 0.500517 |
| Candidate-conditioned LoRA expansion soft agreement | 0.479835 |
| Released label retained or plausible in audit | 0.747500 |
| Blind reannotation accepted after expert review | 0.252500 |
| BM25 Recall@1 | 0.236100 |

The end-to-end orchestration smoke test passed. The final distributed primary, expansion, and resource-quality tables were also regenerated with 5,000 bootstrap resamples; output validation matched the paper point estimates within the configured tolerance.

## Unit tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

Status: **3 passed**.

## Release validation

Command:

```bash
python scripts/validate_release.py
```

Status: **PASS**.

The validation checks required files, author-identifying text, hard-coded personal Drive paths, common API-key patterns, Git history, Python caches, notebook checkpoints, and test caches.

## Regeneration boundary

The deterministic command reproduces reported tables from archived predictions and annotations without invoking commercial APIs. The repository also includes code and configuration for the supervised encoder and LoRA inference, but full LoRA regeneration requires the base model, GPU resources, and adapter training; commercial API results may vary because providers do not expose immutable snapshots.
