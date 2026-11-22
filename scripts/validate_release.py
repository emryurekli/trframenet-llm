#!/usr/bin/env python3
"""Run public-release integrity and expected-file checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".jinja", ".yml", ".yaml", ".toml", ".ipynb"}
REQUIRED = [
    "README.md",
    "LICENSE",
    "DATA_LICENSE.md",
    "requirements.txt",
    "requirements-models.txt",
    "data/lexicon/train.csv",
    "data/lexicon/test.csv",
    "data/lexicon/turkish_framenet_2_annotated.csv",
    "data/lexicon/expansion_target_unlabelled.csv",
    "data/expansion_eval/expert_annotations.csv",
    "data/expansion_eval/model_predictions.csv",
    "data/resource_quality/audit_records.csv",
    "data/resource_quality/blind_reannotation_expert_adjudication_400.xlsx",
    "results/primary/primary_predictions.csv",
    "scripts/reproduce_paper_results.py",
    "sensitivity/README.md",
    "sensitivity/requirements.txt",
    "sensitivity/code/run_label_sensitivity.py",
    "sensitivity/data/audit_400_crosswalk.csv",
    "sensitivity/data/audit_101_decisions.csv",
    "sensitivity/results/sensitivity_metrics.csv",
    "sensitivity/results/model_pairwise_tests.csv",
    "sensitivity/results/run_manifest.json",
]
IGNORED_PATH_PARTS = {".git", ".venv", "venv", "reproduced_results"}
FORBIDDEN_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".virtual_documents"}
FORBIDDEN_TEXT = [
    re.compile(r"/content/drive/MyDrive", re.I),
    re.compile(r"[A-Za-z]:\\\\Users\\\\", re.I),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
]


def release_paths():
    """Walk release content without entering Git metadata or local environments."""
    for parent, directories, filenames in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in IGNORED_PATH_PARTS)
        for name in directories + sorted(filenames):
            yield Path(parent) / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []

    # Remove local test/interpreter artefacts before auditing the release tree.
    for cache in release_paths():
        if cache.name in FORBIDDEN_PATH_PARTS and cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)

    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f"Missing required file: {rel}")

    for path in release_paths():
        rel = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in rel.parts):
            errors.append(f"Forbidden generated path: {rel}")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path.name != "validate_release.py":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_TEXT:
                if pattern.search(text):
                    errors.append(f"Potential local-path/secret issue in {rel}: {pattern.pattern}")

    if args.write_checksums:
        rows = []
        for path in sorted(p for p in release_paths() if p.is_file() and p.name != "SHA256SUMS.txt"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
        (ROOT / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    status = {"status": "FAIL" if errors else "PASS", "errors": errors}
    print(json.dumps(status, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
