#!/usr/bin/env python3
"""Reproduce all deterministic tables reported in the submission.

This command does not call commercial APIs or rerun LoRA training. It recomputes
metrics, confidence intervals, significance tests, subgroup analyses, retrieval
summaries, expansion agreement, and the blind-reannotation/expert-adjudication
resource audit from archived predictions and audit records.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_outputs(out: Path) -> dict[str, float]:
    primary = pd.read_csv(out / "primary" / "primary_metrics_with_ci.csv")
    expansion = pd.read_csv(out / "expansion" / "expansion_metrics_with_ci.csv")
    resource = json.load(open(out / "resource_quality" / "resource_quality_summary.json", encoding="utf-8"))
    retrieval = pd.read_csv(out / "retrieval" / "diagnostic_recall_at_k.csv")

    def metric(df: pd.DataFrame, model: str, column: str) -> float:
        row = df.loc[df["model"] == model]
        if len(row) != 1:
            raise AssertionError(f"Expected one row for {model}; found {len(row)}")
        return float(row.iloc[0][column])

    observed = {
        "lora_unconstrained_accuracy": metric(primary, "Qwen2.5-7B, unconstrained", "accuracy"),
        "lora_candidate_conditioned_accuracy": metric(primary, "Qwen2.5-7B, candidate-conditioned", "accuracy"),
        "encoder_accuracy": metric(primary, "mMPNET + LinearSVC", "accuracy"),
        "gemini_accuracy": metric(primary, "Gemini 2.5 Flash, few-shot", "accuracy"),
        "expansion_gpt_soft": metric(expansion, "GPT-4o-mini", "soft"),
        "expansion_candidate_lora_soft": metric(expansion, "Candidate-conditioned LoRA", "soft"),
        "resource_released_retained_or_plausible": float(resource["released_retained_or_plausible"]["rate"]),
        "resource_new_label_accepted": float(resource["new_label_accepted_rate"]["rate"]),
        "retrieval_bm25_r1": float(retrieval.loc[retrieval["retriever"] == "BM25", "recall_at_1"].iloc[0]),
    }
    expected = {
        "lora_unconstrained_accuracy": 0.3927077,
        "lora_candidate_conditioned_accuracy": 0.3676031,
        "encoder_accuracy": 0.3347280,
        "gemini_accuracy": 0.3454871,
        "expansion_gpt_soft": 0.5005171,
        "expansion_candidate_lora_soft": 0.4798345,
        "resource_released_retained_or_plausible": 0.7475000,
        "resource_new_label_accepted": 0.2525000,
        "retrieval_bm25_r1": 0.2361000,
    }
    tolerance = 5e-4
    failures = {
        key: {"observed": observed[key], "expected": expected[key]}
        for key in expected
        if abs(observed[key] - expected[key]) > tolerance
    }
    if failures:
        raise AssertionError(f"Reproduction check failed: {failures}")
    with open(out / "reproduction_summary.json", "w", encoding="utf-8") as f:
        json.dump({"status": "PASS", "observed": observed, "tolerance": tolerance}, f, indent=2)
    return observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reproduced_results")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Use 500 bootstrap samples for a faster smoke test")
    args = parser.parse_args()
    n_boot = 500 if args.quick else args.n_boot
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "scripts/evaluate_primary.py",
        "--input", "results/primary/primary_predictions.csv",
        "--train", "data/lexicon/train.csv",
        "--output-dir", str(out / "primary"),
        "--n-boot", str(n_boot),
    ])
    run([
        sys.executable, "scripts/evaluate_expansion.py",
        "--annotations", "data/expansion_eval/expert_annotations.csv",
        "--predictions", "data/expansion_eval/model_predictions.csv",
        "--output-dir", str(out / "expansion"),
        "--n-boot", str(n_boot),
    ])
    run([
        sys.executable, "scripts/evaluate_resource_quality.py",
        "--audit-records", "data/resource_quality/audit_records.csv",
        "--output-dir", str(out / "resource_quality"),
        "--n-boot", str(n_boot),
    ])
    run([
        sys.executable, "scripts/export_retrieval_table.py",
        "--input", "results/retrieval/diagnostic_recall_at_k.json",
        "--output", str(out / "retrieval" / "diagnostic_recall_at_k.csv"),
    ])
    observed = validate_outputs(out)
    print("\nREPRODUCTION STATUS: PASS")
    for key, value in observed.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
