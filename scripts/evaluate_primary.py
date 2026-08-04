#!/usr/bin/env python3
"""Recompute the primary lexical benchmark tables from archived predictions."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trframenet_revision.metrics import (  # noqa: E402
    bootstrap_mean_ci,
    exact_mcnemar,
    paired_bootstrap_diff,
    safe_kappa,
)

DISPLAY = {
    "bm25_pred": ("Retrieval top-1", "BM25"),
    "cosine_tr_pred": ("Retrieval top-1", "Turkish cosine mMPNET"),
    "cosine_en_pred": ("Retrieval top-1", "English cosine MPNet"),
    "fn_def_pred": ("Retrieval top-1", "FrameNet-definition embedding"),
    "encoder_mmpnet_linearsvc_pred": ("Supervised encoder", "mMPNET + LinearSVC"),
    "setup_a_gemma3_12b_pred": ("Local few-shot", "Gemma-3-12B"),
    "setup_a_llama3_8b_pred": ("Local few-shot", "LLaMA-3-8B"),
    "setup_a_qwen2.5_7b_pred": ("Local few-shot", "Qwen2.5-7B"),
    "setup_a_gemini_2.5_flash_pred": ("API prompted", "Gemini 2.5 Flash, few-shot"),
    "setup_a_gpt_4o_mini_pred": ("API prompted", "GPT-4o-mini, few-shot"),
    "setup_a_zeroshot_gpt_4o_mini_pred": ("API prompted", "GPT-4o-mini, zero-shot"),
    "lora_unconstrained_pred": ("LoRA", "Qwen2.5-7B, unconstrained"),
    "lora_candidate_conditioned_pred": ("LoRA", "Qwen2.5-7B, candidate-conditioned"),
}


def evaluate_group(df: pd.DataFrame, pred_col: str, group: str, value: str, n_boot: int) -> dict:
    d = df.dropna(subset=[pred_col, "FRAME1"])
    correct = (d[pred_col].astype(str) == d["FRAME1"].astype(str)).to_numpy()
    acc, lo, hi = bootstrap_mean_ci(correct, n_boot=n_boot)
    return {
        "prediction_column": pred_col,
        "model": DISPLAY[pred_col][1],
        "group": group,
        "group_value": value,
        "n": len(d),
        "accuracy": acc,
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "kappa": safe_kappa(d["FRAME1"], d[pred_col]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Archived primary prediction bundle")
    parser.add_argument("--train", required=True, help="Training split used for subgroup definitions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-predictions", help="Optional encoder predictions if absent from --input")
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    train = pd.read_csv(args.train)

    encoder_col = "encoder_mmpnet_linearsvc_pred"
    if encoder_col not in df.columns and args.encoder_predictions:
        enc = pd.read_csv(args.encoder_predictions)
        df = df.merge(enc[["ID", encoder_col]], on="ID", how="left", validate="one_to_one")

    missing = [c for c in DISPLAY if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction bundle is missing required columns: {missing}")

    metric_rows = []
    for pred_col, (category, model) in DISPLAY.items():
        d = df.dropna(subset=[pred_col, "FRAME1"])
        correct = (d[pred_col].astype(str) == d["FRAME1"].astype(str)).to_numpy()
        acc, lo, hi = bootstrap_mean_ci(correct, n_boot=args.n_boot)
        row = {
            "prediction_column": pred_col,
            "category": category,
            "model": model,
            "n": len(d),
            "accuracy": acc,
            "accuracy_ci_low": lo,
            "accuracy_ci_high": hi,
            "kappa": safe_kappa(d["FRAME1"], d[pred_col]),
        }
        if "gold_in_primary_candidates" in d.columns:
            reached = d[d["gold_in_primary_candidates"].astype(bool)]
            row["selector_accuracy_given_gold_retrieved"] = float(
                (reached[pred_col].astype(str) == reached["FRAME1"].astype(str)).mean()
            )
            row["n_gold_retrieved"] = len(reached)
        raw_stem = pred_col.removesuffix("_pred")
        valid_col = f"{raw_stem}_raw_valid"
        candidate_col = f"{raw_stem}_in_candidate"
        if valid_col in df.columns:
            valid = pd.to_numeric(df[valid_col], errors="coerce").fillna(0).astype(bool)
            row["raw_parse_valid_rate"] = float(valid.mean())
            if candidate_col in df.columns and valid.any():
                inc = pd.to_numeric(df.loc[valid, candidate_col], errors="coerce").fillna(0).astype(bool)
                row["prediction_in_candidate_rate"] = float(inc.mean())
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(out / "primary_metrics_with_ci.csv", index=False)

    pair_cols = [c for c in DISPLAY if c.startswith("setup_a_") or c.startswith("lora_") or c == encoder_col]
    pair_rows = []
    for a, b in itertools.combinations(pair_cols, 2):
        d = df.dropna(subset=[a, b, "FRAME1"])
        a_correct = (d[a].astype(str) == d["FRAME1"].astype(str)).to_numpy()
        b_correct = (d[b].astype(str) == d["FRAME1"].astype(str)).to_numpy()
        pair_rows.append({
            "model_a": DISPLAY[a][1],
            "model_b": DISPLAY[b][1],
            "n": len(d),
            **paired_bootstrap_diff(a_correct, b_correct, n_boot=args.n_boot),
            **exact_mcnemar(a_correct, b_correct),
        })
    pd.DataFrame(pair_rows).to_csv(out / "primary_pairwise_tests.csv", index=False)

    train_counts = train["FRAME1"].value_counts()
    train_surfaces = set(train.get("SYNSET_norm", train["SYNSET"]).fillna("").astype(str))
    work = df.copy()
    work["surface_seen_in_train"] = work.get("SYNSET_norm", work["SYNSET"]).fillna("").astype(str).isin(train_surfaces)
    work["frame_train_count"] = work["FRAME1"].map(train_counts).fillna(0).astype(int)
    work["rare_frame_lt10"] = work["frame_train_count"] < 10
    work["is_mwe"] = work["SYNSET"].fillna("").astype(str).str.split().str.len() > 1
    specs = [
        ("surface", "disjoint", ~work["surface_seen_in_train"]),
        ("surface", "overlap", work["surface_seen_in_train"]),
        ("frequency", "rare_<10", work["rare_frame_lt10"]),
        ("frequency", "common_>=10", ~work["rare_frame_lt10"]),
        ("lexical_form", "MWE", work["is_mwe"]),
        ("lexical_form", "single_token", ~work["is_mwe"]),
    ]
    subgroup_rows = [
        evaluate_group(work.loc[mask], pred_col, group, value, args.n_boot)
        for pred_col in DISPLAY
        for group, value, mask in specs
    ]
    pd.DataFrame(subgroup_rows).to_csv(out / "primary_subgroup_metrics.csv", index=False)
    print(f"Primary evaluation written to {out}")


if __name__ == "__main__":
    main()
