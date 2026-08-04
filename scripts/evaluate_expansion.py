#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trframenet_revision.io_utils import split_candidates
from trframenet_revision.metrics import bootstrap_mean_ci, exact_mcnemar, paired_bootstrap_diff

PRED_MAP = {
    "gpt_4o_mini_pred": "GPT-4o-mini",
    "gemma3_12b_pred": "Gemma-3-12B",
    "llama3_8b_pred": "LLaMA-3-8B",
    "qwen2_5_7b_pred": "Qwen2.5-7B",
    "lora_constrained_pred": "Candidate-conditioned LoRA",
    "lora_pred": "Unconstrained LoRA",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ann = pd.read_excel(args.annotations) if str(args.annotations).lower().endswith("xlsx") else pd.read_csv(args.annotations)
    pred = pd.read_csv(args.predictions)
    id_col = "sample_id"
    cols = [c for c in ["annotator_frame1", "annotator_frame2", "annotator_frame3"] if c in ann.columns]
    ann["ann_filled"] = ann[cols].notna().sum(axis=1)
    df = ann.merge(pred, on=id_col, how="inner", suffixes=("", "_predfile"))
    df = df[df["ann_filled"] >= 1].copy()
    if "candidates" not in df.columns and "candidates_predfile" in df.columns:
        df["candidates"] = df["candidates_predfile"]
    metric_rows = []
    correctness = {}
    for c, name in PRED_MAP.items():
        if c not in df.columns:
            continue
        d = df.dropna(subset=[c, "annotator_frame1"]).copy()
        exact = (d[c].astype(str) == d["annotator_frame1"].astype(str))
        soft = d.apply(lambda r: str(r[c]) in [str(r[x]) for x in cols if pd.notna(r[x])], axis=1)
        inc = d.apply(lambda r: str(r[c]) in split_candidates(r.get("candidates", "")), axis=1)
        e, elo, ehi = bootstrap_mean_ci(exact, n_boot=args.n_boot)
        s, slo, shi = bootstrap_mean_ci(soft, n_boot=args.n_boot)
        metric_rows.append({"prediction_column": c, "model": name, "n": len(d), "exact": e, "exact_ci_low": elo, "exact_ci_high": ehi,
                            "soft": s, "soft_ci_low": slo, "soft_ci_high": shi, "prediction_in_candidate_rate": float(inc.mean())})
        correctness[c] = pd.DataFrame({id_col: d[id_col], "exact": exact.to_numpy(), "soft": soft.to_numpy()})
    pd.DataFrame(metric_rows).to_csv(out / "expansion_metrics_with_ci.csv", index=False)
    pairs = []
    for a, b in itertools.combinations(correctness, 2):
        m = correctness[a].merge(correctness[b], on=id_col, suffixes=("_a", "_b"))
        for metric in ["exact", "soft"]:
            ac = m[f"{metric}_a"].to_numpy(); bc = m[f"{metric}_b"].to_numpy()
            pairs.append({"model_a": PRED_MAP[a], "model_b": PRED_MAP[b], "metric": metric, "n": len(m),
                          **paired_bootstrap_diff(ac, bc, n_boot=args.n_boot), **exact_mcnemar(ac, bc)})
    pd.DataFrame(pairs).to_csv(out / "expansion_pairwise_tests.csv", index=False)
    print(f"Saved expansion evaluation outputs to {out}")


if __name__ == "__main__":
    main()
