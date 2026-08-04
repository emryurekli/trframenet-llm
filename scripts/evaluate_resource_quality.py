#!/usr/bin/env python3
"""Evaluate the stratified blind-reannotation and expert-adjudication audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trframenet_revision.metrics import bootstrap_mean_ci  # noqa: E402

OUTCOMES = [
    "Direct match — released label supported",
    "Released label retained",
    "Both plausible",
    "New label accepted",
]
STRATA = [
    "rare_frame",
    "frequent_frame",
    "polysemous_surface",
    "multiword_expression",
]
GROUP_MAP = {
    "LABEL_ERROR": "Released-label error",
    "INVENTORY_NAVIGATION": "Inventory/access difficulty",
    "RARE_FRAME": "Inventory/access difficulty",
    "NEAR_FRAME": "Semantic-neighbour distinction",
    "PRIMARY_ORDER": "Semantic-neighbour distinction",
    "GRANULARITY": "Semantic-neighbour distinction",
    "CONSTRUAL": "Interpretive ambiguity",
    "BOTH_PLAUSIBLE": "Interpretive ambiguity",
    "UNDERSPECIFIED": "Interpretive ambiguity",
    "DIRECT_MATCH": "Direct match",
    "NO_SUITABLE_FRAME": "Other",
    "MWE_IDIOM": "Other",
    "OTHER": "Other",
}
GROUP_ORDER = [
    "Released-label error",
    "Inventory/access difficulty",
    "Semantic-neighbour distinction",
    "Interpretive ambiguity",
    "Direct match",
    "Other",
]


def bootstrap_rate(series: pd.Series, n_boot: int) -> dict[str, float]:
    estimate, low, high = bootstrap_mean_ci(series.astype(bool).tolist(), n_boot=n_boot)
    return {"rate": estimate, "ci_low": low, "ci_high": high}


def crosstab_counts(df: pd.DataFrame, row: str, categories: list[str]) -> pd.DataFrame:
    table = pd.crosstab(df[row], df["sampling_stratum"])
    table = table.reindex(index=categories, columns=STRATA, fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.audit_records)

    required = {
        "sample_id", "sampling_stratum", "blind_primary_decision",
        "released_frame", "adjudicated_frame", "adjudication_outcome",
        "adjudication_category", "expert_adjudicator", "record_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required audit columns: {sorted(missing)}")
    if len(df) != 400 or df["sample_id"].duplicated().any():
        raise ValueError("Audit must contain 400 unique sample IDs")
    stratum_sizes = df["sampling_stratum"].value_counts().to_dict()
    if stratum_sizes != {s: 100 for s in STRATA}:
        raise ValueError(f"Expected 100 items per stratum; observed {stratum_sizes}")
    unknown_outcomes = set(df["adjudication_outcome"].dropna()) - set(OUTCOMES)
    if unknown_outcomes:
        raise ValueError(f"Unknown adjudication outcomes: {sorted(unknown_outcomes)}")
    if not (df["expert_adjudicator"].astype(str) == "A2").all():
        raise ValueError("Expert adjudicator must be anonymised as A2")

    df["released_strictly_supported"] = df["adjudication_outcome"].isin([
        "Direct match — released label supported", "Released label retained"
    ])
    df["released_retained_or_plausible"] = df["adjudication_outcome"].isin([
        "Direct match — released label supported", "Released label retained", "Both plausible"
    ])
    df["new_label_accepted"] = df["adjudication_outcome"].eq("New label accepted")
    df["grouped_reason"] = df["adjudication_category"].map(GROUP_MAP).fillna("Other")

    outcome_table = crosstab_counts(df, "adjudication_outcome", OUTCOMES)
    outcome_table.to_csv(out / "adjudication_outcomes_by_stratum.csv", index=False)

    category_order = sorted(df["adjudication_category"].dropna().unique())
    category_table = crosstab_counts(df, "adjudication_category", category_order)
    category_table.to_csv(out / "adjudication_categories_by_stratum.csv", index=False)

    grouped_table = crosstab_counts(df, "grouped_reason", GROUP_ORDER)
    grouped_table.to_csv(out / "grouped_reasons_by_stratum.csv", index=False)
    df.to_csv(out / "resource_quality_audit_item_results.csv", index=False)

    outcome_counts = df["adjudication_outcome"].value_counts()
    summary = {
        "n": int(len(df)),
        "sampling_design": {s: int(stratum_sizes[s]) for s in STRATA},
        "direct_matches": int(outcome_counts.get(OUTCOMES[0], 0)),
        "released_label_retained_after_review": int(outcome_counts.get(OUTCOMES[1], 0)),
        "both_labels_plausible": int(outcome_counts.get(OUTCOMES[2], 0)),
        "new_label_accepted": int(outcome_counts.get(OUTCOMES[3], 0)),
        "released_strictly_supported_count": int(df["released_strictly_supported"].sum()),
        "released_retained_or_plausible_count": int(df["released_retained_or_plausible"].sum()),
        "released_retained_or_plausible": bootstrap_rate(df["released_retained_or_plausible"], args.n_boot),
        "new_label_accepted_rate": bootstrap_rate(df["new_label_accepted"], args.n_boot),
        "design_note": "Stratified diagnostic audit; rates are not corpus-wide prevalence estimates.",
        "agreement_note": "Blind reannotation followed by independent expert adjudication; not parallel two-annotator IAA.",
    }
    with open(out / "resource_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
