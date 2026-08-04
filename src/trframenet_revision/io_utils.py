from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for order, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            parsed = obj.get("parsed") or {}
            records.append({
                "row_order": order,
                "df_idx": obj.get("df_idx"),
                "run_id": obj.get("run_id"),
                "model": obj.get("model"),
                "setup": obj.get("setup"),
                "prediction": parsed.get("predicted_frame"),
                "confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning"),
                "in_candidate": parsed.get("in_candidate"),
                "gold_frame_raw": obj.get("gold_frame"),
                "gold_in_candidates": obj.get("gold_in_cands"),
                "api_error": obj.get("api_error"),
                "raw_output": obj.get("raw_output"),
            })
    return pd.DataFrame(records)


def split_candidates(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]
