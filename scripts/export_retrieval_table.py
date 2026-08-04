#!/usr/bin/env python3
"""Convert archived retrieval JSON values into the paper's diagnostic table."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

LABELS = {
    "BM25": "BM25",
    "Cosine_TR": "Turkish cosine",
    "FN_Def": "FN-definition",
    "Hybrid": "BM25+FN-definition",
}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    values = json.load(open(args.input, encoding="utf-8"))
    rows=[]
    for key in ["BM25","Cosine_TR","FN_Def","Hybrid"]:
        row={"retriever":LABELS[key]}
        row.update({f"recall_at_{k}":float(v) for k,v in values[key].items()})
        rows.append(row)
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Retrieval table written to {out}")
if __name__ == "__main__": main()
