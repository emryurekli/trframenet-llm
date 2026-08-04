#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


def text(df: pd.DataFrame) -> list[str]:
    return [f"synset: {s} definition: {d}" for s, d in zip(df["SYNSET"].fillna(""), df["DEFINITION"].fillna(""))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--force-reencode", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train); test = pd.read_csv(args.test)
    trp = out / "train_embeddings.npy"; tep = out / "test_embeddings.npy"
    if trp.exists() and tep.exists() and not args.force_reencode:
        xtr = np.load(trp); xte = np.load(tep)
    else:
        enc = SentenceTransformer(args.model, device=args.device)
        xtr = enc.encode(text(train), batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        xte = enc.encode(text(test), batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
        np.save(trp, xtr); np.save(tep, xte)
    le = LabelEncoder(); ytr = le.fit_transform(train["FRAME1"].astype(str))
    clf = LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=20000)
    clf.fit(xtr, ytr)
    pred_ids = clf.predict(xte); preds = le.inverse_transform(pred_ids)
    result = test[["ID", "SYNSET", "DEFINITION", "FRAME1"]].copy()
    result["encoder_mmpnet_linearsvc_pred"] = preds
    result.to_csv(out / "encoder_baseline_predictions.csv", index=False)
    metrics = {
        "model": args.model, "classifier": "LinearSVC(C=1.0, class_weight=None)", "N": len(test),
        "accuracy": float(accuracy_score(test["FRAME1"].astype(str), preds)),
        "kappa": float(cohen_kappa_score(test["FRAME1"].astype(str), preds)),
        "macro_f1": float(f1_score(test["FRAME1"].astype(str), preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test["FRAME1"].astype(str), preds, average="weighted", zero_division=0)),
    }
    with open(out / "encoder_baseline_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    joblib.dump(clf, out / "linearsvc.joblib"); joblib.dump(le, out / "label_encoder.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
