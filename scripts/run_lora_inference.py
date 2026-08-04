#!/usr/bin/env python
"""Regenerate row-level LoRA predictions using the canonical BM25+Turkish-cosine candidate generator.

Requires a GPU and the saved adapter directory. The script is resumable and saves after every batch.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "You are a FrameNet expert for Turkish. Given a Turkish verb expression (SYNSET) and its definition (DEFINITION), output the single most appropriate FrameNet frame name. Output ONLY the frame name, nothing else."
SYSTEM_CC = "You are a FrameNet expert for Turkish. Given a Turkish verb expression (SYNSET), its definition (DEFINITION), and a candidate frame list, output ONLY the single most appropriate frame name. Output ONLY the frame name, nothing else."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", str(text).lower()) or ["<unk>"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--test", required=True)
    p.add_argument("--bm25-index", required=True); p.add_argument("--cosine-tr-embeddings", required=True)
    p.add_argument("--adapter", required=True); p.add_argument("--output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--max-new-tokens", type=int, default=20)
    args = p.parse_args()
    train = pd.read_csv(args.train); test = pd.read_csv(args.test)
    with open(args.bm25_index, "rb") as f: bm25 = pickle.load(f)["index"]
    with open(args.cosine_tr_embeddings, "rb") as f: cos = pickle.load(f)
    encoder = SentenceTransformer(cos["model_name"])
    test_text = (test["SYNSET"].fillna("") + " " + test["DEFINITION"].fillna("")).tolist()
    test_emb = encoder.encode(test_text, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sims = test_emb @ cos["embeddings"].T
    candidates = []
    for i, row in test.iterrows():
        scores = bm25.get_scores(tokenize(f"{row['SYNSET']} {row['DEFINITION']}")); rb = np.argsort(scores)[::-1]
        b=[]; seen=set()
        for idx in rb:
            f=train.iloc[idx]["FRAME1"]
            if f not in seen: seen.add(f); b.append(f)
            if len(b)>=15: break
        c=[]; seen2=set()
        for idx in np.argsort(sims[i])[::-1]:
            f=train.iloc[idx]["FRAME1"]
            if f not in seen2: seen2.add(f); c.append(f)
            if len(c)>=15: break
        candidates.append(list(dict.fromkeys(b+c))[:30])

    print(f"Tokenizer base modelden yükleniyor: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter); model.eval()
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        result = pd.read_csv(out_path)
    else:
        result = test[["ID","SYNSET","DEFINITION","FRAME1"]].copy()
        result["candidates"] = [" | ".join(x) for x in candidates]
        result["lora_unconstrained_pred"] = pd.NA; result["lora_candidate_conditioned_pred"] = pd.NA
    
    def prompts(rows, conditioned: bool) -> list[str]:
        ps=[]
        for i,r in rows.iterrows():
            if conditioned:
                user=f"SYNSET: {r['SYNSET']}\nDEFINITION: {r['DEFINITION']}\nCandidate frames: {', '.join(candidates[i])}"
                system=SYSTEM_CC
            else:
                user=f"SYNSET: {r['SYNSET']}\nDEFINITION: {r['DEFINITION']}"; system=SYSTEM
            ps.append(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        return ps

    for conditioned, col in [(False,"lora_unconstrained_pred"),(True,"lora_candidate_conditioned_pred")]:
        pending = result.index[result[col].isna() | (result[col].astype(str).str.strip()=="")].tolist()
        for start in range(0,len(pending),args.batch_size):
            idxs=pending[start:start+args.batch_size]; rows=test.loc[idxs]
            inp=tok(prompts(rows,conditioned),return_tensors="pt",padding=True,truncation=True,max_length=768).to(model.device)
            with torch.no_grad(): gen=model.generate(**inp,max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            input_len=inp["input_ids"].shape[1]
            preds=[tok.decode(x[input_len:],skip_special_tokens=True).strip().split("\n")[0].strip() for x in gen]
            result.loc[idxs,col]=preds; result.to_csv(out_path,index=False)
            print(f"{col}: {min(start+len(idxs),len(pending))}/{len(pending)}")
    for col in ["lora_unconstrained_pred","lora_candidate_conditioned_pred"]:
        valid=result[col].notna(); acc=(result.loc[valid,col].astype(str)==result.loc[valid,"FRAME1"].astype(str)).mean()
        print(col, "N=",int(valid.sum()),"accuracy=",float(acc))
    with open(out_path.with_suffix(".json"),"w") as f:
        json.dump({"candidate_generator":"BM25 top-15 + Turkish cosine top-15 ordered union, deduplicated, max 30",
                   "base_model":args.base_model,"adapter":str(args.adapter)},f,indent=2)


if __name__ == "__main__":
    main()
