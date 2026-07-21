#!/usr/bin/env python3
"""Build a single-turn SCOTUS issue-area classification parquet for verl GRPO.

Large-input / short-output classification: a full U.S. Supreme Court opinion (lex_glue/scotus;
p50 ~5k tok, tail >10k) classified into ONE of 13 Spaeth issue areas. Output is a single label.
Reward = normalized EM via data_source="searchR1_nq" (the EM reward alias),
reward_model.ground_truth={"target": [code, name]} so EITHER the numeric code or the issue-area
name scores (robust to the exact Spaeth name wording).

Single-turn (no retriever/tool): flows through the same path as hotpotqa/quality.

Usage: make_scotus.py [--local_dir DIR] [--max_train N] [--max_test N]
"""
import argparse
import os

import datasets
import pandas as pd

DATA_SOURCE = "searchR1_nq"  # routes to the normalized-EM reward

SYSTEM_CONTENT = "You are a legal expert who classifies U.S. Supreme Court opinions by issue area."

# lex_glue/scotus ClassLabel names are the Spaeth issueArea codes "1".."13"; these are the
# standard Spaeth Supreme Court Database issue-area names for those codes.
SPAETH = {
    "1": "Criminal Procedure",
    "2": "Civil Rights",
    "3": "First Amendment",
    "4": "Due Process",
    "5": "Privacy",
    "6": "Attorneys",
    "7": "Unions",
    "8": "Economic Activity",
    "9": "Judicial Power",
    "10": "Federalism",
    "11": "Interstate Relations",
    "12": "Federal Taxation",
    "13": "Miscellaneous",
}


def _load(split):
    return datasets.load_dataset("coastalcph/lex_glue", "scotus", split=split, trust_remote_code=True)


def build(split, limit=None, no_cot=False):
    ds = _load(split)
    codes = ds.features["label"].names  # ["1",...,"13"] (Spaeth issueArea codes)
    options = "\n".join(f"{c}: {SPAETH.get(c, c)}" for c in codes)
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        code = codes[ex["label"]]
        name = SPAETH.get(code, code)
        if no_cot:
            # Direct single-label classification (the standard LexGLUE formulation): no CoT.
            # /no_think = Qwen3 soft switch to skip the <think> block (it reasons by default), so the
            # model emits only the label and output is naturally short.
            user = (
                "/no_think\n"
                "Read the following U.S. Supreme Court opinion and classify it into exactly ONE "
                "issue area from the list below. Answer with ONLY the chosen issue area (its "
                "number or its name) inside <answer> and </answer>, e.g. <answer> Criminal "
                "Procedure </answer> or <answer> 1 </answer>. Do not explain.\n\n"
                f"Issue areas:\n{options}\n\nOpinion:\n{ex['text']}"
            )
        else:
            user = (
                "Read the following U.S. Supreme Court opinion and classify it into exactly ONE issue "
                "area from the list below. Reason briefly inside <think> and </think>, then give ONLY "
                "the chosen issue area (its number or its name) inside <answer> and </answer>, e.g. "
                "<answer> Criminal Procedure </answer> or <answer> 1 </answer>.\n\n"
                f"Issue areas:\n{options}\n\nOpinion:\n{ex['text']}"
            )
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [
                    {"role": "system", "content": SYSTEM_CONTENT},
                    {"role": "user", "content": user},
                ],
                "ability": "classification",
                "reward_model": {"style": "rule", "ground_truth": {"target": [code, name]}},
                "extra_info": {"index": i, "split": split, "code": code, "name": name},
            }
        )
    return pd.DataFrame(rows)


def _prompt_len(tok, msgs):
    """Prompt token count as verl sees it (chat template + generation prompt)."""
    try:
        s = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
        return len(tok(s, add_special_tokens=False).input_ids)
    except Exception:
        return len(tok(msgs[-1]["content"]).input_ids)


def filter_band(df, lo, hi, model="/tmp/verl/models/Qwen3-4B"):
    """Keep only rows whose prompt token length is in [lo, hi]. lo/hi<=0 disable that bound.
    Drops small opinions and the extreme tail that busts training memory."""
    if lo <= 0 and hi <= 0:
        return df
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    keep, n0 = [], len(df)
    for i in range(n0):
        L = _prompt_len(tok, df.iloc[i]["prompt"])
        if (lo <= 0 or L >= lo) and (hi <= 0 or L <= hi):
            keep.append(i)
    out = df.iloc[keep].reset_index(drop=True)
    print(f"[filter_band] lo={lo} hi={hi}: kept {len(out)}/{n0} ({100*len(out)/max(n0,1):.1f}%)")
    return out


def token_stats(df, model="/tmp/verl/models/Qwen3-4B", n=300):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001
        print(f"[token_stats] tokenizer unavailable ({e}); skipping")
        return
    lens = []
    for i in range(min(len(df), n)):
        msgs = df.iloc[i]["prompt"]
        try:
            s = tok.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
            ids = tok(s, add_special_tokens=False).input_ids
        except Exception:
            ids = tok(msgs[-1]["content"]).input_ids
        lens.append(len(ids))
    lens.sort()
    m = len(lens)
    over16k = sum(1 for x in lens if x > 16384)
    print(f"[token_stats] prompt tokens: p50={lens[m // 2]} p90={lens[int(0.9 * m)]} "
          f"p99={lens[int(0.99 * m)]} max={lens[-1]} min={lens[0]}  (>16384: {over16k}/{m})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", default="/tmp/verl/data/scotus")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--no_cot", action="store_true", help="direct single-label output, no <think> CoT")
    ap.add_argument("--min_prompt_tokens", type=int, default=24562,
                    help="drop opinions shorter than this many prompt tokens (0 = no lower bound)")
    ap.add_argument("--max_prompt_tokens", type=int, default=2048,
                    help="drop opinions longer than this many prompt tokens (0 = no upper bound)")
    args = ap.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)
    for split, out, lim in (("train", "train.parquet", args.max_train),
                            ("test", "test.parquet", args.max_test)):
        df = build(split, lim or None, no_cot=args.no_cot)
        df = filter_band(df, args.min_prompt_tokens, args.max_prompt_tokens)
        p = os.path.join(args.local_dir, out)
        df.to_parquet(p, index=False)
        print(f"{split}: wrote {len(df)} rows -> {p}")
        if split == "train":
            token_stats(df)