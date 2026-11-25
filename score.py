"""Score predictions: final-answer accuracy per register + paired tests vs 'kk'.

    python score.py --data data/pilot.jsonl --preds preds/sherkala.jsonl

A prediction is correct if its final number is within 1% (relative) of the
gold answer. Because every instance exists in every register with the same
gold answer, registers are compared pairwise with an exact McNemar test.

Note: FinChain's ChainEval step alignment uses English-only embedders, so for
Kazakh we report final-answer accuracy; step-level scoring needs a
multilingual embedder (e.g. LaBSE or multilingual-e5) - a later step.
"""
import argparse
import json
import re
from collections import defaultdict
from math import comb

ANSWER_MARK = re.compile(r"(?:answer|ответ|жауабы|жауап)\s*[:：]?", re.I)
NUM = re.compile(r"-?\d[\d\s  ,.]*")


def parse_number(text: str):
    """Take the number after the last answer marker, else the last number."""
    m = list(ANSWER_MARK.finditer(text))
    tail = text[m[-1].end():] if m else text
    cands = NUM.findall(tail) if m else NUM.findall(text)
    if not cands:
        return None
    raw = (cands[0] if m else cands[-1]).strip().rstrip(".,")
    raw = re.sub(r"[\s  ]", "", raw)
    if "," in raw and "." in raw:            # 1,250,000.50  or 1.250.000,50
        raw = raw.replace(",", "") if raw.rfind(".") > raw.rfind(",") else \
            raw.replace(".", "").replace(",", ".")
    elif "," in raw:                         # 1250000,50 (decimal) or 1,250,000
        parts = raw.split(",")
        raw = raw.replace(",", ".") if len(parts) == 2 and len(parts[1]) != 3 \
            else raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def is_correct(pred, gold, rel_tol=0.01) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) <= rel_tol * max(abs(gold), 1e-9)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot.jsonl")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--baseline", default="kk")
    args = ap.parse_args()

    gold = {(r["uid"], r["register"]): r["answer"]
            for r in map(json.loads, open(args.data, encoding="utf-8"))}
    ok = {}
    for p in map(json.loads, open(args.preds, encoding="utf-8")):
        key = (p["uid"], p["register"])
        ok[key] = is_correct(parse_number(p["output"]), gold[key])

    by_reg = defaultdict(list)
    for (uid, reg), v in ok.items():
        by_reg[reg].append(v)
    uids = sorted({u for u, _ in ok})
    print(f"{'register':10} {'acc':>6} {'n':>5}   vs {args.baseline}: b/c   p(McNemar)")
    for reg in ["en", "ru", "kk", "kk_kb", "kk_cs", "kk_cs_kb"]:
        if reg not in by_reg:
            continue
        acc = sum(by_reg[reg]) / len(by_reg[reg])
        line = f"{reg:10} {acc:6.1%} {len(by_reg[reg]):5}"
        if reg != args.baseline:
            b = sum(ok.get((u, args.baseline), False) and not ok.get((u, reg), False) for u in uids)
            c = sum(not ok.get((u, args.baseline), False) and ok.get((u, reg), False) for u in uids)
            line += f"   {b:>3}/{c:<3}   {mcnemar_exact(b, c):.3f}"
        print(line)


if __name__ == "__main__":
    main()
