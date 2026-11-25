"""Arm B: does Russian-keyboard spelling hurt models on KazMMLU (Kazakh part)?

Every Kazakh-language KazMMLU question is asked twice: as published and with
its question and options typed on a Russian keyboard (kk_surface). The
instruction line stays standard. Paired accuracy difference = cost of
keyboard spelling on someone else's benchmark.

    python kazmmlu_flatten.py --model inceptionai/Llama-3.1-Sherkala-8B-Chat --limit 100

KazMMLU: huggingface.co/datasets/MBZUAI/KazMMLU (CC BY-NC 4.0, research use).
Subjects are discovered from the repository ("... in kaz)" folders); column
names are detected automatically and printed on the first run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import hf_runner
from kk_surface import keyboard_flatten

REPO = "MBZUAI/KazMMLU"
LETTERS = "ABCDE"
INSTR = "Тек дұрыс жауаптың әрпін жазыңыз (A, B, C, D немесе E)."


def kazakh_files() -> list[str]:
    from huggingface_hub import list_repo_files
    files = list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if "in kaz)" in f and f.endswith("test.csv"))


def _columns(df):
    q = next(c for c in df.columns if "question" in c.lower())
    opts = [c for c in df.columns if re.search(r"option", c, re.I)]
    ans = next(c for c in df.columns if "answer" in c.lower())
    return q, opts, ans


def _prompt(question: str, options: list, flatten: bool) -> str:
    f = keyboard_flatten if flatten else (lambda s: s)
    lines = [f(str(question).strip())]
    for letter, opt in zip(LETTERS, options):
        if isinstance(opt, str) and opt.strip():
            lines.append(f"{letter}. {f(opt.strip())}")
    lines.append(INSTR)
    return "\n".join(lines)


def build_items(limit: int = 100) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    items, shown = [], False
    for path in kazakh_files():
        df = pd.read_csv(hf_hub_download(REPO, path, repo_type="dataset"))
        q, opts, ans = _columns(df)
        if not shown:
            print("KazMMLU columns:", list(df.columns))
            shown = True
        subject = path.split("/")[0]
        for k, row in df.head(limit).iterrows():
            gold = str(row[ans]).strip()[:1].upper()
            if gold not in LETTERS:
                continue
            options = [row[c] for c in opts]
            items.append({"uid": f"{subject}#{k}", "subject": subject, "gold": gold,
                          "std": _prompt(row[q], options, False),
                          "kb": _prompt(row[q], options, True)})
    print(f"KazMMLU: {len(items)} Kazakh questions from {len(kazakh_files())} subjects")
    return items


# Kazakh models often answer with Cyrillic option letters in alphabet order
# (А, Б, В, Г, Д = A..E; Sherkala's "Б" matches gold B in 54% of cases), and
# sometimes with look-alikes (С = C, Е = E).
_LET = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
        "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E", "С": "C", "Е": "E"}
_RX = re.compile(r"(?<![^\W\d_])([A-EАБВГДСЕ])(?![^\W\d_])")


def _letter(text: str) -> str:
    """First stand-alone upper-case option letter (Latin or Cyrillic)."""
    m = _RX.search(text or "")
    return _LET[m.group(1)] if m else "?"


def run(model_id: str, items: list[dict], out_path: str, batch_size: int = 32,
        tok_model=None) -> dict:
    tok, model = tok_model or hf_runner.load(model_id)
    raw = {}
    for form in ("std", "kb"):
        raw[form] = hf_runner.generate(tok, model,
                                       [[{"role": "user", "content": it[form]}] for it in items],
                                       max_new_tokens=16, batch_size=batch_size,
                                       desc=f"KazMMLU {form}")
    if tok_model is None:
        hf_runner.unload(model)
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it, s, k in zip(items, raw["std"], raw["kb"]):
            f.write(json.dumps({"uid": it["uid"], "subject": it["subject"], "gold": it["gold"],
                                "pred_std": _letter(s), "pred_kb": _letter(k),
                                "raw_std": s, "raw_kb": k, "model": model_id},
                               ensure_ascii=False) + "\n")
    bad = sum(_letter(x) == "?" for x in raw["std"] + raw["kb"])
    print(f"  unparsed answers: {bad} of {2 * len(items)}", flush=True)
    return summarize(out_path)


def summarize(path: str) -> dict:
    from score import mcnemar_exact
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = len(rows)
    ok_s = [r["pred_std"] == r["gold"] for r in rows]
    ok_k = [r["pred_kb"] == r["gold"] for r in rows]
    b = sum(s and not k for s, k in zip(ok_s, ok_k))
    c = sum(k and not s for s, k in zip(ok_s, ok_k))
    out = {"model": rows[0]["model"] if rows else "", "n": n,
           "acc_std": sum(ok_s) / max(n, 1), "acc_kb": sum(ok_k) / max(n, 1),
           "lost": b, "gained": c, "p_mcnemar": mcnemar_exact(b, c)}
    print(f"{out['model']}: n={n} standard={out['acc_std']:.1%} keyboard={out['acc_kb']:.1%} "
          f"lost={b} gained={c} p={out['p_mcnemar']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=100, help="questions per subject")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"preds/kazmmlu_{args.model.split('/')[-1]}.jsonl"
    run(args.model, build_items(args.limit), out)


if __name__ == "__main__":
    main()
