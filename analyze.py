"""Final numbers for one run: accuracy with 95% CIs, paired tests, KazMMLU re-scored.

    python analyze.py runs/2026-11-20

Reads  <run>/preds/*.jsonl and data/pilot.jsonl
Writes <run>/analysis/pilot_accuracy.csv   model x register: accuracy, Wilson 95% CI, share with no final answer
       <run>/analysis/paired_tests.csv     register pairs per model and pooled: change in pp, McNemar p
       <run>/analysis/by_template.csv      template x register, pooled over models
       <run>/analysis/kazmmlu.csv          KazMMLU standard vs keyboard, re-scored from raw answers
       <run>/analysis/accuracy.png         one panel per model
"""
import csv
import json
import math
import pathlib
import re
import sys

from kazmmlu_flatten import _letter
from score import ANSWER_MARK, is_correct, mcnemar_exact, parse_number

MODELS = [("Llama-3.1-Sherkala-8B-Chat", "Sherkala-8B"),
          ("LLama-3.1-KazLLM-1.0-8B", "KazLLM-8B"),
          ("Qwen2.5-7B-Instruct", "Qwen2.5-7B")]
REGS = ["en", "ru", "kk", "kk_kb", "kk_cs", "kk_cs_kb"]
LABEL = {"en": "English", "ru": "Russian", "kk": "Kazakh", "kk_kb": "Kazakh,\nRU keyboard",
         "kk_cs": "Kazakh–Russian\nmix", "kk_cs_kb": "Mix,\nRU keyboard"}
PAIRS = [("kk", "kk_kb"), ("kk", "kk_cs"), ("kk_cs", "kk_cs_kb"), ("kk", "kk_cs_kb"),
         ("en", "kk"), ("ru", "kk")]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def has_final(text):
    """An answer marker followed by a number (no marker = truncated or looping)."""
    m = list(ANSWER_MARK.finditer(text))
    return bool(m) and re.search(r"\d", text[m[-1].end():m[-1].end() + 80]) is not None


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(run):
    run = pathlib.Path(run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    data = {(r["uid"], r["register"]): r for r in map(json.loads, open("data/pilot.jsonl", encoding="utf-8"))}
    uids = sorted({u for u, _ in data})
    ok, fin = {}, {}
    for fname, s in MODELS:
        for p in map(json.loads, open(run / "preds" / f"{fname}.jsonl", encoding="utf-8")):
            g = data[(p["uid"], p["register"])]["answer"]
            ok[(s, p["uid"], p["register"])] = is_correct(parse_number(p["output"]), g)
            fin[(s, p["uid"], p["register"])] = has_final(p["output"])
    names = [s for _, s in MODELS]

    acc = []
    for s in names:
        for r in REGS:
            k = sum(ok[(s, u, r)] for u in uids)
            lo, hi = wilson(k, len(uids))
            acc.append({"model": s, "register": r, "n": len(uids), "correct": k,
                        "accuracy_pct": round(100 * k / len(uids), 1),
                        "ci95_low": round(100 * lo, 1), "ci95_high": round(100 * hi, 1),
                        "no_final_answer_pct": round(100 * sum(not fin[(s, u, r)] for u in uids) / len(uids), 1)})
    write(out / "pilot_accuracy.csv", acc)

    def pair(models, a, b):
        lost = sum(ok[(s, u, a)] and not ok[(s, u, b)] for s in models for u in uids)
        gained = sum(ok[(s, u, b)] and not ok[(s, u, a)] for s in models for u in uids)
        n = len(models) * len(uids)
        ka = sum(ok[(s, u, a)] for s in models for u in uids)
        kb = sum(ok[(s, u, b)] for s in models for u in uids)
        return {"from": a, "to": b, "acc_from_pct": round(100 * ka / n, 1), "acc_to_pct": round(100 * kb / n, 1),
                "change_pp": round(100 * (kb - ka) / n, 1), "became_wrong": lost, "became_right": gained,
                "mcnemar_p": round(mcnemar_exact(lost, gained), 4)}
    paired = [{"model": s, **pair([s], a, b)} for s in names for a, b in PAIRS]
    paired += [{"model": "all three", **pair(names, a, b)} for a, b in PAIRS]
    write(out / "paired_tests.csv", paired)

    tmpl = {u: data[(u, "en")]["template"] for u in uids}
    by_t = []
    for t in sorted(set(tmpl.values())):
        us = [u for u in uids if tmpl[u] == t]
        by_t.append({"template": t, **{r: round(100 * sum(ok[(s, u, r)] for s in names for u in us)
                                              / (len(names) * len(us))) for r in REGS}})
    write(out / "by_template.csv", by_t)

    kz = []
    for fname, s in MODELS:
        rows = [json.loads(l) for l in open(run / "preds" / f"kazmmlu_{fname}.jsonl", encoding="utf-8")]
        ps = [_letter(r["raw_std"]) for r in rows]
        pk = [_letter(r["raw_kb"]) for r in rows]
        a = [p == r["gold"] for p, r in zip(ps, rows)]
        b = [p == r["gold"] for p, r in zip(pk, rows)]
        lost = sum(x and not y for x, y in zip(a, b))
        gained = sum(y and not x for x, y in zip(a, b))
        n = len(rows)
        kz.append({"model": s, "n": n, "acc_standard_pct": round(100 * sum(a) / n, 1),
                   "acc_keyboard_pct": round(100 * sum(b) / n, 1),
                   "change_pp": round(100 * (sum(b) - sum(a)) / n, 1),
                   "became_wrong": lost, "became_right": gained, "mcnemar_p": mcnemar_exact(lost, gained),
                   "no_letter_standard_pct": round(100 * ps.count("?") / n, 1),
                   "no_letter_keyboard_pct": round(100 * pk.count("?") / n, 1)})
    write(out / "kazmmlu.csv", kz)
    chart(acc, out / "accuracy.png")
    for row in kz:
        print(row)
    print("written to", out)


def chart(acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grey, accent, ref = "#a7a59b", "#2a78d6", "#6f6e69"
    names = [s for _, s in MODELS]
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 8.2), sharex=True)
    for ax, s in zip(axes, names):
        rows = [r for r in acc if r["model"] == s]
        kk = next(r for r in rows if r["register"] == "kk")
        kb = next(r for r in rows if r["register"] == "kk_kb")
        ax.axhline(kk["accuracy_pct"], color=ref, lw=1, ls=(0, (4, 3)))
        for i, r in enumerate(rows):
            c = accent if r["register"] == "kk_kb" else grey
            ax.plot([i, i], [r["ci95_low"], r["ci95_high"]], color=c, lw=2)
            ax.plot(i, r["accuracy_pct"], "o", color=c, ms=7)
            if r["register"] in ("kk", "kk_kb"):
                ax.annotate(f"{r['accuracy_pct']:.1f}", (i, r["accuracy_pct"]), xytext=(9, 4 if r is kb else 7),
                            textcoords="offset points", fontsize=9, color="#1f1f1d",
                            fontweight="bold" if r is kb else "normal")
        drop = 100 * (kb["correct"] - kk["correct"]) / kk["n"]
        ax.set_title(f"{s}   {drop:+.1f} pp on RU keyboard", loc="left", fontsize=10, color="#1f1f1d")
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 20, 40, 60])
        ax.grid(axis="y", color="#e6e5e0")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b9b8b1")
        ax.tick_params(axis="y", colors="#6f6e69", length=0)
    axes[-1].set_xticks(range(len(REGS)), [LABEL[r] for r in REGS], fontsize=9, color="#6f6e69")
    fig.suptitle("Typing Kazakh on a Russian keyboard costs all three models 4–7 points\n"
                 "% correct final answers, 240 tasks per register, 95% CI; dashed = standard Kazakh",
                 x=0.06, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/2026-11-20")
