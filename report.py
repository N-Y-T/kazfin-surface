"""Summarise all model runs: tables + one chart.

    python report.py --data data/pilot.jsonl --preds preds --out results

Writes
  results/summary.csv      accuracy per model x register, drop vs kk, McNemar p
  results/by_template.csv  accuracy per model x template x register
  results/accuracy.png     grouped bars: accuracy by register, one colour per model
"""
import argparse
import csv
import glob
import json
import pathlib
from collections import defaultdict

from score import is_correct, mcnemar_exact, parse_number

REGISTERS = ["en", "ru", "kk", "kk_kb", "kk_cs", "kk_cs_kb"]
LABELS = {"en": "English", "ru": "Russian", "kk": "Kazakh\n(standard)",
          "kk_kb": "Kazakh\n(RU keyboard)", "kk_cs": "Kazakh–Russian\nmix",
          "kk_cs_kb": "Mix +\nRU keyboard"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order


def load_scores(data_path: str, pred_files: list[str]):
    gold = {}
    meta = {}
    for r in map(json.loads, open(data_path, encoding="utf-8")):
        gold[(r["uid"], r["register"])] = r["answer"]
        meta[r["uid"]] = r["template"]
    ok = defaultdict(dict)                      # model -> (uid, reg) -> bool
    for path in pred_files:
        for p in map(json.loads, open(path, encoding="utf-8")):
            key = (p["uid"], p["register"])
            if key in gold:
                ok[p["model"]][key] = is_correct(parse_number(p["output"]), gold[key])
    return ok, meta


def summarise(ok, meta, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, trows = [], []
    for model, res in ok.items():
        uids = sorted({u for u, _ in res})
        base = {u: res.get((u, "kk"), False) for u in uids}
        for reg in REGISTERS:
            vals = [res[(u, reg)] for u in uids if (u, reg) in res]
            if not vals:
                continue
            acc = sum(vals) / len(vals)
            b = sum(base[u] and not res.get((u, reg), False) for u in uids)
            c = sum(not base[u] and res.get((u, reg), False) for u in uids)
            rows.append({"model": model.split("/")[-1], "register": reg, "n": len(vals),
                         "accuracy": round(acc, 4),
                         "drop_vs_kk_pp": "" if reg == "kk" else round(
                             100 * (sum(base.values()) / len(uids) - acc), 1),
                         "lost_vs_kk": "" if reg == "kk" else b,
                         "gained_vs_kk": "" if reg == "kk" else c,
                         "p_mcnemar": "" if reg == "kk" else round(mcnemar_exact(b, c), 4)})
        by_t = defaultdict(list)
        for (u, reg), v in res.items():
            by_t[(meta[u], reg)].append(v)
        for (t, reg), vals in sorted(by_t.items()):
            trows.append({"model": model.split("/")[-1], "template": t, "register": reg,
                          "n": len(vals), "accuracy": round(sum(vals) / len(vals), 4)})
    for name, data in (("summary.csv", rows), ("by_template.csv", trows)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    return rows


def chart(rows, out_png: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in rows))
    acc = {(r["model"], r["register"]): r["accuracy"] for r in rows}
    regs = [g for g in REGISTERS if any((m, g) in acc for m in models)]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        xs = [i - 0.4 + width * (k + 0.5) for i in range(len(regs))]
        ys = [100 * acc.get((m, g), 0) for g in regs]
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[k % len(SERIES)], label=m,
               edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(regs)), [LABELS[g] for g in regs], fontsize=9, color="#3d3d3a")
    ax.set_ylabel("Final-answer accuracy, %", color="#3d3d3a")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b1")
    ax.tick_params(axis="y", colors="#6f6e69", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Same financial tasks, different surface forms", loc="left",
                 fontsize=12, color="#1f1f1d")
    ax.legend(frameon=False, ncol=len(models), loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    print(f"chart -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pilot.jsonl")
    ap.add_argument("--preds", default="preds")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(f"{args.preds}/*.jsonl")
                   if not pathlib.Path(f).name.startswith("kazmmlu_"))
    ok, meta = load_scores(args.data, files)
    rows = summarise(ok, meta, pathlib.Path(args.out))
    for r in rows:
        print(r)
    chart(rows, pathlib.Path(args.out) / "accuracy.png")


if __name__ == "__main__":
    main()
