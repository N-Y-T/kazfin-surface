"""Generate the pilot set: every instance in every register, one JSONL row each.

    python generate.py --per_template 20 --out data/pilot.jsonl
"""
import argparse
import json
import pathlib
import random

from templates_kk import REGISTERS, TEMPLATES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_template", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/pilot.jsonl")
    args = ap.parse_args()

    master = random.Random(args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for tpl in TEMPLATES:
            for k in range(args.per_template):
                seed = master.randint(1, 2**31 - 1)
                item = tpl(random.Random(seed))
                uid = f"{item['template']}-{k:03d}"
                for reg in REGISTERS:
                    f.write(json.dumps({
                        "uid": uid, "register": reg, "template": item["template"],
                        "level": item["level"], "seed": seed,
                        "question": item["question"][reg],
                        "answer": item["answer"], "steps": item["steps"],
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} rows ({n // len(REGISTERS)} instances x {len(REGISTERS)} registers) -> {out}")


if __name__ == "__main__":
    main()
