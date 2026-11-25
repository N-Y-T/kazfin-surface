"""Run one model on the pilot set and save its answers.

    python run_model.py --model inceptionai/Llama-3.1-Sherkala-8B-Chat \
        --data data/pilot.jsonl --out preds/sherkala.jsonl

Gated models (Sherkala, KazLLM) need an accepted licence on Hugging Face and
HF_TOKEN in the environment.
"""
import argparse
import json
import pathlib

import hf_runner

# One instruction per prompt language. The instruction itself is NOT
# perturbed: only the task text differs between registers.
INSTRUCTION = {
    "en": "Solve the problem step by step. End with a line 'Answer: <number>'.",
    "ru": "Решите задачу пошагово. В конце напишите строку 'Ответ: <число>'.",
    "kk": "Есепті қадам бойынша шығарыңыз. Соңында 'Жауабы: <сан>' деп жазыңыз.",
}


def prompt_lang(register: str) -> str:
    return "en" if register == "en" else "ru" if register == "ru" else "kk"


def build_messages(row: dict) -> list:
    return [{"role": "user",
             "content": f"{row['question']}\n\n{INSTRUCTION[prompt_lang(row['register'])]}"}]


def run(model_id: str, rows: list[dict], out_path: str, max_new_tokens: int = 768,
        batch_size: int = 16, tok_model=None) -> None:
    tok, model = tok_model or hf_runner.load(model_id)
    outs = hf_runner.generate(tok, model, [build_messages(r) for r in rows],
                              max_new_tokens=max_new_tokens, batch_size=batch_size,
                              desc=model_id.split("/")[-1])
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"uid": r["uid"], "register": r["register"],
                                "model": model_id, "output": o}, ensure_ascii=False) + "\n")
    if tok_model is None:
        hf_runner.unload(model)
    print(f"saved {len(outs)} answers -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/pilot.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    run(args.model, rows, args.out, args.max_new_tokens, args.batch_size)


if __name__ == "__main__":
    main()
