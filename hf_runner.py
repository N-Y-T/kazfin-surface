"""Batched greedy generation with Hugging Face transformers.

Written for Kaggle "GPU T4 x2" (2 x 15 GB): an 8B model in fp16 is split
across both cards with device_map="auto", leaving room on each card for
activations and the KV cache. Works on any CUDA GPU, and on CPU for tiny
smoke-test models.

Batches are capped both by count and by tokens (prompt + new tokens), so long
prompts automatically get smaller batches. A batch that runs out of GPU memory
is split in half and retried; a generation error that is not about memory is
logged, its prompts get an empty answer, and after two such errors in a row the rest of
the run for this model is skipped instead of crashing the whole notebook.
"""
from __future__ import annotations

import gc
import os
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Fallback for models whose tokenizer ships without a chat template
# (Llama-3 style, which Sherkala and KazLLM are built on).
LLAMA3_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>"
    "{% endfor %}{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def _max_memory(share: float = 0.70):
    """Cap weights per GPU so every card keeps ~30% free for activations."""
    import torch
    n = torch.cuda.device_count()
    if n < 2:
        return None
    return {i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30 * share)}GiB"
            for i in range(n)}


def free_old_models() -> None:
    """Release GPU memory of every model still alive in this process.

    In a notebook a finished model can stay referenced (reference cycles,
    leftover variables); its weights would then block the next model.
    """
    import torch
    from transformers import PreTrainedModel
    gc.collect()
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, PreTrainedModel):
                continue
        except Exception:                           # noqa: BLE001
            continue
        for t in list(obj.parameters()) + list(obj.buffers()):
            t.data = torch.empty(0)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load(model_id: str, dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    free_old_models()

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    use_cuda = torch.cuda.is_available()
    kw = {}
    if use_cuda:
        kw["device_map"] = "auto"
        mm = _max_memory()
        if mm:
            kw["max_memory"] = mm
    dt = getattr(torch, dtype) if use_cuda else torch.float32
    try:                                   # transformers >= 4.56 uses `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:                      # older releases: `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    model.eval()
    if use_cuda:
        used = {d: round(torch.cuda.memory_allocated(d) / 2**30, 1)
                for d in range(torch.cuda.device_count())}
        print(f"{model_id.split('/')[-1]} loaded, weights per GPU (GiB): {used}", flush=True)
    return tok, model


def _stop_ids(tok, model) -> list[int]:
    ids = set()
    for src in (model.generation_config.eos_token_id, tok.eos_token_id):
        if isinstance(src, int):
            ids.add(src)
        elif src:
            ids.update(src)
    for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id and tid is not None:
            ids.add(tid)
    return sorted(ids)


def _batches(lengths: list[int], order: list[int], batch_size: int, budget: int,
             max_new_tokens: int) -> list[list[int]]:
    """Group indices (already sorted long-first) under a count and a token cap."""
    out, cur, cur_max = [], [], 0
    for i in order:
        need = max(cur_max, lengths[i] + max_new_tokens)
        if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * need > budget):
            out.append(cur)
            cur, cur_max = [], 0
            need = lengths[i] + max_new_tokens
        cur.append(i)
        cur_max = need
    if cur:
        out.append(cur)
    return out


def _is_oom(e: BaseException) -> bool:
    import torch
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (oom and isinstance(e, oom)) or "out of memory" in str(e).lower()


def generate(tok, model, conversations: list[list[dict]], max_new_tokens: int = 768,
             batch_size: int = 16, desc: str = "", token_budget: int = 32_000) -> list[str]:
    """Greedy-decode every conversation; returns outputs in input order."""
    import torch

    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
               for c in conversations]
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    # Long prompts first: similar lengths share a batch, OOM shows up early.
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    batches = _batches(lengths, order, batch_size, token_budget, max_new_tokens)
    stop = _stop_ids(tok, model)
    device = next(model.parameters()).device
    outs: list[str] = [""] * len(prompts)
    state = {"errors": 0}

    def run(idx: list[int]) -> None:
        if state["errors"] >= 2:
            return
        try:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     eos_token_id=stop, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:].cpu()
            del gen, enc
            for j, i in enumerate(idx):
                outs[i] = tok.decode(new[j], skip_special_tokens=True).strip()
            state["errors"] = 0
        except Exception as e:                      # noqa: BLE001
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            if _is_oom(e) and len(idx) > 1:
                half = len(idx) // 2
                print(f"  out of GPU memory on {len(idx)} prompts -> retry as "
                      f"{half} + {len(idx) - half}", flush=True)
                run(idx[:half])
                run(idx[half:])
            elif _is_oom(e):
                print("  a single prompt does not fit in GPU memory; left empty", flush=True)
            else:
                state["errors"] += 1
                print(f"  generation error ({type(e).__name__}); {len(idx)} prompts left empty",
                      flush=True)
                traceback.print_exc()
                if state["errors"] >= 2:
                    print("  two errors in a row: skipping the rest of this model's run",
                          flush=True)

    try:
        from tqdm.auto import tqdm
        it = tqdm(batches, desc=desc or "generate")
    except ImportError:
        it = batches
    for idx in it:
        run(idx)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outs


def unload(model) -> None:
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
