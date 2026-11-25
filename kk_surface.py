"""Surface-form variants of Kazakh text.

keyboard_flatten() reproduces how people type Kazakh on a Russian keyboard:
the nine Kazakh-specific Cyrillic letters are replaced by their nearest
Russian letters. The mapping is deterministic, so the meaning and the gold
answer of a task do not change - only the spelling does.

Code-switched variants are NOT generated here: naturalistic Kazakh-Russian
mixing (Russian stems with Kazakh suffixes, Russian terms inside Kazakh
syntax) is authored by a native speaker in the templates themselves.
"""
from __future__ import annotations

import random
import re

# Russian-keyboard substitutions (lower case). Upper case is derived below.
KEYBOARD_RU = {
    "ә": "а",
    "ғ": "г",
    "қ": "к",
    "ң": "н",
    "ө": "о",
    "ұ": "у",
    "ү": "у",
    "һ": "х",
    "і": "и",
}

# Common alternative: і is typed as Latin "i" (it sits on many phone layouts).
KEYBOARD_RU_LATIN_I = dict(KEYBOARD_RU, **{"і": "i"})

KAZAKH_SPECIFIC = set(KEYBOARD_RU) | {c.upper() for c in KEYBOARD_RU}


def _with_upper(mapping: dict[str, str]) -> dict[str, str]:
    full = dict(mapping)
    for src, dst in mapping.items():
        full[src.upper()] = dst.upper()
    return full


def keyboard_flatten(text: str, latin_i: bool = False, rate: float = 1.0,
                     seed: int | None = None) -> str:
    """Replace Kazakh-specific letters with Russian-keyboard equivalents.

    rate < 1.0 flattens only that share of eligible characters (people are
    often inconsistent within one message); rate = 1.0 flattens all of them.
    """
    mapping = _with_upper(KEYBOARD_RU_LATIN_I if latin_i else KEYBOARD_RU)
    if rate >= 1.0:
        return text.translate(str.maketrans(mapping))
    rng = random.Random(seed)
    out = []
    for ch in text:
        if ch in mapping and rng.random() < rate:
            out.append(mapping[ch])
        else:
            out.append(ch)
    return "".join(out)


def kazakh_letter_share(text: str) -> float:
    """Share of Cyrillic letters that are Kazakh-specific (0 for flattened text)."""
    cyr = [c for c in text if re.match(r"[Ѐ-ӿ]", c)]
    if not cyr:
        return 0.0
    return sum(c in KAZAKH_SPECIFIC for c in cyr) / len(cyr)


if __name__ == "__main__":
    sample = "Ағымдағы өтімділік коэффициентін есептеңіз. Несие сомасы қанша?"
    print(sample)
    print(keyboard_flatten(sample))
    print(keyboard_flatten(sample, latin_i=True))
    print(keyboard_flatten(sample, rate=0.5, seed=1))
