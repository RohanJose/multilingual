from collections import Counter

import jiwer


def wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)


def cer(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis)


def _f1(overlap: int, ref_n: int, hyp_n: int) -> float:
    if not overlap:
        return 0.0
    p, r = overlap / hyp_n, overlap / ref_n
    return 2 * p * r / (p + r)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measure on whitespace tokens.

    Implemented here because the rouge-score package's default tokenizer drops non-ASCII text.
    """
    ref, hyp = reference.split(), hypothesis.split()
    out = {}
    for n, key in ((1, "rouge1"), (2, "rouge2")):
        rg = Counter(zip(*[ref[i:] for i in range(n)]))
        hg = Counter(zip(*[hyp[i:] for i in range(n)]))
        out[key] = _f1(sum((rg & hg).values()), sum(rg.values()), sum(hg.values())) if rg and hg else 0.0
    out["rougeL"] = _f1(_lcs(ref, hyp), len(ref), len(hyp)) if ref and hyp else 0.0
    return out
