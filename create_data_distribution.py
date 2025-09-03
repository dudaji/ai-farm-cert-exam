#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sentence length distributions for vi-en.json

Input:
  - JSON file with [{"source_text": "...", "reference_text": "..."}]

Outputs (under data/distributions/):
  - length_distribution.png        : histogram of token lengths (source vs reference)
  - ratio_distribution.png         : histogram of reference/source token length ratio
  - sentence_length_stats.csv      : per-sentence stats (tokens, chars, ratios)
  - unigram_top.csv, bigram_top.csv, trigram_top.csv, fourgram_top.csv : n-gram frequency
"""

import argparse
import json
import math
import statistics as stats
import re
from pathlib import Path
from typing import List, Tuple, Iterable
from collections import Counter

import matplotlib.pyplot as plt
import csv

WORD_RE = re.compile(r"\w+([’']\w+)?", re.UNICODE)  # simple regex tokenizer


def tokenize(text: str) -> List[str]:
    """Tokenize text using regex word boundaries."""
    return [m.group(0) for m in WORD_RE.finditer(text)]


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens."""
    if n <= 0:
        return []
    return (tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


def describe(name: str, data: List[float]) -> dict:
    """Print and return descriptive statistics for a list of values."""
    if not data:
        return {}
    data_sorted = sorted(data)

    def pct(p: float) -> float:
        k = (len(data_sorted) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(data_sorted[int(k)])
        return float(data_sorted[f] * (c - k) + data_sorted[c] * (k - f))

    desc = {
        "count": len(data),
        "mean": float(stats.fmean(data)),
        "median": float(stats.median(data)),
        "stdev": float(stats.pstdev(data)),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "min": float(min(data)),
        "max": float(max(data)),
    }
    print(f"\n[{name}] summary statistics")
    for k in [
        "count",
        "mean",
        "median",
        "stdev",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "min",
        "max",
    ]:
        print(
            f" - {k:>6}: {desc[k]:.4f}"
            if isinstance(desc[k], float)
            else f" - {k:>6}: {desc[k]}"
        )
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to vi-en.json")
    ap.add_argument("--limit", type=int, default=1000, help="Maximum number of samples")
    ap.add_argument("--ngram", type=int, default=4, help="Maximum n-gram order")
    ap.add_argument("--hist_bins", type=int, default=50, help="Histogram bin count")
    ap.add_argument(
        "--use_whitespace",
        action="store_true",
        help="Use whitespace tokenization instead of regex",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    assert inp.exists(), f"Input file not found: {inp}"

    outdir = Path("data/distributions")
    outdir.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Top-level JSON must be a list.")

    rows = data[: args.limit] if args.limit else data
    print(f"Analyzing {len(rows)} samples (limit={args.limit})")

    token_func = (lambda s: s.split()) if args.use_whitespace else tokenize

    src_tok_lens, ref_tok_lens = [], []
    src_char_lens, ref_char_lens = [], []
    ratios_tok, ratios_char = [], []

    ngram_counters = {n: Counter() for n in range(1, args.ngram + 1)}
    detailed_rows = []

    for i, item in enumerate(rows):
        src = (item.get("source_text") or "").strip()
        ref = (item.get("reference_text") or "").strip()

        src_toks = token_func(src)
        ref_toks = token_func(ref)

        s_tok = len(src_toks)
        r_tok = len(ref_toks)
        s_chr = len(src)
        r_chr = len(ref)

        src_tok_lens.append(s_tok)
        ref_tok_lens.append(r_tok)
        src_char_lens.append(s_chr)
        ref_char_lens.append(r_chr)

        ratio_t = (r_tok / s_tok) if s_tok > 0 else float("nan")
        ratio_c = (r_chr / s_chr) if s_chr > 0 else float("nan")
        ratios_tok.append(ratio_t)
        ratios_char.append(ratio_c)

        detailed_rows.append(
            {
                "idx": i,
                "source_tokens": s_tok,
                "reference_tokens": r_tok,
                "source_chars": s_chr,
                "reference_chars": r_chr,
                "ratio_tokens_ref_over_src": ratio_t,
                "ratio_chars_ref_over_src": ratio_c,
            }
        )

        for n in range(1, args.ngram + 1):
            for g in ngrams(ref_toks, n):
                ngram_counters[n][g] += 1

    # Summary stats
    describe("Source token length", src_tok_lens)
    describe("Reference token length", ref_tok_lens)
    describe(
        "Reference/Source token length ratio",
        [x for x in ratios_tok if not math.isnan(x)],
    )

    # Histogram: token length
    plt.figure()
    plt.hist(src_tok_lens, bins=args.hist_bins, alpha=0.6, label="source (tokens)")
    plt.hist(ref_tok_lens, bins=args.hist_bins, alpha=0.6, label="reference (tokens)")
    plt.xlabel("Token length")
    plt.ylabel("Sentence count")
    plt.title("Sentence token length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "length_distribution.png", dpi=150)
    plt.close()

    # Histogram: length ratio
    valid_ratios = [
        x for x in ratios_tok if not math.isnan(x) and x < 10
    ]  # clip extreme outliers
    plt.figure()
    plt.hist(valid_ratios, bins=args.hist_bins, alpha=0.85)
    plt.xlabel("Reference/Source token length ratio")
    plt.ylabel("Sentence count")
    plt.title("Token length ratio distribution")
    plt.tight_layout()
    plt.savefig(outdir / "ratio_distribution.png", dpi=150)
    plt.close()

    # Save CSV: per-sentence stats
    with open(outdir / "sentence_length_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        w.writeheader()
        w.writerows(detailed_rows)

    # Save n-gram counts
    for n in range(1, args.ngram + 1):
        outp = outdir / (
            f"{['uni','bi','tri','four','five','six'][n-1] if n-1 < 6 else str(n)}gram_top.csv"
        )
        with open(outp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{n}-gram", "count"])
            for gram, cnt in ngram_counters[n].most_common(1000):
                w.writerow([" ".join(gram), cnt])
        print(f"Saved top {n}-grams to: {outp}")

    print("\nGenerated files in:", outdir)


if __name__ == "__main__":
    main()
