#!/usr/bin/env python3
"""
Cohen's Kappa & Fleiss' Kappa inter-annotator agreement analysis
for the Doccano Twitter Sentiment project.

Compares 7 annotation passes (A1..A7) on the same 100 tweets:
  - pairwise Cohen's kappa between every pair of annotators
  - per-tweet agreement (how many annotators gave the same label)
  - each annotator vs the Kaggle gold labels
  - Fleiss' kappa across all 7 annotators
"""
import json
import os
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(HERE, "annotations")
LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]
ANNOTATORS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    agreed = sum(1 for x, y in zip(a, b) if x == y)
    po = agreed / n
    fa, fb = Counter(a), Counter(b)
    pe = sum((fa[k] / n) * (fb[k] / n) for k in set(fa) | set(fb))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def fleiss_kappa(matrix: list, n_raters: int) -> float:
    n_items, n_cat = len(matrix), len(matrix[0])
    p = [sum((c * (c - 1)) for c in row) / (n_raters * (n_raters - 1)) for row in matrix]
    p_bar = sum(p) / n_items
    total = sum(sum(row) for row in matrix)
    p_cat = [sum(row[j] for row in matrix) / total for j in range(n_cat)]
    p_e = sum(x * x for x in p_cat)
    return (p_bar - p_e) / (1 - p_e)


def kappa_scale(k: float) -> str:
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def main() -> None:
    with open(os.path.join(ANN_DIR, "tweets_100.json"), encoding="utf-8") as f:
        rows = json.load(f)
    gold = [r["gold"] for r in rows]

    ann = {}
    for name in ANNOTATORS:
        with open(os.path.join(ANN_DIR, f"{name}.json"), encoding="utf-8") as f:
            arr = json.load(f)
        assert len(arr) == len(rows), f"{name} has {len(arr)} labels, expected {len(rows)}"
        ann[name] = arr

    out = []
    push = out.append
    push("=" * 62)
    push("SENTIMENT ANNOTATION AGREEMENT  (100 tweets, 7 annotators)")
    push("=" * 62)

    push("\n[1] Per-annotator label distribution")
    push(f"{'annotator':10} " + "  ".join(f"{l[:6]:>7}" for l in LABELS) + "   total")
    for name in ANNOTATORS:
        c = Counter(ann[name])
        push(f"{name:10} " + "  ".join(f"{c.get(l, 0):>7}" for l in LABELS) + f"   {len(ann[name]):>5}")

    push("\n[2] Pairwise Cohen's kappa (compare every annotation with every other)")
    push(f"{'':12}" + "".join(f"{a:>8}" for a in ANNOTATORS))
    kappas = {}
    for a, b in combinations(ANNOTATORS, 2):
        kappas[(a, b)] = cohen_kappa(ann[a], ann[b])
    for a in ANNOTATORS:
        row = f"{a:12}"
        for b in ANNOTATORS:
            val = kappas.get((a, b)) or kappas.get((b, a))
            if a == b:
                row += f"{'--':>8}"
            else:
                row += f"{val:>8.3f}"
        push(row)

    pair_vals = list(kappas.values())
    avg = sum(pair_vals) / len(pair_vals)
    min_k = min(pair_vals)
    max_k = max(pair_vals)
    match = list(filter(lambda v: v >= 0.60, pair_vals))
    push(f"\nPairwise summary  → mean {avg:.3f}  | min {min_k:.3f}  | max {max_k:.3f}  | pairs >= 0.60: {len(match)}/{len(pair_vals)}")
    push(f"Mean kappa is '{kappa_scale(avg)}' agreement.")

    push(f"\n[3] Fleiss' kappa (all {len(ANNOTATORS)} annotators at once)")
    matrix = []
    for i in range(len(rows)):
        cnt = Counter(ann[a][i] for a in ANNOTATORS)
        matrix.append([cnt.get(l, 0) for l in LABELS])
    fk = fleiss_kappa(matrix, len(ANNOTATORS))
    push(f"{fk:.3f}  → {kappa_scale(fk)} agreement")

    push("\n[4] Per-tweet agreement (how many of the 7 annotators gave the same label)")
    push("Your explanation example: if tweet #N is negative, check whether the other")
    push("annotations also marked it negative.")
    push(f"{'tweet#':6} {'entity':18} {'labels':70} {'n_agree':7} {'majority':9} {'gold':9} {'match?':6}")
    counts = Counter()
    gold_hits = Counter()
    for i, row in enumerate(rows):
        votes = [ann[a][i] for a in ANNOTATORS]
        c = Counter(votes)
        top_label, top_n = c.most_common(1)[0]
        agreement = top_n / len(ANNOTATORS)
        bucket = "all7" if top_n == 7 else ("6" if top_n == 6 else ("5" if top_n == 5 else "<=4"))
        counts[bucket] += 1
        hit = "yes" if top_label == row["gold"] else "no"
        if top_label == row["gold"]:
            gold_hits[bucket] += 1
        push(f"{i:6} {row['entity'][:18]:18} {str(votes):70} {top_n:^7} {top_label:9} {row['gold']:9} {hit:6}")
    push(f"\nMajority-agreement distribution: {dict(counts)}")
    push(f"Majority-vs-gold: {dict(gold_hits)} (tweets where the majority label matches gold)")

    push("\n[5] Each annotator vs Kaggle gold labels (Cohen's kappa)")
    push(f"{'annotator':10} {'kappa':8} {'agreement %':12} {'quality'}")
    user_gold = [(cohen_kappa(ann[a], gold), sum(1 for x, y in zip(ann[a], gold) if x == y) / len(gold)) for a in ANNOTATORS]
    for name, (k, po) in zip(ANNOTATORS, user_gold):
        push(f"{name:10} {k:8.3f} {po * 100:11.1f}% {kappa_scale(k)}")

    path = os.path.join(HERE, "kappa_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()