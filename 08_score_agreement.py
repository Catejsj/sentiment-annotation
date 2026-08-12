#!/usr/bin/env python3
"""
STEP 8 - Score inter-annotator agreement across ANY number of annotators.

This is the "scoring" script: Cohen's Kappa between every pair of annotators,
plus Fleiss' Kappa for all of them at once.

    python 08_score_agreement.py exports/*.csv
    python 08_score_agreement.py mine.csv friend.csv classmate3.csv

Each input is a Doccano CSV export. They must all cover the same tweets; the
script proves that before scoring anything and refuses if they do not line up.

MULTI-LABEL HANDLING
    Our project allows two labels on one tweet ("Irrelevant#Positive"). Cohen's
    and Fleiss' Kappa are both defined for ONE label per item, so the headline
    numbers use the PRIMARY label (the first one). Two extra views are printed
    so nothing is hidden:
      * exact-match agreement, which respects the full label set
      * per-label kappa, treating each label as its own yes/no question

Libraries: scikit-learn (Cohen), statsmodels (Fleiss).
"""
import os
import sys
from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def interpret(k: float) -> str:
    """Landis & Koch (1977)."""
    if k != k:
        return "undefined"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("text", "label"):
        if col not in df.columns:
            raise SystemExit(f"{path}: missing '{col}' column. Found {list(df.columns)}")
    key = "id" if "id" in df.columns else ("row" if "row" in df.columns else None)
    if key is None:
        raise SystemExit(f"{path}: needs an 'id' or 'row' column to align annotators.")
    out = pd.DataFrame(
        {
            "key": df[key],
            "text": df["text"].astype(str),
            "raw": df["label"].fillna("").astype(str),
        }
    )
    out["labels"] = out["raw"].apply(lambda v: frozenset(p for p in v.split("#") if p))
    # primary = first label as Doccano wrote it; blank stays blank
    out["primary"] = out["raw"].apply(lambda v: v.split("#")[0] if v else "")
    return out.set_index("key")


def main() -> None:
    paths = sys.argv[1:]
    if len(paths) < 2:
        raise SystemExit(__doc__)

    names, frames = [], {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0].replace("_annotations", "")
        while name in frames:
            name += "_2"
        names.append(name)
        frames[name] = load(p)

    out = []
    push = out.append
    push("=" * 78)
    push(f"INTER-ANNOTATOR AGREEMENT  -  {len(names)} annotators")
    push("Cohen's Kappa (scikit-learn) + Fleiss' Kappa (statsmodels)")
    push("=" * 78)
    push("\nfiles:")
    for n, p in zip(names, paths):
        push(f"    {n:20} {p}   ({len(frames[n])} rows)")

    # ------------------------------------------------------------- align
    push("\n[1] ALIGNMENT CHECK")
    common = set.intersection(*(set(frames[n].index) for n in names))
    push(f"    document ids present in every file : {len(common)}")
    for n in names:
        missing = len(frames[n]) - len(common)
        if missing:
            push(f"    [!] {n} has {missing} rows that the others do not")
    if len(common) < 2:
        push("\n    Not enough shared documents to score. Stopping.")
        print("\n".join(out))
        return
    ids = sorted(common)

    ref = frames[names[0]].loc[ids, "text"]
    mismatched = []
    for n in names[1:]:
        diff = (frames[n].loc[ids, "text"].str.strip() != ref.str.strip()).sum()
        if diff:
            mismatched.append((n, diff))
    if mismatched:
        for n, d in mismatched:
            push(f"    [!] {n}: {d} shared ids carry DIFFERENT text than {names[0]}")
        push("\n    The files are not the same corpus. Scores would be meaningless. Stopping.")
        print("\n".join(out))
        return
    push("    ok  every shared id carries identical text in every file")

    data = pd.DataFrame({n: frames[n].loc[ids, "primary"] for n in names})
    sets = pd.DataFrame({n: frames[n].loc[ids, "labels"] for n in names})

    blank = (data == "").sum()
    if blank.any():
        push("\n    [!] unannotated rows found:")
        for n in names:
            if blank[n]:
                push(f"        {n}: {blank[n]} blank")
        push("        Those rows still count as a category; finish them for a clean score.")

    # ------------------------------------------------- label distribution
    push("\n\n[2] LABEL DISTRIBUTION (primary label)")
    dist = pd.DataFrame({n: data[n].value_counts() for n in names})
    dist = dist.reindex([l for l in LABELS if l in dist.index] +
                        [i for i in dist.index if i not in LABELS]).fillna(0).astype(int)
    push(dist.to_string())

    # ------------------------------------------------ pairwise Cohen
    push("\n\n[3] PAIRWISE COHEN'S KAPPA  (primary label)")
    cats = sorted(set(data.to_numpy().ravel()))
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    pairs = {}
    for a, b in combinations(names, 2):
        k = cohen_kappa_score(data[a], data[b], labels=cats)
        pairs[(a, b)] = k
        matrix.loc[a, b] = matrix.loc[b, a] = k
    for n in names:
        matrix.loc[n, n] = 1.0
    push(matrix.round(3).to_string())

    push("\n    pair-by-pair, with raw agreement for context:")
    for (a, b), k in sorted(pairs.items(), key=lambda kv: kv[1], reverse=True):
        raw = (data[a] == data[b]).mean()
        push(f"      {a:14} vs {b:14} kappa={k:6.3f}  raw={raw:5.1%}  {interpret(k)}")

    mean_k = sum(pairs.values()) / len(pairs)
    push(f"\n    mean pairwise kappa : {mean_k:.3f}  ({interpret(mean_k)})")

    # ------------------------------------------------------ Fleiss
    if len(names) >= 3:
        push(f"\n\n[4] FLEISS' KAPPA  (all {len(names)} annotators at once)")
        table, categories = aggregate_raters(data.to_numpy())
        fk = fleiss_kappa(table, method="fleiss")
        push(f"    categories : {list(categories)}")
        push(f"    Fleiss' kappa : {fk:.3f}  ({interpret(fk)})")
        push("    Use this as the single headline number for the whole group.")
    else:
        push("\n\n[4] FLEISS' KAPPA")
        push("    Needs 3+ annotators; with 2 files Cohen's Kappa above is the right measure.")

    # --------------------------------------------- multi-label views
    push("\n\n[5] MULTI-LABEL VIEWS  (because we allow two labels per tweet)")
    push("    exact-match agreement - both annotators picked the identical label SET:")
    for a, b in combinations(names, 2):
        em = (sets[a] == sets[b]).mean()
        push(f"      {a:14} vs {b:14} {em:6.1%}")

    push("\n    per-label Cohen's kappa - 'did we both apply this label?'")
    push(f"      {'label':12} {'mean kappa':>14}   interpretation")
    for label in LABELS:
        ks = []
        for a, b in combinations(names, 2):
            ya = sets[a].apply(lambda s: label in s)
            yb = sets[b].apply(lambda s: label in s)
            if ya.any() or yb.any():
                ks.append(cohen_kappa_score(ya, yb))
        if ks:
            m = sum(ks) / len(ks)
            push(f"      {label:12} {m:14.3f}   {interpret(m)}")
        else:
            push(f"      {label:12} {'--':>14}   never used by anyone")

    # ------------------------------------------- consensus / hard tweets
    push("\n\n[6] WHERE WE DISAGREE")
    majority = data.mode(axis=1)[0]
    n_agree = data.eq(majority, axis=0).sum(axis=1)
    push(f"    unanimous            : {(n_agree == len(names)).sum():3d} / {len(ids)} tweets")
    push(f"    majority but not all : {n_agree.between(2, len(names) - 1).sum():3d}")
    push(f"    no majority          : {(n_agree == 1).sum():3d}")

    hard = n_agree.nsmallest(10).index
    push("\n    10 most contested tweets:")
    for i in hard:
        votes = ", ".join(f"{n}={data.loc[i, n] or '(blank)'}" for n in names)
        push(f"      id={i}  [{n_agree[i]}/{len(names)} agree]")
        push(f"        {ref.loc[i][:96]}")
        push(f"        {votes}")

    push("\n\n[7] READING THESE NUMBERS")
    push("    Cohen's Kappa is agreement AFTER removing the agreement you would")
    push("    expect from two people guessing with the same label frequencies.")
    push("    That is why it is always lower than the raw percentage, and why it")
    push("    is the honest figure to quote.")
    push("      < 0.20 slight | 0.21-0.40 fair | 0.41-0.60 moderate")
    push("      0.61-0.80 substantial | > 0.80 almost perfect   (Landis & Koch 1977)")

    report = "\n".join(out)
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, "agreement_scores.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
