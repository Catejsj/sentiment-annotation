#!/usr/bin/env python3
"""
STEP 6 - Check a Doccano export, and compare two of them.

Written for the MULTI-LABEL setup created by 05_setup_doccano_project.py, where
a tweet may carry two labels (Irrelevant+Positive, Neutral+Positive, ...) but
where Positive and Negative may never appear together.

Validation catches:
    * Positive + Negative on the same tweet   (breaks the agreed guideline)
    * tweets left unannotated
    * label names that are not in the agreed set

Comparison reports:
    * exact-match agreement  - both annotators chose the identical label set
    * per-label Cohen's kappa - for each label, did we both apply it? (yes/no)
      This is the standard way to score multi-label agreement, because plain
      Cohen's kappa only works when every item has exactly one label.
    * Jaccard overlap        - partial credit when label sets overlap but differ

Usage:
    python 06_validate_and_compare.py mine.csv
    python 06_validate_and_compare.py mine.csv theirs.csv
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID = ["Positive", "Negative", "Neutral", "Irrelevant"]
FORBIDDEN_PAIR = {"Positive", "Negative"}


def interpret(k: float) -> str:
    """Landis & Koch (1977)."""
    if k != k:
        return "undefined (label never used by either annotator)"
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
    if "text" not in df.columns:
        raise SystemExit(f"{path}: no 'text' column. Found {list(df.columns)}")
    if "label" not in df.columns:
        raise SystemExit(f"{path}: no 'label' column. Found {list(df.columns)}")
    # Doccano joins multiple labels with '#'
    df["labels"] = (
        df["label"].fillna("").astype(str)
        .apply(lambda v: frozenset(p for p in v.split("#") if p))
    )
    key = "id" if "id" in df.columns else ("row" if "row" in df.columns else None)
    if key is None:
        raise SystemExit(f"{path}: needs an 'id' or 'row' column to match documents.")
    df["_key"] = df[key]
    return df


def validate(df: pd.DataFrame, path: str) -> bool:
    print(f"\n--- validating {path} ({len(df)} rows) ---")
    clean = True

    blank = df[df["labels"].apply(len) == 0]
    if len(blank):
        clean = False
        print(f"  [!] {len(blank)} tweets are NOT annotated yet")
        for _, r in blank.head(5).iterrows():
            print(f"      id={r['_key']}  {str(r['text'])[:60]!r}")
        if len(blank) > 5:
            print(f"      ... and {len(blank) - 5} more")
    else:
        print("  ok  every tweet is annotated")

    bad = df[df["labels"].apply(lambda s: FORBIDDEN_PAIR <= set(s))]
    if len(bad):
        clean = False
        print(f"  [!] {len(bad)} tweets have BOTH Positive and Negative - not allowed")
        for _, r in bad.iterrows():
            print(f"      id={r['_key']}  {str(r['text'])[:60]!r}")
        print("      Fix these in Doccano: keep whichever feeling dominates.")
    else:
        print("  ok  no Positive+Negative conflicts")

    unknown = sorted({l for s in df["labels"] for l in s} - set(VALID))
    if unknown:
        clean = False
        print(f"  [!] unexpected label names: {unknown}")
    else:
        print("  ok  all label names are in the agreed set")

    multi = df[df["labels"].apply(len) > 1]
    print(f"  fyi {len(multi)} tweets carry two labels")
    counts = pd.Series([l for s in df["labels"] for l in s]).value_counts()
    print("  label usage:", counts.to_dict())
    return clean


def compare(a: pd.DataFrame, b: pd.DataFrame, pa: str, pb: str) -> None:
    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)

    merged = a.merge(b, on="_key", suffixes=("_a", "_b"))
    print(f"  matched {len(merged)} documents on their id")
    if len(merged) != len(a) or len(merged) != len(b):
        print(f"  [!] {pa} has {len(a)} rows, {pb} has {len(b)}, only {len(merged)} line up.")
        print("      The two projects were not built from the same CSV.")
        if not len(merged):
            return
    mismatch = merged[merged["text_a"].astype(str) != merged["text_b"].astype(str)]
    if len(mismatch):
        print(f"  [!] {len(mismatch)} matched ids have DIFFERENT text - wrong corpus, stopping.")
        return
    print("  ok  matched ids all carry identical text")

    exact = (merged["labels_a"] == merged["labels_b"]).mean()
    jac = merged.apply(
        lambda r: len(r["labels_a"] & r["labels_b"]) / len(r["labels_a"] | r["labels_b"])
        if (r["labels_a"] | r["labels_b"]) else 1.0,
        axis=1,
    ).mean()

    print(f"\n  exact-match agreement : {exact:.1%}")
    print(f"  Jaccard overlap       : {jac:.3f}   (partial credit for partial overlap)")

    print("\n  per-label Cohen's kappa - 'did we both apply this label?'")
    print(f"    {'label':12} {'kappa':>7}  {'mine':>5} {'theirs':>7}  interpretation")
    kappas = []
    for label in VALID:
        ya = merged["labels_a"].apply(lambda s: label in s)
        yb = merged["labels_b"].apply(lambda s: label in s)
        if not ya.any() and not yb.any():
            print(f"    {label:12} {'--':>7}  {0:5} {0:7}  never used by either of us")
            continue
        k = cohen_kappa_score(ya, yb)
        kappas.append(k)
        print(f"    {label:12} {k:7.3f}  {int(ya.sum()):5} {int(yb.sum()):7}  {interpret(k)}")

    if kappas:
        mean_k = sum(kappas) / len(kappas)
        print(f"\n  mean per-label kappa  : {mean_k:.3f}  ({interpret(mean_k)})")
        print("  Quote this as the headline agreement number for a multi-label task.")

    # plain Cohen's kappa is still valid on the rows where we both used one label
    single = merged[
        (merged["labels_a"].apply(len) == 1) & (merged["labels_b"].apply(len) == 1)
    ]
    if len(single) > 1:
        sa = single["labels_a"].apply(lambda s: next(iter(s)))
        sb = single["labels_b"].apply(lambda s: next(iter(s)))
        k = cohen_kappa_score(sa, sb, labels=VALID)
        print(f"\n  standard Cohen's kappa on the {len(single)} single-label rows: {k:.3f} ({interpret(k)})")

    dis = merged[merged["labels_a"] != merged["labels_b"]]
    print(f"\n  we disagreed on {len(dis)} tweets:")
    for _, r in dis.head(15).iterrows():
        mine = "+".join(sorted(r["labels_a"])) or "(blank)"
        theirs = "+".join(sorted(r["labels_b"])) or "(blank)"
        print(f"    id={r['_key']:<5} me={mine:22} them={theirs:22} {str(r['text_a'])[:45]!r}")
    if len(dis) > 15:
        print(f"    ... and {len(dis) - 15} more")


def main() -> None:
    args = sys.argv[1:]
    if not 1 <= len(args) <= 2:
        raise SystemExit(__doc__)

    a = load(args[0])
    ok = validate(a, args[0])

    if len(args) == 1:
        print("\nValidation only. Pass a second CSV to compare two annotators.")
        if not ok:
            print("Fix the issues above before comparing.")
        return

    b = load(args[1])
    validate(b, args[1])
    compare(a, b, args[0], args[1])


if __name__ == "__main__":
    main()
