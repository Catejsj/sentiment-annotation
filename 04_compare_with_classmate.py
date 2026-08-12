#!/usr/bin/env python3
"""
STEP 4 - Compare my Doccano CSV against somebody else's Doccano CSV.

The important part is the SAFETY CHECK before any number is computed:
two annotation files can only be compared if they cover the same tweets in the
same order. This script proves that first, and refuses to report a kappa if the
files do not line up.

Usage:
    python 04_compare_with_classmate.py their_file.csv
    python 04_compare_with_classmate.py mine.csv theirs.csv
"""
import re
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ENTITY_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
DEFAULT_MINE = "data/doccano_export.csv"


def normalise(series: pd.Series) -> pd.Series:
    """Make two exports comparable: drop the [Entity] tag, case and spacing."""
    return (
        series.fillna("")
        .str.replace(ENTITY_TAG_RE, "", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def find_column(df: pd.DataFrame, candidates: list, kind: str, path: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"Could not find a {kind} column in {path}.\n"
        f"  looked for : {candidates}\n"
        f"  file has   : {list(df.columns)}\n"
        f"Rename the column, or edit the candidate list in this script."
    )


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = find_column(df, ["text", "Text", "raw_text", "sentence", "document"], "text", path)
    label_col = find_column(df, ["label", "Label", "labels", "category", "sentiment"], "label", path)
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": df[label_col].astype(str)})
    out["key"] = normalise(out["text"])
    return out


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 1:
        mine_path, theirs_path = DEFAULT_MINE, args[0]
    elif len(args) == 2:
        mine_path, theirs_path = args
    else:
        raise SystemExit(__doc__)

    mine, theirs = load(mine_path), load(theirs_path)
    print(f"mine   : {mine_path}   ({len(mine)} rows)")
    print(f"theirs : {theirs_path}   ({len(theirs)} rows)")

    # -------------------------------------------------- check 1: row count
    print("\n[check 1] same number of rows?")
    if len(mine) != len(theirs):
        print(f"  NO  - {len(mine)} vs {len(theirs)}. The files annotate different corpora.")
        print("        Fix this before comparing; a kappa here would be meaningless.")
        return
    print(f"  yes - both {len(mine)} rows")

    # -------------------------------------------------- check 2: same tweets, same order
    print("\n[check 2] same tweets, in the same order?")
    aligned = mine["key"] == theirs["key"]
    print(f"  {aligned.sum()}/{len(mine)} rows have matching text")
    if not aligned.all():
        bad = (~aligned).to_numpy().nonzero()[0][:5]
        print("  NOT ALIGNED - first few mismatching rows:")
        for i in bad:
            print(f"    row {i}: mine   = {mine['key'][i][:70]!r}")
            print(f"           theirs = {theirs['key'][i][:70]!r}")
        same_set = set(mine["key"]) == set(theirs["key"])
        if same_set:
            print("\n  The two files contain the SAME tweets but in a DIFFERENT ORDER.")
            print("  Sort both by the text column, then run this script again.")
        else:
            only_mine = len(set(mine["key"]) - set(theirs["key"]))
            print(f"\n  {only_mine} of my tweets do not appear in their file at all.")
            print("  You annotated different data - these cannot be compared.")
        return
    print("  yes - safe to compare row by row")

    # -------------------------------------------------- check 3: label sets
    print("\n[check 3] same label set?")
    my_labels, their_labels = set(mine["label"]), set(theirs["label"])
    print(f"  mine   : {sorted(my_labels)}")
    print(f"  theirs : {sorted(their_labels)}")
    only_mine = my_labels - their_labels
    only_theirs = their_labels - my_labels
    if only_mine or only_theirs:
        if only_mine:
            print(f"  NOTE: only I used {sorted(only_mine)}")
        if only_theirs:
            print(f"  NOTE: only they used {sorted(only_theirs)}")
        print("  This is allowed, but it will lower kappa on its own - the guideline")
        print("  was read differently, not just the individual tweets.")
    else:
        print("  yes - identical label sets")

    # -------------------------------------------------- the actual comparison
    labels = sorted(my_labels | their_labels)
    k = cohen_kappa_score(mine["label"], theirs["label"], labels=labels)
    agree = (mine["label"] == theirs["label"]).mean()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  raw agreement    : {agree:.1%}  ({int(agree * len(mine))}/{len(mine)} tweets)")
    print(f"  Cohen's kappa    : {k:.3f}")
    print(f"  interpretation   : {interpret(k)}")
    print("\n  Raw agreement counts lucky matches; kappa subtracts them.")
    print("  That is why kappa is always the lower - and the honest - number.")

    print("\n  confusion matrix (rows = mine, columns = theirs)")
    cm = confusion_matrix(mine["label"], theirs["label"], labels=labels)
    print(pd.DataFrame(cm, index=[f"me:{l}" for l in labels], columns=[f"them:{l}" for l in labels]).to_string())

    disagree = mine["label"] != theirs["label"]
    print(f"\n  we disagreed on {disagree.sum()} tweets:")
    for i in disagree.to_numpy().nonzero()[0][:15]:
        print(f"    row {i:<3} me={mine['label'][i]:11} them={theirs['label'][i]:11} {mine['key'][i][:60]!r}")
    if disagree.sum() > 15:
        print(f"    ... and {disagree.sum() - 15} more")


def interpret(k: float) -> str:
    """Landis & Koch (1977)."""
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


if __name__ == "__main__":
    main()
