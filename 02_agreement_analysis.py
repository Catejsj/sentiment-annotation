#!/usr/bin/env python3
"""
STEP 2 - Compare the annotations with EXTERNAL PYTHON LIBRARIES.

Libraries used (this is the point of the exercise - no hand-written formulas):
    scikit-learn  -> cohen_kappa_score, confusion_matrix, classification_report
    statsmodels   -> fleiss_kappa, aggregate_raters
    pandas        -> tables

Raters compared:
    Doccano  = my own annotation, exported in step 1 (data/doccano_export.csv)
    A1..A7   = the other annotation passes (annotations/A*.json)
    Gold     = the original Kaggle labels shipped with the tweets

Usage:
    python 02_agreement_analysis.py
Output:
    reports/agreement_report.txt
    data/all_annotations.csv
"""
import csv
import json
import os
from itertools import combinations

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

HERE = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(HERE, "annotations")
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")

LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]
OTHERS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def interpret(k: float) -> str:
    """Landis & Koch (1977) benchmark scale for kappa."""
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


def load_annotations() -> pd.DataFrame:
    """Build one tidy table: one row per tweet, one column per rater."""
    with open(os.path.join(ANN_DIR, "tweets_100.json"), encoding="utf-8") as f:
        tweets = json.load(f)

    with open(os.path.join(DATA_DIR, "doccano_export.csv"), encoding="utf-8") as f:
        doccano = [row["label"] for row in csv.DictReader(f)]

    if len(doccano) != len(tweets):
        raise SystemExit(f"Doccano has {len(doccano)} rows but there are {len(tweets)} tweets")

    df = pd.DataFrame(
        {
            "id": [t["id"] for t in tweets],
            "entity": [t["entity"] for t in tweets],
            "text": [t["text"] for t in tweets],
            "Doccano": doccano,
        }
    )
    for name in OTHERS:
        with open(os.path.join(ANN_DIR, f"{name}.json"), encoding="utf-8") as f:
            df[name] = json.load(f)
    df["Gold"] = [t["gold"] for t in tweets]
    return df


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_annotations()
    raters = ["Doccano"] + OTHERS

    out = []
    push = out.append
    push("=" * 78)
    push("INTER-ANNOTATOR AGREEMENT  -  100 tweets, 8 annotation passes")
    push("computed with scikit-learn + statsmodels")
    push("=" * 78)

    # ---------------------------------------------------------------- 1
    push("\n[1] LABEL DISTRIBUTION PER RATER")
    dist = pd.DataFrame({r: df[r].value_counts() for r in raters + ["Gold"]})
    dist = dist.reindex(LABELS).fillna(0).astype(int)
    push(dist.to_string())

    # ---------------------------------------------------------------- 2
    push("\n\n[2] PAIRWISE COHEN'S KAPPA  (sklearn.metrics.cohen_kappa_score)")
    push("    Every rater compared against every other rater.")
    matrix = pd.DataFrame(index=raters, columns=raters, dtype=float)
    pairs = {}
    for a, b in combinations(raters, 2):
        k = cohen_kappa_score(df[a], df[b], labels=LABELS)
        pairs[(a, b)] = k
        matrix.loc[a, b] = matrix.loc[b, a] = k
    for r in raters:
        matrix.loc[r, r] = 1.0
    push(matrix.round(3).to_string())

    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)
    mean_k = sum(pairs.values()) / len(pairs)
    push(f"\n    mean pairwise kappa : {mean_k:.3f}  ({interpret(mean_k)})")
    push(f"    strongest pair      : {ranked[0][0][0]} vs {ranked[0][0][1]} = {ranked[0][1]:.3f}")
    push(f"    weakest pair        : {ranked[-1][0][0]} vs {ranked[-1][0][1]} = {ranked[-1][1]:.3f}")
    push(f"    pairs >= 0.61 (substantial or better): {sum(1 for v in pairs.values() if v >= 0.61)}/{len(pairs)}")

    push("\n    How MY Doccano annotation compares with each of the others:")
    mine = sorted(
        ((r, pairs.get(("Doccano", r), pairs.get((r, "Doccano")))) for r in OTHERS),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for r, k in mine:
        agree = (df["Doccano"] == df[r]).mean()
        push(f"      Doccano vs {r}: kappa={k:6.3f}  raw agreement={agree:5.1%}  ({interpret(k)})")

    # ---------------------------------------------------------------- 3
    push("\n\n[3] FLEISS' KAPPA  (statsmodels.stats.inter_rater.fleiss_kappa)")
    push("    One number for all 8 raters at once.")
    table, categories = aggregate_raters(df[raters].to_numpy())
    fk = fleiss_kappa(table, method="fleiss")
    push(f"    categories found : {list(categories)}")
    push(f"    Fleiss' kappa    : {fk:.3f}  ({interpret(fk)})")

    # ---------------------------------------------------------------- 4
    push("\n\n[4] AGREEMENT WITH THE KAGGLE GOLD LABELS")
    push("    (Gold has no 'Irrelevant' class, so Irrelevant votes always count as errors.)")
    push(f"    {'rater':10} {'kappa':>8} {'accuracy':>10}   interpretation")
    for r in raters:
        k = cohen_kappa_score(df[r], df["Gold"], labels=LABELS)
        acc = (df[r] == df["Gold"]).mean()
        push(f"    {r:10} {k:8.3f} {acc:9.1%}   {interpret(k)}")

    # ---------------------------------------------------------------- 5
    push("\n\n[5] MY ANNOTATION vs GOLD  -  confusion matrix (sklearn)")
    push("    rows = what I said in Doccano, columns = the gold label")
    cm = confusion_matrix(df["Doccano"], df["Gold"], labels=LABELS)
    push(pd.DataFrame(cm, index=[f"me:{l}" for l in LABELS], columns=[f"gold:{l}" for l in LABELS]).to_string())
    push("\n" + classification_report(df["Gold"], df["Doccano"], labels=LABELS, zero_division=0))

    # ---------------------------------------------------------------- 6
    push("\n[6] PER-TWEET AGREEMENT")
    push("    e.g. if a tweet is Negative, did the other annotators also say Negative?")
    votes = df[raters]
    majority = votes.mode(axis=1)[0]
    n_agree = votes.eq(majority, axis=0).sum(axis=1)
    df["majority"] = majority
    df["n_agree"] = n_agree
    push(f"    unanimous (8/8)      : {(n_agree == 8).sum():3d} tweets")
    push(f"    strong    (6-7/8)    : {n_agree.between(6, 7).sum():3d} tweets")
    push(f"    split     (<= 5/8)   : {(n_agree <= 5).sum():3d} tweets")
    push(f"    majority label matches gold on {(majority == df['Gold']).sum()}/100 tweets")

    push("\n    10 most disagreed-on tweets:")
    worst = df.nsmallest(10, "n_agree")
    for _, row in worst.iterrows():
        push(f"      #{row['id']:<3} [{row['entity'][:14]:14}] {row['n_agree']}/8 agree -> majority={row['majority']:10} gold={row['Gold']}")
        push(f"           {row['text'][:96]}")
        push(f"           votes: {', '.join(f'{r}={row[r]}' for r in raters)}")

    push("\n\n[7] WHAT THIS MEANS")
    push(f"    * Mean pairwise Cohen's kappa {mean_k:.3f} and Fleiss' kappa {fk:.3f} both land in the")
    push(f"      '{interpret(fk)}' band, so the 8 passes broadly agree but are not interchangeable.")
    push("    * The disagreement is concentrated in Neutral vs Irrelevant: raters who used")
    push("      'Irrelevant' heavily (A5, and to a lesser extent A4/A6/Doccano) drag kappa down,")
    push("      because Gold and A1/A2/A3/A7 never use that class at all.")
    push("    * Sarcasm and violent-but-playful gaming talk ('i will murder you all') are the")
    push("      other systematic source of disagreement.")
    push("    * Fix for a real project: sharpen the annotation guideline so 'Irrelevant' has an")
    push("      explicit test, then re-annotate the split tweets.")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, "agreement_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    df.to_csv(os.path.join(DATA_DIR, "all_annotations.csv"), index=False)
    print(report)
    print(f"\nSaved: {path}")
    print(f"Saved: {os.path.join(DATA_DIR, 'all_annotations.csv')}")


if __name__ == "__main__":
    main()
