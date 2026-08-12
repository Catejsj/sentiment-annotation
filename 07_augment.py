#!/usr/bin/env python3
"""
STEP 7 - Data augmentation, run BEFORE preprocessing (Group 2).

The brief:
    "please also perform data augmentation as well before preprocessing.
     Group 1 and Group 2: Please use random swapping, random insertion, and
     random deletion. (you can use TextAttack framework)"

So this reads the RAW annotated export and writes an enlarged raw corpus.
03_preprocessing.py is then run on the OUTPUT of this script, giving the order
the brief asks for:

    annotate -> AUGMENT -> preprocess

TextAttack transformations used, one per required technique:
    random swapping  -> WordInnerSwapRandom
    random insertion -> WordInsertionRandomSynonym   (WordNet synonyms)
    random deletion  -> WordDeletion

NEGATION GUARD
    Random deletion can delete the word "not", which silently flips a Negative
    tweet into a Positive one and poisons the training data. Every variant is
    checked: if the augmentation dropped a negation that was in the original,
    it is retried, and the variant is discarded if it never comes back clean.

Usage:
    python 07_augment.py
    python 07_augment.py --csv exports/sherlock_annotations.csv --variants 2
"""
import argparse
import os
import random

import pandas as pd
from textattack.augmentation import Augmenter
from textattack.transformations import (
    WordDeletion,
    WordInnerSwapRandom,
    WordInsertionRandomSynonym,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")
EXPORT_DIR = os.path.join(HERE, "exports")


def default_csv() -> str:
    """Use the single CSV in exports/ so renaming the export does not break this."""
    found = sorted(f for f in os.listdir(EXPORT_DIR) if f.endswith(".csv")) if os.path.isdir(EXPORT_DIR) else []
    if len(found) == 1:
        return os.path.join(EXPORT_DIR, found[0])
    return os.path.join(EXPORT_DIR, "my_annotations.csv")

SEED = 42
# Words whose loss would invert the sentiment of the tweet.
NEGATIONS = {
    "not", "no", "never", "nor", "none", "nothing", "cannot", "cant", "wont",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt",
    "havent", "shouldnt", "wouldnt", "couldnt", "aint",
}

# WordNet synonym sets contain anatomical and vulgar lemmas, so random synonym
# insertion happily drops "penis" into a tweet about a video game. Any variant
# that INTRODUCES one of these is rejected; words already in the source tweet
# are left alone, because censoring the corpus would change the data.
BLOCKED = {
    "penis", "cock", "prick", "dick", "pussy", "cunt", "tit", "tits", "boob",
    "boobs", "arse", "ass", "asshole", "bastard", "bitch", "whore", "slut",
    "fuck", "shit", "crap", "piss", "turd", "bugger", "wanker", "screw",
    "nigger", "faggot", "queer", "retard", "spastic", "coon", "kike",
}


def negations_in(text: str) -> set:
    cleaned = "".join(c if c.isalpha() or c.isspace() else " " for c in text.lower())
    return {w for w in cleaned.split() if w in NEGATIONS}


def words_of(text: str) -> set:
    cleaned = "".join(c if c.isalpha() or c.isspace() else " " for c in text.lower())
    return set(cleaned.split())


def introduces_slur(original: str, candidate: str) -> bool:
    """True if augmentation added a blocked word that was not already there."""
    return bool((words_of(candidate) & BLOCKED) - (words_of(original) & BLOCKED))


TECHNIQUES = {
    "swap": WordInnerSwapRandom,
    "insert": WordInsertionRandomSynonym,
    "delete": WordDeletion,
}


def build_augmenters(pct: float, variants: int) -> dict:
    return {
        name: Augmenter(
            transformation=cls(),
            pct_words_to_swap=pct,
            transformations_per_example=variants,
        )
        for name, cls in TECHNIQUES.items()
    }


def safe_augment(augmenter, text: str, want: int, attempts: int = 4) -> list:
    """Produce `want` variants that never lose a negation present in the source."""
    keep, source_negs = [], negations_in(text)
    for _ in range(attempts):
        for cand in augmenter.augment(text):
            if cand == text:
                continue                                  # no-op, useless
            if not source_negs <= negations_in(cand):
                continue                                  # dropped a negation
            if introduces_slur(text, cand):
                continue                                  # WordNet vulgarity
            if cand not in keep:
                keep.append(cand)
            if len(keep) == want:
                return keep
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=default_csv())
    ap.add_argument("--variants", type=int, default=2, help="variants per technique per tweet")
    ap.add_argument("--pct", type=float, default=0.15, help="fraction of words each op touches")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "augmented_corpus.csv"))
    args = ap.parse_args()

    random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    for col in ("text", "label"):
        if col not in df.columns:
            raise SystemExit(f"{args.csv}: missing '{col}' column. Found {list(df.columns)}")
    df["label"] = df["label"].fillna("")

    print(f"input      : {args.csv}  ({len(df)} tweets)")
    print(f"techniques : {', '.join(TECHNIQUES)}   (TextAttack)")
    print(f"variants   : {args.variants} per technique per tweet")
    print(f"pct words  : {args.pct}")
    print("order      : AUGMENT first, preprocessing runs afterwards\n")

    augmenters = build_augmenters(args.pct, args.variants)

    rows, stats = [], {k: 0 for k in TECHNIQUES}
    dropped = {k: 0 for k in TECHNIQUES}

    for i, row in df.iterrows():
        text = str(row["text"])
        base = {
            "orig_id": row.get("id", i),
            "entity": row.get("entity", ""),
            "source_id": row.get("source_id", ""),
            "label": row["label"],
        }
        # the original always stays in the corpus
        rows.append({**base, "technique": "original", "variant": 0, "text": text})

        for name, aug in augmenters.items():
            variants = safe_augment(aug, text, args.variants)
            for v_i, v in enumerate(variants, start=1):
                rows.append({**base, "technique": name, "variant": v_i, "text": v})
            stats[name] += len(variants)
            dropped[name] += args.variants - len(variants)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(df)} tweets augmented")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    # ------------------------------------------------------------- report
    lines = []
    push = lines.append
    push("=" * 74)
    push("DATA AUGMENTATION REPORT  (Group 2: random swap / insertion / deletion)")
    push("=" * 74)
    push(f"\nsource corpus : {args.csv}")
    push(f"framework     : TextAttack 0.3.10")
    push(f"random seed   : {SEED}  (rerunning reproduces this file exactly)")
    push("\nPIPELINE ORDER")
    push("    annotate  ->  AUGMENT (this script)  ->  preprocess (03_preprocessing.py)")
    push("    The brief says augmentation comes BEFORE preprocessing, so the")
    push("    augmenters see natural English. Synonym insertion in particular needs")
    push("    real words - run it after lemmatization and 'was' is already 'be',")
    push("    so WordNet lookups degrade.")

    push("\n[1] TECHNIQUES")
    push(f"    {'technique':12} {'TextAttack class':30} {'rows made':>10} {'discarded':>10}")
    for name, cls in TECHNIQUES.items():
        push(f"    {name:12} {cls.__name__:30} {stats[name]:>10} {dropped[name]:>10}")
    push(f"\n    'discarded' = variants thrown away because they were identical to the")
    push(f"    original or would have deleted a negation word.")

    push("\n[2] CORPUS SIZE")
    push(f"    original tweets   : {len(df)}")
    push(f"    augmented rows    : {len(out) - len(df)}")
    push(f"    total corpus      : {len(out)}   ({len(out) / len(df):.1f}x bigger)")

    push("\n[3] LABEL BALANCE  (augmentation must not change the class ratios)")
    before = df["label"].value_counts()
    after = out[out.technique != "original"]["label"].value_counts()
    comp = pd.DataFrame({"original": before, "augmented": after}).fillna(0).astype(int)
    comp["orig %"] = (comp["original"] / comp["original"].sum() * 100).round(1)
    comp["aug %"] = (comp["augmented"] / comp["augmented"].sum() * 100).round(1)
    push(comp.to_string())

    push("\n[4] NEGATION SAFETY CHECK")
    neg_rows = out[out["text"].apply(lambda t: bool(negations_in(str(t))))]
    orig_neg = df[df["text"].apply(lambda t: bool(negations_in(str(t))))]
    push(f"    original tweets containing a negation : {len(orig_neg)}")
    push(f"    rows in final corpus with a negation  : {len(neg_rows)}")
    lost = 0
    for _, r in out[out.technique != "original"].iterrows():
        src = df[df.get("id", df.index) == r["orig_id"]]
        if len(src) and not negations_in(str(src.iloc[0]["text"])) <= negations_in(str(r["text"])):
            lost += 1
    push(f"    variants that LOST a negation         : {lost}   <- must be 0")

    push("\n[4b] SYNONYM HYGIENE CHECK")
    push("    WordNet synonym sets include vulgar lemmas, so raw random insertion")
    push("    injects profanity into unrelated tweets. Variants that introduce a")
    push("    blocked word are rejected and retried.")
    introduced = sum(
        1
        for _, r in out[out.technique != "original"].iterrows()
        for src in [df[df.get("id", df.index) == r["orig_id"]]]
        if len(src) and introduces_slur(str(src.iloc[0]["text"]), str(r["text"]))
    )
    push(f"    blocklist size                        : {len(BLOCKED)} words")
    push(f"    variants that INTRODUCED a slur       : {introduced}   <- must be 0")

    push("\n[5] EXAMPLES  (same tweet, three techniques)")
    sample_id = out[out.technique == "delete"]["orig_id"].iloc[0]
    grp = out[out["orig_id"] == sample_id]
    for _, r in grp.iterrows():
        push(f"    {r['technique']:9} v{r['variant']} | {str(r['text'])[:105]}")

    push("\n[6] NEXT STEP")
    push("    Run the preprocessing pipeline on THIS file, not on the raw export:")
    push(f"        python 03_preprocessing.py --csv {args.out} --tag augmented")

    report = "\n".join(lines)
    path = os.path.join(REPORT_DIR, "augmentation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nSaved: {args.out}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
