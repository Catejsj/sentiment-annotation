#!/usr/bin/env python3
"""
Preprocessing pipeline for the annotated tweet corpus.

Required by the brief: word tokenization, lemmatization (chosen over stemming),
stopword removal, contraction expansion, plus any further cleaning judged
necessary.

Runs AFTER augmentation - see scripts/augment.py.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --csv data/augmented/tweets_augmented.csv --tag augmented
"""
import argparse
import html
import os
import re
from collections import Counter

import contractions
import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "raw", "annotated_tweets.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")

TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
CASED_TOKENIZER = TweetTokenizer(preserve_case=True)
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# Negations and degree words invert or scale sentiment. Removing them as
# "stopwords" would turn "not good" into "good" and silently flip the polarity
# of every Negative tweet in the corpus.
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "cannot", "n't",
    "very", "too", "but", "against", "off", "down", "up", "over", "under",
    "don", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "shouldn", "wasn", "weren", "won", "wouldn",
}
# Twitter boilerplate that carries no sentiment.
TWITTER_NOISE = {"rt", "via", "im", "u", "ur", "amp", "pic", "twitter", "com"}
STOPWORDS = (set(stopwords.words("english")) | TWITTER_NOISE) - KEEP_WORDS

ENTITY_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]\s*")
URL_RE = re.compile(
    r"(?:https?://|www\.)\s*\S+"
    r"|\b\S+\.(?:com|net|org|ly|gg|tv|io|co|uk)\b(?:\s*/\s*\S+)?",
    re.IGNORECASE,
)
MENTION_RE = re.compile(r"@\s*\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
# Apostrophes go only AFTER contraction expansion, so "don't" is already
# "do not" and only possessives ("maya's" -> "mayas") are affected.
NON_LETTER_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")
HAS_LETTER_RE = re.compile(r"[a-z]")
WORDNET_POS = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}


# --------------------------------------------------------------- text stages
def strip_entity_tag(text):
    return ENTITY_TAG_RE.sub("", text)


def unescape_html(text):
    return html.unescape(text)


def remove_urls(text):
    return URL_RE.sub(" ", text)


def remove_mentions(text):
    return MENTION_RE.sub(" ", text)


def unwrap_hashtags(text):
    """#Borderlands carries meaning, the '#' does not."""
    return HASHTAG_RE.sub(r"\1", text)


def lowercase(text):
    return text.lower()


def expand_contractions(text):
    """don't -> do not.  MUST run before punctuation removal."""
    return contractions.fix(text)


def squash_elongation(text):
    """soooo -> soo, keeps the emphasis but normalises the spelling."""
    return ELONGATION_RE.sub(r"\1\1", text)


def strip_symbols(text):
    """Removes emoji and punctuation. Digits survive here, filtered at token level."""
    return NON_LETTER_RE.sub(" ", text)


def normalise_space(text):
    return MULTISPACE_RE.sub(" ", text).strip()


TEXT_STAGES = [
    ("00_raw", None),
    ("01_strip_entity_tag", strip_entity_tag),
    ("02_unescape_html", unescape_html),
    ("03_remove_urls", remove_urls),
    ("04_remove_mentions", remove_mentions),
    ("05_unwrap_hashtags", unwrap_hashtags),
    ("06_lowercase", lowercase),
    ("07_expand_contractions", expand_contractions),
    ("08_squash_elongation", squash_elongation),
    ("09_strip_symbols", strip_symbols),
    ("10_normalise_space", normalise_space),
]


# -------------------------------------------------------------- token stages
def tokenize(text):
    return TOKENIZER.tokenize(text)


def drop_numeric(tokens):
    """Keep tokens containing a letter: 'ps5' and '2k' survive, '2020' does not.

    Bare years and byte counts are noise, but alphanumeric product names are
    exactly the entities this corpus is annotated against.
    """
    return [t for t in tokens if HAS_LETTER_RE.search(t)]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def protected_words(raw_text, entity):
    """Proper nouns must not be lemmatized: 'Borderlands' is a product name, and
    lemmatizing it to 'borderland' breaks the link to the entity column."""
    protected = {w.lower() for w, tag in pos_tag(CASED_TOKENIZER.tokenize(raw_text))
                 if tag in ("NNP", "NNPS")}
    protected.update(re.findall(r"\w+", str(entity).lower()))
    return protected


def lemmatize(tokens, protected):
    """POS-aware lemmatisation; without the tag 'was' stays 'was' instead of 'be'."""
    return [
        w if w in protected else LEMMATIZER.lemmatize(w, WORDNET_POS.get(tag[0], wordnet.NOUN))
        for w, tag in pos_tag(tokens)
    ]


def drop_short(tokens):
    return [t for t in tokens if len(t) > 1]


def preprocess(raw_text, entity):
    """Run every stage and return all intermediate results."""
    stages, current = {}, raw_text
    for name, fn in TEXT_STAGES:
        current = current if fn is None else fn(current)
        stages[name] = current

    tokens = tokenize(current)
    stages["11_tokenized"] = tokens
    tokens = drop_numeric(tokens)
    stages["12_numeric_dropped"] = tokens
    tokens = remove_stopwords(tokens)
    stages["13_stopwords_removed"] = tokens
    tokens = lemmatize(tokens, protected_words(raw_text, entity))
    stages["14_lemmatized"] = tokens
    tokens = drop_short(tokens)
    stages["15_short_dropped"] = tokens
    stages["tokens"] = tokens
    stages["cleaned_text"] = " ".join(tokens)
    return stages


def pick_demo_rows(df):
    """Pick tweets that actually exercise the interesting stages."""
    wanted = []
    for pattern in [r"'", r"http", r"@\w", r"#\w", r"&\w+;", r"ooo|aaa|!!!|\.\.\."]:
        hits = df.index[df["text"].str.contains(pattern, regex=True, na=False)]
        if len(hits):
            wanted.append(int(hits[0]))
    wanted.append(int(df["n_tokens_raw"].idxmax()))
    seen, ordered = set(), []
    for i in wanted:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise SystemExit(f"{args.csv}: no 'text' column. Found {list(df.columns)}")
    df["label"] = df.get("label", pd.Series([""] * len(df))).fillna("")
    if "entity" not in df.columns:
        df["entity"] = df["text"].str.extract(ENTITY_TAG_RE)
    df["entity"] = df["entity"].fillna("")
    if "id" not in df.columns:
        df["id"] = df["orig_id"] if "orig_id" in df.columns else df.index

    results = [preprocess(str(t), e) for t, e in zip(df["text"], df["entity"])]
    df["cleaned_text"] = [r["cleaned_text"] for r in results]
    df["tokens"] = [r["tokens"] for r in results]
    df["n_tokens_raw"] = [len(TOKENIZER.tokenize(str(t))) for t in df["text"]]
    df["n_tokens_clean"] = [len(r["tokens"]) for r in results]

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(PROCESSED_DIR, f"tweets_preprocessed{suffix}.csv")
    stages_csv = os.path.join(PROCESSED_DIR, f"preprocessing_stages{suffix}.csv")
    out_report = os.path.join(REPORT_DIR, f"preprocessing_report{suffix}.txt")

    out, push = [], None
    push = out.append
    push("=" * 78)
    push(f"PREPROCESSING REPORT  -  {len(df)} documents")
    push(f"source: {args.csv}")
    push(f"entities covered: {df['entity'].nunique()}")
    push("=" * 78)

    push("\n[1] PIPELINE")
    push("""
      01 strip [Entity] tag        08 squash elongations (soooo -> soo)
      02 unescape HTML             09 remove emoji / punctuation
      03 remove URLs               10 normalise whitespace
      04 remove @mentions          11 WORD TOKENIZATION      <-- required
      05 unwrap #hashtags          12 drop pure-numeric tokens
      06 lowercase                 13 STOPWORD REMOVAL       <-- required
      07 EXPAND CONTRACTIONS <--   14 LEMMATIZATION          <-- required
         required                  15 drop 1-character tokens

    Ordering decisions worth defending:
      * contraction expansion (07) MUST precede punctuation removal (09).
        Reverse them and "don't" becomes "dont", which no expander recognises.
      * lemmatization (14) runs after stopword removal (13), so fewer tokens
        need tagging, and after lowercasing so WordNet lookups hit.""")

    push("\n[2] STAGE-BY-STAGE ON REAL TWEETS  (the 'cleaned text' view)")
    for idx in pick_demo_rows(df):
        row, stages = df.iloc[idx], results[idx]
        push("\n" + "-" * 78)
        push(f"id={row['id']}   entity={row['entity']}   label={row['label']}")
        push("-" * 78)
        for name, _ in TEXT_STAGES:
            push(f"  {name:24} | {stages[name]!r}")
        for name in ["11_tokenized", "12_numeric_dropped", "13_stopwords_removed",
                     "14_lemmatized", "15_short_dropped"]:
            push(f"  {name:24} | {stages[name]}")
        push(f"  {'>> CLEANED TEXT':24} | {stages['cleaned_text']!r}")

    push("\n\n[3] LEMMATIZATION, NOT STEMMING")
    push("    The brief says choose one. Porter stemming mangles words into")
    push("    non-words, destroying the terms that carry the sentiment:")
    push(f"\n    {'word':16} {'PorterStemmer':18} {'WordNetLemmatizer':18}")
    for w, pos in [("murdering", "v"), ("games", "n"), ("better", "a"), ("really", "r"),
                   ("terrible", "a"), ("communities", "n"), ("was", "v"), ("studying", "v")]:
        push(f"    {w:16} {STEMMER.stem(w):18} {LEMMATIZER.lemmatize(w, pos):18}")
    push("\n    'terrible' -> 'terribl' and 'really' -> 'realli' are the clearest")
    push("    losses: both are strong sentiment carriers, both survive lemmatization.")

    push("\n\n[4] PROPER NOUNS ARE PROTECTED")
    push("    Product and game names are POS-tagged NNP/NNPS and skipped by the")
    push("    lemmatizer, plus every word of the row's entity value. Without this,")
    push("    'Borderlands' becomes 'borderland' and no longer matches the entity")
    push("    column the corpus is annotated against.")

    push("\n\n[5] STOPWORDS - removed vs deliberately kept")
    push(f"    nltk english list      : {len(stopwords.words('english'))}")
    push(f"    + twitter noise        : {len(TWITTER_NOISE)}  {sorted(TWITTER_NOISE)}")
    push(f"    - negations put back   : {len(KEEP_WORDS)}")
    push(f"    effective stopword list: {len(STOPWORDS)}")
    push("\n    Kept on purpose: " + ", ".join(sorted(KEEP_WORDS)))
    push("    Negation flips polarity: dropping 'not' turns \"not good\" into \"good\".")

    push("\n\n[6] CORPUS STATISTICS")
    raw_tokens = [t for text in df["text"] for t in TOKENIZER.tokenize(str(text))]
    clean_tokens = [t for toks in df["tokens"] for t in toks]
    stats = pd.DataFrame(
        {
            "before": [len(raw_tokens), len(set(raw_tokens)),
                       round(df["n_tokens_raw"].mean(), 2), int(df["n_tokens_raw"].max())],
            "after": [len(clean_tokens), len(set(clean_tokens)),
                      round(df["n_tokens_clean"].mean(), 2), int(df["n_tokens_clean"].max())],
        },
        index=["total tokens", "vocabulary size", "mean tokens/doc", "longest doc"],
    )
    stats["reduction"] = (1 - stats["after"] / stats["before"]).map(lambda v: f"{v:.1%}")
    push(stats.to_string())
    empty = int((df["n_tokens_clean"] == 0).sum())
    push(f"\n    documents empty after cleaning: {empty}")
    if empty:
        for _, r in df[df["n_tokens_clean"] == 0].iterrows():
            push(f"      id={r['id']}  label={r['label']}  raw={str(r['text'])[:60]!r}")

    push("\n\n[7] TOP 15 LEMMAS PER CLASS")
    push("    (a document labelled 'Irrelevant#Neutral' counts under BOTH headings)")
    for label in ["Positive", "Negative", "Neutral", "Irrelevant"]:
        subset = df[df["label"].astype(str).apply(lambda v: label in v.split("#"))]
        toks = [t for toks in subset["tokens"] for t in toks]
        if not toks:
            continue
        top = ", ".join(f"{w}({n})" for w, n in Counter(toks).most_common(15))
        push(f"\n    {label} ({len(subset)} docs, {len(toks)} tokens)")
        push(f"      {top}")

    report = "\n".join(out)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    cols = [c for c in ["id", "entity", "label", "text", "cleaned_text", "tokens",
                        "n_tokens_raw", "n_tokens_clean"] if c in df.columns]
    df[cols].rename(columns={"text": "raw_text"}).to_csv(out_csv, index=False)

    pd.DataFrame([
        {"id": df["id"].iloc[i], "label": df["label"].iloc[i],
         **{k: (" ".join(v) if isinstance(v, list) else v)
            for k, v in r.items() if k != "tokens"}}
        for i, r in enumerate(results)
    ]).to_csv(stages_csv, index=False)

    print(report)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {stages_csv}")
    print(f"Saved: {out_report}")


if __name__ == "__main__":
    main()
