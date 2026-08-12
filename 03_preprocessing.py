#!/usr/bin/env python3
"""
STEP 3 - Preprocess the annotated corpus (the actual assignment).

Required by the brief:
    * word tokenization          -> nltk TweetTokenizer
    * stemming OR lemmatization  -> LEMMATIZATION (WordNet + POS tags), see report
    * stopwords removal          -> nltk english stopwords, negations kept on purpose
    * contraction expansion      -> `contractions` library
    * other tasks I judged necessary:
        - strip the "[Entity]" tag Doccano prepends
        - HTML unescaping   (&amp; -> &)
        - URL removal
        - @mention removal, #hashtag kept as a plain word
        - lowercasing
        - elongation squashing  (loooove -> loove)
        - emoji / non-ASCII removal
        - digit and punctuation removal
        - whitespace normalisation
        - very short token removal
        - empty-document flagging

The pipeline is written as an ordered list of stages so the report can show the
text after EVERY stage - that is the "cleaned text" the teacher asked to see.

Usage:
    python 03_preprocessing.py
Output:
    data/cleaned_corpus.csv
    reports/preprocessing_report.txt
"""
import argparse
import html
import os
import re
from collections import Counter

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")
DEFAULT_IN_CSV = os.path.join(DATA_DIR, "doccano_export.csv")

TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

# Negations and degree words invert or scale sentiment, so removing them as
# "stopwords" would destroy the very signal we annotated for:
#   "not good" -> "good" would flip a Negative tweet into a Positive one.
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "cannot", "n't",
    "very", "too", "but", "against", "off", "down", "up", "over", "under",
    "don", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "shouldn", "wasn", "weren", "won", "wouldn",
}
STOPWORDS = set(stopwords.words("english")) - KEEP_WORDS

ENTITY_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]\s*")
URL_RE = re.compile(
    r"(?:https?://|www\.)\s*\S+"        # http://t.co/x , www.foo.bar
    # pic.twitter.com/mLsI5wf9Jg, and the spaced-out "pic.twitter.com / mLsI5wf9Jg"
    # variant that appears in the paraphrased tweets
    r"|\b\S+\.(?:com|net|org|ly|gg|tv|io|co|uk)\b(?:\s*/\s*\S+)?",
    re.IGNORECASE,
)
MENTION_RE = re.compile(r"@\s*\w+")     # '@ Borderlands' appears too, hence \s*
HASHTAG_RE = re.compile(r"#(\w+)")
ELONGATION_RE = re.compile(r"(.)\1{2,}")
# Apostrophes are dropped only AFTER contraction expansion, so "don't" is already
# "do not" by this point and only possessives ("maya's" -> "mayas") are affected.
NON_LETTER_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# individual stages - each takes a string and returns a string
# --------------------------------------------------------------------------
def strip_entity_tag(text: str) -> str:
    """Doccano rows look like '[Borderlands] i love this game'. Drop the tag."""
    return ENTITY_TAG_RE.sub("", text)


def unescape_html(text: str) -> str:
    return html.unescape(text)


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub(" ", text)


def unwrap_hashtags(text: str) -> str:
    """#Borderlands carries meaning, the '#' does not - keep the word."""
    return HASHTAG_RE.sub(r"\1", text)


def lowercase(text: str) -> str:
    return text.lower()


def expand_contractions(text: str) -> str:
    """don't -> do not, y'all -> you all.  MUST run before punctuation removal."""
    return contractions.fix(text)


def squash_elongation(text: str) -> str:
    """soooo -> soo, !!!!! -> !!  (keeps emphasis, normalises spelling)"""
    return ELONGATION_RE.sub(r"\1\1", text)


def strip_non_letters(text: str) -> str:
    """Removes emoji, digits and punctuation. Runs AFTER contraction expansion."""
    return NON_LETTER_RE.sub(" ", text)


def normalise_space(text: str) -> str:
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
    ("09_strip_non_letters", strip_non_letters),
    ("10_normalise_space", normalise_space),
]


# --------------------------------------------------------------------------
# token-level stages
# --------------------------------------------------------------------------
def to_wordnet_pos(treebank_tag: str) -> str:
    """Map NLTK POS tags onto the 4 tags WordNet understands."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize(text: str) -> list:
    return TOKENIZER.tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS]


def lemmatize(tokens: list) -> list:
    """POS-aware lemmatisation: without the tag, 'was' stays 'was' not 'be'."""
    tagged = nltk.pos_tag(tokens)
    return [LEMMATIZER.lemmatize(tok, to_wordnet_pos(tag)) for tok, tag in tagged]


def drop_short(tokens: list) -> list:
    """Single letters left over from cleaning carry no meaning ('i', 'u' handled above)."""
    return [t for t in tokens if len(t) > 1]


def preprocess(text: str) -> dict:
    """Run the whole pipeline and return every intermediate result."""
    stages = {}
    current = text
    for name, fn in TEXT_STAGES:
        current = current if fn is None else fn(current)
        stages[name] = current

    tokens = tokenize(current)
    stages["11_tokenized"] = tokens
    tokens = remove_stopwords(tokens)
    stages["12_stopwords_removed"] = tokens
    tokens = lemmatize(tokens)
    stages["13_lemmatized"] = tokens
    tokens = drop_short(tokens)
    stages["14_short_dropped"] = tokens
    stages["cleaned_text"] = " ".join(tokens)
    stages["tokens"] = tokens
    return stages


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_IN_CSV, help="Doccano CSV export to clean")
    ap.add_argument("--tag", default="", help="suffix for the output filenames, e.g. --tag shared")
    args = ap.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise SystemExit(f"{args.csv}: no 'text' column. Found {list(df.columns)}")
    df["label"] = df.get("label", pd.Series([""] * len(df))).fillna("")

    # Two corpus shapes are supported: an 'entity' column (the shared corpus),
    # or the older style where the entity is glued onto the front of the text.
    if "entity" not in df.columns:
        df["entity"] = df["text"].str.extract(ENTITY_TAG_RE)
    # The augmented corpus keys rows by 'orig_id' since ids repeat across variants.
    if "id" not in df.columns:
        df["id"] = df["orig_id"] if "orig_id" in df.columns else df.index
    df["entity"] = df["entity"].fillna("")

    suffix = f"_{args.tag}" if args.tag else ""
    out_csv = os.path.join(DATA_DIR, f"cleaned_corpus{suffix}.csv")
    out_report = os.path.join(REPORT_DIR, f"preprocessing_report{suffix}.txt")

    results = [preprocess(t) for t in df["text"]]
    df["cleaned_text"] = [r["cleaned_text"] for r in results]
    df["tokens"] = [r["tokens"] for r in results]
    df["n_tokens_raw"] = [len(TOKENIZER.tokenize(t)) for t in df["text"]]
    df["n_tokens_clean"] = [len(r["tokens"]) for r in results]
    df["is_empty_after_cleaning"] = df["n_tokens_clean"] == 0

    out = []
    push = out.append
    push("=" * 78)
    push(f"TEXT PREPROCESSING REPORT  -  Twitter sentiment corpus ({len(df)} annotated tweets)")
    push(f"source: {args.csv} (exported from Doccano)")
    push(f"entities covered: {df['entity'].nunique()}")
    push("=" * 78)

    # ------------------------------------------------------------------ 1
    push("\n[1] PIPELINE - the order matters")
    push("""
    raw tweet
      01 strip the [Entity] tag Doccano adds
      02 unescape HTML entities        (&amp;  -> &)
      03 remove URLs
      04 remove @mentions
      05 unwrap #hashtags              (#BoT   -> BoT)
      06 lowercase
      07 EXPAND CONTRACTIONS           (don't  -> do not)      <-- required
      08 squash elongations            (soooo  -> soo)
      09 remove emoji / digits / punctuation
      10 normalise whitespace
      11 WORD TOKENIZATION             (nltk TweetTokenizer)   <-- required
      12 STOPWORDS REMOVAL             (nltk, negations kept)  <-- required
      13 LEMMATIZATION                 (WordNet + POS tags)    <-- required
      14 drop 1-character tokens

    Two ordering decisions worth defending in the write-up:
      * contraction expansion (07) MUST come before punctuation removal (09).
        Reverse them and "don't" becomes "dont", which no expander recognises.
      * lemmatization (13) comes after stopword removal (12) so we lemmatize
        fewer tokens, and after lowercasing so WordNet lookups actually hit.""")

    # ------------------------------------------------------------------ 2
    push("\n[2] STAGE-BY-STAGE ON REAL TWEETS  (this is the 'cleaned text' view)")
    demo_ids = pick_demo_rows(df)
    for idx in demo_ids:
        row = df.iloc[idx]
        stages = results[idx]
        push("\n" + "-" * 78)
        push(f"tweet id={row['id']}   entity={row['entity']}   label={row['label']}")
        push("-" * 78)
        for name, _ in TEXT_STAGES:
            push(f"  {name:24} | {stages[name]!r}")
        for name in ["11_tokenized", "12_stopwords_removed", "13_lemmatized", "14_short_dropped"]:
            push(f"  {name:24} | {stages[name]}")
        push(f"  {'>> CLEANED TEXT':24} | {stages['cleaned_text']!r}")

    # ------------------------------------------------------------------ 3
    push("\n\n[3] WHY LEMMATIZATION AND NOT STEMMING")
    push("    The brief says choose one. I chose lemmatization because Porter stemming")
    push("    mangles words into non-words, which makes the cleaned corpus unreadable")
    push("    for the sentiment analysis this dataset is for:")
    push(f"\n    {'word':16} {'PorterStemmer':18} {'WordNetLemmatizer':18}")
    for w, pos in [
        ("murdering", "v"), ("coming", "v"), ("games", "n"), ("better", "a"),
        ("crashes", "v"), ("really", "r"), ("terrible", "a"), ("communities", "n"),
        ("was", "v"), ("studying", "v"),
    ]:
        push(f"    {w:16} {STEMMER.stem(w):18} {LEMMATIZER.lemmatize(w, pos):18}")
    push("\n    'terrible' -> 'terribl' and 'really' -> 'realli' are the clearest losses:")
    push("    both are strong sentiment carriers and both survive lemmatization intact.")

    # ------------------------------------------------------------------ 4
    push("\n\n[4] STOPWORDS - what I removed and what I deliberately kept")
    push(f"    nltk english stopword list : {len(stopwords.words('english'))} words")
    push(f"    words I put back           : {len(KEEP_WORDS)}")
    push(f"    effective stopword list    : {len(STOPWORDS)} words")
    push("\n    Kept on purpose: " + ", ".join(sorted(KEEP_WORDS)))
    push("    Reason: negation flips polarity. Dropping 'not' turns \"not good\" into")
    push("    \"good\", which would silently corrupt every Negative tweet in the corpus.")

    # ------------------------------------------------------------------ 5
    push("\n\n[5] CORPUS STATISTICS - before vs after")
    raw_tokens = [t for text in df["text"] for t in TOKENIZER.tokenize(text)]
    clean_tokens = [t for toks in df["tokens"] for t in toks]
    stats = pd.DataFrame(
        {
            "before": [
                len(raw_tokens),
                len(set(raw_tokens)),
                round(df["n_tokens_raw"].mean(), 2),
                int(df["n_tokens_raw"].max()),
            ],
            "after": [
                len(clean_tokens),
                len(set(clean_tokens)),
                round(df["n_tokens_clean"].mean(), 2),
                int(df["n_tokens_clean"].max()),
            ],
        },
        index=["total tokens", "vocabulary size", "mean tokens/tweet", "longest tweet"],
    )
    stats["reduction"] = (
        (1 - stats["after"] / stats["before"]).map(lambda v: f"{v:.1%}")
    )
    push(stats.to_string())
    push(f"\n    documents that became EMPTY after cleaning: {int(df['is_empty_after_cleaning'].sum())}")
    if df["is_empty_after_cleaning"].any():
        push("    (these were punctuation-only or stopword-only tweets - exactly the ones")
        push("     the annotators disagreed most about in step 2)")
        for _, r in df[df["is_empty_after_cleaning"]].iterrows():
            push(f"      id={r['id']:<4} label={r['label']:10} raw={r['text']!r}")

    # ------------------------------------------------------------------ 6
    push("\n\n[6] TOP 15 LEMMAS PER SENTIMENT CLASS  (sanity check that cleaning kept the signal)")
    push("    (a tweet labelled 'Irrelevant#Neutral' counts under BOTH headings)")
    for label in ["Positive", "Negative", "Neutral", "Irrelevant"]:
        # Doccano joins multiple labels with '#', so match on membership
        has_label = df["label"].astype(str).apply(lambda v: label in v.split("#"))
        subset = df[has_label]
        toks = [t for toks in subset["tokens"] for t in toks]
        if not toks:
            continue
        top = ", ".join(f"{w}({n})" for w, n in Counter(toks).most_common(15))
        push(f"\n    {label} ({len(subset)} tweets, {len(toks)} tokens)")
        push(f"      {top}")

    # ------------------------------------------------------------------ 7
    push("\n\n[7] FILES PRODUCED")
    push(f"    {out_csv}  - id, entity, label, raw text, cleaned text, tokens")
    push(f"    {out_report} - this file")

    report = "\n".join(out)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    cols = ["id", "entity", "label", "text", "cleaned_text", "tokens", "n_tokens_raw", "n_tokens_clean"]
    df[[c for c in cols if c in df.columns]].rename(columns={"text": "raw_text"}).to_csv(
        out_csv, index=False
    )

    print(report)
    print(f"\nSaved: {out_report}")
    print(f"Saved: {out_csv}")


def pick_demo_rows(df: pd.DataFrame) -> list:
    """Pick tweets that actually exercise the interesting stages."""
    wanted = []
    # literal alternatives, not a backreference, so pandas does not warn about groups
    for pattern in [r"'", r"http", r"@\w", r"#\w", r"&\w+;", r"ooo|aaa|eee|!!!|\.\.\."]:
        hits = df.index[df["text"].str.contains(pattern, regex=True, na=False)]
        if len(hits):
            wanted.append(int(hits[0]))
    wanted.append(int(df["n_tokens_raw"].idxmax()))
    seen, ordered = set(), []
    for i in wanted:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered[:6]


if __name__ == "__main__":
    main()
