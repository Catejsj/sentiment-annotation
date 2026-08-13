#!/usr/bin/env python3
"""
Text representation: Bag-of-Words with CountVectorizer, unigrams through trigrams.

Turns cleaned text into a numeric matrix a model can use. Runs on the
`cleaned_text` column produced by scripts/preprocess.py - vectorising raw text
would throw away the tokenization, stopword removal and lemmatization.

Usage:
    python scripts/vectorize.py
    python scripts/vectorize.py --csv data/processed/tweets_preprocessed_augmented.csv --tag augmented
    python scripts/vectorize.py --min-df 2
"""
import argparse
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "data", "processed", "tweets_preprocessed.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
REPORT_DIR = os.path.join(ROOT, "reports")

LABELS = ["Positive", "Negative", "Neutral", "Irrelevant"]

# The four representations compared in the report.
SETTINGS = [
    ("unigram", (1, 1), "single words: 'good', 'game'"),
    ("bigram", (2, 2), "word pairs: 'not good', 'red dead'"),
    ("trigram", (3, 3), "word triples: 'red dead redemption'"),
    ("uni+bigram", (1, 2), "both, the usual choice for sentiment"),
]


def top_terms(matrix, vocab, n=20):
    """Total count of each term across the whole corpus, highest first."""
    totals = matrix.sum(axis=0).A1
    return sorted(zip(vocab, totals), key=lambda kv: kv[1], reverse=True)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--min-df", type=int, default=1,
                    help="ignore terms appearing in fewer than N documents")
    ap.add_argument("--save-matrix", default="uni+bigram",
                    help="which setting's document-term matrix to write to CSV")
    args = ap.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "cleaned_text" not in df.columns:
        raise SystemExit(
            f"{args.csv}: no 'cleaned_text' column. Run scripts/preprocess.py first.\n"
            f"Found: {list(df.columns)}"
        )
    df["cleaned_text"] = df["cleaned_text"].fillna("")
    df["label"] = df.get("label", pd.Series([""] * len(df))).fillna("")
    corpus = df["cleaned_text"].tolist()

    suffix = f"_{args.tag}" if args.tag else ""
    out = []
    push = out.append
    push("=" * 78)
    push(f"TEXT REPRESENTATION - CountVectorizer (Bag of Words)")
    push(f"source: {args.csv}   ({len(df)} documents)")
    push("=" * 78)

    push("\n[1] WHAT THIS DOES")
    push("""
    A model cannot read text, only numbers. CountVectorizer builds a table:

        one ROW per document (tweet)
        one COLUMN per distinct term in the whole corpus
        each CELL = how many times that term appears in that document

    That table is the "document-term matrix", and the representation is called
    Bag of Words because it records WHICH words occur and HOW OFTEN, but throws
    away the order they came in.

    N-grams are how word order is partly recovered. An n-gram is a run of n
    consecutive words, so the vectoriser can treat "not good" as its own column
    instead of separate "not" and "good" columns that lose the connection.""")

    push("\n[2] WORKED EXAMPLE  (3 tiny documents, unigrams)")
    demo = ["good game", "bad game", "not good"]
    dv = CountVectorizer()
    dm = dv.fit_transform(demo)
    demo_table = pd.DataFrame(dm.toarray(), columns=dv.get_feature_names_out(),
                              index=[f"doc{i+1}: {d!r}" for i, d in enumerate(demo)])
    push(demo_table.to_string())
    push("\n    doc3 scores 1 for 'good' and 1 for 'not', but nothing links them -")
    push("    to a unigram model doc1 and doc3 both simply 'contain good'.")
    push("    That is exactly the weakness bigrams fix:")
    dv2 = CountVectorizer(ngram_range=(2, 2))
    dm2 = dv2.fit_transform(demo)
    push(pd.DataFrame(dm2.toarray(), columns=dv2.get_feature_names_out(),
                      index=["doc1", "doc2", "doc3"]).to_string())
    push("\n    'not good' is now its own feature, and doc3 is distinguishable.")

    push(f"\n\n[3] THE FOUR REPRESENTATIONS  (min_df={args.min_df})")
    push(f"    {'setting':12} {'ngram_range':13} {'features':>9} {'matrix':>14} {'sparsity':>9}   meaning")
    fitted = {}
    for name, rng, meaning in SETTINGS:
        vec = CountVectorizer(ngram_range=rng, min_df=args.min_df, token_pattern=r"\S+")
        mat = vec.fit_transform(corpus)
        fitted[name] = (vec, mat)
        cells = mat.shape[0] * mat.shape[1]
        sparsity = 1 - (mat.nnz / cells) if cells else 0
        push(f"    {name:12} {str(rng):13} {mat.shape[1]:>9} "
             f"{f'{mat.shape[0]}x{mat.shape[1]}':>14} {sparsity:>8.1%}   {meaning}")
    push("\n    'sparsity' = share of cells that are zero. Most tweets contain almost")
    push("    none of the corpus vocabulary, so the matrix is mostly empty - which is")
    push("    why scikit-learn stores it in a sparse format rather than a full table.")
    push("    Notice the feature count explodes as n grows while each feature gets")
    push("    rarer: that is why unigrams+bigrams is the usual compromise.")

    for name, _, _ in SETTINGS:
        vec, mat = fitted[name]
        push(f"\n\n[4:{name}] TOP 20 {name.upper()} FEATURES BY TOTAL COUNT")
        rows = top_terms(mat, vec.get_feature_names_out(), 20)
        for i in range(0, len(rows), 2):
            chunk = rows[i:i + 2]
            push("    " + "".join(f"{t:<32}{int(c):>5}   " for t, c in chunk))

    push("\n\n[5] MOST DISTINCTIVE TERMS PER SENTIMENT CLASS  (unigrams)")
    push("    Counted within each class, so these are the words that characterise it.")
    vec, mat = fitted["unigram"]
    vocab = vec.get_feature_names_out()
    for label in LABELS:
        mask = df["label"].astype(str).apply(lambda v: label in v.split("#")).to_numpy()
        if not mask.any():
            continue
        sub = mat[mask]
        rows = top_terms(sub, vocab, 12)
        push(f"\n    {label} ({int(mask.sum())} documents)")
        push("      " + ", ".join(f"{t}({int(c)})" for t, c in rows))

    push("\n\n[6] ONE DOCUMENT AS A VECTOR")
    vec, mat = fitted["unigram"]
    idx = int(df["cleaned_text"].str.split().str.len().idxmax())
    row = mat[idx].toarray()[0]
    nz = [(vocab[j], int(row[j])) for j in row.nonzero()[0]]
    push(f"    document #{idx}  label={df['label'].iloc[idx]}")
    push(f"    cleaned text : {df['cleaned_text'].iloc[idx][:150]}")
    push(f"    vector length: {len(row)} (one slot per corpus term)")
    push(f"    non-zero     : {len(nz)}  ->  {sum(row)} total words")
    push(f"    the non-zero slots: {', '.join(f'{t}={c}' for t, c in nz[:18])}")
    push("    every other slot in that row is 0.")

    # ------------------------------------------------------------ save files
    vec, mat = fitted[args.save_matrix]
    vocab = vec.get_feature_names_out()

    matrix_path = os.path.join(PROCESSED_DIR, f"count_matrix_{args.save_matrix.replace('+', '_')}{suffix}.csv")
    dtm = pd.DataFrame(mat.toarray(), columns=vocab)
    # Prefixed so they cannot collide with a real term - the corpus genuinely
    # contains tokens called "id" and "label".
    dtm.insert(0, "doc_label", df["label"].values)
    dtm.insert(0, "doc_id", df["id"].values if "id" in df.columns else range(len(df)))
    dtm.to_csv(matrix_path, index=False)

    vocab_path = os.path.join(PROCESSED_DIR, f"vocabulary_{args.save_matrix.replace('+', '_')}{suffix}.csv")
    doc_freq = (mat > 0).sum(axis=0).A1
    pd.DataFrame(
        {"term": vocab, "total_count": mat.sum(axis=0).A1, "document_frequency": doc_freq}
    ).sort_values("total_count", ascending=False).to_csv(vocab_path, index=False)

    push("\n\n[7] FILES WRITTEN")
    push(f"    {matrix_path}")
    push(f"        the document-term matrix for '{args.save_matrix}': {mat.shape[0]} rows x {mat.shape[1]} term columns,")
    push("        plus id and label columns so it can be fed straight to a classifier")
    push(f"    {vocab_path}")
    push("        every term with its total count and how many documents contain it")

    report = "\n".join(out)
    path = os.path.join(REPORT_DIR, f"vectorization_report{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
