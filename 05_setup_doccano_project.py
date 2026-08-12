#!/usr/bin/env python3
"""
STEP 5 - Build an IDENTICAL Doccano project from a shared CSV.

Both annotators run this exact script against the exact same CSV. Everything
that could make two projects differ is pinned here:

    * project settings   - multi-label on, random_order OFF
    * the 4 label types  - same names, same colours, same keyboard shortcuts,
                           created in the same order
    * the 100 documents  - uploaded one by one in CSV row order, so document 1
                           is the same tweet for both people
    * id / entity / source_id - carried into each document's metadata, so the
                           two exports can be matched back together afterwards

Nothing is pre-labelled. Both people annotate all 100 tweets from scratch,
which is the only way the agreement numbers mean anything.

Usage:
    python 05_setup_doccano_project.py
    python 05_setup_doccano_project.py --csv /path/to/file.csv --name "My Project"
"""
import argparse
import sys

import pandas as pd
import requests

# Local Docker defaults; override with DOCCANO_USER / DOCCANO_PASS if yours differ.
HOST = os.environ.get("DOCCANO_HOST", "http://localhost:8000")
USERNAME = os.environ.get("DOCCANO_USER", "admin")
PASSWORD = os.environ.get("DOCCANO_PASS", "password")

DEFAULT_CSV = "/home/sherlock/Desktop/annotated_tweets_50.csv"
DEFAULT_NAME = "Twitter Sentiment 100 (shared)"

GUIDELINE = """Label the sentiment each tweet expresses TOWARDS ITS ENTITY.

Labels
  Positive    - the tweet is favourable towards the entity
  Negative    - the tweet is critical / hostile towards the entity
  Neutral     - the tweet is about the entity but carries no clear feeling
  Irrelevant  - the tweet does not really talk about the entity at all
                (link dumps, spam, unrelated chatter)

COMBINING LABELS
  You MAY select two labels when the tweet genuinely sits in both places, e.g.
      Irrelevant + Positive
      Neutral + Positive
  You MUST NOT select Positive and Negative together. A tweet cannot be both
  favourable and hostile towards the same entity - pick the dominant one.

Prefer a single label. Only reach for a second one when you would otherwise be
guessing between the two.
"""

# name, colour, keyboard shortcut - pinned so both projects look identical
LABEL_TYPES = [
    ("Positive", "#3ba272", "p"),
    ("Negative", "#ee6666", "n"),
    ("Neutral", "#5470c6", "u"),
    ("Irrelevant", "#9a60b4", "i"),
]


def login() -> requests.Session:
    r = requests.post(
        f"{HOST}/v1/auth/login/",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=30,
    )
    r.raise_for_status()
    s = requests.Session()
    s.headers["Authorization"] = f"Token {r.json()['key']}"
    return s


def project_exists(s: requests.Session, name: str):
    r = s.get(f"{HOST}/v1/projects", timeout=30)
    r.raise_for_status()
    for p in r.json()["results"]:
        if p["name"] == name:
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--name", default=DEFAULT_NAME)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("text", "entity", "source_id"):
        if col not in df.columns:
            raise SystemExit(f"CSV is missing the '{col}' column. Found: {list(df.columns)}")
    if df["text"].isna().any():
        raise SystemExit("Some rows have an empty text column - fix the CSV first.")

    print(f"CSV      : {args.csv}")
    print(f"documents: {len(df)}")
    print("labels   : NOT imported - both annotators start from a blank slate")

    s = login()

    existing = project_exists(s, args.name)
    if existing:
        print(f"\nA project called {args.name!r} already exists (id={existing['id']}).")
        print("Refusing to create a second copy - delete it in the web UI first,")
        print("or pass a different --name.")
        sys.exit(1)

    # ------------------------------------------------------------- project
    r = s.post(
        f"{HOST}/v1/projects",
        json={
            "name": args.name,
            "description": "Shared 100-tweet sentiment corpus. Both annotators use identical settings.",
            "guideline": GUIDELINE,
            "project_type": "DocumentClassification",
            "resourcetype": "TextClassificationProject",
            # OFF so both people see the tweets in the same order as the CSV
            "random_order": False,
            "collaborative_annotation": False,
            # False = multi-label allowed (needed for Irrelevant+Positive etc.)
            "single_class_classification": False,
            "allow_member_to_create_label_type": False,
        },
        timeout=30,
    )
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"\ncreated project id={pid}  {args.name!r}")
    print("  multi-label   : ON  (Positive+Negative still forbidden by the guideline)")
    print("  random order  : OFF (both annotators see the same order)")

    # -------------------------------------------------------------- labels
    for text, colour, key in LABEL_TYPES:
        r = s.post(
            f"{HOST}/v1/projects/{pid}/category-types",
            json={
                "text": text,
                "prefix_key": None,
                "suffix_key": key,
                "background_color": colour,
                "text_color": "#ffffff",
            },
            timeout=30,
        )
        r.raise_for_status()
        print(f"  label {text:11} colour={colour}  shortcut='{key}'")

    # ----------------------------------------------------------- documents
    print("\nuploading documents...")
    for i, row in df.iterrows():
        r = s.post(
            f"{HOST}/v1/projects/{pid}/examples",
            json={
                "text": str(row["text"]),
                # a real dict, NOT the string "{}" - this is what keeps
                # Doccano's own CSV export working later
                "meta": {
                    "row": int(i),
                    "id": int(row["id"]),
                    "entity": str(row["entity"]),
                    "source_id": int(row["source_id"]),
                },
            },
            timeout=30,
        )
        r.raise_for_status()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)}")

    check = s.get(f"{HOST}/v1/projects/{pid}/examples?limit=1", timeout=30).json()
    print(f"\ndone. {check['count']} documents in project {pid}.")
    print(f"open: {HOST}/projects/{pid}/text-classification")


if __name__ == "__main__":
    main()
