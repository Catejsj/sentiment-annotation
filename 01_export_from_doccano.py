#!/usr/bin/env python3
"""
STEP 1 - Export the annotated corpus from Doccano as a CSV sheet.

Doccano's own "Export dataset" button writes a CSV with the columns
    id, text, label
This script talks to the Doccano REST API and writes exactly that file, so it
works even when the built-in exporter fails.

Usage:
    python 01_export_from_doccano.py
Output:
    data/doccano_export.csv
"""
import csv
import os

import requests

# Local Docker defaults; override with DOCCANO_USER / DOCCANO_PASS if yours differ.
HOST = os.environ.get("DOCCANO_HOST", "http://localhost:8000")
USERNAME = os.environ.get("DOCCANO_USER", "admin")
PASSWORD = os.environ.get("DOCCANO_PASS", "password")
PROJECT_ID = int(os.environ.get("DOCCANO_PROJECT", "2"))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data")
OUT_CSV = os.path.join(OUT_DIR, "doccano_export.csv")


def login() -> str:
    """Log in and return an API token."""
    r = requests.post(
        f"{HOST}/v1/auth/login/",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["key"]


def get_all(session: requests.Session, url: str) -> list:
    """Follow Doccano's pagination and return every result row."""
    rows = []
    while url:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list):  # some endpoints are not paginated
            return payload
        rows.extend(payload["results"])
        url = payload["next"]
    return rows


def main() -> None:
    token = login()
    s = requests.Session()
    s.headers["Authorization"] = f"Token {token}"

    # label id -> label text, e.g. 5 -> "Positive"
    label_types = get_all(s, f"{HOST}/v1/projects/{PROJECT_ID}/category-types")
    id2label = {lt["id"]: lt["text"] for lt in label_types}
    print(f"label set: {sorted(id2label.values())}")

    examples = get_all(s, f"{HOST}/v1/projects/{PROJECT_ID}/examples?limit=100")
    print(f"examples : {len(examples)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    n_labelled = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label"])
        for ex in examples:
            cats = s.get(
                f"{HOST}/v1/projects/{PROJECT_ID}/examples/{ex['id']}/categories",
                timeout=30,
            ).json()
            labels = [id2label[c["label"]] for c in cats]
            if labels:
                n_labelled += 1
            w.writerow([ex["id"], ex["text"], "#".join(labels)])

    print(f"annotated: {n_labelled}/{len(examples)}")
    print(f"written  : {OUT_CSV}")


if __name__ == "__main__":
    main()
