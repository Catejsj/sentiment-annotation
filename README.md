# Twitter Sentiment — Annotation, Augmentation & Preprocessing

Text mining coursework: annotate a tweet corpus in Doccano, score inter-annotator
agreement, augment the data, and preprocess it.

**Group 2** — augmentation uses random swapping, random insertion, and random
deletion via the TextAttack framework.

## Pipeline

```
annotate (Doccano)  ->  score agreement  ->  AUGMENT  ->  preprocess
```

Augmentation runs **before** preprocessing, as the brief specifies. That order
matters: the augmenters see natural English, so WordNet synonym insertion has
real words to work with. Run it after lemmatization and `was` is already `be`,
which degrades the synonym lookups.

## Scripts

| Script | Purpose |
|---|---|
| `01_export_from_doccano.py` | Pull annotations out of Doccano via its REST API |
| `02_agreement_analysis.py` | Cohen's + Fleiss' kappa over the original 8-pass corpus |
| `03_preprocessing.py` | 14-stage cleaning pipeline (`--csv`, `--tag`) |
| `04_compare_with_classmate.py` | Two-annotator comparison with alignment checks |
| `05_setup_doccano_project.py` | Build a reproducible Doccano project from a CSV |
| `06_validate_and_compare.py` | Validate an export, compare two annotators |
| `07_augment.py` | **Data augmentation** — TextAttack swap / insert / delete |
| `08_score_agreement.py` | **Scoring** — Cohen's + Fleiss' kappa for N annotators |

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

```bash
./.venv/bin/python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger_eng']]"
```

## Usage

```bash
./.venv/bin/python 07_augment.py
```

```bash
./.venv/bin/python 03_preprocessing.py --csv data/augmented_corpus.csv --tag augmented
```

```bash
./.venv/bin/python 08_score_agreement.py exports/*.csv
```

## Augmentation

| Technique | TextAttack class |
|---|---|
| Random swapping | `WordInnerSwapRandom` |
| Random insertion | `WordInsertionRandomSynonym` |
| Random deletion | `WordDeletion` |

Seed 42, so reruns reproduce the output exactly. 100 tweets → 698 rows
(2 variants per technique per tweet, plus the originals).

Two guards run on every generated variant:

**Negation preservation.** Random deletion can delete `not`, which silently
flips a Negative tweet to Positive and poisons the training data. Variants that
drop a negation present in the source are rejected and retried. Verified: 0 lost.

**Synonym hygiene.** WordNet synsets contain vulgar lemmas, so random synonym
insertion drops profanity into unrelated tweets — the first run inserted
`penis` into a tweet about a community event. Variants introducing a blocked
word not already in the source are rejected. Verified: 0 introduced.

Label balance is preserved to within 0.4 percentage points across all classes.

## Scoring

`08_score_agreement.py` takes any number of Doccano exports and reports:

- pairwise **Cohen's kappa** (scikit-learn) for every pair
- **Fleiss' kappa** (statsmodels) across all annotators at once
- exact-match agreement and per-label kappa, for the multi-label rows
- the most contested tweets, with each annotator's vote

It refuses to score until it has proven every file covers the same documents
with identical text — misaligned files would produce a meaningless number.

Cohen's kappa is agreement after subtracting the agreement two people would
reach by guessing with the same label frequencies, which is why it always sits
below the raw percentage. Landis & Koch (1977) bands: `<0.20` slight,
`0.21–0.40` fair, `0.41–0.60` moderate, `0.61–0.80` substantial, `>0.80` almost
perfect.

## Annotation scheme

Four labels: `Positive`, `Negative`, `Neutral`, `Irrelevant`, judged **towards
the entity** named in each row.

Two labels may be combined where a tweet genuinely sits in both places
(`Irrelevant+Positive`, `Neutral+Positive`). `Positive+Negative` is forbidden —
a tweet cannot be both favourable and hostile towards the same entity.
`06_validate_and_compare.py` enforces this on every export.

See `SHARED_SETUP.md` for the full protocol used to keep multiple annotators'
projects identical.

## Preprocessing

14 stages: entity-tag stripping, HTML unescaping, URL/mention removal, hashtag
unwrapping, lowercasing, **contraction expansion**, elongation squashing,
punctuation/digit removal, whitespace normalisation, **word tokenization**
(NLTK `TweetTokenizer`), **stopword removal**, **lemmatization** (WordNet with
POS tags), and short-token removal.

**Lemmatization over stemming** — Porter turns `terrible` into `terribl` and
`really` into `realli`, destroying the words that carry the sentiment.

**Negations are kept, not stopped** — NLTK's stopword list contains `not`, `no`,
`never`. Removing them turns *"not good"* into *"good"*, inverting the polarity
of exactly the tweets labelled Negative. 31 words are restored, leaving 172
active stopwords.

## Outputs

| File | Contents |
|---|---|
| `data/augmented_corpus.csv` | 698 rows — originals plus augmented variants |
| `data/cleaned_corpus_augmented.csv` | The augmented corpus, preprocessed |
| `reports/augmentation_report.txt` | Techniques, balance, safety checks, examples |
| `reports/preprocessing_report_*.txt` | Stage-by-stage output on real tweets |
| `reports/agreement_scores.txt` | Kappa scores across annotators |
