# Submission — Text Mining Assignment

**Group 2** — augmentation by random swapping, random insertion, random deletion.

Pipeline order, as instructed: **annotate → augment → preprocess.**

---

## Requirement check

### "So by this Thursday please submit all these three steps"

| # | Required | File | Status |
|---|---|---|---|
| 1 | Your annotation | `1_my_annotation.csv` | 100 tweets, all labelled |
| 1 | Your inter-annotator agreements | `1_inter_annotator_agreement.txt` | Cohen's + Fleiss' kappa, 5 annotators |
| 2 | Your augmented dataset (**added to the original**) | `2_augmented_dataset.csv` | 699 rows = 100 original + 599 augmented |
| 3 | Your cleaned dataset after preprocessing | `3_cleaned_dataset.csv` | 699 rows preprocessed |

### "perform data augmentation as well before preprocessing"

Augmentation runs on the **raw** annotated corpus; preprocessing runs on the
**output of augmentation**. Evidence: `3_cleaned_dataset.csv` has 699 rows — it
could only have that many if augmentation happened first.

### "Group 1 and Group 2: random swapping, random insertion, random deletion"

| Technique | TextAttack class | Rows produced |
|---|---|---|
| Random swapping | `WordInnerSwapRandom` | 200 |
| Random insertion | `WordInsertionRandomSynonym` | 199 |
| Random deletion | `WordDeletion` | 200 |

Full detail in `2_augmentation_report.txt`.

### Preprocessing tasks

| Required | Where | Method |
|---|---|---|
| Word tokenization | stage 11 | NLTK `TweetTokenizer` |
| Stemming **or** lemmatization | stage 14 | **Lemmatization** (WordNet + POS tags) |
| Stopwords removal | stage 13 | NLTK English list, negations retained |
| Contraction expansion | stage 07 | `contractions` library |
| Other tasks deemed necessary | stages 01–06, 08–10, 12, 15 | entity-tag stripping, HTML unescaping, URL removal, @mention removal, #hashtag unwrapping, lowercasing, elongation squashing, punctuation/emoji removal, whitespace normalisation, numeric-token dropping, short-token dropping |

### "Please show me the version of cleaned text"

`3_preprocessing_report.txt` section [2] shows every stage on 5 real tweets.
`3_cleaned_dataset.csv` has `raw_text` and `cleaned_text` side by side.
`3_preprocessing_stages.csv` has the output of all 15 stages for all 699 rows.

---

## Results

### 1. Inter-annotator agreement

5 genuine annotators over the same 100 tweets (see note below).

| Metric | Value | Interpretation |
|---|---|---|
| **Fleiss' kappa** | **0.427** | moderate |
| Mean pairwise Cohen's kappa | 0.434 | moderate |
| Strongest pair | 0.826 | almost perfect |
| Weakest pair | 0.152 | slight |

The disagreement is concentrated in the **Neutral** class. One annotator used
Neutral 30 times; the rest used it 7–12 times. A second leaned Positive (45)
over Negative (24) where others were near-even. The guideline never defined the
boundary between "Neutral" and a weak polarity, and that single ambiguity
accounts for most of the lost agreement.

**Note on rater count.** Six files were submitted, but two of them carry
identical primary labels on all 100 tweets, differing only in second labels on
7 rows. That pair scores kappa 1.000 against each other, which pulls the group
average up. Both figures are reported for transparency:

| Raters counted | Fleiss' kappa | Mean pairwise Cohen's kappa |
|---|---|---|
| All 6 files as submitted | 0.495 | 0.502 |
| 5, treating the identical pair as one rater | **0.427** | **0.434** |

The 5-rater figure is the honest one, since two independent annotators do not
produce identical labels on 100 tweets.

### 2. Augmented dataset

100 original tweets → **699 rows** (originals retained, as instructed).
Random seed 42, so the file reproduces exactly on a rerun.

Label balance is preserved to within 0.4 percentage points across all classes —
augmentation must not change the class distribution.

Two safety guards run on every generated variant:

- **Negation preservation.** Random deletion can delete `not`, which flips a
  Negative tweet to Positive and corrupts the label. Variants losing a negation
  present in the source are rejected and retried. **Verified: 0 lost.**
- **Synonym hygiene.** WordNet synsets contain vulgar lemmas, so random synonym
  insertion injected profanity into unrelated tweets on the first run. Variants
  introducing a blocked word are rejected. **Verified: 0 introduced.**

### 3. Cleaned dataset

| | Before | After | Reduction |
|---|---|---|---|
| Total tokens | 16,147 | 7,857 | 51.3% |
| Vocabulary size | 1,289 | 981 | 23.9% |
| Mean tokens/document | 23.1 | 11.2 | 51.3% |

---

## Two decisions to defend

**Lemmatization, not stemming.** Porter stemming produces `terribl` from
`terrible` and `realli` from `really`. Both are strong sentiment carriers and
both survive lemmatization intact. A comparison table is printed in
`3_preprocessing_report.txt` section [3].

**Negation words are kept, not removed as stopwords.** NLTK's English stopword
list contains `not`, `no`, `never`, `nor`. Removing them turns *"not good"* into
*"good"*, inverting the polarity of precisely the tweets labelled Negative. 31
words are restored to the vocabulary, leaving 172 active stopwords.

Proper nouns are also protected from lemmatization, so `Borderlands` does not
become `borderland` and lose its link to the entity column.

---

## Reproducing

```bash
python scripts/augment.py
python scripts/preprocess.py --csv data/augmented/tweets_augmented.csv --tag augmented
python scripts/score.py data/raw/annotated_tweets.csv exports/team/*.csv
```
