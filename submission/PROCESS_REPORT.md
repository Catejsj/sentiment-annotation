# Process Report: Tweet Corpus Preprocessing and Augmentation

This report documents the full pipeline applied to a Twitter sentiment corpus for an
NLP assignment: from corpus acquisition and annotation, through text preprocessing,
to data augmentation — including every decision made and the reasoning behind it.

---

## 1. Corpus & Annotation

| Item | Value |
|---|---|
| Dataset | `twitter-entity-sentiment-analysis` (Kaggle), 74,681 tweets |
| Columns (raw) | `id`, `entity`, `sentiment`, `text` |
| Task | Entity-level sentiment: judge the tweet's sentiment toward the mentioned entity |
| Labels | Positive, Negative, Neutral, Irrelevant |
| Sample | 100 tweets, balanced 25 per sentiment class (random seed 42) |
| Annotation tool | Doccano (self-hosted via Docker) |
| Exported file | `annotated_tweets.csv` (100 rows, all labeled) |

### Decisions & rationale
- **100-row sample with a 100-row limit** — assignment constraint; balanced classes
  avoid a degenerate (always-Positive) dataset.
- **Doccano for annotation** — team workflow required; also enabled a blind
  re-annotation round with a classmate (shared 50-row CSV) for later
  inter-annotator agreement (Cohen's Kappa).
- **Cleaning before annotation:** rows with text shorter than 15 chars were dropped
  as junk (e.g., single-word tweets like `was`), since they are unannotatable.

---

## 2. Preprocessing Pipeline

Pipeline order (`preprocess.py`), and why each stage sits where it does:

1. **Contraction expansion** (`I'll → i will`) — done *first* so every later stage
   sees full words; URLs/handles removed before expansion so punctuation does not
   confuse it.
2. **Cleaning / normalization**
   - URLs (`https://…`, `t.co/…`) removed — carry no sentiment
   - `@mentions` removed — user handles are noise for sentiment
   - `#` stripped from hashtags (content kept)
   - emojis and smart quotes removed (unicode ranges)
   - punctuation → space, whitespace collapsed, lowercase
3. **Tokenization** — NLTK `TweetTokenizer` with `preserve_case=False`,
   `strip_handles=True`, `reduce_len=True` (collapses `sooo → soo`)
4. **Noise filter** — drop tokens containing `/`, `=`, `&`, `…` (broken-link
   fragments that survive URL removal) and tokens with no alphanumeric character
5. **Stopword removal** — NLTK English stopwords **minus** a negation whitelist,
   plus corpus-specific additions (`rt`, `via`, `amp`, `com`)
6. **Lemmatization** — NLTK `WordNetLemmatizer` after POS tagging (J/N/V/R mapped
   to WordNet tags)

### The big decisions

| Decision | Choice | Why |
|---|---|---|
| Stemming vs lemmatization | **Lemmatization** | Meaning-preserving (`running → run`), stable for downstream classification; stemming is cruder (`studies → studi`) |
| Negation words | **Kept, not stopped** | Sentiment corpus — `not` is the strongest signal. NLTK's stoplist would delete it |
| Proper nouns | **Protected from lemmatization** | Game/product names (`Borderlands`, `PlayStation`) must match the `entity` metadata for later joins |
| `_`-joined hashtag words | **Kept** (removed filter) | e.g. `borderlands_sf` carries entity context |

### v1 → v2 fixes (evidence-based)

Fixes were applied after inspecting v1 output:

| Fix | v1 (broken) | v2 (fixed) | Evidence |
|---|---|---|---|
| Negation preservation | `Thanks to Verizon, you're too expensive *not* to work` → `thanks verizon expensive work` | `… expensive **not** work` | rows keeping a negation: **7 → 21** |
| Proper-noun protection | `Borderlands` → `borderland` | `Borderlands` kept | POS tagger says `NNS`, not `NNP` — so protection also whitelists the entity column |
| Noise filter | `_`-words dropped | kept | hashtag context restored |

**Measured result:** avg tokens per tweet **20.2 → 11.5** (~43% reduction), 35/100
rows changed between versions, no empty outputs.

---

## 3. Text Augmentation

Three EDA-style techniques (Wei & Zou, 2019), each producing 2 variants per tweet:

| Technique | Implementation |
|---|---|
| Random swap | Swap positions of two random non-negation words |
| Random insertion | Insert a WordNet synonym of a random word at a random position |
| Random deletion | Remove one random non-negation word (TextAttack `WordDeletion`) |

Framework: **TextAttack** (v0.3.10) — chosen for its ready-made transforms
(`WordDeletion`, synonym insertion) consistent with "use TextAttack" requirement.

### Decisions & rationale

1. **Synonym hygiene filters** — TextAttack's `WordInsertionRandomSynonym` picked
   obscene/awkward WordNet senses (`come → penis`, `say → mouth`). Replaced with a
   filtered NLTK synonym inserter: alphabetic only, len > 2, not stopwords, not in
   a profanity blocklist. **Verified: 0 inserted profanity.** (35 augmented rows
   contain profanity that already exists in the *source* corpus — real data, kept.)
2. **v1 → v2 hardening** (audit found: 58/600 no-op rows; 3 negation losses; 2
   degenerate tweets; 4 multi-tag labels):

| Fix | Change | Result |
|---|---|---|
| A | Insert retries up to 5 words before giving up | unchanged rows **58 → 1** (0.17%) |
| B | Swap/delete never touch negation words | negation losses **3 → 0** |
| C | Skip tweets with < 3 tokens | 2 degenerate tweets dropped (600 → **588 rows**) |
| D | Multi-tag labels normalized to first label (`Irrelevant#Positive → Irrelevant`) | 4 dirty rows cleaned, **no data loss** |

Final augmented dataset (`tweets_augmented.csv`, 588 rows):
`source_id, entity, label, technique, augmented_text`
Label distribution: Positive 222, Negative 222, Irrelevant 78, Neutral 66.

---

## 4. Files

| File | Description |
|---|---|
| `data/raw/annotated_tweets.csv` | 100 annotated tweets from Doccano |
| `scripts/preprocess.py` | Preprocessing pipeline (v2) |
| `data/processed/preprocessing_stages.csv` | Per-stage output for every tweet |
| `data/processed/tweets_preprocessed.csv` | Final preprocessed corpus |
| `scripts/augment.py` | Augmentation pipeline (v2, hardened) |
| `data/augmented/tweets_augmented.csv` | 588 augmented rows (final) |
| `data/augmented/tweets_augmented_v1.csv` | Pre-hardening snapshot (600 rows) for comparison |

---

## 5. Next Steps

- Score inter-annotator agreement (Cohen's Kappa) against the classmate's
  re-annotation of the shared 50 tweets (`shared_annotation_50.csv`)
- Optionally train a sentiment classifier on `tweets_augmented.csv` to measure the
  augmentation/negation-preservation effect quantitatively