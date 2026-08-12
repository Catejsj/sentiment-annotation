# Shared annotation setup — read this before you start

Two people are annotating the same 100 tweets independently, then comparing.
For the comparison to mean anything, **everything except the labels must be identical.**

## The one file that must be shared

```
annotated_tweets_50.csv
```

100 rows, columns `id, text, entity, source_id, label, Comments`.
Send this exact file. Do not re-save it from Excel, do not re-sort it, do not
delete rows. The `id` column (6–105) is what links the two exports back together.

**The `label` column in this file is ignored.** Both annotators start blank.
53 rows happen to have old labels in them; the setup script does not import them,
because seeing someone else's answer first would bias you and inflate the
agreement score.

## Setup — both people run the same command

Doccano must be running:

```bash
docker start doccano
```

Then, with `05_setup_doccano_project.py` and the CSV in place:

```bash
python 05_setup_doccano_project.py --csv annotated_tweets_50.csv
```

That script pins every setting that could otherwise drift:

| Setting | Value | Why it matters |
|---|---|---|
| Project type | DocumentClassification | — |
| Multi-label | **ON** | so `Irrelevant+Positive` is possible |
| Random order | **OFF** | both people see tweet 1 first, tweet 100 last |
| Labels | Positive, Negative, Neutral, Irrelevant | created in that order, same colours |
| Shortcuts | `p` `n` `u` `i` | so muscle memory matches |
| Metadata | `id`, `entity`, `source_id`, `row` | carried into every document, survives export |

If it says a project with that name already exists, delete the old one in the
web UI first — do not make a second copy alongside it.

## The labelling rule we agreed

Label the sentiment **towards the entity** named in the `entity` field.

- **Positive** — favourable towards the entity
- **Negative** — critical or hostile towards the entity
- **Neutral** — about the entity, but no clear feeling
- **Irrelevant** — doesn't really talk about the entity (link dumps, spam, chatter)

**Two labels are allowed** when the tweet genuinely sits in both places:

- ✅ `Irrelevant + Positive`
- ✅ `Neutral + Positive`
- ❌ `Positive + Negative` — **never.** A tweet can't be both favourable and
  hostile towards the same entity. Pick whichever dominates.

Prefer one label. Only reach for a second when you'd otherwise be guessing.

## When you're both done

Export from Doccano: **Datasets → Export Dataset → CSV**. You'll get `admin.csv`.
Rename it so you know whose it is, e.g. `sherlock.csv` and `friend.csv`.

Check your own file before sending it:

```bash
python 06_validate_and_compare.py sherlock.csv
```

It tells you if you left anything unannotated or broke the Positive+Negative rule.
Fix those in Doccano and re-export.

Then compare the two:

```bash
python 06_validate_and_compare.py sherlock.csv friend.csv
```

You get exact-match agreement, per-label Cohen's kappa, and the list of tweets
you disagreed on.

## Why per-label kappa and not plain kappa

Plain Cohen's kappa assumes every item has **exactly one** label. Since we allow
two, it doesn't apply directly. Instead the script asks, for each label
separately, "did we both apply this one?" — four yes/no questions, four kappas,
and the mean of those is the headline number. The script also prints plain
Cohen's kappa computed only on the rows where you both used a single label, so
you have both figures available.

## Common ways this goes wrong

| Problem | Symptom | Fix |
|---|---|---|
| One person re-sorted the CSV | script says text doesn't match | re-import from the original file |
| One person left random order ON | you annotated different tweets in different orders | it still exports by id, so this is survivable — but use the script's settings |
| Someone added their own label type | "unexpected label names" | stick to the four |
| Exported before finishing | "N tweets are NOT annotated yet" | finish, re-export |
