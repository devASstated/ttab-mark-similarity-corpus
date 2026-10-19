# LASA Pair Extraction from TTAB Decisions

This repository contains the current working pipeline for extracting adjudicated name-similarity pairs from decisions of the U.S. Trademark Trial and Appeal Board (TTAB).

The purpose is to build a useful reference set for LASA-style name confusability research.

The important distinction is that we are not treating the final outcome of a trademark case as a name-similarity label.

A TTAB likelihood-of-confusion decision can depend on many things besides the names themselves, including the relatedness of goods or services, trade channels, purchaser sophistication, strength of the earlier mark, and other DuPont factors.

For this dataset, the part we care about is the Board's analysis of the similarity or dissimilarity of the marks themselves. This is commonly handled under the first DuPont factor and covers things such as appearance, sound, connotation, and overall commercial impression.

So the intended label is:

- `1`: the Board treated the two marks as similar under the first DuPont factor
- `0`: the Board treated the two marks as dissimilar under the first DuPont factor

Procedural dismissals, withdrawals, settlements, defaults, and cases where the relevant pair cannot be identified safely are not forced into either class.

## Why the Reading Room path exists

An earlier approach started from sequential TTAB proceeding records and tried to locate useful decisions from there.

That worked technically, but it was extremely inefficient because most proceedings never produced a substantive final likelihood-of-confusion decision.

The current pipeline instead starts from the TTAB Reading Room search API and targets final decisions containing:

```text
"likelihood of confusion" AND "du Pont"
```

This produces a much denser set of decisions that are actually useful for the dataset.

In the current test run:

```text
500 targeted final decisions
146 automatically verified rows
121 similar
25 dissimilar
0 label conflicts
```

The purpose of the remaining review set is to keep the automatic labels conservative. Many review cases already contain a usable first-DuPont-factor conclusion but have ambiguous, multiple, graphical, or otherwise unresolved marks.

## Repository files

### `fetch_ttab_reading_room.py`

Queries the TTAB Reading Room search API, keeps final decisions matching the likelihood-of-confusion and DuPont search, and downloads the corresponding decision PDFs.

This is the acquisition step.

### `extract_reading_room_dupont.py`

Reads the downloaded decisions and Reading Room metadata.

It:

- extracts the first DuPont factor discussion
- classifies the Board's mark-similarity conclusion
- resolves the textual marks from Reading Room metadata where possible
- falls back to decision text conservatively when needed
- separates automatically verified rows from review cases
- excludes conflicting labels
- calculates basic string-similarity features for the verified pair

### `ttab_pipeline_dupont.py`

Contains the core DuPont extraction, classification, similarity, and PDF-processing helpers used by the Reading Room extractor.

It also contains the earlier XML-oriented TTAB pipeline. The Reading Room path is currently the preferred acquisition route, but this file is still required because `extract_reading_room_dupont.py` imports its tested extraction logic.

### `requirements.txt`

Lists the Python packages required by the working pipeline.

## Setup after cloning

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

A quick syntax check can be run with:

```bash
python -m py_compile \
  ttab_pipeline_dupont.py \
  fetch_ttab_reading_room.py \
  extract_reading_room_dupont.py
```

## Step 1: Download targeted TTAB decisions

For a small test:

```bash
python fetch_ttab_reading_room.py \
  --target 50 \
  --page-size 100
```

For a larger run:

```bash
python fetch_ttab_reading_room.py \
  --target 500 \
  --page-size 100
```

The fetcher creates:

```text
reading_room_2d_manifest.csv
reading_room_2d_matches.jsonl
reading_room_decisions_2d/
```

### `reading_room_2d_manifest.csv`

A compact index of the decisions that were selected and downloaded.

It includes information such as:

- proceeding number
- case type
- decision date
- decision type
- Reading Room categories
- PDF URL
- local PDF path
- download status

### `reading_room_2d_matches.jsonl`

The richer raw metadata returned by the Reading Room API for the selected decisions.

The extractor uses this file because it contains structured fields for the applicant mark, cited mark, opposer mark, goods and services, proceeding identifiers, and other useful metadata.

### `reading_room_decisions_2d/`

Contains the actual TTAB decision PDFs.

These PDFs are the evidence behind the labels.

They should be kept locally while building or auditing the dataset, because any automatically generated pair can be traced back to the Board's written reasoning.

The directory is intentionally excluded from Git because it is generated data and can be recreated by running the fetcher.

## Step 2: Extract DuPont name-similarity pairs

Once the decisions have been downloaded, run:

```bash
python extract_reading_room_dupont.py
```

The extractor uses:

```text
reading_room_2d_matches.jsonl
reading_room_decisions_2d/
```

and produces several output files.

## Output files

### `reading_room_dupont_audit.csv`

The complete audit table.

There is one row for every decision that was processed, including cases that could not be automatically verified.

This is the first place to look when debugging the pipeline or checking why a particular decision was accepted or rejected.

Useful fields include:

- proceeding number
- case type
- Factor 1 extraction result
- overall Section 2(d) extraction result
- mark candidates
- selected marks
- mark-resolution source
- review reason
- similarity features

### `reading_room_dupont_verified.csv`

Rows that passed the automatic verification rules.

For a row to reach this file, the pipeline must have:

1. found a directional first-DuPont-factor result
2. resolved exactly one mark on each side
3. avoided any conflicting label for the same textual pair

These rows are the main automatically accepted evidence set.

### `reading_room_dupont_review.csv`

Cases that were not safe to label automatically.

A review row does not necessarily mean that the decision is unusable.

Common reasons include:

- one mark could not be recovered
- both marks could not be recovered
- multiple marks appear on one side
- a mark is graphical or design-only
- the first DuPont factor wording is unclear
- both similarity and dissimilarity language appears in the extracted section

This file is useful for improving extraction coverage without weakening the automatic acceptance rules.

### `reading_room_dupont_conflicts.csv`

Contains textual pairs that receive contradictory first-DuPont-factor labels across decisions.

These pairs are excluded from the training set automatically.

A conflict should be inspected rather than resolved by guessing.

### `reading_room_dupont_training.csv`

The compact training-oriented output.

It contains the accepted mark pair, the first-DuPont-factor label, proceeding information, and similarity features.

The important label is:

```text
label = 1   marks treated as similar
label = 0   marks treated as dissimilar
```

This label refers only to the Board's treatment of the marks under the first DuPont factor.

It does not mean that the Board ultimately found or did not find a likelihood of confusion.

### `reading_room_dupont_training.parquet`

The same training data in Parquet form when the Parquet dependency is available.

This is convenient for larger-scale analysis and model work.

## Evidence rule

The central rule of this project is simple:

**Do not turn the final trademark case outcome into a name-similarity label.**

For example, an opposition can be dismissed even when the marks are similar, because another DuPont factor defeats the overall likelihood-of-confusion claim.

Likewise, a case can succeed for reasons that should not be allowed to contaminate a pure name-confusability dataset.

The first DuPont factor is therefore kept separate from the final Section 2(d) outcome.

## Current limitations

The automatic pipeline is deliberately conservative.

It performs best when:

- the decision contains a clearly headed mark-similarity section
- one textual mark exists on each side
- Reading Room metadata exposes those marks cleanly
- the Board states whether the first DuPont factor favors or weighs against confusion

Cases involving multiple registrations, mark families, graphical marks, unusual decision structure, or complicated prose are often sent to review instead.

This is intentional. The aim is to build a defensible reference set rather than maximize the row count at the expense of label quality.

## Generated data and Git

Downloaded decisions and generated outputs are excluded through `.gitignore`.

A clean clone should contain the code and dependency list, then recreate the working data locally by running:

```bash
python fetch_ttab_reading_room.py --target 500 --page-size 100
python extract_reading_room_dupont.py
```

That keeps the repository small while preserving a reproducible path from TTAB decisions to the derived LASA-style name-similarity dataset.
