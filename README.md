# TTAB Factor 1 Mark-Similarity Corpus

This repository contains the production pipeline and current V1 corpus for extracting adjudicated mark-similarity pairs from decisions of the U.S. Trademark Trial and Appeal Board (TTAB).

The project was built to provide an external reference set for LASA-style name-confusability research and for evaluating string-similarity algorithms against real adjudicated examples.

The central rule is:

> **The final trademark case outcome is not used as the name-similarity label.**

A TTAB likelihood-of-confusion decision can depend on many factors besides the names themselves, including relatedness of goods or services, trade channels, purchaser sophistication, mark strength, and other DuPont factors.

For this corpus, the label comes only from the Board's treatment of the marks under the **first DuPont factor**, which concerns similarity or dissimilarity in appearance, sound, connotation, and overall commercial impression.

```text
label = 1
    the Board treated the two marks as similar under DuPont Factor 1

label = 0
    the Board treated the two marks as dissimilar under DuPont Factor 1
```

Ambiguous, unresolved, procedural, conflicting, graphical, or otherwise unsafe cases are not forced into either class.

---

## Current V1 corpus

The current authoritative global corpus contains:

```text
319 unique pair records
186 similar
133 dissimilar
0 cross-source label conflicts
```

Main pair-level deliverable:

```text
ttab_global_factor1_pairs.csv
```

Minimal evaluation export:

```text
ttab_global_factor1_training.csv
```

Evidence-level corpus:

```text
ttab_global_factor1_evidence.csv
```

Conflict quarantine:

```text
ttab_global_factor1_conflicts.csv
```

---

## What TTAB is

TTAB is the **Trademark Trial and Appeal Board** of the United States Patent and Trademark Office, USPTO.

Relevant proceeding types include:

```text
EXA    ex parte appeal
OPP    opposition
CAN    cancellation
```

The project focuses on decisions involving likelihood of confusion under Section 2(d).

TTAB likelihood-of-confusion analysis commonly applies the factors from:

```text
In re E. I. du Pont de Nemours & Co.
```

The first DuPont factor is the part most directly relevant to name similarity.

---

## Why DuPont Factor 1 is the label

The overall Section 2(d) result is not a pure name-similarity judgment.

A pair of marks can be similar under Factor 1 while the overall refusal or opposition fails because other factors weigh against confusion.

Likewise, a case can succeed even when some individual similarity evidence is weak.

Therefore the production ground-truth rule is:

```text
Factor 1 says similar      -> label 1
Factor 1 says dissimilar   -> label 0
Factor 1 unclear/mixed     -> no binary label, review/exclude
```

Disposition text such as:

```text
Refusal Reversed
Refusal Affirmed
Opposition Dismissed
Opposition Sustained
```

is used only as an **acquisition hint**, never as the similarity label.

---

## Why the Reading Room path is used

An earlier approach started from sequential TTAB proceedings and searched for useful decisions afterward.

That was technically possible but inefficient because most proceedings did not lead to a substantive final likelihood-of-confusion decision.

The production path instead starts from the USPTO TTAB Reading Room search API.

The full-text query used is:

```text
"likelihood of confusion" AND "du Pont"
```

The search endpoint is:

```text
POST https://ttab-reading-room.uspto.gov/ttab-efoia-api/decision/search
```

Representative request body:

```json
{
  "dateRangeData": {},
  "facetData": {},
  "parameterData": {},
  "recordTotalQuantity": 100,
  "searchText": "\"likelihood of confusion\" AND \"du Pont\"",
  "sortDataBag": [
    {
      "issueDate": "desc"
    }
  ],
  "recordStartNumber": 0
}
```

At acquisition time, this query returned a Reading Room result universe of approximately:

```text
8,715 records
```

This does **not** mean 8,715 total TTAB cases or all Section 2(d) cases.

It means approximately 8,715 Reading Room records matched the exact full-text query at the time of acquisition.

The production pipeline then filters that raw search universe for:

```text
decisionTypeCategory == Final Decision
```

and optionally applies acquisition-only enrichment filters.

---

## Production architecture

```text
USPTO TTAB Reading Room
        |
        v
fetch_ttab_reading_room.py
        |
        | JSON metadata + decision PDFs
        v
extract_reading_room_dupont.py
        |
        | high-confidence Factor 1 evidence
        v
merge_ttab_verified_shards.py
        |
        v
ttab_global_factor1_pairs.csv
        |
        +--> ttab_global_factor1_training.csv
```

The three core production programs are:

```text
fetch_ttab_reading_room.py
extract_reading_room_dupont.py
merge_ttab_verified_shards.py
```

---

# Production files

## `fetch_ttab_reading_room.py`

This is the acquisition layer.

It:

- queries the TTAB Reading Room search API
- paginates through raw search results
- keeps final decisions
- supports deterministic raw-result sharding
- optionally applies acquisition-only enrichment filters
- preserves the original Reading Room JSON metadata
- downloads the corresponding decision PDFs
- writes a human-readable manifest

It does **not** assign mark-similarity labels.

### Fetcher outputs

A fetch run creates:

```text
*_matches.jsonl
*_manifest.csv
*_pdfs/
```

Example:

```text
reversed_0000_2905_matches.jsonl
reversed_0000_2905_manifest.csv
reversed_0000_2905_pdfs/
```

### Important Reading Room metadata fields

```text
proceedingNumberDisplay
proceedingNumber
proceedingTypeCd
decision
decisionTypeCategory
groundsForRefusalCategory
opposerMarkGoodService
applicantMarkGoodService
examiningAttorneyMarkGoodCited
appealDecisionId
documentImageId
documentId
```

### Manifest fields

Typical normalized fields include:

```text
proceeding_number
case_type
decision
decision_type_category
grounds_for_refusal_category
appeal_decision_id
pdf_url
pdf_path
download_status
query_index
query_start_index
query_end_index
decision_filter
```

---

## Decision PDF retrieval

Decision PDFs are retrieved from:

```text
https://ttab-reading-room.uspto.gov/cms/rest/legal-proceeding/<PROCEEDING>/decision/<TYPE>_<APPEAL_DECISION_ID>.pdf
```

Example:

```text
https://ttab-reading-room.uspto.gov/cms/rest/legal-proceeding/97565017/decision/EXA_8.pdf
```

The fetcher primarily uses:

```text
appealDecisionId
```

with fallback handling for:

```text
documentImageId
documentId
```

Downloaded PDFs use deterministic filenames such as:

```text
97565017-EXA-8.pdf
```

---

## Deterministic sharding

The fetcher supports:

```text
--start-index
--end-index
```

with:

```text
start index inclusive
end index exclusive
```

These are raw Reading Room result offsets.

For the approximately 8,715-result universe, the production three-machine split was:

```text
[0, 2905)
[2905, 5810)
[5810, 8715)
```

This provides:

```text
no overlap
no gaps
reproducible shard ownership
```

---

## Acquisition filters

Acquisition filters are sampling strategies only.

They never determine the final label.

Current production filters include:

```text
all
opposition-dismissed
exa-refusal-reversed
```

### `all`

Broad final-decision acquisition without disposition enrichment.

### `opposition-dismissed`

Approximate logic:

```python
case_type == "OPP"
and "dismiss" in decision
and "sustain" not in decision
```

Lane-level result:

```text
127 unique clean pairs
79 similar
48 dissimilar
0 conflicts
```

### `exa-refusal-reversed`

Approximate logic:

```python
case_type == "EXA"
and "revers" in decision
and "affirm" not in decision
```

Lane-level result:

```text
192 unique clean pairs
107 similar
85 dissimilar
0 conflicts
```

A reversed refusal does not automatically imply dissimilar marks. The disposition only helps find useful cases efficiently.

---

## `extract_reading_room_dupont.py`

This is the ground-truth extraction layer.

Inputs:

```text
*_matches.jsonl
*_pdfs/
```

It performs two main tasks:

1. resolve the two marks being compared
2. classify the DuPont Factor 1 conclusion

A row becomes automatically verified only when both tasks succeed.

---

## Mark resolution

Preferred source: structured Reading Room metadata.

For EXA:

```text
side A = applicantMarkGoodService
side B = examiningAttorneyMarkGoodCited
```

For OPP/CAN:

```text
side A = opposerMarkGoodService
side B = applicantMarkGoodService
```

If metadata does not safely resolve both sides, the extractor can fall back conservatively to decision-text patterns such as:

```text
Applicant's mark ...
Opposer's mark ...
Petitioner's mark ...
Respondent's mark ...
```

The production philosophy is precision-first.

If exactly one usable mark cannot be resolved on each side, the case should go to review rather than being forced into the corpus.

---

## Factor 1 classification

The extractor identifies the first DuPont factor discussion from the decision PDF and classifies the Board's conclusion.

```text
Factor 1 says marks are similar
    -> name_similarity_label = 1

Factor 1 says marks are dissimilar
    -> name_similarity_label = 0

Factor 1 unclear, mixed, or conflicting
    -> name_similarity_label = None
    -> review / exclusion
```

The extractor keeps fields such as:

```text
factor1_text
factor1_heading
factor1_result
name_similarity_label
factor1_evidence
```

It separately retains the overall Section 2(d) result in fields such as:

```text
overall_2d_result
overall_confusion_label
overall_evidence
```

These overall fields are for auditability only and do not create the name-similarity label.

---

## Extractor outputs

For an output prefix such as:

```text
reversed_0000_2905
```

it writes:

```text
reversed_0000_2905_audit.csv
reversed_0000_2905_verified.csv
reversed_0000_2905_review.csv
reversed_0000_2905_conflicts.csv
reversed_0000_2905_training.csv
```

### `*_audit.csv`

One row per acquired decision. Useful for debugging, provenance, mark resolution, Factor 1 evidence, and review reasons.

### `*_verified.csv`

High-confidence evidence rows accepted by the automatic pipeline.

### `*_review.csv`

Cases not safe to convert automatically into a binary textual pair.

Typical reasons include:

- unresolved mark
- multiple marks
- graphical/design-only mark
- unclear Factor 1 conclusion
- conflicting language
- PDF or extraction failure

A review row does not mean negative.

### `*_conflicts.csv`

Pairs receiving contradictory labels within that extraction run.

### `*_training.csv`

Compact per-run training-oriented output.

---

## `merge_ttab_verified_shards.py`

This is the consolidation layer.

It:

- merges verified evidence from multiple shards or lanes
- normalizes mark identity
- treats pair order as irrelevant
- deduplicates repeated evidence
- preserves proceeding and source-file provenance
- detects contradictory labels
- quarantines conflicts
- emits one unique pair per normalized textual pair
- produces a minimal training export

Conceptually:

```text
ABC vs XYZ
```

and:

```text
XYZ vs ABC
```

are the same pair.

---

## Evidence rows vs pair rows

An evidence row represents:

```text
one TTAB adjudicative observation
```

A pair row represents:

```text
one unique normalized unordered mark pair
```

The final evaluation corpus is pair-level so repeatedly litigated names do not receive disproportionate weight.

---

## Conflict handling

If a normalized pair receives both:

```text
label 1
and
label 0
```

it is quarantined rather than silently resolved.

Conflict evidence is written to:

```text
*_conflicts.csv
```

The current V1 global corpus has:

```text
0 cross-source conflicts
```

---

# Current global V1 merge

The authoritative global merge used:

```text
reading_room_dupont_verified.csv
ttab_opposition_dismissed_full_evidence.csv
ttab_refusal_reversed_full_evidence.csv
```

Command:

```bash
python merge_ttab_verified_shards.py \
    reading_room_dupont_verified.csv \
    ttab_opposition_dismissed_full_evidence.csv \
    ttab_refusal_reversed_full_evidence.csv \
    --out-prefix ttab_global_factor1
```

Result:

```text
Input files:             3
Input verified rows:     399
Deduped evidence rows:   349
Conflict evidence rows:  0
Clean evidence rows:     349
Unique clean pairs:      319

Unique pair label breakdown:
label
1    186
0    133
```

---

# Final corpus files

## `ttab_global_factor1_pairs.csv`

Canonical pair-level research corpus.

Typical fields include:

```text
mark_a
mark_b
label
pair_key
evidence_count
proceeding_count
proceeding_numbers
source_files
```

Use this file when provenance matters.

Current size:

```text
319 unique pairs
```

## `ttab_global_factor1_training.csv`

Minimal evaluation export.

Schema:

```text
mark_a
mark_b
label
```

This is the easiest file to use when benchmarking similarity algorithms.

## `ttab_global_factor1_evidence.csv`

Evidence-level corpus after global evidence deduplication.

Current size:

```text
349 clean evidence rows
```

Use this for provenance, auditing, and repeated-adjudication analysis.

## `ttab_global_factor1_conflicts.csv`

Conflict quarantine.

Current state:

```text
0 conflicting pair rows
```

---

# External similarity validation

The repository also contains a reproducible validation harness for comparing similarity algorithms against the TTAB corpus.

Main validation file:

```text
validate_ttab.py
```

Algorithm modules:

```text
feline.py
kbigram.py
lumera.py
```

Model artifacts:

```text
kbigram_model.npz
lumera_model.npz
```

These model files are intentionally committed because they are small and required to reproduce the trained algorithms exactly.

Approximate artifact sizes:

```text
kbigram_model.npz    75 KB
lumera_model.npz     18 KB
```

No Git LFS is required.

---

## Validation command

```bash
python validate_ttab.py \
    ttab_global_factor1_training.csv \
    --view all \
    --show-excluded
```

The script reports:

```text
raw view
clean view
strict view
```

and evaluates:

```text
edit distance
Jaro-Winkler
bigram Dice
KBIGRAM
LUMERA
FELINE
```

It also reports:

- ROC-AUC
- bootstrap confidence intervals
- PR-AUC
- optimistic best-F1 operating points
- paired-bootstrap AUC differences
- score correlations
- identical-string edge cases
- orthography-discordant Factor 1 cases

---

# Current validation result

The current strict textual view contains:

```text
278 pairs
161 similar
117 dissimilar
```

Latest reproduced strict ROC-AUCs:

```text
Jaro-Winkler     0.6354
FELINE           0.6325
Edit             0.6004
Bigram Dice      0.5837
KBIGRAM          0.5781
LUMERA           0.5641
```

These numbers should be interpreted as performance against:

```text
US TTAB Factor 1 adjudication
```

not as direct Indian medication-error ground truth.

Factor 1 includes:

- appearance
- sound
- connotation
- commercial impression

so a purely orthographic algorithm is not expected to perfectly reproduce the legal label.

---

# Canonical corpus vs strict textual benchmark

The canonical legal corpus currently has:

```text
319 unique pair records
```

The strict textual validation view currently keeps:

```text
278 rows
```

and drops:

```text
41 rows
```

because they are not cleanly one textual mark versus one textual mark.

Current strict exclusions:

```text
29 multi-mark fields
7 non-textual marks
4 extraction failures
1 row with nothing usable after cleaning
```

These 41 rows are not all the same kind of problem.

```text
multi-mark or mark-family cases
    may be legally valid but are not safe single-name benchmark rows

design/trade-dress/configuration cases
    may be legally valid but are not text-only similarity problems

extraction failures
    genuine upstream parsing defects

empty-after-cleaning case
    insufficient textual representation for string scoring
```

The intended distinction is:

```text
TTAB legal corpus
    319 current pair records

TTAB strict textual benchmark
    278 current scoreable name-pair rows
```

A future V1.1 may revise this after a provenance audit.

Do not silently delete the 41 rows from the canonical legal corpus without first distinguishing valid non-textual legal evidence from extraction mistakes.

---

# Identical-string negative cases

The strict textual audit currently finds 18 identical cleaned-string pairs, including five labelled dissimilar:

```text
NAUTICA / NAUTICA
REALTOR / REALTOR
DOCUVIEW / DOCUVIEW
UNIFRAME / UNIFRAME
SPECTRUM / SPECTRUM
```

These should not automatically be assumed to be incorrect TTAB labels.

They may reflect:

- design or stylization differences
- disclaimed matter
- commercial impression
- metadata reduction that collapses legally distinct marks into the same plain text

These cases require provenance-level inspection before any corpus correction.

---

# Ground-truth invariants

Future changes should preserve these unless the research design is explicitly changed.

## 1. Factor 1 is the similarity label

Do not use overall Section 2(d) outcome as `name_similarity_label`.

## 2. Acquisition disposition is not ground truth

`Opposition Dismissed` and `Refusal Reversed` are acquisition strategies only.

## 3. Unclear does not mean negative

Ambiguous cases must go to review or exclusion.

## 4. Mark resolution must remain conservative

Do not blindly Cartesian-product multi-mark proceedings.

## 5. Do not prefilter with similarity metrics

Do not use Jaro-Winkler, edit distance, phonetic similarity, or related measures to decide which TTAB examples enter the corpus.

That would bias the benchmark.

## 6. Preserve provenance

Final pair rows should remain traceable to TTAB proceeding evidence.

## 7. Pair identity is unordered

A/B and B/A are the same pair.

## 8. Conflicts are quarantined

Contradictory pair labels should not be silently resolved.

---

# Setup after cloning

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Syntax check:

```bash
python -m py_compile \
    fetch_ttab_reading_room.py \
    extract_reading_room_dupont.py \
    merge_ttab_verified_shards.py \
    validate_ttab.py \
    feline.py \
    kbigram.py \
    lumera.py
```

---

# Reproducing the current global corpus

If the three V1 evidence inputs are present:

```bash
python merge_ttab_verified_shards.py \
    reading_room_dupont_verified.csv \
    ttab_opposition_dismissed_full_evidence.csv \
    ttab_refusal_reversed_full_evidence.csv \
    --out-prefix ttab_global_factor1
```

Expected result:

```text
399 input verified rows
349 deduplicated clean evidence rows
0 conflict evidence rows
319 unique clean pairs
186 similar
133 dissimilar
```

---

# Reproducing validation

```bash
python validate_ttab.py \
    ttab_global_factor1_training.csv \
    --view all \
    --show-excluded
```

The committed algorithm modules and model artifacts allow FELINE, KBIGRAM, and LUMERA to be scored without copying files from another repository.

---

# Generated data and Git policy

The repository intentionally commits compact artifacts required for reproducibility.

Committed production deliverables include:

```text
ttab_global_factor1_pairs.csv
ttab_global_factor1_training.csv
ttab_global_factor1_conflicts.csv
```

and the validation modules/model artifacts.

Large or easily reproducible working artifacts should generally remain local, such as:

```text
downloaded PDF directories
raw acquisition JSONL
large manifests
individual shard audit files
individual shard review files
temporary diagnostics
```

Guiding rule:

> Commit compact artifacts required to reproduce or consume the official V1 result. Keep bulky reacquirable source material out of Git unless there is a strong provenance reason to retain it.

---

# Relevant production Git milestones

```text
fd22b2f
Add global TTAB Factor 1 corpus V1

a572c58
Add reversed-refusal TTAB acquisition lane

384b144
Add opposition-dismissed TTAB corpus milestone

a889f15
Add deterministic sharding for TTAB corpus acquisition

be4f618
Stabilize high-confidence TTAB Factor 1 corpus pipeline

1b61d0e
Add targeted TTAB DuPont corpus pipeline
```

Conceptually:

```text
targeted Factor 1 acquisition
        ↓
high-confidence extraction stabilization
        ↓
deterministic multi-machine sharding
        ↓
negative-enriched opposition lane
        ↓
negative-enriched reversed-refusal lane
        ↓
global V1 corpus
        ↓
external algorithm validation
```

---

# Recommended reading order for new contributors

```text
1. README.md
2. fetch_ttab_reading_room.py
3. extract_reading_room_dupont.py
4. merge_ttab_verified_shards.py
5. ttab_global_factor1_pairs.csv
6. ttab_global_factor1_training.csv
7. ttab_global_factor1_evidence.csv
8. validate_ttab.py
9. feline.py
10. kbigram.py
11. lumera.py
```

Historical experiments and older prototype scripts should not be treated as the production path unless specifically needed for debugging or archaeology.

---

# Current limitations

The current corpus is:

```text
US trademark adjudication based
Factor 1 mark-similarity grounded
precision-first
automatically extracted
```

It is not:

```text
a complete TTAB universe
a pharmaceutical-only dataset
an Indian drug-name dataset
a medication-error benchmark
a universal human-confusability benchmark
```

The TTAB corpus is best used as an external adjudicated similarity benchmark and as a stress test for algorithms trained or tuned on synthetic/random-negative datasets.

---

# Current next step

The highest-value next step is a provenance audit of:

```text
41 strict textual exclusions
5 identical-string dissimilar pairs
```

The goal is to distinguish:

```text
legitimate non-textual legal evidence
from
true extraction or representation defects
```

That audit can support a future:

```text
TTAB Factor 1 textual benchmark V1.1
```

while preserving the original legal corpus and its provenance.

---

# Summary

The current production system is:

```text
Reading Room search API
    ↓
final-decision acquisition
    ↓
optional enrichment lanes
    ↓
decision PDFs + structured metadata
    ↓
DuPont Factor 1 extraction
    ↓
conservative mark resolution
    ↓
verified evidence
    ↓
global pair normalization and deduplication
    ↓
319 unique pair V1 corpus
    ↓
reproducible external similarity validation
```

The most important files are:

```text
fetch_ttab_reading_room.py
extract_reading_room_dupont.py
merge_ttab_verified_shards.py

ttab_global_factor1_pairs.csv
ttab_global_factor1_training.csv
ttab_global_factor1_evidence.csv
ttab_global_factor1_conflicts.csv

validate_ttab.py
feline.py
kbigram.py
kbigram_model.npz
lumera.py
lumera_model.npz
```

The most important methodological rule remains:

> **Use DuPont Factor 1 for the mark-similarity label. Never substitute the overall trademark-case result.**
