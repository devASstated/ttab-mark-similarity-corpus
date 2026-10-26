from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

# Reuse the now-tested DuPont extraction logic.
from ttab_pipeline_dupont import (
    clean_ws,
    feature_dict,
    extract_factor1_window,
    classify_binary,
    FACTOR1_POS_RE,
    FACTOR1_NEG_RE,
    extract_overall_2d_window,
    OVERALL_POS_RE,
    OVERALL_NEG_RE,
)


def meta_text(value) -> str:
    # Convert Reading Room metadata values (including lists/dicts) to text.
    if value is None:
        return ""

    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            vv = meta_text(v)
            if vv:
                parts.append(f"{k}: {vv}")
        return " | ".join(parts)

    if isinstance(value, (list, tuple, set)):
        return " | ".join(
            x for x in (meta_text(v) for v in value) if x
        )

    return clean_ws(str(value))


def norm(s: str | None) -> str:
    return clean_ws(s).upper().replace("’", "'")


def identity_key(s: str | None) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", norm(s)))


def dedupe_marks(values: list[str]) -> list[str]:
    out = {}

    for v in values:
        v = clean_ws(v)

        if not v:
            continue

        k = identity_key(v)

        if not k:
            continue

        out.setdefault(k, v)

    return list(out.values())


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))

    return "\n\n".join(
        (p.extract_text() or "")
        for p in reader.pages
    )


def collect_markish_values(obj, key_hint="") -> list[str]:
    """
    Recursively pull strings from metadata structures whose keys look
    mark-related.

    This supports Reading Room schema variations without hard-coding one
    representation.
    """

    vals = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            kh = str(k).lower()

            if "mark" in kh:
                vals.extend(
                    collect_markish_values(
                        v,
                        kh,
                    )
                )

            elif isinstance(v, (dict, list)):
                vals.extend(
                    collect_markish_values(
                        v,
                        key_hint,
                    )
                )

    elif isinstance(obj, list):
        for v in obj:
            vals.extend(
                collect_markish_values(
                    v,
                    key_hint,
                )
            )

    elif isinstance(obj, str):
        s = clean_ws(obj)

        if s and (
            "mark" in key_hint
            or len(s) <= 200
        ):
            vals.append(s)

    return vals


def sanitize_mark_candidate(value: str) -> str:
    s = clean_ws(value)

    if not s:
        return ""

    # Strip inline goods/services block.
    s = re.split(
        r"\s*\[",
        s,
        maxsplit=1,
    )[0].strip()

    # Strip common metadata/design suffixes.
    s = re.sub(
        r"(?i)\s+and\s+(?:related\s+)?design(?:\s+mark)?\s*$",
        "",
        s,
    ).strip()

    s = re.sub(
        r"(?i)\s+and\s*$",
        "",
        s,
    ).strip()

    low = s.lower()

    bad_starts = (
        "is affirmed",
        "is reversed",
        "was affirmed",
        "was reversed",
        "the refusal",
        "refusal is",
        "registration is",
        "decision is",
    )

    if low.startswith(bad_starts):
        return ""

    if len(s) > 120:
        return ""

    if len(s.split()) > 20:
        return ""

    return s.strip(" ,;:-")


def looks_like_fallback_mark(value: str) -> bool:
    """
    Cheap precision guard for generic decision-text fallback extraction.

    This is intentionally conservative.

    The fallback exists only after structured metadata fails, so
    sentence-like prose is discarded instead of trying to rescue every
    possible mark.
    """

    s = sanitize_mark_candidate(value)

    if not s:
        return False

    # A genuine mark captured by these generic patterns should normally
    # begin with an uppercase alphabetic character or a digit.
    first_alpha = next(
        (
            c
            for c in s
            if c.isalpha()
        ),
        "",
    )

    if (
        first_alpha
        and not first_alpha.isupper()
    ):
        return False

    low = s.lower()

    prose_starts = (
        "affects ",
        "appears ",
        "by ",
        "contains ",
        "does ",
        "has ",
        "in standard ",
        "is ",
        "that ",
        "the ",
        "this ",
        "which ",
        "would ",
    )

    if low.startswith(prose_starts):
        return False

    prose_fragments = (
        " commercial impression",
        " likelihood of confusion",
        " scope of protection",
        " examining attorney",
        " evidence of ",
        " standard character form",
    )

    if any(
        fragment in low
        for fragment in prose_fragments
    ):
        return False

    # Long strings with several common prose words are almost certainly
    # text leakage rather than a trademark.
    words = re.findall(
        r"[A-Za-z]+",
        low,
    )

    prose_words = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "which",
        "with",
        "would",
    }

    if (
        len(words) >= 4
        and sum(
            word in prose_words
            for word in words
        ) >= 2
    ):
        return False

    return True


def split_structured_mark_field(raw) -> list[str]:
    if raw is None:
        return []

    if isinstance(
        raw,
        (dict, list),
    ):
        vals = [
            sanitize_mark_candidate(v)
            for v in collect_markish_values(raw)
        ]

        return dedupe_marks(
            [
                v
                for v in vals
                if v
            ]
        )

    s = str(raw)

    s = s.replace(
        chr(13) + chr(10),
        chr(10),
    )

    s = s.replace(
        chr(13),
        chr(10),
    ).strip(chr(10))

    if not s.strip():
        return []

    st = s.strip()

    if st[:1] in "[{":
        try:
            obj = json.loads(st)

            vals = [
                sanitize_mark_candidate(v)
                for v in collect_markish_values(obj)
            ]

            vals = [
                v
                for v in vals
                if v
            ]

            if vals:
                return dedupe_marks(vals)

        except Exception:
            pass

    header_lines = []

    for raw_line in s.split(chr(10)):
        line = clean_ws(raw_line)

        if line.startswith("["):
            break

        header_lines.append(line)

    # A standalone conjunction usually means multiple marks.
    # Leave these cases for review.
    if any(
        line.lower() == "and"
        for line in header_lines
    ):
        return []

    vals = []

    reject_prefixes = (
        "numerous registered ",
        "multiple ",
        "various ",
        "several ",
        "claim of ",
    )

    note_prefixes = (
        "translation:",
        "translation of ",
        "name/portrait/consent",
        "scope of ",
        "goods in ",
        "services in ",
    )

    for line in header_lines:
        if (
            not line
            or line.startswith("(")
        ):
            continue

        low = line.lower()

        if low in {"and", "or"}:
            continue

        if (
            low.startswith(note_prefixes)
            or low.startswith(reject_prefixes)
        ):
            continue

        if re.fullmatch(
            r"(?i)(?:common law|two registrations?|supplemental register)",
            line,
        ):
            continue

        cand = sanitize_mark_candidate(
            line
        )

        if cand:
            vals.append(cand)

    return dedupe_marks(vals)


def extract_text_marks_exa(
    text: str,
) -> tuple[list[str], list[str]]:
    """
    Conservative fallback for ex parte decisions.

    Returns:
        applicant_marks, cited_marks
    """

    applicant = []
    cited = []

    applicant_patterns = [
        (
            r'(?is)\bApplicant\s+(?:seeks|seek)\s+'
            r'registration\s+of\s+(?:the\s+)?mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
        (
            r'(?is)\bApplicant(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
    ]

    cited_patterns = [
        (
            r'(?is)\bcited\s+registration.{0,180}?'
            r'\bmark\s+["“]?([^"”\n,;]{1,100})'
        ),
        (
            r'(?is)\bregistrant(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
        (
            r'(?is)\bcited\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
    ]

    for pat in applicant_patterns:
        for match in re.finditer(
            pat,
            text,
        ):
            cand = sanitize_mark_candidate(
                match.group(1)
            )

            if looks_like_fallback_mark(cand):
                applicant.append(cand)

    for pat in cited_patterns:
        for match in re.finditer(
            pat,
            text,
        ):
            cand = sanitize_mark_candidate(
                match.group(1)
            )

            if looks_like_fallback_mark(cand):
                cited.append(cand)

    return (
        dedupe_marks(
            [
                x
                for x in applicant
                if x
            ]
        ),
        dedupe_marks(
            [
                x
                for x in cited
                if x
            ]
        ),
    )


def extract_text_marks_interpartes(
    text: str,
) -> tuple[list[str], list[str]]:
    opposer = []
    applicant = []

    opp_patterns = [
        (
            r'(?is)\bOpposer(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
        (
            r'(?is)\bPetitioner(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
    ]

    app_patterns = [
        (
            r'(?is)\bApplicant(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
        (
            r'(?is)\bRespondent(?:\'s)?\s+mark\s+'
            r'["“]?([^"”\n,;]{1,100})'
        ),
    ]

    for pat in opp_patterns:
        for match in re.finditer(
            pat,
            text,
        ):
            cand = sanitize_mark_candidate(
                match.group(1)
            )

            if looks_like_fallback_mark(cand):
                opposer.append(cand)

    for pat in app_patterns:
        for match in re.finditer(
            pat,
            text,
        ):
            cand = sanitize_mark_candidate(
                match.group(1)
            )

            if looks_like_fallback_mark(cand):
                applicant.append(cand)

    return (
        dedupe_marks(
            [
                x
                for x in opposer
                if x
            ]
        ),
        dedupe_marks(
            [
                x
                for x in applicant
                if x
            ]
        ),
    )


# ---------------------------------------------------------------------
# Precision-first Factor 1 mark recovery
# ---------------------------------------------------------------------

FACTOR1_MARK_A_RULES = [
    (
        "a_applicant_standard_character_mark_is",
        re.compile(
            r"\bApplicant[’']s\s+"
            r"standard[- ]character\s+mark\s+is\s+"
            r"("
            r"[A-Z0-9][A-Z0-9&'’.\-/]*"
            r"(?:\s+[A-Z0-9][A-Z0-9&'’.\-/]*){0,7}"
            r")"
            r"(?=\s*(?:,|\.|and\b|with\b|in\b))",
            re.I,
        ),
    ),

    (
        "a_word_is_dominant_portion",
        re.compile(
            r"\b(?:word|term)\s+"
            r"("
            r"[A-Z0-9][A-Z0-9&'’.\-/]*"
            r"(?:\s+[A-Z0-9][A-Z0-9&'’.\-/]*){0,5}"
            r")"
            r"\s+is\s+the\s+dominant\s+"
            r"(?:literal\s+)?portion\s+"
            r"of\s+(?:the\s+)?mark\b",
            re.I,
        ),
    ),

    (
        "a_literal_elements_in_applicant_mark",
        re.compile(
            r"\bliteral\s+elements?\s+"
            r"("
            r"[A-Z0-9][A-Z0-9&'’.\-/]*"
            r"(?:\s+[A-Z0-9][A-Z0-9&'’.\-/]*){0,7}"
            r")"
            r"\s+in\s+Applicant[’']s\s+mark\b",
            re.I,
        ),
    ),

    (
        "a_dominant_portion_remains",
        re.compile(
            r"\bdominant\s+portion\s+"
            r"of\s+Applicant[’']s\s+mark\s+"
            r"remains\s+"
            r"("
            r"[A-Z0-9][A-Z0-9&'’.\-/]*"
            r"(?:\s+[A-Z0-9][A-Z0-9&'’.\-/]*){0,5}"
            r")"
            r"(?=\s*(?:,|\.|which\b))",
            re.I,
        ),
    ),
]


FACTOR1_MARK_B_RULES = [
    (
        "b_cited_mark_literal_elements",
        re.compile(
            r"\b(?:the\s+)?cited\s+mark\s+"
            r"consists\s+of.{0,60}?"
            r"\bliteral\s+elements?\s+"
            r"("
            r"[A-Z0-9][A-Z0-9&'’.\-/]*"
            r"(?:\s+[A-Z0-9][A-Z0-9&'’.\-/]*){0,7}"
            r")"
            r"(?=\s*(?:\.|,|Again\b|which\b))",
            re.I,
        ),
    ),
]


def normalize_factor1_mark(
    s: str,
) -> str:
    s = clean_ws(s)

    s = re.sub(
        r"\s+\((?:stylized|in standard characters?)\)\s*$",
        "",
        s,
        flags=re.I,
    )

    return s.strip(
        " ,.;:()[]{}"
    )


def looks_like_written_mark(
    s: str,
) -> bool:
    """
    Precision-first guard for Factor 1 grammar recovery.

    A candidate must appear overwhelmingly uppercase in the decision text.
    This prevents ordinary prose from being promoted to a mark.
    """

    s = normalize_factor1_mark(s)

    if not s:
        return False

    if len(s) > 80:
        return False

    if len(s.split()) > 8:
        return False

    letters = [
        c
        for c in s
        if c.isalpha()
    ]

    if not letters:
        return False

    uppercase = sum(
        c.isupper()
        for c in letters
    )

    if (
        uppercase / len(letters)
        < 0.90
    ):
        return False

    low = s.lower()

    reject_exact = {
        "applicant",
        "registrant",
        "opposer",
        "petitioner",
        "registered",
        "cited",
        "mark",
        "registered mark",
        "cited mark",
        "standard character",
        "standard characters",
    }

    if low in reject_exact:
        return False

    reject_fragments = (
        "factor",
        "confusion",
        "commercial impression",
        "examining attorney",
        "ttabvue",
        "serial no",
        "registration no",
        "likelihood",
        "section 2",
    )

    if any(
        fragment in low
        for fragment in reject_fragments
    ):
        return False

    return True


def dedupe_factor1_candidates(
    candidates,
):
    out = []
    seen = set()

    for (
        rule_name,
        candidate,
    ) in candidates:
        candidate = normalize_factor1_mark(
            candidate
        )

        if not looks_like_written_mark(
            candidate
        ):
            continue

        key = identity_key(
            candidate
        )

        if (
            not key
            or key in seen
        ):
            continue

        seen.add(key)

        out.append(
            (
                rule_name,
                candidate,
            )
        )

    return out


def extract_factor1_mark_candidates(
    text: str,
    rules,
):
    candidates = []

    for (
        rule_name,
        pattern,
    ) in rules:
        for match in pattern.finditer(
            text
        ):
            candidates.append(
                (
                    rule_name,
                    match.group(1),
                )
            )

    return dedupe_factor1_candidates(
        candidates
    )


def recover_marks_from_factor1(
    factor1_text: str,
    side_a: list[str],
    side_b: list[str],
):
    """
    Precision-first second-pass mark recovery.

    A side is filled only when:
      1. it is currently unresolved, and
      2. exactly one grammar candidate survives.

    The Factor 1 grammar only recovers mark text.
    It never determines the similarity label.
    """

    recovery_rules = []

    if len(side_a) != 1:
        a_candidates = (
            extract_factor1_mark_candidates(
                factor1_text,
                FACTOR1_MARK_A_RULES,
            )
        )

        if len(a_candidates) == 1:
            rule_name, mark = (
                a_candidates[0]
            )

            side_a = [mark]

            recovery_rules.append(
                rule_name
            )

    if len(side_b) != 1:
        b_candidates = (
            extract_factor1_mark_candidates(
                factor1_text,
                FACTOR1_MARK_B_RULES,
            )
        )

        if len(b_candidates) == 1:
            rule_name, mark = (
                b_candidates[0]
            )

            side_b = [mark]

            recovery_rules.append(
                rule_name
            )

    return (
        side_a,
        side_b,
        recovery_rules,
    )


def classify_factor1(text: str):
    f1_text, f1_heading = (
        extract_factor1_window(text)
    )

    f1_cls, f1_evidence = (
        classify_binary(
            FACTOR1_POS_RE,
            FACTOR1_NEG_RE,
            f1_text,
        )
    )

    if f1_cls == "positive":
        label = 1
        result = "similar"

    elif f1_cls == "negative":
        label = 0
        result = "dissimilar"

    else:
        label = None
        result = f1_cls

    overall_window = (
        extract_overall_2d_window(
            text
        )
    )

    (
        overall_cls,
        overall_evidence,
    ) = classify_binary(
        OVERALL_POS_RE,
        OVERALL_NEG_RE,
        overall_window,
    )

    if overall_cls == "positive":
        overall_label = 1
        overall_result = (
            "confusion_likely"
        )

    elif overall_cls == "negative":
        overall_label = 0
        overall_result = (
            "confusion_not_likely"
        )

    else:
        overall_label = None
        overall_result = (
            overall_cls
        )

    return {
        "factor1_text":
            clean_ws(f1_text),

        "factor1_heading":
            f1_heading,

        "factor1_result":
            result,

        "name_similarity_label":
            label,

        "factor1_evidence":
            f1_evidence,

        "overall_2d_result":
            overall_result,

        "overall_confusion_label":
            overall_label,

        "overall_evidence":
            overall_evidence,
    }


def metadata_marks(
    row: dict,
    case_type: str,
) -> tuple[
    list[str],
    list[str],
    str,
]:
    """
    Returns:
        side_a,
        side_b,
        source

    EXA:
        applicant vs examining-attorney cited mark

    OPP/CAN:
        opposer/petitioner vs applicant/respondent
    """

    if case_type == "EXA":
        a = split_structured_mark_field(
            row.get(
                "applicantMarkGoodService",
                "",
            )
        )

        b = split_structured_mark_field(
            row.get(
                "examiningAttorneyMarkGoodCited",
                "",
            )
        )

        return (
            a,
            b,
            "reading_room_metadata",
        )

    a = split_structured_mark_field(
        row.get(
            "opposerMarkGoodService",
            "",
        )
    )

    b = split_structured_mark_field(
        row.get(
            "applicantMarkGoodService",
            "",
        )
    )

    return (
        a,
        b,
        "reading_room_metadata",
    )


def process_one(
    row: dict,
    pdf_dir: Path,
) -> dict:
    pno = str(
        row.get(
            "proceedingNumberDisplay"
        )
        or row.get(
            "proceedingNumber"
        )
        or ""
    )

    pno = re.sub(
        r"\D",
        "",
        pno,
    )

    case_type = clean_ws(
        row.get(
            "proceedingTypeCd",
            "",
        )
    ).upper()

    aid = clean_ws(
        str(
            row.get(
                "appealDecisionId",
                "",
            )
        )
    )

    pdf_path = (
        pdf_dir
        / f"{pno}-{case_type}-{aid}.pdf"
    )

    base = {
        "proceeding_number":
            pno,

        "case_type":
            case_type,

        "appeal_decision_id":
            aid,

        "decision":
            meta_text(
                row.get(
                    "decision",
                    "",
                )
            ),

        "decision_type_category":
            meta_text(
                row.get(
                    "decisionTypeCategory",
                    "",
                )
            ),

        "grounds_for_refusal_category":
            meta_text(
                row.get(
                    "groundsForRefusalCategory",
                    "",
                )
            ),

        "pdf_path":
            str(pdf_path),
    }

    if not pdf_path.exists():
        return {
            **base,
            "status":
                "pdf_missing",
            "auto_verified":
                False,
        }

    try:
        text = pdf_text(
            pdf_path
        )

    except Exception as ex:
        return {
            **base,
            "status":
                "pdf_text_error",
            "error":
                type(ex).__name__,
            "auto_verified":
                False,
        }

    analysis = classify_factor1(
        text
    )

    (
        side_a,
        side_b,
        source,
    ) = metadata_marks(
        row,
        case_type,
    )

    # Generic prose fallback only when structured metadata does not
    # resolve both sides.
    if (
        len(side_a) != 1
        or len(side_b) != 1
    ):
        if case_type == "EXA":
            ta, tb = (
                extract_text_marks_exa(
                    text
                )
            )

        else:
            ta, tb = (
                extract_text_marks_interpartes(
                    text
                )
            )

        if (
            len(side_a) != 1
            and len(ta) == 1
        ):
            side_a = ta

            source = (
                "decision_text_fallback"
            )

        if (
            len(side_b) != 1
            and len(tb) == 1
        ):
            side_b = tb

            source = (
                "decision_text_fallback"
            )

    # Precision-first second pass over the Factor 1 discussion.
    #
    # Grammar recovery may recover mark text, but it never determines
    # the similarity label.
    grammar_rules = []

    if (
        analysis[
            "name_similarity_label"
        ] is not None
        and (
            len(side_a) != 1
            or len(side_b) != 1
        )
    ):
        (
            side_a,
            side_b,
            grammar_rules,
        ) = recover_marks_from_factor1(
            analysis[
                "factor1_text"
            ],
            side_a,
            side_b,
        )

        if grammar_rules:
            source = (
                "factor1_grammar_fallback"
            )

    mark_a = (
        side_a[0]
        if len(side_a) == 1
        else ""
    )

    mark_b = (
        side_b[0]
        if len(side_b) == 1
        else ""
    )

    auto_verified = bool(
        analysis[
            "name_similarity_label"
        ] is not None
        and mark_a
        and mark_b
    )

    review_reasons = []

    if (
        analysis[
            "name_similarity_label"
        ] is None
    ):
        review_reasons.append(
            "factor1_"
            + str(
                analysis[
                    "factor1_result"
                ]
            )
        )

    if len(side_a) != 1:
        review_reasons.append(
            f"side_a_marks_{len(side_a)}"
        )

    if len(side_b) != 1:
        review_reasons.append(
            f"side_b_marks_{len(side_b)}"
        )

    out = {
        **base,
        **analysis,

        "status":
            "analyzed",

        "mark_a_candidates":
            " || ".join(side_a),

        "mark_b_candidates":
            " || ".join(side_b),

        "mark_a":
            mark_a,

        "mark_b":
            mark_b,

        "mark_resolution_source":
            source,

        "mark_recovery_rules":
            " || ".join(
                grammar_rules
            ),

        "auto_verified":
            auto_verified,

        "review_reason":
            ";".join(
                review_reasons
            ),
    }

    if (
        mark_a
        and mark_b
    ):
        out.update(
            feature_dict(
                mark_a,
                mark_b,
            )
        )

    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run the existing DuPont Factor-1 extractor over Reading Room "
            "final likelihood-of-confusion decisions."
        )
    )

    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path(
            "reading_room_2d_matches.jsonl"
        ),
    )

    ap.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(
            "reading_room_decisions_2d"
        ),
    )

    ap.add_argument(
        "--out-prefix",
        default=(
            "reading_room_dupont"
        ),
    )

    args = ap.parse_args()

    if not args.jsonl.exists():
        raise SystemExit(
            f"Missing: {args.jsonl}"
        )

    rows = []

    with args.jsonl.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    print(
        "Reading Room decisions loaded: "
        f"{len(rows):,}"
    )

    results = []

    for i, row in enumerate(
        rows,
        1,
    ):
        results.append(
            process_one(
                row,
                args.pdf_dir,
            )
        )

        if (
            i % 25 == 0
            or i == len(rows)
        ):
            print(
                f"Processed "
                f"{i}/{len(rows)} decisions..."
            )

    audit = pd.DataFrame(
        results
    )

    audit.to_csv(
        f"{args.out_prefix}_audit.csv",
        index=False,
    )

    if audit.empty:
        print(
            "No rows produced."
        )
        return

    verified = audit[
        audit[
            "auto_verified"
        ] == True
    ].copy()

    # Remove duplicate evidence rows occurring within the same
    # proceeding for the same normalized pair and label.
    if not verified.empty:
        verified["_pair_a"] = (
            verified[
                "mark_a"
            ].map(
                identity_key
            )
        )

        verified["_pair_b"] = (
            verified[
                "mark_b"
            ].map(
                identity_key
            )
        )

        verified = (
            verified.drop_duplicates(
                subset=[
                    "proceeding_number",
                    "_pair_a",
                    "_pair_b",
                    "name_similarity_label",
                ],
                keep="first",
            ).copy()
        )

        verified = (
            verified.drop(
                columns=[
                    "_pair_a",
                    "_pair_b",
                ]
            )
        )

    review = audit[
        audit[
            "auto_verified"
        ] != True
    ].copy()

    # Conflict protection across repeated textual pairs.
    if not verified.empty:
        verified[
            "pair_key"
        ] = verified.apply(
            lambda r: " || ".join(
                sorted(
                    [
                        identity_key(
                            r["mark_a"]
                        ),
                        identity_key(
                            r["mark_b"]
                        ),
                    ]
                )
            ),
            axis=1,
        )

        label_counts = (
            verified.groupby(
                "pair_key"
            )[
                "name_similarity_label"
            ].nunique()
        )

        bad_keys = set(
            label_counts[
                label_counts > 1
            ].index
        )

        conflicts = verified[
            verified[
                "pair_key"
            ].isin(
                bad_keys
            )
        ].copy()

        verified = verified[
            ~verified[
                "pair_key"
            ].isin(
                bad_keys
            )
        ].copy()

    else:
        conflicts = (
            pd.DataFrame()
        )

    verified.to_csv(
        f"{args.out_prefix}_verified.csv",
        index=False,
    )

    review.to_csv(
        f"{args.out_prefix}_review.csv",
        index=False,
    )

    conflicts.to_csv(
        f"{args.out_prefix}_conflicts.csv",
        index=False,
    )

    training_cols = [
        "mark_a",
        "mark_b",
        "name_similarity_label",
        "proceeding_number",
        "case_type",
        "appeal_decision_id",
        "factor1_result",
        "overall_2d_result",
        "jaro_winkler_sim",
        "norm_levenshtein_sim",
        "metaphone_match",
        "pdf_path",
    ]

    for c in training_cols:
        if c not in verified.columns:
            verified[c] = ""

    training = verified[
        training_cols
    ].rename(
        columns={
            "name_similarity_label":
                "label"
        }
    )

    training.to_csv(
        f"{args.out_prefix}_training.csv",
        index=False,
    )

    try:
        training.to_parquet(
            f"{args.out_prefix}_training.parquet",
            index=False,
        )

    except Exception:
        pass

    print()

    print(
        "Reading Room DuPont extraction complete"
    )

    print(
        "======================================="
    )

    print(
        f"Audit rows:             "
        f"{len(audit):,}"
    )

    print(
        f"Auto-verified pairs:    "
        f"{len(verified):,}"
    )

    print(
        f"Needs review:           "
        f"{len(review):,}"
    )

    print(
        f"Conflicting pair rows:  "
        f"{len(conflicts):,}"
    )

    if not verified.empty:
        print()

        print(
            "Verified label breakdown:"
        )

        print(
            verified[
                "name_similarity_label"
            ].value_counts().to_string()
        )

    print()

    print(
        "Outputs:"
    )

    print(
        f"  {args.out_prefix}_audit.csv"
    )

    print(
        f"  {args.out_prefix}_verified.csv"
    )

    print(
        f"  {args.out_prefix}_review.csv"
    )

    print(
        f"  {args.out_prefix}_conflicts.csv"
    )

    print(
        f"  {args.out_prefix}_training.csv"
    )


if __name__ == "__main__":
    main()
