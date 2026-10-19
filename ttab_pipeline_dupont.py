from __future__ import annotations

import argparse
import glob
import html
import io
import os
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urljoin

import jellyfish
import pandas as pd
import requests
from pypdf import PdfReader
from rapidfuzz import fuzz, utils


# =============================================================================
# CONFIG
# =============================================================================

TTABVUE_BASE = "https://ttabvue.uspto.gov/ttabvue/"
TTABVUE_EVENT = urljoin(TTABVUE_BASE, "v")
UA = "LASA-TTAB-DuPont/2.0 research corpus builder"

# Procedural terms are NOT negative similarity labels.  They are only audit flags.
PROCEDURAL_HISTORY_RE = re.compile(
    r"(?i)\b("
    r"w/draw|withdraw(?:al|n)?|settlement|settle|"
    r"dismissed\s+w/o\s+prej|dismissed\s+without\s+prejudice|"
    r"default|failure\s+to\s+prosecute|consent"
    r")\b"
)

DECISION_EVENT_RE = re.compile(
    r"(?i)\b("
    r"bd\s+decision|board\s+decision|final\s+decision|decision"
    r")\b"
)

# A decision must contain substantive likelihood-of-confusion analysis.
SECTION2D_RE = re.compile(
    r"(?i)("
    r"section\s+2\s*\(\s*d\s*\)|"
    r"likelihood\s+of\s+confusion|"
    r"du\s*pont"
    r")"
)

DUPONT_RE = re.compile(r"(?i)\bdu\s*pont\b")

FACTOR1_HEADING_PATTERNS = [
    re.compile(r"(?i)similarity\s+(?:or\s+dissimilarity\s+)?of\s+the\s+marks"),
    re.compile(r"(?i)similarity\s*/\s*dissimilarity\s+of\s+the\s+marks"),
    re.compile(r"(?i)comparison\s+of\s+the\s+marks"),
    re.compile(r"(?i)(?:the\s+)?first\s+du\s*pont\s+factor"),
    re.compile(
        r"(?i)similarity\s+or\s+dissimilarity\s+of\s+the\s+marks\s+"
        r"(?:in\s+their\s+entiret(?:y|ies)\s+)?as\s+to\s+"
        r"(?:appearance|sound|meaning|connotation|commercial\s+impression)"
    ),
]

NEXT_SECTION_RE = re.compile(
    r"(?im)^\s*(?:[A-Z]\.|[IVXLC]+\.|\d+\.)?\s*(?:"
    r"similarity\s+(?:or\s+dissimilarity\s+)?of\s+(?:the\s+)?goods|"
    r"relatedness\s+of\s+(?:the\s+)?goods|"
    r"goods\s+and\s+services|"
    r"channels?\s+of\s+trade|"
    r"conditions?\s+of\s+(?:sale|purchase)|"
    r"sophistication|"
    r"strength\s+of\s+(?:the\s+)?mark|"
    r"fame\s+of\s+(?:the\s+)?mark|"
    r"actual\s+confusion|"
    r"conclusion|"
    r"balancing\s+the\s+factors"
    r")\b"
)

# Strong Factor-1 conclusions.  These are intentionally conservative.
FACTOR1_POS_PATTERNS = [
    r"\bmarks?\s+(?:are|is)\s+(?:highly\s+|very\s+|substantially\s+)?similar\b",
    r"\bmarks?\s+(?:are|is)\s+more\s+similar\s+than\s+dissimilar\b",
    r"\bmarks?\s+(?:are|is)\s+similar\s+in\s+(?:appearance|sound|meaning|connotation|commercial\s+impression)",
    r"\bwe\s+(?:find|conclude)\b.{0,260}\bmarks?\b.{0,120}\bsimilar\b",
    r"\bsimilarit(?:y|ies)\b.{0,120}\boutweigh\b.{0,120}\bdifference(?:s)?\b",
    r"\boverall\b.{0,120}\b(?:look|sound|appear)\b.{0,120}\bsimilar\b",
    r"\bfirst\s+du\s*pont\s+factor\b.{0,320}\bweighs?\s+(?:somewhat\s+|slightly\s+|strongly\s+)?in\s+favor\s+of\s+(?:a\s+finding\s+of\s+)?likelihood\s+of\s+confusion\b",
    r"\bsimilarity\s+of\s+the\s+marks\b.{0,260}\bweighs?\s+(?:somewhat\s+|slightly\s+|strongly\s+)?in\s+favor\s+of\s+(?:a\s+finding\s+of\s+)?likelihood\s+of\s+confusion\b",
]
FACTOR1_NEG_PATTERNS = [
    r"\bmarks?\s+(?:are|is)\s+(?:highly\s+|very\s+|substantially\s+)?dissimilar\b",
    r"\bmarks?\s+(?:are|is)\s+more\s+dissimilar\s+than\s+similar\b",
    r"\bmarks?\s+(?:are|is)\s+(?:sufficiently\s+)?different\b",
    r"\bwe\s+(?:find|conclude)\b.{0,260}\bmarks?\b.{0,120}\b(?:dissimilar|different)\b",
    r"\bdifference(?:s)?\b.{0,120}\boutweigh\b.{0,120}\bsimilarit(?:y|ies)\b",
    r"\bfirst\s+du\s*pont\s+factor\b.{0,320}\bweighs?\s+(?:somewhat\s+|slightly\s+|strongly\s+)?against\s+(?:a\s+finding\s+of\s+)?likelihood\s+of\s+confusion\b",
    r"\bdissimilarity\s+of\s+the\s+marks\b.{0,260}\bweighs?\s+(?:somewhat\s+|slightly\s+|strongly\s+)?against\s+(?:a\s+finding\s+of\s+)?likelihood\s+of\s+confusion\b",
]
FACTOR1_POS_RE = [re.compile(x, re.I | re.S) for x in FACTOR1_POS_PATTERNS]
FACTOR1_NEG_RE = [re.compile(x, re.I | re.S) for x in FACTOR1_NEG_PATTERNS]

OVERALL_POS_PATTERNS = [
    r"\bwe\s+(?:find|conclude|hold)\b.{0,220}\blikelihood\s+of\s+confusion\b",
    r"\bconfusion\s+(?:is|would\s+be)\s+likely\b",
    r"\blikelihood\s+of\s+confusion\s+(?:exists|has\s+been\s+established|is\s+established)\b",
    r"\bsection\s+2\s*\(\s*d\s*\)\b.{0,300}\b(?:refusal\s+is\s+affirmed|opposition\s+is\s+sustained|sustain(?:ed)?)\b",
]
OVERALL_NEG_PATTERNS = [
    r"\bwe\s+(?:find|conclude|hold)\b.{0,220}\bno\s+likelihood\s+of\s+confusion\b",
    r"\bconfusion\s+(?:is|would\s+be)\s+not\s+likely\b",
    r"\bno\s+likelihood\s+of\s+confusion\b",
    r"\bfailed\s+to\s+(?:establish|prove)\b.{0,180}\blikelihood\s+of\s+confusion\b",
    r"\bsection\s+2\s*\(\s*d\s*\)\b.{0,300}\b(?:refusal\s+is\s+reversed|opposition\s+is\s+dismissed)\b",
]
OVERALL_POS_RE = [re.compile(x, re.I | re.S) for x in OVERALL_POS_PATTERNS]
OVERALL_NEG_RE = [re.compile(x, re.I | re.S) for x in OVERALL_NEG_PATTERNS]


# =============================================================================
# HELPERS
# =============================================================================

def clean_ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_mark(mark: str | None) -> str:
    # Conservative normalization for identity/dedup only.
    s = clean_ws(mark).upper().replace("’", "'")
    return s


def iter_input_xml(raw_dir: Path):
    """Yield (source_id, xml_bytes) from .xml and .zip inputs without extracting zips."""
    for fp in sorted(raw_dir.glob("*.xml")):
        try:
            yield fp.name, fp.read_bytes()
        except OSError:
            continue

    for zp in sorted(raw_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as z:
                for member in z.namelist():
                    if member.lower().endswith(".xml"):
                        try:
                            yield f"{zp.name}:{member}", z.read(member)
                        except Exception:
                            continue
        except zipfile.BadZipFile:
            continue


def text_of(node, path: str) -> str:
    e = node.find(path)
    return clean_ws(e.text if e is not None else "")


def mark_occurrences(text: str, mark: str) -> int:
    """
    Count reasonably literal mark mentions.
    For alphanumeric marks, use token-ish boundaries so e.g. CAT does not match EDUCATION.
    """
    mark = clean_ws(mark)
    if not mark:
        return 0
    escaped = re.escape(mark)
    escaped = escaped.replace(r"\ ", r"\s+")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '\-/.&+]*", mark):
        pat = re.compile(rf"(?i)(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
    else:
        pat = re.compile(escaped, re.I)
    return len(pat.findall(text))


def first_match(patterns, text: str):
    hits = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m)
    return min(hits, key=lambda x: x.start()) if hits else None


def classify_binary(patterns_pos, patterns_neg, text: str) -> tuple[str, str]:
    """
    Returns (classification, evidence).
    conflicting -> both strong directions found.
    """
    pos_hits = [m for p in patterns_pos if (m := p.search(text))]
    neg_hits = [m for p in patterns_neg if (m := p.search(text))]

    if pos_hits and neg_hits:
        ev = clean_ws(
            text[max(0, min(pos_hits[0].start(), neg_hits[0].start()) - 250):
                 min(len(text), max(pos_hits[0].end(), neg_hits[0].end()) + 400)]
        )
        return "conflicting", ev
    if pos_hits:
        m = min(pos_hits, key=lambda x: x.start())
        return "positive", clean_ws(text[max(0, m.start()-350):min(len(text), m.end()+650)])
    if neg_hits:
        m = min(neg_hits, key=lambda x: x.start())
        return "negative", clean_ws(text[max(0, m.start()-350):min(len(text), m.end()+650)])
    return "unclear", ""


def extract_factor1_window(text: str, max_chars: int = 14000) -> tuple[str, str]:
    candidate_patterns = list(FACTOR1_HEADING_PATTERNS) + [
        re.compile(
            r"(?im)^\\s*(?:[A-Z]\\.|[IVXLC]+\\.|\\d+\\.)\\s*"
            r"Similarity\\s+or\\s+Dissimilarity\\s+of\\s+"
            r"(?:the\\s+)?(?:Parties['’]\\s+)?Marks\\b"
        ),
        re.compile(
            r"(?im)^\\s*(?:[A-Z]\\.|[IVXLC]+\\.|\\d+\\.)\\s*"
            r"(?:The\\s+)?Similarity\\s+of\\s+(?:the\\s+)?Marks\\b"
        ),
    ]

    candidates = []
    seen = set()

    for pat in candidate_patterns:
        for m in pat.finditer(text):
            start = m.start()
            if start in seen:
                continue
            seen.add(start)

            tail = text[start:start + max_chars]
            min_chars = min(4500, len(tail))
            mnext = NEXT_SECTION_RE.search(tail[min_chars:])
            if mnext:
                tail = tail[:min_chars + mnext.start()]

            score = 0
            heading = clean_ws(m.group(0))

            if re.match(
                r"(?i)^\\s*(?:[A-Z]\\.|[IVXLC]+\\.|\\d+\\.)\\s*",
                heading
            ):
                score += 6

            if re.search(r"(?i)parties['’]\\s+marks", heading):
                score += 3

            if re.search(r"(?i)first\\s+du\\s*pont\\s+factor", tail):
                score += 5

            if re.search(
                r"(?i)appearance.{0,120}sound.{0,120}"
                r"(?:connotation|meaning).{0,160}commercial\\s+impression",
                tail,
                re.S,
            ):
                score += 3

            pos = any(p.search(tail) for p in FACTOR1_POS_RE)
            neg = any(p.search(tail) for p in FACTOR1_NEG_RE)
            if pos or neg:
                score += 8
            if pos and neg:
                score -= 2

            if re.search(
                r"(?i)similarity\\s+or\\s+dissimilarity\\s+of\\s+"
                r"the\\s+marks\\s+and\\s+the\\s+goods\\s+or\\s+services",
                heading,
            ):
                score -= 8

            candidates.append((score, start, tail, heading))

    if not candidates:
        return "", ""

    score, start, tail, heading = max(candidates, key=lambda x: (x[0], x[1]))
    return tail, heading

def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n\n".join(chunks), len(reader.pages)


def feature_dict(a: str, b: str) -> dict:
    t1, t2 = a.strip().lower(), b.strip().lower()
    fuzzy_ratio = fuzz.ratio(t1, t2, processor=utils.default_process)
    fuzzy_partial = fuzz.partial_ratio(t1, t2, processor=utils.default_process)
    fuzzy_token_sort = fuzz.token_sort_ratio(t1, t2, processor=utils.default_process)
    fuzzy_token_set = fuzz.token_set_ratio(t1, t2, processor=utils.default_process)
    lev = jellyfish.levenshtein_distance(t1, t2)
    max_len = max(len(t1), len(t2))
    jw = jellyfish.jaro_winkler_similarity(t1, t2)
    m1, m2 = jellyfish.metaphone(t1), jellyfish.metaphone(t2)
    return {
        "exact_match": int(t1 == t2),
        "fuzzy_ratio": round(fuzzy_ratio, 2),
        "fuzzy_partial_ratio": round(fuzzy_partial, 2),
        "fuzzy_token_sort_ratio": round(fuzzy_token_sort, 2),
        "fuzzy_token_set_ratio": round(fuzzy_token_set, 2),
        "jaro_winkler_sim": round(jw * 100.0, 2),
        "norm_levenshtein_sim": round((1.0 - lev / max_len) * 100.0, 2) if max_len else 100.0,
        "levenshtein_dist": lev,
        "metaphone_match": int(bool(m1 and m2 and m1 == m2)),
    }



def mark_identity_key(mark: str | None) -> str:
    s = clean_ws(mark).upper().replace("’", "'")
    return " ".join(re.findall(r"[A-Z0-9]+", s))


def mark_search_key(mark: str | None) -> str:
    return "".join(re.findall(r"[A-Z0-9]+", clean_ws(mark).upper()))


def compact_search_text(text: str) -> str:
    return "".join(re.findall(r"[A-Z0-9]+", (text or "").upper()))


def mark_occurrences_flexible(text: str, mark: str) -> int:
    literal = mark_occurrences(text, mark)
    if literal:
        return literal

    key = mark_search_key(mark)
    if len(key) < 6:
        return 0

    return compact_search_text(text).count(key)

# =============================================================================
# XML CASE MODEL
# =============================================================================

@dataclass
class DecisionEvent:
    identifier: str
    event_text: str
    entry_date: str
    object_id: str


@dataclass
class Case:
    proceeding_number: str
    proceeding_type: str
    source_file: str
    plaintiff_marks: list[str]
    defendant_marks: list[str]
    decision_events: list[DecisionEvent]
    all_events: list[str]
    procedural_history_flag: bool


def parse_case(xml_bytes: bytes, source_file: str) -> Case | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    pno = clean_ws(root.findtext(".//proceeding-number", default=""))
    ptype = clean_ws(root.findtext(".//proceeding-type", default=""))
    if not pno or not ptype:
        return None

    def get_marks(xpath: str):
        seen, out = set(), []
        for n in root.findall(xpath):
            if not n.text:
                continue
            m = clean_ws(n.text)
            k = norm_mark(m)
            if m and k not in seen:
                seen.add(k)
                out.append(m)
        return out

    plaintiffs = get_marks(".//plaintiffs//ttab-property/mark-explanation")
    defendants = get_marks(".//defendants//ttab-property/mark-explanation")

    events = []
    decisions = []
    for e in root.findall(".//prosecution-history-event"):
        txt = clean_ws(e.attrib.get("event-text", ""))
        if txt:
            events.append(txt)
        ident = clean_ws(e.attrib.get("identifier", ""))
        if ident and DECISION_EVENT_RE.search(txt):
            decisions.append(DecisionEvent(
                identifier=ident,
                event_text=txt,
                entry_date=clean_ws(e.attrib.get("entry-date", "")),
                object_id=clean_ws(e.attrib.get("object-id", "")),
            ))

    # Highest event number first: final substantive decisions are commonly late in history.
    def intish(x):
        try:
            return int(x.identifier)
        except Exception:
            return -1
    decisions.sort(key=intish, reverse=True)

    return Case(
        proceeding_number=pno,
        proceeding_type=ptype,
        source_file=source_file,
        plaintiff_marks=plaintiffs,
        defendant_marks=defendants,
        decision_events=decisions,
        all_events=events,
        procedural_history_flag=bool(PROCEDURAL_HISTORY_RE.search(" ".join(events))),
    )


# =============================================================================
# DECISION RETRIEVAL
# =============================================================================

def direct_pdf_url(case: Case, eno: str) -> str:
    return urljoin(
        TTABVUE_BASE,
        f"ttabvue-{case.proceeding_number}-{case.proceeding_type}-{eno}.pdf"
    )


def event_url(case: Case, eno: str) -> str:
    return (
        f"{TTABVUE_EVENT}?eno={eno}"
        f"&pno={case.proceeding_number}"
        f"&pty={case.proceeding_type}"
    )


def is_pdf_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("content-type") or "").lower()
    return resp.status_code == 200 and (
        resp.content.startswith(b"%PDF") or "application/pdf" in ctype
    )


def html_pdf_links(base_url: str, body: str) -> list[str]:
    body = html.unescape(body)
    # href/src/data and raw URL strings.
    found = re.findall(
        r"""(?i)(?:href|src|data)\s*=\s*["']([^"']+\.pdf(?:\?[^"']*)?)["']""",
        body
    )
    found += re.findall(r"""(?i)(https?://[^\s"'<>]+\.pdf(?:\?[^\s"'<>]*)?)""", body)
    out = []
    seen = set()
    for x in found:
        u = urljoin(base_url, x)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


READING_ROOM_BASE = "https://ttab-reading-room.uspto.gov/cms/rest/legal-proceeding/"


def reading_room_pdf_url(case: Case, eno: str) -> str:
    return (
        f"{READING_ROOM_BASE}"
        f"{case.proceeding_number}/decision/"
        f"{case.proceeding_type}_{eno}.pdf"
    )


def fetch_decision_pdf(
    session: requests.Session,
    case: Case,
    event: DecisionEvent,
    timeout: int = 35,
) -> tuple[bytes | None, str, str]:
    # Retrieval order:
    #   1. TTAB Reading Room direct PDF
    #   2. TTABVUE direct PDF convention
    #   3. TTABVUE event endpoint / HTML wrapper

    rr_url = reading_room_pdf_url(case, event.identifier)
    tv_pdf_url = direct_pdf_url(case, event.identifier)
    tv_event_url = event_url(case, event.identifier)

    errors = []

    try:
        resp = session.get(rr_url, timeout=timeout, allow_redirects=True)
        if is_pdf_response(resp):
            return resp.content, resp.url, "ok_reading_room"
        errors.append(f"reading_room_http_{resp.status_code}")
    except requests.RequestException as ex:
        errors.append(f"reading_room_{type(ex).__name__}")

    try:
        resp = session.get(tv_pdf_url, timeout=timeout, allow_redirects=True)
        if is_pdf_response(resp):
            return resp.content, resp.url, "ok_ttabvue_direct"
        errors.append(f"ttabvue_direct_http_{resp.status_code}")
    except requests.RequestException as ex:
        errors.append(f"ttabvue_direct_{type(ex).__name__}")

    try:
        resp = session.get(tv_event_url, timeout=timeout, allow_redirects=True)

        if is_pdf_response(resp):
            return resp.content, resp.url, "ok_ttabvue_event"

        ctype = (resp.headers.get("content-type") or "").lower()
        if resp.status_code == 200 and (
            "html" in ctype or resp.text.lstrip().startswith("<")
        ):
            for pdf_url in html_pdf_links(resp.url, resp.text):
                try:
                    pr = session.get(
                        pdf_url,
                        timeout=timeout,
                        allow_redirects=True,
                    )
                    if is_pdf_response(pr):
                        return pr.content, pr.url, "ok_ttabvue_html_link"
                except requests.RequestException:
                    continue

        errors.append(f"ttabvue_event_http_{resp.status_code}")

    except requests.RequestException as ex:
        errors.append(f"ttabvue_event_{type(ex).__name__}")

    return None, "", ";".join(errors)

# =============================================================================
# SUBSTANTIVE DECISION ANALYSIS
# =============================================================================

def is_substantive_2d(text: str) -> tuple[bool, str]:
    """
    Require meaningful text plus Section 2(d)/likelihood-of-confusion language
    and a DuPont signal.  This excludes short withdrawal/default orders.
    """
    cleaned = clean_ws(text)
    if len(cleaned) < 1500:
        return False, "decision_text_too_short"
    if not SECTION2D_RE.search(text):
        return False, "no_section2d_language"
    if not DUPONT_RE.search(text):
        return False, "no_dupont_language"
    f1, _ = extract_factor1_window(text)
    if not f1:
        return False, "no_factor1_section"
    return True, "substantive_2d"


def resolve_pair(case: Case, decision_text: str, factor1_text: str) -> tuple[str, str, str, str]:
    def collapse_variants(marks):
        groups = {}
        for mark in marks:
            key = mark_identity_key(mark)
            if not key:
                continue
            groups.setdefault(key, []).append(mark)

        collapsed = []
        for key, variants in groups.items():
            rep = sorted(
                variants,
                key=lambda s: (
                    1 if re.search(r"[^\\w)]$", s.strip()) else 0,
                    len(s),
                    s,
                ),
            )[0]
            collapsed.append(rep)
        return collapsed

    P = collapse_variants(case.plaintiff_marks)
    D = collapse_variants(case.defendant_marks)

    if len(P) == 1 and len(D) == 1:
        return (
            P[0],
            D[0],
            "xml_singleton_normalized",
            "one unique normalized textual mark on each side",
        )

    p_hits = [(m, mark_occurrences_flexible(factor1_text, m)) for m in P]
    d_hits = [(m, mark_occurrences_flexible(factor1_text, m)) for m in D]

    p_present = [(m, c) for m, c in p_hits if c > 0]
    d_present = [(m, c) for m, c in d_hits if c > 0]

    if len(p_present) == 1 and len(d_present) == 1:
        a, ac = p_present[0]
        b, bc = d_present[0]
        return (
            a,
            b,
            "factor1_unique_mentions_normalized",
            f"Factor1 mentions: {a} x{ac}; {b} x{bc}",
        )

    evidence = (
        "plaintiff Factor1 mentions="
        + "; ".join(f"{m}:{c}" for m, c in p_hits)
        + " | defendant Factor1 mentions="
        + "; ".join(f"{m}:{c}" for m, c in d_hits)
    )
    return "", "", "unresolved_multi_mark", evidence


def extract_overall_2d_window(text: str) -> str:
    anchors = [
        re.compile(
            r"(?im)^\s*(?:[A-Z]\.|[IVXLC]+\.|\d+\.)?\s*"
            r"Balancing\s+the\s+Factors\b"
        ),
        re.compile(
            r"(?im)^\s*(?:[A-Z]\.|[IVXLC]+\.|\d+\.)?\s*"
            r"Conclusion\b"
        ),
        re.compile(r"(?im)^\s*Decision\b"),
    ]

    starts = []
    for pat in anchors:
        starts.extend(m.start() for m in pat.finditer(text))

    if starts:
        start = max(starts)
        return text[start:start + 10000]

    return text[-8000:]

def analyze_decision(case: Case, text: str) -> dict:
    substantive, substantive_reason = is_substantive_2d(text)
    f1_text, f1_heading = extract_factor1_window(text)

    f1_cls, f1_evidence = classify_binary(FACTOR1_POS_RE, FACTOR1_NEG_RE, f1_text)
    overall_window = extract_overall_2d_window(text)
    overall_cls, overall_evidence = classify_binary(
        OVERALL_POS_RE, OVERALL_NEG_RE, overall_window
    )

    mark_a, mark_b, pair_resolution, pair_evidence = resolve_pair(
        case, text, f1_text
    ) if f1_text else ("", "", "no_factor1", "")

    if f1_cls == "positive":
        similarity_label = 1
    elif f1_cls == "negative":
        similarity_label = 0
    else:
        similarity_label = None

    if overall_cls == "positive":
        overall_label = 1
    elif overall_cls == "negative":
        overall_label = 0
    else:
        overall_label = None

    auto_verified = bool(
        substantive
        and similarity_label is not None
        and mark_a and mark_b
        and pair_resolution in {
            "xml_singleton",
            "factor1_unique_mentions",
            "xml_singleton_normalized",
            "factor1_unique_mentions_normalized",
        }
    )

    review_reasons = []
    if not substantive:
        review_reasons.append(substantive_reason)
    if not f1_text:
        review_reasons.append("factor1_missing")
    if f1_cls in {"unclear", "conflicting"}:
        review_reasons.append(f"factor1_{f1_cls}")
    if not mark_a or not mark_b:
        review_reasons.append(pair_resolution)
    if overall_cls in {"unclear", "conflicting"}:
        review_reasons.append(f"overall_{overall_cls}")
    if case.procedural_history_flag:
        # Informational only: substantive PDF can override procedural history.
        review_reasons.append("procedural_history_present")

    return {
        "substantive_2d": substantive,
        "substantive_reason": substantive_reason,
        "factor1_heading": f1_heading,
        "factor1_result": (
            "similar" if f1_cls == "positive"
            else "dissimilar" if f1_cls == "negative"
            else f1_cls
        ),
        "name_similarity_label": similarity_label,
        "overall_2d_result": (
            "confusion_likely" if overall_cls == "positive"
            else "confusion_not_likely" if overall_cls == "negative"
            else overall_cls
        ),
        "overall_confusion_label": overall_label,
        "mark_plaintiff": mark_a,
        "mark_defendant": mark_b,
        "pair_resolution": pair_resolution,
        "pair_evidence": pair_evidence,
        "factor1_evidence": f1_evidence,
        "overall_evidence": overall_evidence,
        "factor1_text": clean_ws(f1_text),
        "auto_verified": auto_verified,
        "review_reason": ";".join(dict.fromkeys(review_reasons)),
    }


# =============================================================================
# PER-CASE PROCESSING
# =============================================================================

def process_case(case: Case, cache_dir: Path, session_factory, timeout: int) -> list[dict]:
    base = {
        "proceeding_number": case.proceeding_number,
        "proceeding_type": case.proceeding_type,
        "source_file": case.source_file,
        "plaintiff_marks_xml": " || ".join(case.plaintiff_marks),
        "defendant_marks_xml": " || ".join(case.defendant_marks),
        "plaintiff_unique_mark_count": len(case.plaintiff_marks),
        "defendant_unique_mark_count": len(case.defendant_marks),
        "procedural_history_flag": case.procedural_history_flag,
    }

    if not case.plaintiff_marks or not case.defendant_marks:
        return [{**base, "stage_status": "missing_marks", "auto_verified": False}]
    if not case.decision_events:
        return [{**base, "stage_status": "no_decision_event", "auto_verified": False}]

    session = session_factory()
    rows = []

    # Try decision-like events newest first and keep every substantive candidate;
    # final dedup later chooses the best auto-verified row per proceeding/pair.
    for event in case.decision_events:
        stem = f"{case.proceeding_number}-{case.proceeding_type}-{event.identifier}"
        pdf_path = cache_dir / f"{stem}.pdf"
        txt_path = cache_dir / f"{stem}.txt"

        pdf_bytes = None
        resolved_url = ""
        fetch_status = ""

        if pdf_path.exists():
            try:
                pdf_bytes = pdf_path.read_bytes()
                fetch_status = "cached_pdf"
                resolved_url = direct_pdf_url(case, event.identifier)
            except OSError:
                pass

        if pdf_bytes is None:
            pdf_bytes, resolved_url, fetch_status = fetch_decision_pdf(
                session, case, event, timeout=timeout
            )
            if pdf_bytes:
                try:
                    pdf_path.write_bytes(pdf_bytes)
                except OSError:
                    pass

        if not pdf_bytes:
            rows.append({
                **base,
                "decision_event_id": event.identifier,
                "decision_event_text": event.event_text,
                "decision_event_date": event.entry_date,
                "decision_object_id": event.object_id,
                "decision_url": resolved_url or direct_pdf_url(case, event.identifier),
                "stage_status": "decision_fetch_failed",
                "fetch_status": fetch_status,
                "auto_verified": False,
            })
            continue

        if txt_path.exists():
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace")
                page_count = ""
            except OSError:
                text = ""
                page_count = ""
        else:
            try:
                text, page_count = extract_pdf_text(pdf_bytes)
                try:
                    txt_path.write_text(text, encoding="utf-8")
                except OSError:
                    pass
            except Exception as ex:
                rows.append({
                    **base,
                    "decision_event_id": event.identifier,
                    "decision_event_text": event.event_text,
                    "decision_event_date": event.entry_date,
                    "decision_object_id": event.object_id,
                    "decision_url": resolved_url,
                    "stage_status": "pdf_text_error",
                    "fetch_status": fetch_status,
                    "pdf_error": type(ex).__name__,
                    "auto_verified": False,
                })
                continue

        analysis = analyze_decision(case, text)
        row = {
            **base,
            "decision_event_id": event.identifier,
            "decision_event_text": event.event_text,
            "decision_event_date": event.entry_date,
            "decision_object_id": event.object_id,
            "decision_url": resolved_url or direct_pdf_url(case, event.identifier),
            "stage_status": "analyzed",
            "fetch_status": fetch_status,
            "page_count": page_count,
            "decision_text_chars": len(text),
            **analysis,
        }

        if analysis.get("mark_plaintiff") and analysis.get("mark_defendant"):
            row.update(feature_dict(
                analysis["mark_plaintiff"],
                analysis["mark_defendant"]
            ))
        rows.append(row)

        # If this is a clean substantive auto-label, prefer it and stop trying older
        # decision events for the same proceeding.
        if analysis["auto_verified"]:
            break

    return rows


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


# =============================================================================
# EXPORT
# =============================================================================

def safe_parquet(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def export_results(rows: list[dict], out_prefix: str):
    all_df = pd.DataFrame(rows)
    all_path = Path(f"{out_prefix}_audit.csv")
    all_df.to_csv(all_path, index=False)

    if all_df.empty:
        print("No analyzed rows.")
        return

    verified = all_df[
        all_df.get("auto_verified", pd.Series(False, index=all_df.index)).astype(str).str.lower()
        .isin(["true", "1"])
    ].copy()

    # Optional no-op if boolean dtype retained.
    if "auto_verified" in all_df.columns and all_df["auto_verified"].dtype == bool:
        verified = all_df[all_df["auto_verified"]].copy()

    # Keep one authoritative row per exact proceeding/pair/label.
    if not verified.empty:
        verified["pair_key"] = verified.apply(
            lambda r: " || ".join(sorted([
                norm_mark(r.get("mark_plaintiff", "")),
                norm_mark(r.get("mark_defendant", "")),
            ])),
            axis=1
        )
        verified = verified.drop_duplicates(
            subset=["proceeding_number", "pair_key", "name_similarity_label"]
        )

        # Cross-proceeding contradictory labels are preserved outside training output.
        label_counts = verified.groupby("pair_key")["name_similarity_label"].nunique()
        conflict_keys = set(label_counts[label_counts > 1].index)
        conflicts = verified[verified["pair_key"].isin(conflict_keys)].copy()
        verified = verified[~verified["pair_key"].isin(conflict_keys)].copy()
    else:
        conflicts = pd.DataFrame()

    review = all_df[
        ~all_df.index.isin(
            all_df[all_df.get("auto_verified", False) == True].index
            if "auto_verified" in all_df.columns and all_df["auto_verified"].dtype == bool
            else []
        )
    ].copy()

    # More reliable review split for mixed/object dtype.
    if "auto_verified" in all_df.columns:
        mask = all_df["auto_verified"].astype(str).str.lower().isin(["true", "1"])
        review = all_df[~mask].copy()

    verified_path = Path(f"{out_prefix}_verified.csv")
    review_path = Path(f"{out_prefix}_review.csv")
    conflict_path = Path(f"{out_prefix}_conflicts.csv")
    verified.to_csv(verified_path, index=False)
    review.to_csv(review_path, index=False)
    conflicts.to_csv(conflict_path, index=False)

    # Training subset: NO prefilter on Jaro-Winkler.
    training_cols = [
        "mark_plaintiff", "mark_defendant", "name_similarity_label",
        "proceeding_number", "decision_url",
        "factor1_result", "overall_2d_result",
        "jaro_winkler_sim", "norm_levenshtein_sim",
        "metaphone_match",
    ]
    for c in training_cols:
        if c not in verified.columns:
            verified[c] = ""
    training = verified[training_cols].copy()
    training = training.rename(columns={"name_similarity_label": "label"})
    training_path = Path(f"{out_prefix}_training.csv")
    training.to_csv(training_path, index=False)
    safe_parquet(training, Path(f"{out_prefix}_training.parquet"))

    print("\nDuPont extraction complete")
    print("==========================")
    print(f"Audit rows:             {len(all_df):,}")
    print(f"Auto-verified pairs:    {len(verified):,}")
    print(f"Needs review:           {len(review):,}")
    print(f"Conflicting pair rows:  {len(conflicts):,}")
    if not verified.empty:
        print("\nVerified name-similarity labels:")
        print(verified["name_similarity_label"].value_counts(dropna=False).to_string())

    print("\nOutputs:")
    print(f"  {all_path}")
    print(f"  {verified_path}")
    print(f"  {review_path}")
    print(f"  {conflict_path}")
    print(f"  {training_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(
    raw_dir: str = "raw",
    out_prefix: str = "ttab_dupont",
    cache_dir: str = "decision_cache",
    download_workers: int = 4,
    limit_cases: int = 0,
    timeout: int = 35,
):
    raw = Path(raw_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    cases = []
    source_count = 0
    parse_fail = 0

    print(f"Scanning '{raw}' for XML and ZIP inputs...")
    for source_id, xml_bytes in iter_input_xml(raw):
        source_count += 1
        c = parse_case(xml_bytes, source_id)
        if c is None:
            parse_fail += 1
        else:
            cases.append(c)

    # Deduplicate repeated copies of the same proceeding, keeping the richest XML
    # (more mark properties + more decision events).
    by_pno = {}
    for c in cases:
        score = len(c.plaintiff_marks) + len(c.defendant_marks) + len(c.decision_events)
        old = by_pno.get((c.proceeding_number, c.proceeding_type))
        if old is None:
            by_pno[(c.proceeding_number, c.proceeding_type)] = c
        else:
            old_score = len(old.plaintiff_marks) + len(old.defendant_marks) + len(old.decision_events)
            if score > old_score:
                by_pno[(c.proceeding_number, c.proceeding_type)] = c

    cases = list(by_pno.values())
    cases.sort(key=lambda x: x.proceeding_number)
    if limit_cases:
        cases = cases[:limit_cases]

    print(f"XML entries read:       {source_count:,}")
    print(f"Unique proceedings:     {len(cases):,}")
    print(f"Parse failures:         {parse_fail:,}")
    print(f"Download workers:       {download_workers}")
    print("No similarity threshold is applied before ground-truth labeling.")

    all_rows = []
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as ex:
        futs = {
            ex.submit(process_case, c, cache, make_session, timeout): c
            for c in cases
        }
        done = 0
        for fut in as_completed(futs):
            c = futs[fut]
            done += 1
            try:
                all_rows.extend(fut.result())
            except Exception as exn:
                all_rows.append({
                    "proceeding_number": c.proceeding_number,
                    "proceeding_type": c.proceeding_type,
                    "source_file": c.source_file,
                    "stage_status": "case_error",
                    "case_error": repr(exn),
                    "auto_verified": False,
                })
            if done % 50 == 0 or done == len(cases):
                print(f"Processed {done}/{len(cases)} proceedings...")

    export_results(all_rows, out_prefix)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a conservative TTAB DuPont Factor-1 name-similarity corpus. "
            "Procedural case outcomes are never used as 0/1 similarity labels."
        )
    )
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--out-prefix", default="ttab_dupont")
    ap.add_argument("--cache-dir", default="decision_cache")
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=35)
    args = ap.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        out_prefix=args.out_prefix,
        cache_dir=args.cache_dir,
        download_workers=args.download_workers,
        limit_cases=args.limit_cases,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
