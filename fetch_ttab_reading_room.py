from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests


API = "https://ttab-reading-room.uspto.gov/ttab-efoia-api/decision/search"
PDF_BASE = "https://ttab-reading-room.uspto.gov/cms/rest/legal-proceeding"
UA = "LASA-TTAB-ReadingRoom-API/2.0 research corpus acquisition"

FINAL_RE = re.compile(r"(?i)\bfinal\s+decision\b")


def clean(v) -> str:
    if v is None:
        return ""

    if isinstance(v, (list, tuple)):
        return " | ".join(clean(x) for x in v)

    return re.sub(r"\s+", " ", str(v)).strip()


def is_final(row: dict) -> bool:
    return bool(
        FINAL_RE.search(
            clean(row.get("decisionTypeCategory"))
        )
    )


def passes_decision_filter(
    row: dict,
    decision_filter: str,
) -> bool:
    """
    Acquisition-only filtering.

    This determines which TTAB decisions are downloaded.
    It never determines the similarity label.
    """

    if decision_filter == "all":
        return True

    case_type = clean(
        row.get("proceedingTypeCd")
    ).upper()

    decision = clean(
        row.get("decision")
    ).lower()

    if decision_filter == "opposition-dismissed":
        return (
            case_type == "OPP"
            and "dismiss" in decision
            and "sustain" not in decision
        )

    raise ValueError(
        f"Unknown decision filter: {decision_filter}"
    )


def candidate_pdf_urls(
    row: dict,
) -> list[str]:
    pno = clean(
        row.get("proceedingNumberDisplay")
        or row.get("proceedingNumber")
    )

    pno = re.sub(
        r"\D",
        "",
        pno,
    )

    ptype = clean(
        row.get("proceedingTypeCd")
    ).upper()

    if not pno or not ptype:
        return []

    candidates = []

    # appealDecisionId has proven to map to the
    # Reading Room decision event number.
    aid = clean(
        row.get("appealDecisionId")
    )

    if aid.isdigit():
        candidates.append(
            f"{PDF_BASE}/{pno}/decision/"
            f"{ptype}_{aid}.pdf"
        )

    # Fallback candidates for schema variations.
    for field in (
        "documentImageId",
        "documentId",
    ):
        raw = clean(
            row.get(field)
        )

        if not raw:
            continue

        m = re.search(
            r"(?i)\b(EXA|OPP|CAN|CNU|RE)[_-]?(\d+)\b",
            raw,
        )

        if m:
            candidates.append(
                f"{PDF_BASE}/{pno}/decision/"
                f"{m.group(1).upper()}_"
                f"{m.group(2)}.pdf"
            )

        if raw.isdigit():
            candidates.append(
                f"{PDF_BASE}/{pno}/decision/"
                f"{ptype}_{raw}.pdf"
            )

    out = []
    seen = set()

    for url in candidates:
        if url in seen:
            continue

        seen.add(url)
        out.append(url)

    return out


def fetch_valid_pdf(
    session: requests.Session,
    row: dict,
    out_dir: Path,
    timeout: int,
) -> tuple[str, str, str]:
    urls = candidate_pdf_urls(row)

    if not urls:
        return "", "", "no_pdf_candidates"

    errors = []

    for url in urls:
        try:
            response = session.get(
                url,
                timeout=timeout,
                allow_redirects=True,
            )

        except requests.RequestException as ex:
            errors.append(
                f"{type(ex).__name__}:{url}"
            )
            continue

        ctype = (
            response.headers.get(
                "content-type"
            )
            or ""
        ).lower()

        if (
            response.status_code == 200
            and (
                response.content.startswith(b"%PDF")
                or "application/pdf" in ctype
            )
        ):
            pno = re.sub(
                r"\D",
                "",
                clean(
                    row.get(
                        "proceedingNumberDisplay"
                    )
                    or row.get(
                        "proceedingNumber"
                    )
                ),
            )

            ptype = clean(
                row.get("proceedingTypeCd")
            ).upper()

            aid = (
                clean(
                    row.get(
                        "appealDecisionId"
                    )
                )
                or "decision"
            )

            dest = (
                out_dir
                / f"{pno}-{ptype}-{aid}.pdf"
            )

            dest.write_bytes(
                response.content
            )

            return (
                response.url,
                str(dest),
                "downloaded",
            )

        errors.append(
            f"http_{response.status_code}:{url}"
        )

    return (
        "",
        "",
        ";".join(errors)[:1500],
    )


def post_search(
    session: requests.Session,
    start: int,
    count: int,
    search_text: str,
    timeout: int,
) -> dict:
    payload = {
        "dateRangeData": {},
        "facetData": {},
        "parameterData": {},
        "recordTotalQuantity": count,
        "searchText": search_text,
        "sortDataBag": [
            {
                "issueDate": "desc"
            }
        ],
        "recordStartNumber": start,
    }

    response = session.post(
        API,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Query TTAB Reading Room full-text search for "
            "likelihood-of-confusion / du Pont decisions."
        )
    )

    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help=(
            "Maximum accepted decisions to keep. "
            "If omitted, bounded --end-index shards "
            "scan their entire assigned range."
        ),
    )

    ap.add_argument(
        "--page-size",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--max-pages",
        type=int,
        default=100,
    )

    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Inclusive raw Reading Room query offset "
            "at which scanning begins."
        ),
    )

    ap.add_argument(
        "--end-index",
        type=int,
        default=None,
        help=(
            "Exclusive raw Reading Room query offset "
            "at which scanning stops."
        ),
    )

    ap.add_argument(
        "--decision-filter",
        choices=[
            "all",
            "opposition-dismissed",
        ],
        default="all",
        help=(
            "Acquisition filter only. "
            "It never determines the corpus label."
        ),
    )

    ap.add_argument(
        "--delay",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--timeout",
        type=int,
        default=35,
    )

    ap.add_argument(
        "--query",
        default=(
            '"likelihood of confusion" '
            'AND "du Pont"'
        ),
        help=(
            "Reading Room search syntax."
        ),
    )

    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "reading_room_decisions_2d"
        ),
    )

    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reading_room_2d_manifest.csv"
        ),
    )

    ap.add_argument(
        "--raw-jsonl",
        type=Path,
        default=Path(
            "reading_room_2d_matches.jsonl"
        ),
    )

    ap.add_argument(
        "--no-download",
        action="store_true",
    )

    args = ap.parse_args()

    if args.start_index < 0:
        raise SystemExit(
            "--start-index must be >= 0"
        )

    if (
        args.end_index is not None
        and args.end_index <= args.start_index
    ):
        raise SystemExit(
            "--end-index must be greater "
            "than --start-index"
        )

    if (
        args.target is not None
        and args.target <= 0
    ):
        raise SystemExit(
            "--target must be > 0"
        )

    # Open-ended historical behavior remains convenient.
    # Bounded shards have no artificial target by default.
    target_limit = args.target

    if (
        target_limit is None
        and args.end_index is None
    ):
        target_limit = 50

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": UA,
            "Accept":
                "application/json, text/plain, */*",
            "Content-Type":
                "application/json",
            "Origin":
                "https://ttab-reading-room.uspto.gov",
            "Referer":
                "https://ttab-reading-room.uspto.gov/cms/",
        }
    )

    accepted = []
    seen_ids = set()

    scanned = 0
    query_total = None

    with args.raw_jsonl.open(
        "w",
        encoding="utf-8",
    ) as raw_f:

        for page_no in range(
            args.max_pages
        ):
            start = (
                args.start_index
                + page_no * args.page_size
            )

            if (
                args.end_index is not None
                and start >= args.end_index
            ):
                break

            request_count = args.page_size

            if args.end_index is not None:
                request_count = min(
                    request_count,
                    args.end_index - start,
                )

            data = post_search(
                session,
                start=start,
                count=request_count,
                search_text=args.query,
                timeout=args.timeout,
            )

            results = (
                data.get("results")
                or []
            )

            if query_total is None:
                query_total = data.get(
                    "recordTotalQuantity"
                )

            if not results:
                print(
                    f"Page {page_no + 1}: "
                    "no results; stopping."
                )
                break

            scanned += len(results)

            accepted_this_page = 0

            for result_offset, row in enumerate(
                results
            ):
                query_index = (
                    start + result_offset
                )

                # Search query establishes the LOC +
                # du Pont evidence universe.
                if not is_final(row):
                    continue

                if not passes_decision_filter(
                    row,
                    args.decision_filter,
                ):
                    continue

                stable_id = clean(
                    row.get("id")
                    or row.get("documentId")
                )

                if (
                    stable_id
                    and stable_id in seen_ids
                ):
                    continue

                if stable_id:
                    seen_ids.add(
                        stable_id
                    )

                raw_f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if args.no_download:
                    (
                        pdf_url,
                        pdf_path,
                        status,
                    ) = (
                        "",
                        "",
                        "discovered",
                    )

                else:
                    (
                        pdf_url,
                        pdf_path,
                        status,
                    ) = fetch_valid_pdf(
                        session,
                        row,
                        args.out_dir,
                        args.timeout,
                    )

                accepted.append(
                    {
                        "id":
                            clean(
                                row.get("id")
                            ),

                        "proceeding_number":
                            clean(
                                row.get(
                                    "proceedingNumberDisplay"
                                )
                                or row.get(
                                    "proceedingNumber"
                                )
                            ),

                        "case_type":
                            clean(
                                row.get(
                                    "proceedingTypeCd"
                                )
                            ),

                        "date_issued":
                            clean(
                                row.get(
                                    "issueDateStr"
                                )
                            ),

                        "party_name":
                            clean(
                                row.get(
                                    "partyName"
                                )
                            ),

                        "issue":
                            clean(
                                row.get(
                                    "issue"
                                )
                            ),

                        "decision":
                            clean(
                                row.get(
                                    "decision"
                                )
                            ),

                        "decision_type_category":
                            clean(
                                row.get(
                                    "decisionTypeCategory"
                                )
                            ),

                        "grounds_for_refusal_category":
                            clean(
                                row.get(
                                    "groundsForRefusalCategory"
                                )
                            ),

                        "issue_type_category":
                            clean(
                                row.get(
                                    "issueTypeCategory"
                                )
                            ),

                        "decision_issues":
                            clean(
                                row.get(
                                    "decisionIssues"
                                )
                            ),

                        "opposer_mark_goods_services":
                            clean(
                                row.get(
                                    "opposerMarkGoodService"
                                )
                            ),

                        "applicant_mark_goods_services":
                            clean(
                                row.get(
                                    "applicantMarkGoodService"
                                )
                            ),

                        "examining_attorney_mark_goods_cited":
                            clean(
                                row.get(
                                    "examiningAttorneyMarkGoodCited"
                                )
                            ),

                        "appeal_decision_id":
                            clean(
                                row.get(
                                    "appealDecisionId"
                                )
                            ),

                        "document_image_id":
                            clean(
                                row.get(
                                    "documentImageId"
                                )
                            ),

                        "document_id":
                            clean(
                                row.get(
                                    "documentId"
                                )
                            ),

                        "pdf_url":
                            pdf_url,

                        "pdf_path":
                            pdf_path,

                        "download_status":
                            status,

                        "query":
                            args.query,

                        "query_index":
                            query_index,

                        "query_start_index":
                            args.start_index,

                        "query_end_index":
                            (
                                args.end_index
                                if args.end_index
                                is not None
                                else ""
                            ),

                        "decision_filter":
                            args.decision_filter,
                    }
                )

                accepted_this_page += 1

                if (
                    target_limit is not None
                    and len(accepted)
                    >= target_limit
                ):
                    break

            if target_limit is None:
                accepted_text = (
                    f"accepted={len(accepted)}"
                )

            else:
                accepted_text = (
                    f"accepted={len(accepted)}"
                    f"/{target_limit}"
                )

            print(
                f"API page {page_no + 1}: "
                f"query_scanned={scanned:,}, "
                f"accepted_added="
                f"{accepted_this_page}, "
                f"{accepted_text}"
            )

            if (
                target_limit is not None
                and len(accepted)
                >= target_limit
            ):
                break

            time.sleep(
                args.delay
            )

    fields = [
        "id",
        "proceeding_number",
        "case_type",
        "date_issued",
        "party_name",
        "issue",
        "decision",
        "decision_type_category",
        "grounds_for_refusal_category",
        "issue_type_category",
        "decision_issues",
        "opposer_mark_goods_services",
        "applicant_mark_goods_services",
        "examining_attorney_mark_goods_cited",
        "appeal_decision_id",
        "document_image_id",
        "document_id",
        "pdf_url",
        "pdf_path",
        "download_status",
        "query",
        "query_index",
        "query_start_index",
        "query_end_index",
        "decision_filter",
    ]

    with args.manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            accepted
        )

    downloaded = sum(
        x["download_status"]
        == "downloaded"
        for x in accepted
    )

    print()
    print(
        "TTAB Reading Room 2(d) acquisition complete"
    )
    print(
        "==========================================="
    )

    print(
        f"Search query:           "
        f"{args.query}"
    )

    print(
        f"Decision filter:        "
        f"{args.decision_filter}"
    )

    print(
        f"Query start index:      "
        f"{args.start_index:,}"
    )

    if args.end_index is not None:
        print(
            f"Query end index:        "
            f"{args.end_index:,} "
            "(exclusive)"
        )

    if query_total is not None:
        print(
            f"Query result universe:  "
            f"{query_total:,}"
        )

    print(
        f"Query records scanned:  "
        f"{scanned:,}"
    )

    print(
        f"Final decisions kept:   "
        f"{len(accepted):,}"
    )

    print(
        f"PDFs downloaded:        "
        f"{downloaded:,}"
    )

    print(
        f"Manifest:               "
        f"{args.manifest}"
    )

    print(
        f"Raw matched JSONL:      "
        f"{args.raw_jsonl}"
    )

    print(
        f"PDF directory:          "
        f"{args.out_dir}"
    )

    if accepted:
        print()
        print(
            "First retained decision:"
        )

        first = accepted[0]

        for key in (
            "proceeding_number",
            "case_type",
            "decision_type_category",
            "grounds_for_refusal_category",
            "decision",
            "pdf_url",
            "download_status",
        ):
            print(
                f"  {key}: "
                f"{first.get(key, '')}"
            )


if __name__ == "__main__":
    main()
