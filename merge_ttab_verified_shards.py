from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def identity_key(value) -> str:
    s = clean(value).upper().replace("’", "'")
    return " ".join(re.findall(r"[A-Z0-9]+", s))


def unordered_pair_key(mark_a, mark_b) -> str:
    a = identity_key(mark_a)
    b = identity_key(mark_b)
    return " || ".join(sorted((a, b)))


def normalize_label(value):
    if pd.isna(value):
        return pd.NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return pd.NA
    if number == 0:
        return 0
    if number == 1:
        return 1
    return pd.NA


def unique_join(values) -> str:
    out = []
    seen = set()
    for value in values:
        s = clean(value)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return " || ".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Merge high-confidence TTAB verified shard CSVs, globally deduplicate "
            "textual pairs, preserve evidence provenance, and quarantine conflicts."
        )
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more reading_room_dupont_verified.csv shard outputs.",
    )
    ap.add_argument(
        "--out-prefix",
        default="ttab_merged",
        help="Output prefix. Default: ttab_merged",
    )
    args = ap.parse_args()

    frames = []
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

        df = pd.read_csv(path)
        required = {"mark_a", "mark_b", "name_similarity_label"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )

        df = df.copy()
        df["source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise SystemExit("No inputs supplied.")

    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows["label"] = all_rows["name_similarity_label"].map(normalize_label)
    all_rows = all_rows[all_rows["label"].isin([0, 1])].copy()

    all_rows["pair_key"] = all_rows.apply(
        lambda r: unordered_pair_key(r["mark_a"], r["mark_b"]), axis=1
    )

    # Remove accidental duplicate evidence caused by overlapping acquisition
    # shards or by merging broad and targeted runs containing the same case.
    evidence_identity = ["pair_key", "label"]
    for col in ("proceeding_number", "appeal_decision_id"):
        if col in all_rows.columns:
            evidence_identity.append(col)

    all_rows = all_rows.drop_duplicates(
        subset=evidence_identity,
        keep="first",
    ).copy()

    label_counts = all_rows.groupby("pair_key")["label"].nunique()
    conflict_keys = set(label_counts[label_counts > 1].index)

    conflicts = all_rows[all_rows["pair_key"].isin(conflict_keys)].copy()
    clean_evidence = all_rows[~all_rows["pair_key"].isin(conflict_keys)].copy()

    pair_rows = []
    for pair_key, group in clean_evidence.groupby("pair_key", sort=False):
        first = group.iloc[0]
        label = int(first["label"])

        proceedings = (
            unique_join(group["proceeding_number"])
            if "proceeding_number" in group.columns
            else ""
        )
        source_files = unique_join(group["source_file"])

        pair_rows.append(
            {
                "mark_a": clean(first["mark_a"]),
                "mark_b": clean(first["mark_b"]),
                "label": label,
                "pair_key": pair_key,
                "evidence_count": len(group),
                "proceeding_count": (
                    group["proceeding_number"].astype(str).nunique()
                    if "proceeding_number" in group.columns
                    else len(group)
                ),
                "proceeding_numbers": proceedings,
                "source_files": source_files,
            }
        )

    pairs = pd.DataFrame(pair_rows)

    evidence_path = Path(f"{args.out_prefix}_evidence.csv")
    pairs_path = Path(f"{args.out_prefix}_pairs.csv")
    conflicts_path = Path(f"{args.out_prefix}_conflicts.csv")
    training_path = Path(f"{args.out_prefix}_training.csv")

    clean_evidence.to_csv(evidence_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    conflicts.to_csv(conflicts_path, index=False)

    if pairs.empty:
        training = pd.DataFrame(columns=["mark_a", "mark_b", "label"])
    else:
        training = pairs[["mark_a", "mark_b", "label"]].copy()
    training.to_csv(training_path, index=False)

    print("TTAB shard merge complete")
    print("=========================")
    print(f"Input files:             {len(args.inputs):,}")
    print(f"Input verified rows:     {sum(len(x) for x in frames):,}")
    print(f"Deduped evidence rows:   {len(all_rows):,}")
    print(f"Conflict evidence rows:  {len(conflicts):,}")
    print(f"Clean evidence rows:     {len(clean_evidence):,}")
    print(f"Unique clean pairs:      {len(pairs):,}")

    if not pairs.empty:
        print()
        print("Unique pair label breakdown:")
        print(pairs["label"].value_counts().sort_index(ascending=False).to_string())

    print()
    print("Outputs:")
    print(f"  {evidence_path}")
    print(f"  {pairs_path}")
    print(f"  {conflicts_path}")
    print(f"  {training_path}")


if __name__ == "__main__":
    main()
