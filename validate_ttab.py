#!/usr/bin/env python3
"""
validate_ttab.py — validate the similarity measures against adjudicated TTAB
DuPont Factor 1 labels.

WHY THIS SET MATTERS
    Every AUC this project has quoted so far came from lasa_refset.csv, whose
    negatives are RANDOM pairs. Separating documented LASA pairs from random
    pairs is an easy task, so those figures were only ever valid for ranking
    measures against each other. The TTAB Factor 1 corpus is the first set with
    negatives that a human adjudicator actually looked at and called
    dissimilar. Its labels were also not derived from any string metric
    (the corpus pipeline deliberately refuses to prefilter on Jaro-Winkler,
    edit distance or phonetic match), so the evaluation is not circular.

    That makes these the first numbers here that mean something in absolute
    terms. It also makes them much lower, and that is the point.

WHAT THIS SET IS NOT
    It is US trademark adjudication, not Indian drug names, and DuPont Factor 1
    weighs appearance, sound, CONNOTATION and overall commercial impression.
    The last two are outside what an orthographic measure can model at all:
    PHANTOM / APPARITION is labelled similar, BACIO / KISS is labelled
    dissimilar, THE PERFECT FACE / PARFAIT VISAGE is labelled similar. No
    character-alignment score will ever get those right. The script therefore
    measures the ceiling that semantics imposes, so the result is read as
    "how much of Factor 1 is orthographic" rather than as a grade.

THE DATA NEEDS AUDITING BEFORE IT NEEDS SCORING
    The mark strings come from USPTO metadata and PDF text, and they carry
    legal annotation along with the mark: '(HOTEL disclaimed)', '(and design)',
    '(two registrations)', '(Supplemental Register)'. 49% of rows contain a
    parenthetical. Scoring those strings raw measures the boilerplate as much
    as the marks — two rows that both end in '(and design)' share a long
    literal substring that has nothing to do with the marks. So three views
    are reported:

        raw     — the strings exactly as delivered
        clean   — quotes and parenthetical annotation removed
        strict  — clean, minus rows where there is no textual mark to score
                  (design marks, trade dress, multi-mark lists, visible
                  extraction failures). Every excluded row is PRINTED so the
                  exclusion rule can be argued with rather than trusted.

    Reporting all three is deliberate. If raw and clean disagree a lot, the raw
    number was partly measuring annotation.

ERROR FLOOR
    Ten rows have identical strings on both sides; four of them are labelled 0
    (NAUTICA/NAUTICA, REALTOR/REALTOR, DOCUVIEW/DOCUVIEW, UNIFRAME/UNIFRAME).
    Every similarity measure returns 1.0 on identical input, so those four are
    false positives at every threshold for every algorithm. They are an
    irreducible floor, not an algorithm failure, and the script quantifies them
    separately instead of letting them silently depress all six columns.

USAGE
    python validate_ttab.py ttab_global_factor1_training.csv
    python validate_ttab.py ttab_global_factor1_training.csv --view strict
    python validate_ttab.py ttab_global_factor1_training.csv --show-excluded
    python validate_ttab.py ttab_global_factor1_training.csv --boot 4000
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def roc_auc(y, s):
    """Rank-based AUC with mid-ranks for ties — the same estimator the rest of
    the project uses, so numbers stay comparable with bench_measures.py."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        i = np.where(s == v)[0]
        ranks[i] = ranks[i].mean()
    pos = y == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def average_precision(y, s):
    """PR-AUC. Ties are resolved as a block rather than in arbitrary order; a
    measure that gives many pairs the same score would otherwise be flattered
    or punished by however argsort happened to break them."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    order = np.argsort(-s)
    s, y = s[order], y[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    ap = 0.0
    tp = fp = 0
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        tp += int(y[i:j].sum())
        fp += (j - i) - int(y[i:j].sum())
        prec = tp / (tp + fp)
        ap += prec * int(y[i:j].sum()) / n_pos
        i = j
    return ap


def best_f1(y, s):
    """Sweep every achievable threshold and report the best F1 and where it is.

    This is an OPTIMISTIC, in-sample operating point — it is chosen using the
    labels it is then scored on. It says what the measure could do with a
    perfectly tuned cutoff, not what it would do on unseen pairs. Quoted as
    such, never as an accuracy claim.
    """
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    best = (0.0, 0.5, 0.0, 0.0, 0.0)
    for thr in np.unique(np.r_[s, 0.0, 1.0]):
        pred = s >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best[0]:
            best = (f, float(thr), p, r, (tp + tn) / len(y))
    return best


def boot_ci(y, s, fn, n=2000, seed=0):
    """Stratified bootstrap CI. With 319 pairs a point estimate on its own
    invites over-reading, so no AUC is printed here without its interval."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        ix = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
        v = fn(y[ix], s[ix])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------
# mark-string hygiene
# ----------------------------------------------------------------------
_SMART = str.maketrans({"‘": "'", "’": "'", "“": '"',
                        "”": '"', "–": "-", "—": "-"})

# A parenthetical in these strings is USPTO annotation, not part of the mark:
# '(HOTEL disclaimed)', '(and design)', '(stylized)', '(two registrations)'.
_PARENS = re.compile(r"\([^()]*\)")

# Visible extraction damage — PDF sentence fragments that reached the mark
# field. Each pattern was written against a row actually present in V1.
_BROKEN = [
    (r"^\s*[.,;:]", "starts with punctuation (sentence fragment)"),
    (r"(?i)\bwe (may|do) not\b", "decision prose, not a mark"),
    (r"(?i)\bis\s*$", "trailing verb (sentence fragment)"),
    (r"\]", "goods/services fragment"),
    (r"(?i)^as (applicant|registrant)\s*:", "role prefix left in field"),
]

# No textual mark exists to score at all.
_NONTEXT = [
    (r"(?i)^\s*design mark\s*$", "design mark"),
    (r"(?i)\bconfiguration of\b", "product/package configuration"),
    (r"(?i)\btrade dress\b", "trade dress"),
    (r"(?i)three dimensional", "3-D configuration"),
    (r"(?i)^\s*design\b", "design-only mark"),
    (r"(?i)stylized human figures", "figurative mark"),
    (r"(?i)^\s*(a |an )?design mark incorporating", "design mark"),
]

# Several marks packed into one field. The corpus spec sends multi-mark
# proceedings to review (handoff section 29); these survived into V1 anyway,
# and there is no principled way to pick which of the listed marks the Factor 1
# finding attached to.
_MULTI = [
    (r"(?i)family of marks", "family-of-marks claim"),
    (r"(?i)-formative marks", "formative-marks group"),
    (r"(?i)\b\d+ cited registrations", "multiple cited registrations"),
    (r"(?i)\b(two|three|four|numerous|several) (registrations|applications|"
     r"design marks|crest designs)", "multiple registrations"),
    (r"(?i)\ban alleged\b", "alleged mark family"),
    (r"(?i)\ba number of marks\b", "mark group"),
    (r"(?i)\bincluding the term\b", "mark group"),
]


def clean_mark(s):
    """Strip quoting and USPTO annotation; keep the mark text."""
    t = str(s).translate(_SMART).strip()
    t = t.strip('"').strip()
    prev = None
    while prev != t:                      # nested annotation needs repeating
        prev = t
        t = _PARENS.sub(" ", t)
    t = t.replace('"', " ")
    t = re.sub(r"\s+", " ", t).strip(" ,;:-").strip()
    return t


def classify(a_raw, b_raw, a_cl, b_cl):
    """Return a reason this row cannot be scored as a name pair, or None."""
    for side, raw, cl in (("a", a_raw, a_cl), ("b", b_raw, b_cl)):
        for pat, why in _NONTEXT:
            if re.search(pat, str(raw)):
                return f"non-textual mark ({why}, side {side})"
        for pat, why in _BROKEN:
            if re.search(pat, str(raw)):
                return f"extraction failure ({why}, side {side})"
        for pat, why in _MULTI:
            if re.search(pat, str(raw)):
                return f"multi-mark field ({why}, side {side})"
        if len(cl) < 2:
            return f"nothing left after cleaning (side {side})"
    return None


# ----------------------------------------------------------------------
# measures
# ----------------------------------------------------------------------
def m_edit(a, b):
    import jellyfish
    m = max(len(a), len(b))
    return 1.0 - jellyfish.levenshtein_distance(a, b) / m if m else 1.0


def m_jw(a, b):
    import jellyfish
    return jellyfish.jaro_winkler_similarity(a, b)


def m_dice(a, b):
    A = {a[i:i + 2] for i in range(len(a) - 1)} or {a}
    B = {b[i:i + 2] for i in range(len(b) - 1)} or {b}
    return 2 * len(A & B) / (len(A) + len(B)) if (A or B) else 0.0


BASELINES = {"edit": m_edit, "jaro_winkler": m_jw, "bigram_dice": m_dice}
OURS = ("kbigram", "lumera", "feline")


def load_ours(missing_ok=True):
    import importlib
    out = {}
    for name in OURS:
        try:
            fn = importlib.import_module(name).score
            fn("abc", "abd")      # force the lazy model load now, not mid-run
            out[name] = fn
        except Exception as e:
            if not missing_ok:
                raise
            first = str(e).splitlines()[0]
            print(f"  {name:<14} SKIPPED — {first}")
    return out


# ----------------------------------------------------------------------
def report(tag, a, b, y, scorers, args):
    print("\n" + "=" * 78)
    print(f"  {tag}")
    print("=" * 78)
    print(f"  pairs {len(y):,}   similar {int((y == 1).sum())}   "
          f"dissimilar {int((y == 0).sum())}   "
          f"base rate {y.mean():.3f}")

    cols = {}
    for name, fn in scorers.items():
        cols[name] = np.array([fn(x, z) for x, z in zip(a, b)], float)

    rows = []
    for name, s in cols.items():
        au = roc_auc(y, s)
        lo, hi = boot_ci(y, s, roc_auc, args.boot, args.seed)
        ap = average_precision(y, s)
        f1, thr, p, r, acc = best_f1(y, s)
        rows.append((name, au, lo, hi, ap, f1, thr, p, r, acc))
    rows.sort(key=lambda t: -t[1])

    print(f"\n  {'measure':<14}{'ROC-AUC':>9}{'95% CI':>18}{'PR-AUC':>9}"
          f"{'bestF1':>8}{'@thr':>7}{'prec':>7}{'rec':>7}{'acc':>7}")
    print(f"  {'-'*92}")
    for (name, au, lo, hi, ap, f1, thr, p, r, acc) in rows:
        print(f"  {name:<14}{au:>9.4f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{ap:>9.4f}{f1:>8.3f}{thr:>7.3f}{p:>7.3f}{r:>7.3f}{acc:>7.3f}")
    print(f"\n  chance ROC-AUC 0.500;  PR-AUC of a constant scorer "
          f"{y.mean():.3f} (the base rate).")
    print(f"  bestF1/@thr/prec/rec/acc are chosen on these same labels, so")
    print(f"  they are an upper bound on a tuned operating point, not an")
    print(f"  out-of-sample accuracy.")

    # Paired comparisons. The marginal CIs above overlap heavily, which is NOT
    # the same as the measures being indistinguishable: resampling each one
    # separately throws away the fact that they are scored on identical pairs.
    # Resampling the pairs once and taking the DIFFERENCE removes the shared
    # sampling noise and is the test that answers "is A actually better than B".
    pairs_to_test = [(x, y_) for x, y_ in
                     (("feline", "jaro_winkler"), ("feline", "kbigram"),
                      ("feline", "lumera"), ("kbigram", "lumera"),
                      ("kbigram", "bigram_dice"), ("lumera", "edit"))
                     if x in cols and y_ in cols]
    if pairs_to_test:
        rng = np.random.default_rng(args.seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        draws = [np.r_[rng.choice(pos, len(pos), True),
                       rng.choice(neg, len(neg), True)]
                 for _ in range(args.boot)]
        print(f"\n  PAIRED AUC DIFFERENCE (same resampled pairs both sides)")
        print(f"  {'comparison':<30}{'delta':>9}{'95% CI':>18}"
              f"{'P(delta>0)':>12}")
        print(f"  {'-'*69}")
        for x, y_ in pairs_to_test:
            d = np.array([roc_auc(y[ix], cols[x][ix]) -
                          roc_auc(y[ix], cols[y_][ix]) for ix in draws])
            d = d[np.isfinite(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {x+' - '+y_:<30}{d.mean():>+9.4f}"
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{(d > 0).mean():>12.3f}")
        print(f"  An interval spanning 0 means this set cannot separate the")
        print(f"  two measures; with ~280 pairs that is the common outcome and")
        print(f"  it is a statement about the set's size, not about the")
        print(f"  measures being equal.")

    # redundancy: a measure that tracks edit distance adds nothing
    if len(cols) > 1:
        names = list(cols)
        print(f"\n  correlation between measures")
        print(f"  {'':<14}" + "".join(f"{n[:10]:>11}" for n in names))
        for n1 in names:
            line = "".join(
                f"{np.corrcoef(cols[n1], cols[n2])[0, 1]:>11.3f}"
                for n2 in names)
            print(f"  {n1:<14}{line}")
    return cols


def ceiling(a, b, y, cols):
    """How much of Factor 1 is simply not an orthographic question."""
    print("\n" + "=" * 78)
    print("  WHAT NO ORTHOGRAPHIC MEASURE CAN REACH")
    print("=" * 78)

    ident = np.array([x.lower() == z.lower() for x, z in zip(a, b)])
    n_id = int(ident.sum())
    n_id0 = int((ident & (y == 0)).sum())
    print(f"\n  identical strings        : {n_id} rows, {n_id0} labelled "
          f"DISSIMILAR")
    if n_id0:
        print(f"    every measure returns 1.0 on identical input, so these "
              f"{n_id0} are")
        print(f"    false positives at every threshold for every algorithm.")
        for i in np.flatnonzero(ident & (y == 0)):
            print(f"      {a[i][:34]:<36} == {b[i][:34]}")

    ref = cols["feline"] if "feline" in cols else cols.get("edit")
    if ref is None:
        return
    ref = np.asarray(ref)
    lo_sim_pos = (ref < 0.35) & (y == 1)
    hi_sim_neg = (ref > 0.80) & (y == 0)
    print(f"\n  labelled SIMILAR but orthographically distant (<0.35): "
          f"{int(lo_sim_pos.sum())} rows")
    for i in np.flatnonzero(lo_sim_pos)[:10]:
        print(f"      {ref[i]:.2f}  {a[i][:30]:<32} / {b[i][:30]}")
    print(f"\n  labelled DISSIMILAR but orthographically close (>0.80): "
          f"{int(hi_sim_neg.sum())} rows")
    for i in np.flatnonzero(hi_sim_neg)[:10]:
        print(f"      {ref[i]:.2f}  {a[i][:30]:<32} / {b[i][:30]}")
    n_hard = int(lo_sim_pos.sum() + hi_sim_neg.sum())
    print(f"\n  {n_hard} of {len(y)} rows ({100*n_hard/len(y):.0f}%) point "
          f"the opposite way to")
    print(f"  their orthography. Factor 1 weighs connotation")
    print(f"  and commercial impression, and discounts disclaimed terms;")
    print(f"  those rows are the part of the label a character measure is not")
    print(f"  entitled to get right. They cap the achievable AUC on this set.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?",
                    default="ttab_global_factor1_training.csv")
    ap.add_argument("--view", default="all",
                    choices=("all", "raw", "clean", "strict"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-excluded", action="store_true",
                    help="print every row the strict view drops, with reason")
    ap.add_argument("--baselines-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["mark_a", "mark_b", "label"])
    df["label"] = df["label"].astype(int)
    print("=" * 78)
    print("  TTAB DuPont FACTOR 1 — EXTERNAL VALIDATION")
    print("=" * 78)
    print(f"source          : {args.csv}")
    print(f"rows            : {len(df):,}  "
          f"(similar {int((df.label == 1).sum())}, "
          f"dissimilar {int((df.label == 0).sum())})")

    a_raw = df.mark_a.astype(str).str.translate(_SMART).tolist()
    b_raw = df.mark_b.astype(str).str.translate(_SMART).tolist()
    a_cl = [clean_mark(x) for x in a_raw]
    b_cl = [clean_mark(x) for x in b_raw]
    y = df.label.to_numpy(int)

    reasons = [classify(ar, br, ac, bc)
               for ar, br, ac, bc in zip(a_raw, b_raw, a_cl, b_cl)]
    keep = np.array([r is None for r in reasons])

    changed = sum(1 for ar, ac, br, bc in zip(a_raw, a_cl, b_raw, b_cl)
                  if ar != ac or br != bc)
    print(f"annotation strip: {changed} of {len(df)} rows had a parenthetical "
          f"or quoting removed")
    print(f"strict view     : keeps {int(keep.sum())}, drops "
          f"{int((~keep).sum())}")
    from collections import Counter
    for why, n in Counter(r.split(" (")[0] for r in reasons
                          if r).most_common():
        print(f"    {why:<28}{n:>4}")
    if args.show_excluded:
        print("\n  rows dropped by the strict view:")
        for i, r in enumerate(reasons):
            if r:
                print(f"    [{y[i]}] {r}")
                print(f"          {a_raw[i][:62]}")
                print(f"          {b_raw[i][:62]}")

    scorers = dict(BASELINES)
    if not args.baselines_only:
        print()
        scorers.update(load_ours())
    if not scorers:
        sys.exit("no measures available")

    views = []
    if args.view in ("all", "raw"):
        views.append(("VIEW 1 — RAW strings, exactly as delivered",
                      a_raw, b_raw, y))
    if args.view in ("all", "clean"):
        views.append(("VIEW 2 — CLEAN (quotes and USPTO annotation removed)",
                      a_cl, b_cl, y))
    if args.view in ("all", "strict"):
        views.append(("VIEW 3 — STRICT (clean, non-name rows dropped)",
                      [x for x, k in zip(a_cl, keep) if k],
                      [x for x, k in zip(b_cl, keep) if k],
                      y[keep]))

    last = None
    for tag, a, b, yy in views:
        last = (report(tag, a, b, np.asarray(yy), scorers, args), a, b,
                np.asarray(yy))

    if last:
        cols, a, b, yy = last
        ceiling(a, b, yy, cols)

    print("\n" + "=" * 78)
    print("  HOW TO QUOTE THIS")
    print("=" * 78)
    print("  The STRICT view is the defensible headline: it is the subset where")
    print("  a name pair genuinely exists to be scored. Report it with its CI,")
    print("  and say the labels are US trademark Factor 1 findings rather than")
    print("  medication-error evidence. These numbers are far below the")
    print("  lasa_refset figures because that set's negatives were random")
    print("  pairs; the drop is the synthetic-negative inflation becoming")
    print("  visible, not a regression in any algorithm.")
    print()


if __name__ == "__main__":
    main()
