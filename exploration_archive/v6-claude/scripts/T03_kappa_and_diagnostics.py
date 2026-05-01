"""Compute Cohen's kappa and additional diagnostics from the T03 crosstabs.

Treats seniority_native as the comparison reference (where available).
"""

from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jihgaboot/gabor/job-research")
TBL = ROOT / "exploration" / "tables" / "T03"

LEVELS = ["entry", "associate", "mid-senior", "director"]


def load_crosstab(path: Path):
    with path.open() as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    return rows


def cohen_kappa(matrix: dict, labels) -> tuple[float, float, int]:
    """Standard Cohen's kappa.

    matrix[(actual, predicted)] = count.
    labels is the shared label set.
    Returns (kappa, observed_agreement, N).
    """
    N = sum(matrix.values())
    if N == 0:
        return (float("nan"), float("nan"), 0)
    # Observed agreement (diagonal)
    po = sum(matrix.get((l, l), 0) for l in labels) / N
    # Marginals
    row_totals = {l: 0 for l in labels}
    col_totals = {l: 0 for l in labels}
    for (a, p), c in matrix.items():
        if a in row_totals and p in col_totals:
            row_totals[a] += c
            col_totals[p] += c
    pe = sum(
        (row_totals[l] / N) * (col_totals[l] / N) for l in labels
    )
    if pe == 1.0:
        return (float("nan"), po, N)
    return ((po - pe) / (1.0 - pe), po, N)


def per_class_recall(matrix: dict, labels):
    """Per-class recall using native as reference: P(final=L | native=L)."""
    out = {}
    for l in labels:
        tot = sum(c for (a, p), c in matrix.items() if a == l)
        if tot == 0:
            out[l] = (None, 0)
            continue
        correct = matrix.get((l, l), 0)
        out[l] = (correct / tot, tot)
    return out


def process(tag: str, path: Path, out_rows: list):
    rows = load_crosstab(path)
    # Build two matrices:
    #   A: strict — excludes rows with native=null and final in {null, unknown}
    #   B: lenient — includes final='unknown' as a wrong answer (still excludes native=null)
    mat_strict = defaultdict(int)
    mat_lenient = defaultdict(int)
    total_rows = 0
    for r in rows:
        native = r["seniority_native"]
        final = r["seniority_final"]
        n = int(r["n"])
        total_rows += n
        if native in (None, "", "<null>"):
            continue
        # Normalize some native synonyms
        native_norm = {
            "intern": "entry",  # intern is effectively entry for this purpose
            "executive": "director",
        }.get(native, native)
        if native_norm not in LEVELS:
            continue
        # Lenient: keep all final values
        if final not in LEVELS and final not in ("unknown", "<null>", None, ""):
            continue
        if final == "unknown" or final in ("<null>", None, ""):
            # for lenient, record as 'unknown' non-match
            mat_lenient[(native_norm, "unknown")] += n
        else:
            mat_strict[(native_norm, final)] += n
            mat_lenient[(native_norm, final)] += n

    # Compute strict kappa (excluding final=unknown)
    k_s, po_s, N_s = cohen_kappa(mat_strict, LEVELS)
    # Lenient kappa: 'unknown' as an extra label
    k_l, po_l, N_l = cohen_kappa(mat_lenient, LEVELS + ["unknown"])

    recall_strict = per_class_recall(mat_strict, LEVELS)
    recall_lenient = per_class_recall(mat_lenient, LEVELS)

    print(f"\n=== {tag} ===")
    print(f"  strict N (native known & final !=unknown): {N_s}")
    print(f"  lenient N (native known, final=unknown counted as wrong): {N_l}")
    print(f"  strict kappa: {k_s:.3f}, obs agreement: {po_s:.3f}")
    print(f"  lenient kappa: {k_l:.3f}, obs agreement: {po_l:.3f}")
    print("  per-class recall (strict, final!=unknown):")
    for l in LEVELS:
        r, t = recall_strict[l]
        if r is None:
            print(f"    {l}: N=0")
        else:
            print(f"    {l}: {r:.3f} (n_native={t})")
    print("  per-class recall (lenient, counting unknown as wrong):")
    for l in LEVELS:
        r, t = recall_lenient[l]
        if r is None:
            print(f"    {l}: N=0")
        else:
            print(f"    {l}: {r:.3f} (n_native={t})")

    out_rows.append({
        "tag": tag,
        "N_strict": N_s,
        "N_lenient": N_l,
        "kappa_strict": round(k_s, 4),
        "kappa_lenient": round(k_l, 4),
        "obs_agreement_strict": round(po_s, 4),
        "obs_agreement_lenient": round(po_l, 4),
        **{
            f"recall_{l}_strict": (None if recall_strict[l][0] is None else round(recall_strict[l][0], 4))
            for l in LEVELS
        },
        **{
            f"recall_{l}_lenient": (None if recall_lenient[l][0] is None else round(recall_lenient[l][0], 4))
            for l in LEVELS
        },
        **{f"n_native_{l}_strict": recall_strict[l][1] for l in LEVELS},
    })


def main() -> None:
    out_rows = []
    process("arshkon", TBL / "03_crosstab_arshkon.csv", out_rows)
    process("scraped_linkedin", TBL / "03_crosstab_scraped_linkedin.csv", out_rows)

    # Write kappa summary
    path = TBL / "03_kappa_summary.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
