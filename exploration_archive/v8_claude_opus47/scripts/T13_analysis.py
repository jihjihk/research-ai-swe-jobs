"""T13 — Linguistic & structural evolution of SWE LinkedIn postings.

Runs:
  (1) readability metrics by period × T30 seniority
  (2) section anatomy via T13_section_classifier
  (3) stacked-bar decomposition + per-section SNR
  (4) tone markers (per 1K chars of cleaned text)
  (5) entry-level focused comparison

Writes tables to exploration/tables/T13/ and figures to exploration/figures/T13/.
"""
from __future__ import annotations

import os
import re
import sys
import json
from collections import defaultdict, Counter

import duckdb
import numpy as np
import pandas as pd
import textstat

sys.path.insert(0, "/home/jihgaboot/gabor/job-research/exploration/scripts")
from T13_section_classifier import classify_description, SECTION_TYPES

ROOT = "/home/jihgaboot/gabor/job-research"
UNIFIED = f"{ROOT}/data/unified.parquet"
CLEANED = f"{ROOT}/exploration/artifacts/shared/swe_cleaned_text.parquet"
SPECIALISTS = f"{ROOT}/exploration/artifacts/shared/entry_specialist_employers.csv"
TABLES = f"{ROOT}/exploration/tables/T13"
FIGS = f"{ROOT}/exploration/figures/T13"
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

FILTER = (
    "source_platform='linkedin' AND is_english=true "
    "AND date_flag='ok' AND is_swe=true"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_base_frame() -> pd.DataFrame:
    """Pull the raw+cleaned joined frame we need for T13."""
    c = duckdb.connect()
    # We need raw description (for section anatomy) and cleaned description +
    # text_source (for tone markers / readability on LLM-only cell).
    # Join on uid.
    c.execute(
        f"""
        CREATE VIEW swe_raw AS
        SELECT uid, description, source, period, seniority_final,
               seniority_3level, is_aggregator, company_name_canonical,
               yoe_extracted
        FROM read_parquet('{UNIFIED}')
        WHERE {FILTER}
        """
    )
    c.execute(
        f"""
        CREATE VIEW swe_cleaned AS
        SELECT uid, description_cleaned, text_source
        FROM read_parquet('{CLEANED}')
        """
    )
    df = c.execute(
        """
        SELECT r.uid, r.description, r.source, r.period, r.seniority_final,
               r.seniority_3level, r.is_aggregator, r.company_name_canonical,
               r.yoe_extracted, c.description_cleaned, c.text_source
        FROM swe_raw r LEFT JOIN swe_cleaned c USING (uid)
        """
    ).fetchdf()
    return df


def _seniority_slice(row) -> str:
    """Return J2 / mid-senior / S1 slice label (else 'other').
    J2 = entry OR associate. S1 = mid-senior OR director (per T30)."""
    s = row.get("seniority_final")
    if s in ("entry", "associate"):
        return "J2_entry_assoc"
    if s == "mid-senior":
        return "mid_senior"
    if s == "director":
        return "director"
    return "other"


def _s1_mask(df: pd.DataFrame) -> pd.Series:
    """S1 = mid-senior + director."""
    return df["seniority_final"].isin(["mid-senior", "director"])


# ---------------------------------------------------------------------------
# Readability (step 1)
# ---------------------------------------------------------------------------
def compute_readability(df: pd.DataFrame, sample_per_cell: int = 2000) -> pd.DataFrame:
    """For each (period, seniority_slice), sample up to N rows and compute
    textstat readability metrics on cleaned text (text_source='llm' only)."""
    # Filter to llm-cleaned, non-empty
    src = df[df["text_source"] == "llm"].copy()
    src = src[src["description_cleaned"].str.len() > 100]
    # Attach three slices: J2, mid_senior, S1 (mid-senior ∪ director)
    src["slice_raw"] = src.apply(_seniority_slice, axis=1)
    # Build a per-row 'slice' column by exploding into a long frame for
    # computing the S1 slice as a union.
    src_j2 = src[src["slice_raw"] == "J2_entry_assoc"].copy()
    src_j2["slice"] = "J2_entry_assoc"
    src_mid = src[src["slice_raw"] == "mid_senior"].copy()
    src_mid["slice"] = "mid_senior"
    src_s1 = src[_s1_mask(src)].copy()  # mid-senior + director
    src_s1["slice"] = "S1_senior_plus"
    src = pd.concat([src_j2, src_mid, src_s1], ignore_index=True)
    # Build period label: pooled_2024 for arshkon+asaniczka, 2026 for scraped.
    src["period_group"] = np.where(
        src["source"] == "scraped", "2026",
        np.where(src["source"] == "kaggle_arshkon", "2024_arshkon", "2024_asaniczka")
    )

    rows_out = []
    for (pg, sl), g in src.groupby(["period_group", "slice"]):
        if len(g) > sample_per_cell:
            g = g.sample(sample_per_cell, random_state=42)
        metrics = defaultdict(list)
        for t in g["description_cleaned"]:
            if not t or len(t) < 50:
                continue
            try:
                metrics["fk_grade"].append(textstat.flesch_kincaid_grade(t))
                metrics["fre"].append(textstat.flesch_reading_ease(t))
                metrics["gunning_fog"].append(textstat.gunning_fog(t))
                metrics["avg_sent_len"].append(textstat.avg_sentence_length(t))
                metrics["lexicon"].append(textstat.lexicon_count(t, removepunct=True))
                metrics["syllables"].append(textstat.syllable_count(t))
                # vocabulary richness: TTR on first 1000 chars (length-normalized)
                head = t[:1000].split()
                if head:
                    metrics["ttr_1k"].append(len(set(head)) / len(head))
            except Exception:
                continue
        row = {"period_group": pg, "slice": sl, "n": len(g)}
        for k, v in metrics.items():
            row[f"{k}_mean"] = float(np.mean(v)) if v else np.nan
            row[f"{k}_median"] = float(np.median(v)) if v else np.nan
        rows_out.append(row)
    out = pd.DataFrame(rows_out).sort_values(["slice", "period_group"])
    return out


# ---------------------------------------------------------------------------
# Section anatomy (step 2-3)
# ---------------------------------------------------------------------------
def compute_section_anatomy(df: pd.DataFrame) -> dict:
    """For every posting compute per-section char counts. Return a DataFrame
    of per-posting section counts (wide)."""
    print(f"[T13] classifying sections for {len(df)} descriptions")
    out_cols = {s: [] for s in SECTION_TYPES}
    totals = []
    for d in df["description"].values:
        secs = classify_description(d or "")
        total = 0
        for s in SECTION_TYPES:
            c = secs[s]["chars"]
            out_cols[s].append(c)
            total += c
        totals.append(total)
    wide = pd.DataFrame(out_cols)
    wide["total_chars_any_section"] = totals
    # Attach per-row metadata we need
    for c in ("uid", "source", "period", "seniority_final", "is_aggregator",
              "company_name_canonical", "description"):
        wide[c] = df[c].values
    wide["raw_len"] = wide["description"].str.len().fillna(0).astype(int)
    return wide


def summarize_sections_by_period(wide: pd.DataFrame, specialists: set[str]) -> pd.DataFrame:
    """Per (source, period, seniority_slice): median / mean char counts per
    section + share-of-total."""
    tmp = wide.copy()
    tmp["slice"] = tmp.apply(_seniority_slice, axis=1)
    # exclude specialists
    tmp = tmp[~tmp["company_name_canonical"].isin(specialists)]
    # exclude aggregators
    tmp = tmp[tmp["is_aggregator"].fillna(False) == False]

    rows = []
    groupers = ["source", "period", "slice"]
    # Also emit pooled 2024 and all-seniority rows
    for keys, g in tmp.groupby(groupers):
        row = {
            "source": keys[0], "period": keys[1], "slice": keys[2], "n": len(g),
            "raw_len_median": float(g["raw_len"].median()),
            "raw_len_mean": float(g["raw_len"].mean()),
        }
        for s in SECTION_TYPES:
            row[f"{s}_median_chars"] = float(g[s].median())
            row[f"{s}_mean_chars"] = float(g[s].mean())
            row[f"{s}_share_median"] = float((g[s] / g["raw_len"].clip(lower=1)).median())
        rows.append(row)
    # Add all-seniority rows per source/period
    for keys, g in tmp.groupby(["source", "period"]):
        row = {
            "source": keys[0], "period": keys[1], "slice": "all", "n": len(g),
            "raw_len_median": float(g["raw_len"].median()),
            "raw_len_mean": float(g["raw_len"].mean()),
        }
        for s in SECTION_TYPES:
            row[f"{s}_median_chars"] = float(g[s].median())
            row[f"{s}_mean_chars"] = float(g[s].mean())
            row[f"{s}_share_median"] = float((g[s] / g["raw_len"].clip(lower=1)).median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["slice", "source", "period"])


def compute_section_snr(wide: pd.DataFrame, specialists: set[str]) -> pd.DataFrame:
    """Calibration SNR per section using the Gate 1 rule:
    within-2024 diff (arshkon − asaniczka) vs cross-period (pooled-2024 vs scraped).
    SNR = |cross| / max(|within|, MIN_FLOOR).

    We report three per-section statistics:
      - mean_chars (primary, stable against zeros in sparse sections)
      - median_chars (legacy comparison; zero-floored for sections sparse in arshkon)
      - share_of_total (per-posting share of total chars in classified sections)

    The floor prevents inflated SNR when within-2024 diff is 0. We use
    floor = 0.05 * pooled_2024 value (i.e. 5% of the pooled level) — noise
    cannot logically be smaller than that.
    """
    tmp = wide.copy()
    tmp = tmp[~tmp["company_name_canonical"].isin(specialists)]
    tmp = tmp[tmp["is_aggregator"].fillna(False) == False]
    # Share-of-total per posting = chars_in_section / max(total_classified, 1)
    total_classified = sum(tmp[s] for s in SECTION_TYPES)
    total_classified = total_classified.replace(0, 1)
    for s in SECTION_TYPES:
        tmp[f"{s}_share"] = tmp[s] / total_classified

    out = []
    for s in SECTION_TYPES:
        for stat_name, col, fn in [
            ("mean_chars", s, np.mean),
            ("median_chars", s, np.median),
            ("mean_share", f"{s}_share", np.mean),
        ]:
            ar = tmp[tmp["source"] == "kaggle_arshkon"][col]
            asz = tmp[tmp["source"] == "kaggle_asaniczka"][col]
            sc = tmp[tmp["source"] == "scraped"][col]
            pooled = pd.concat([ar, asz])
            ar_v = fn(ar) if len(ar) else np.nan
            as_v = fn(asz) if len(asz) else np.nan
            sc_v = fn(sc) if len(sc) else np.nan
            pooled_v = fn(pooled) if len(pooled) else np.nan
            within = ar_v - as_v
            cross = sc_v - pooled_v
            cross_ar = sc_v - ar_v
            # noise floor: max(|within|, 0.05 * max(pooled_v, small))
            floor = max(abs(pooled_v) * 0.05, 1e-6 if "share" in stat_name else 1.0)
            denom = max(abs(within), floor)
            snr_pooled = abs(cross) / denom
            snr_arshkon = abs(cross_ar) / denom
            verdict = "above_noise" if max(snr_pooled, snr_arshkon) >= 2 else (
                "below_noise" if max(snr_pooled, snr_arshkon) < 1 else "marginal")
            out.append({
                "section": s, "stat": stat_name,
                "arshkon": ar_v, "asaniczka": as_v, "scraped": sc_v,
                "pooled_2024": pooled_v, "within_2024": within,
                "cross_pooled": cross, "cross_arshkon": cross_ar,
                "noise_floor_used": denom,
                "snr_pooled": snr_pooled, "snr_arshkon": snr_arshkon,
                "verdict": verdict,
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Tone markers (step 4)
# ---------------------------------------------------------------------------
_TONE_PATTERNS = {
    "imperative": re.compile(r"\b(you\s+will|you'll|must|should|you\s+are\s+required|you\s+need\s+to)\b", re.I),
    "inclusive": re.compile(r"\b(we|our\s+team|our\s+company|you'll\s+join|join\s+us)\b", re.I),
    "passive": re.compile(r"\b(is|are|was|were)\s+\w+ed\b", re.I),
    "marketing": re.compile(r"\b(exciting|innovative|cutting[- ]edge|world[- ]class|dynamic|vibrant|mission[- ]driven)\b", re.I),
}


def compute_tone(df: pd.DataFrame) -> pd.DataFrame:
    """Tone markers per 1K chars of cleaned text (llm-only)."""
    src = df[df["text_source"] == "llm"].copy()
    src = src[src["description_cleaned"].str.len() > 100]
    # Build three-slice long-form frame like readability
    src["slice_raw"] = src.apply(_seniority_slice, axis=1)
    src_j2 = src[src["slice_raw"] == "J2_entry_assoc"].copy(); src_j2["slice"] = "J2_entry_assoc"
    src_mid = src[src["slice_raw"] == "mid_senior"].copy(); src_mid["slice"] = "mid_senior"
    src_s1 = src[_s1_mask(src)].copy(); src_s1["slice"] = "S1_senior_plus"
    src = pd.concat([src_j2, src_mid, src_s1], ignore_index=True)
    src["period_group"] = np.where(
        src["source"] == "scraped", "2026",
        np.where(src["source"] == "kaggle_arshkon", "2024_arshkon", "2024_asaniczka")
    )

    rows = []
    for (pg, sl), g in src.groupby(["period_group", "slice"]):
        if sl == "other":
            continue
        counts = defaultdict(list)
        for t in g["description_cleaned"]:
            n = len(t)
            if n == 0:
                continue
            for name, pat in _TONE_PATTERNS.items():
                counts[name].append(len(pat.findall(t)) / (n / 1000.0))
        row = {"period_group": pg, "slice": sl, "n": len(g)}
        for k, v in counts.items():
            row[f"{k}_per_1k_mean"] = float(np.mean(v)) if v else np.nan
            row[f"{k}_per_1k_median"] = float(np.median(v)) if v else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["slice", "period_group"])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_section_stack(section_summary: pd.DataFrame) -> None:
    """Stacked bar by (source, period) for slice='all' — median char shares."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        ("kaggle_arshkon", "2024-04"),
        ("kaggle_asaniczka", "2024-01"),
        ("scraped", "2026-03"),
        ("scraped", "2026-04"),
    ]
    df = section_summary[section_summary["slice"] == "all"].copy()
    df = df.set_index(["source", "period"])

    # Use mean_chars (absolute) for the stacked bar so lengths reflect reality.
    sec_cols = [f"{s}_mean_chars" for s in SECTION_TYPES]
    data = df.loc[order][sec_cols].fillna(0)
    data.columns = SECTION_TYPES

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(order))
    colors = plt.cm.tab10(np.arange(len(SECTION_TYPES)))
    for i, s in enumerate(SECTION_TYPES):
        vals = data[s].values
        ax.bar(range(len(order)), vals, bottom=bottom, label=s, color=colors[i])
        bottom += vals
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{s}\n{p}" for s, p in order], rotation=0)
    ax.set_ylabel("Mean chars per posting")
    ax.set_title("Section composition by (source, period) — mean absolute chars")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(f"{FIGS}/section_stack_mean_chars.png", dpi=140)
    plt.close(fig)

    # Also plot share-of-total
    share_cols = [f"{s}_share_median" for s in SECTION_TYPES]
    data2 = df.loc[order][share_cols].fillna(0)
    data2.columns = SECTION_TYPES
    # normalize rows to 100%
    data2 = data2.div(data2.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(order))
    for i, s in enumerate(SECTION_TYPES):
        vals = data2[s].values
        ax.bar(range(len(order)), vals, bottom=bottom, label=s, color=colors[i])
        bottom += vals
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{s}\n{p}" for s, p in order], rotation=0)
    ax.set_ylabel("Percent of total posting chars (median across postings)")
    ax.set_title("Section composition by (source, period) — % share (normalized)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(f"{FIGS}/section_stack_share.png", dpi=140)
    plt.close(fig)
    print(f"[T13] wrote section_stack figures to {FIGS}")


# ---------------------------------------------------------------------------
# Entry vs mid-senior anatomy (step 5)
# ---------------------------------------------------------------------------
def entry_vs_midsenior(wide: pd.DataFrame, specialists: set[str]) -> pd.DataFrame:
    """For J2 vs mid-senior vs S1, compare 2024-pooled vs 2026 section proportions."""
    tmp = wide.copy()
    tmp["slice_raw"] = tmp.apply(_seniority_slice, axis=1)
    tmp = tmp[~tmp["company_name_canonical"].isin(specialists)]
    tmp = tmp[tmp["is_aggregator"].fillna(False) == False]
    tmp["period_group"] = np.where(tmp["source"] == "scraped", "2026", "2024")

    # Build long-form with three slices
    pieces = []
    for sl, mask in [
        ("J2_entry_assoc", tmp["slice_raw"] == "J2_entry_assoc"),
        ("mid_senior", tmp["slice_raw"] == "mid_senior"),
        ("S1_senior_plus", _s1_mask(tmp)),
    ]:
        p = tmp[mask].copy()
        p["slice"] = sl
        pieces.append(p)
    tmp = pd.concat(pieces, ignore_index=True)

    rows = []
    for (pg, sl), g in tmp.groupby(["period_group", "slice"]):
        row = {"period_group": pg, "slice": sl, "n": len(g),
               "raw_len_median": float(g["raw_len"].median()),
               "raw_len_mean": float(g["raw_len"].mean())}
        for s in SECTION_TYPES:
            row[f"{s}_median_chars"] = float(g[s].median())
            row[f"{s}_mean_chars"] = float(g[s].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["slice", "period_group"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[T13] loading data")
    df = build_base_frame()
    print(f"[T13] loaded {len(df)} rows; source distribution:")
    print(df["source"].value_counts())

    specialists = set(pd.read_csv(SPECIALISTS)["company"].unique())
    print(f"[T13] {len(specialists)} specialist companies loaded")

    # 1. Readability
    print("[T13] computing readability on 2k-per-cell sample")
    read = compute_readability(df, sample_per_cell=2000)
    read.to_csv(f"{TABLES}/readability_by_period_seniority.csv", index=False)
    print(read.round(2).to_string(index=False))

    # 2-3. Section anatomy + SNR
    print("[T13] classifying all descriptions")
    wide = compute_section_anatomy(df)
    # save per-posting section chars so T12 can join on uid
    wide[["uid"] + SECTION_TYPES + ["raw_len"]].to_parquet(
        f"{TABLES}/per_posting_section_chars.parquet", index=False)
    print(f"[T13] wrote per_posting_section_chars.parquet")

    summary = summarize_sections_by_period(wide, specialists)
    summary.to_csv(f"{TABLES}/section_anatomy_by_period_seniority.csv", index=False)
    print("[T13] section_anatomy (slice='all') median chars:")
    show = summary[summary["slice"] == "all"][["source", "period", "n", "raw_len_median"] +
                                              [f"{s}_median_chars" for s in SECTION_TYPES]]
    print(show.round(0).to_string(index=False))

    snr = compute_section_snr(wide, specialists)
    snr.to_csv(f"{TABLES}/section_snr.csv", index=False)
    print("[T13] section SNR (median-based):")
    print(snr[snr["stat"] == "median"].round(2).to_string(index=False))

    plot_section_stack(summary)

    # 4. Tone
    tone = compute_tone(df)
    tone.to_csv(f"{TABLES}/tone_markers_by_period_seniority.csv", index=False)
    print("[T13] tone markers (per 1K chars):")
    print(tone.round(2).to_string(index=False))

    # 5. Entry vs mid-senior
    e_vs_m = entry_vs_midsenior(wide, specialists)
    e_vs_m.to_csv(f"{TABLES}/entry_vs_midsenior_section.csv", index=False)
    print("[T13] entry vs mid-senior (pooled 2024 vs 2026):")
    print(e_vs_m.round(0).to_string(index=False))

    print("[T13] done.")


if __name__ == "__main__":
    main()
