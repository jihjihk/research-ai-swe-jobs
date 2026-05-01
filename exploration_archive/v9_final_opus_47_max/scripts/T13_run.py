"""T13 — Linguistic & structural evolution.

Runs readability, section anatomy, and tone-marker analysis on the SWE LinkedIn
corpus (arshkon + asaniczka + scraped) and writes tables, figures, and a per-
posting parquet consumed by downstream tasks.

Primary outputs
---------------

tables/T13/
    readability_by_period_seniority.csv
    section_anatomy_by_period_seniority.csv
    section_anatomy_pct_by_period_seniority.csv
    tone_by_period_seniority.csv
    entry_level_period_comparison.csv
    within_2024_calibration.csv
    labeled_vs_not_split.csv
    aggregator_sensitivity.csv
    description_text_source_split.csv

figures/T13/
    section_anatomy_stacked.png        # key decider figure
    section_anatomy_absolute.png       # absolute chars per section
    readability_boxplots.png
    tone_period_bars.png

artifacts/shared/
    T13_readability_metrics.parquet    # consumed by T29

reports/T13.md                         # written by hand after running
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

import textstat

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

from T13_section_classifier import classify_sections, SECTION_LABELS  # type: ignore  # noqa: E402

UNIFIED = REPO / "data" / "unified.parquet"
SENIORITY_PANEL = REPO / "exploration" / "artifacts" / "shared" / "seniority_definition_panel.csv"
OUT_TABLES = REPO / "exploration" / "tables" / "T13"
OUT_FIG = REPO / "exploration" / "figures" / "T13"
OUT_ARTIFACTS = REPO / "exploration" / "artifacts" / "shared"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SQL: base frame and derived columns
# ---------------------------------------------------------------------------

BASE_FILTER = (
    "is_swe AND source_platform='linkedin' AND is_english AND date_flag='ok'"
)


def load_frame(limit: int | None = None) -> pd.DataFrame:
    """Load SWE LinkedIn rows with period, seniority, yoe, raw + cleaned text."""
    con = duckdb.connect()
    q = f"""
    SELECT
      uid,
      source,
      CASE WHEN source LIKE 'kaggle_%' THEN '2024' ELSE '2026' END AS period_year,
      period,
      company_name_canonical,
      is_aggregator,
      seniority_final,
      seniority_3level,
      yoe_min_years_llm,
      llm_classification_coverage,
      llm_extraction_coverage,
      description,
      description_core_llm
    FROM read_parquet('{UNIFIED.as_posix()}')
    WHERE {BASE_FILTER}
      AND description IS NOT NULL
      AND length(description) >= 50
    """
    if limit:
        q += f" LIMIT {limit}"
    df = con.execute(q).df()
    return df


# ---------------------------------------------------------------------------
# Readability + tone metrics
# ---------------------------------------------------------------------------


_IMPERATIVE_RE = __import__("re").compile(
    r"\b(you\s+will|you['’]ll|you\s+are\s+expected|must|should|need\s+to|required\s+to|able\s+to)\b",
    __import__("re").IGNORECASE,
)
_INCLUSIVE_RE = __import__("re").compile(
    r"\b(we['’]re|we\s+are|we\s+do|we\s+believe|our\s+team|our\s+company|together|join\s+(?:us|our)|you['’]ll\s+join)\b",
    __import__("re").IGNORECASE,
)
_MARKETING_RE = __import__("re").compile(
    r"\b(exciting|innovative|cutting[\s-]edge|world[\s-]class|mission[\s-]driven|disrupt(?:ive|ing)?|passionate|best[\s-]in[\s-]class|rock[\s-]?star|ninja|game[\s-]changer|unicorn|world[\s-]leading)\b",
    __import__("re").IGNORECASE,
)
_PASSIVE_RE = __import__("re").compile(
    r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b",
    __import__("re").IGNORECASE,
)


def _ttr_first1k(text: str) -> float:
    tokens = [t for t in __import__("re").split(r"\W+", text[:1000].lower()) if t]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _safe(fn, default=float("nan")):
    try:
        v = fn()
        if isinstance(v, (int, float)) and (v != v or v is None):  # NaN
            return default
        return v
    except Exception:
        return default


def compute_metrics(text: str) -> dict:
    """All readability + tone metrics on one text block."""
    n_chars = len(text)
    if n_chars == 0:
        return {
            "flesch_kincaid_grade": float("nan"),
            "flesch_reading_ease": float("nan"),
            "gunning_fog": float("nan"),
            "avg_sentence_length": float("nan"),
            "sentence_length_sd": float("nan"),
            "type_token_ratio": float("nan"),
            "syllable_count": 0,
            "lexicon_count": 0,
            "imperative_density": 0.0,
            "inclusive_density": 0.0,
            "marketing_density": 0.0,
            "passive_density": 0.0,
            "raw_length": 0,
        }
    fk = _safe(lambda: float(textstat.flesch_kincaid_grade(text)))
    fre = _safe(lambda: float(textstat.flesch_reading_ease(text)))
    fog = _safe(lambda: float(textstat.gunning_fog(text)))
    syl = _safe(lambda: int(textstat.syllable_count(text)), default=0)
    lex = _safe(lambda: int(textstat.lexicon_count(text, removepunct=True)), default=0)
    # Sentence-level stats
    try:
        sents = textstat.sentence_count(text)
    except Exception:
        sents = 0
    import re as _re
    sent_list = [s.strip() for s in _re.split(r"[.!?]\s+", text) if s.strip()]
    sent_lens = [len(_re.split(r"\W+", s)) for s in sent_list if s]
    if sent_lens:
        avg_sl = float(np.mean(sent_lens))
        sd_sl = float(np.std(sent_lens))
    else:
        avg_sl = float("nan")
        sd_sl = float("nan")
    ttr = _ttr_first1k(text)
    # Per-1K-char densities
    scale = max(n_chars, 1) / 1000
    imp = len(_IMPERATIVE_RE.findall(text)) / scale
    inc = len(_INCLUSIVE_RE.findall(text)) / scale
    mkt = len(_MARKETING_RE.findall(text)) / scale
    pas = len(_PASSIVE_RE.findall(text)) / scale
    return {
        "flesch_kincaid_grade": fk,
        "flesch_reading_ease": fre,
        "gunning_fog": fog,
        "avg_sentence_length": avg_sl,
        "sentence_length_sd": sd_sl,
        "type_token_ratio": float(ttr),
        "syllable_count": int(syl),
        "lexicon_count": int(lex),
        "imperative_density": float(imp),
        "inclusive_density": float(inc),
        "marketing_density": float(mkt),
        "passive_density": float(pas),
        "raw_length": int(n_chars),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def compute_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row readability, tone, and section counts on RAW description."""
    rows = []
    t0 = time.time()
    for i, row in df.reset_index(drop=True).iterrows():
        desc = row["description"] or ""
        sec = classify_sections(desc)
        metrics = compute_metrics(desc)
        out = {"uid": row["uid"]}
        out.update(metrics)
        for k in SECTION_LABELS:
            out[f"sec_{k}_chars"] = sec[k]
        total = sec["total"] or 1
        for k in SECTION_LABELS:
            out[f"sec_{k}_share"] = sec[k] / total
        out["sec_total_chars"] = sec["total"]
        rows.append(out)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  [features] {i+1}/{len(df)}  {rate:.1f} rows/s  ETA {eta:.0f}s")
    return pd.DataFrame(rows)


def attach_labels(features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "uid",
        "source",
        "period_year",
        "period",
        "company_name_canonical",
        "is_aggregator",
        "seniority_final",
        "seniority_3level",
        "yoe_min_years_llm",
        "llm_classification_coverage",
        "llm_extraction_coverage",
    ]
    return features.merge(df[keep], on="uid", how="left")


def add_t30_seniority(df: pd.DataFrame) -> pd.DataFrame:
    """Add J3/S4 (YOE-based) and J1/S1 (label-based) flags."""
    df = df.copy()
    yoe = df["yoe_min_years_llm"]
    labeled = df["llm_classification_coverage"] == "labeled"
    df["is_J3"] = (yoe <= 2) & yoe.notna() & labeled
    df["is_S4"] = (yoe >= 5) & yoe.notna() & labeled
    df["is_J1"] = df["seniority_final"].eq("entry")
    df["is_S1"] = df["seniority_final"].eq("mid-senior")
    return df


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


SENIORITY_BUCKETS = ["all", "J3_yoe_le2", "S4_yoe_ge5", "J1_entry_label", "S1_midsenior_label"]
READABILITY_COLS = [
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "gunning_fog",
    "avg_sentence_length",
    "sentence_length_sd",
    "type_token_ratio",
    "syllable_count",
    "lexicon_count",
    "raw_length",
]
TONE_COLS = [
    "imperative_density",
    "inclusive_density",
    "marketing_density",
    "passive_density",
]
SECTION_CHAR_COLS = [f"sec_{k}_chars" for k in SECTION_LABELS]
SECTION_SHARE_COLS = [f"sec_{k}_share" for k in SECTION_LABELS]


def seniority_mask(df: pd.DataFrame, label: str) -> pd.Series:
    if label == "all":
        return pd.Series(True, index=df.index)
    if label == "J3_yoe_le2":
        return df["is_J3"]
    if label == "S4_yoe_ge5":
        return df["is_S4"]
    if label == "J1_entry_label":
        return df["is_J1"]
    if label == "S1_midsenior_label":
        return df["is_S1"]
    raise KeyError(label)


def agg_by_period_seniority(df: pd.DataFrame, cols: list[str], how: str = "mean") -> pd.DataFrame:
    rows = []
    for period in ("2024", "2026"):
        dp = df[df["period_year"] == period]
        for sen in SENIORITY_BUCKETS:
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            r = {"period": period, "seniority": sen, "n": int(len(dps))}
            for c in cols:
                if how == "mean":
                    r[c] = float(dps[c].mean()) if len(dps) else float("nan")
                elif how == "sum":
                    r[c] = float(dps[c].sum()) if len(dps) else 0.0
                elif how == "median":
                    r[c] = float(dps[c].median()) if len(dps) else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def agg_by_source_seniority(df: pd.DataFrame, cols: list[str], how: str = "mean") -> pd.DataFrame:
    rows = []
    for src in ("kaggle_arshkon", "kaggle_asaniczka", "scraped"):
        dp = df[df["source"] == src]
        for sen in SENIORITY_BUCKETS:
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            r = {"source": src, "seniority": sen, "n": int(len(dps))}
            for c in cols:
                if how == "mean":
                    r[c] = float(dps[c].mean()) if len(dps) else float("nan")
                elif how == "median":
                    r[c] = float(dps[c].median()) if len(dps) else float("nan")
            rows.append(r)
    return pd.DataFrame(rows)


def within_2024_vs_cross_period(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-2024 calibration: arshkon vs asaniczka vs cross-period (pooled 2024 vs scraped)."""
    rows = []
    a = df[df["source"] == "kaggle_arshkon"]
    s = df[df["source"] == "kaggle_asaniczka"]
    pooled2024 = df[df["period_year"] == "2024"]
    scraped = df[df["period_year"] == "2026"]
    for c in cols:
        w24 = float(abs((a[c].mean() if len(a) else 0) - (s[c].mean() if len(s) else 0)))
        cross = float(abs((pooled2024[c].mean() if len(pooled2024) else 0) - (scraped[c].mean() if len(scraped) else 0)))
        ratio = cross / w24 if w24 > 0 else float("inf")
        rows.append({
            "metric": c,
            "arshkon_mean": float(a[c].mean()) if len(a) else float("nan"),
            "asaniczka_mean": float(s[c].mean()) if len(s) else float("nan"),
            "pooled_2024_mean": float(pooled2024[c].mean()) if len(pooled2024) else float("nan"),
            "scraped_mean": float(scraped[c].mean()) if len(scraped) else float("nan"),
            "within_2024_abs": w24,
            "cross_period_abs": cross,
            "snr_ratio": ratio,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figures(df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- KEY FIGURE: section share 2024 vs 2026 stacked bar (pooled all SWE + J3 + S4) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (sen, title) in zip(
        axes,
        [("all", "All SWE"), ("J3_yoe_le2", "J3 junior (YOE ≤ 2)"), ("S4_yoe_ge5", "S4 senior (YOE ≥ 5)")],
    ):
        data = []
        for period in ("2024", "2026"):
            dp = df[df["period_year"] == period]
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            total = dps[SECTION_CHAR_COLS].sum().sum()
            shares = dps[SECTION_CHAR_COLS].sum() / total if total else pd.Series([0] * len(SECTION_CHAR_COLS), index=SECTION_CHAR_COLS)
            data.append(shares.values * 100)
        data = np.array(data)  # rows: period, cols: sections
        bottoms = np.zeros(2)
        colors = plt.cm.tab10.colors
        for i, k in enumerate(SECTION_LABELS):
            ax.bar(["2024", "2026"], data[:, i], bottom=bottoms, label=k, color=colors[i % len(colors)])
            bottoms += data[:, i]
        ax.set_title(f"{title}\n(section share of raw description chars)")
        ax.set_ylabel("Share %")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.suptitle("T13 — Section anatomy share by period (raw description)")
    fig.tight_layout()
    fig.savefig(out_dir / "section_anatomy_stacked.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- ABSOLUTE chars per section (mean per posting) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (sen, title) in zip(
        axes,
        [("all", "All SWE"), ("J3_yoe_le2", "J3 junior (YOE ≤ 2)"), ("S4_yoe_ge5", "S4 senior (YOE ≥ 5)")],
    ):
        means_2024 = []
        means_2026 = []
        for period, dest in (("2024", means_2024), ("2026", means_2026)):
            dp = df[df["period_year"] == period]
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            for k in SECTION_LABELS:
                dest.append(float(dps[f"sec_{k}_chars"].mean()) if len(dps) else 0)
        x = np.arange(len(SECTION_LABELS))
        w = 0.4
        ax.bar(x - w / 2, means_2024, width=w, label="2024 pooled", alpha=0.8)
        ax.bar(x + w / 2, means_2026, width=w, label="2026 scraped", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(SECTION_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{title}\n(mean chars per posting by section)")
        ax.set_ylabel("Mean chars")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("T13 — Section anatomy absolute chars by period (raw description)")
    fig.tight_layout()
    fig.savefig(out_dir / "section_anatomy_absolute.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Readability boxplots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["flesch_kincaid_grade", "gunning_fog", "type_token_ratio"]):
        data = []
        labels = []
        for period in ("2024", "2026"):
            dp = df[df["period_year"] == period][metric].dropna()
            data.append(dp.values)
            labels.append(f"{period}\nn={len(dp)}")
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("T13 — Readability by period (raw description)")
    fig.tight_layout()
    fig.savefig(out_dir / "readability_boxplots.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Tone bars ---
    fig, ax = plt.subplots(figsize=(10, 5))
    means_2024 = []
    means_2026 = []
    for c in TONE_COLS:
        means_2024.append(df[df["period_year"] == "2024"][c].mean())
        means_2026.append(df[df["period_year"] == "2026"][c].mean())
    x = np.arange(len(TONE_COLS))
    w = 0.4
    ax.bar(x - w / 2, means_2024, width=w, label="2024 pooled")
    ax.bar(x + w / 2, means_2026, width=w, label="2026 scraped")
    ax.set_xticks(x)
    ax.set_xticklabels(TONE_COLS, rotation=30, ha="right")
    ax.set_ylabel("Density per 1K chars")
    ax.set_title("T13 — Tone markers by period (raw description)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tone_period_bars.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print("Loading frame...")
    df = load_frame()
    print(f"  n = {len(df):,}")

    print("Computing per-row features (raw description)...")
    feats = compute_row_features(df)
    feats = attach_labels(feats, df)
    feats = add_t30_seniority(feats)

    # Save per-posting parquet
    shared_out = OUT_ARTIFACTS / "T13_readability_metrics.parquet"
    keep_cols = (
        ["uid", "source", "period_year", "seniority_final", "seniority_3level",
         "yoe_min_years_llm", "llm_classification_coverage", "is_aggregator",
         "is_J3", "is_S4", "is_J1", "is_S1"]
        + READABILITY_COLS + TONE_COLS + SECTION_CHAR_COLS + SECTION_SHARE_COLS
        + ["sec_total_chars"]
    )
    feats_out = feats[keep_cols]
    pq.write_table(pa.Table.from_pandas(feats_out), shared_out)
    print(f"  wrote {shared_out}")

    # -----------------
    # Aggregations
    # -----------------
    print("Aggregating...")
    readability_tbl = agg_by_period_seniority(feats, READABILITY_COLS, how="mean")
    readability_tbl.to_csv(OUT_TABLES / "readability_by_period_seniority.csv", index=False)

    tone_tbl = agg_by_period_seniority(feats, TONE_COLS, how="mean")
    tone_tbl.to_csv(OUT_TABLES / "tone_by_period_seniority.csv", index=False)

    # Per-section chars (mean per posting)
    sec_tbl = agg_by_period_seniority(feats, SECTION_CHAR_COLS + ["sec_total_chars"], how="mean")
    sec_tbl.to_csv(OUT_TABLES / "section_anatomy_by_period_seniority.csv", index=False)

    # Per-section shares
    # Aggregate TOTAL chars per (period, seniority) and compute per-section share
    pct_rows = []
    for period in ("2024", "2026"):
        dp = feats[feats["period_year"] == period]
        for sen in SENIORITY_BUCKETS:
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            total = dps[SECTION_CHAR_COLS].sum().sum()
            r = {"period": period, "seniority": sen, "n": int(len(dps)), "total_chars_sum": float(total)}
            for k in SECTION_LABELS:
                col = f"sec_{k}_chars"
                r[f"{k}_share_pct"] = float(dps[col].sum() / total * 100) if total else 0.0
                r[f"{k}_mean_chars"] = float(dps[col].mean()) if len(dps) else 0.0
            pct_rows.append(r)
    pct_tbl = pd.DataFrame(pct_rows)
    pct_tbl.to_csv(OUT_TABLES / "section_anatomy_pct_by_period_seniority.csv", index=False)

    # Entry-level specific comparison (raw vs core_llm might matter — note this
    # run uses raw). J3 and J1 both reported.
    entry_rows = []
    for sen in ("J3_yoe_le2", "J1_entry_label"):
        for period in ("2024", "2026"):
            dp = feats[feats["period_year"] == period]
            mask = seniority_mask(dp, sen)
            dps = dp[mask]
            r = {"period": period, "seniority": sen, "n": int(len(dps))}
            for c in READABILITY_COLS + TONE_COLS + SECTION_CHAR_COLS + ["sec_total_chars"]:
                r[c] = float(dps[c].mean()) if len(dps) else float("nan")
            entry_rows.append(r)
    pd.DataFrame(entry_rows).to_csv(OUT_TABLES / "entry_level_period_comparison.csv", index=False)

    # Within-2024 calibration for readability, tone, and section shares
    calib_cols = (
        READABILITY_COLS + TONE_COLS + [f"sec_{k}_share" for k in SECTION_LABELS]
        + [f"sec_{k}_chars" for k in SECTION_LABELS]
    )
    calib = within_2024_vs_cross_period(feats, calib_cols)
    calib.to_csv(OUT_TABLES / "within_2024_calibration.csv", index=False)

    # Labeled vs not split (scraped)
    scr = feats[feats["source"] == "scraped"]
    lvn_rows = []
    for cov_name, cov_val in (("labeled", "labeled"), ("not_labeled", None)):
        sub = scr[scr["llm_extraction_coverage"] == cov_val] if cov_val == "labeled" else scr[scr["llm_extraction_coverage"] != "labeled"]
        r = {"split": cov_name, "n": int(len(sub))}
        for c in READABILITY_COLS + TONE_COLS + [f"sec_{k}_share" for k in SECTION_LABELS] + ["sec_total_chars"]:
            r[c] = float(sub[c].mean()) if len(sub) else float("nan")
        lvn_rows.append(r)
    pd.DataFrame(lvn_rows).to_csv(OUT_TABLES / "labeled_vs_not_split.csv", index=False)

    # Aggregator sensitivity: compare metrics with vs without aggregators
    agg_rows = []
    for excl in (False, True):
        sub = feats if not excl else feats[~feats["is_aggregator"].astype(bool)]
        for period in ("2024", "2026"):
            dps = sub[sub["period_year"] == period]
            r = {"exclude_aggregators": excl, "period": period, "n": int(len(dps))}
            for c in ["raw_length"] + [f"sec_{k}_share" for k in SECTION_LABELS] + TONE_COLS:
                r[c] = float(dps[c].mean()) if len(dps) else float("nan")
            agg_rows.append(r)
    pd.DataFrame(agg_rows).to_csv(OUT_TABLES / "aggregator_sensitivity.csv", index=False)

    # Text source split (description vs description_core_llm) not applicable on feats
    # since we only computed on raw. Report source-level summary:
    src_rows = []
    for src in ("kaggle_arshkon", "kaggle_asaniczka", "scraped"):
        dp = feats[feats["source"] == src]
        r = {"source": src, "n": int(len(dp))}
        for c in ["raw_length"] + [f"sec_{k}_share" for k in SECTION_LABELS] + TONE_COLS:
            r[c] = float(dp[c].mean()) if len(dp) else float("nan")
        src_rows.append(r)
    pd.DataFrame(src_rows).to_csv(OUT_TABLES / "description_text_source_split.csv", index=False)

    # -----------------
    # Figures
    # -----------------
    print("Drawing figures...")
    save_figures(feats, OUT_FIG)

    print("Done. Artifacts:")
    print(f"  {shared_out}")
    for f in sorted(OUT_TABLES.iterdir()):
        print(f"  {f}")
    for f in sorted(OUT_FIG.iterdir()):
        print(f"  {f}")


if __name__ == "__main__":
    main()
