"""T13 — Linguistic & structural evolution.

Runs readability metrics, section anatomy, tone markers, and length-growth
decomposition on SWE LinkedIn postings. Uses the section classifier in
`T13_section_classifier.py`.

Outputs under exploration/tables/T13/ and exploration/figures/T13/.
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from T13_section_classifier import classify_sections, section_char_proportions, SECTIONS  # noqa: E402

try:
    import textstat  # type: ignore
except Exception as e:
    print("textstat missing:", e)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "exploration" / "tables" / "T13"
FIG_DIR = ROOT / "exploration" / "figures" / "T13"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def period_label(source, period):
    if source == "scraped":
        return "2026"
    return "2024"


def load_data(limit_per_cell=None):
    """Load SWE LinkedIn rows with description_core_llm and metadata.
    Primary frame: text_source='llm' (llm_extraction_coverage='labeled').
    Also load raw description for the sensitivity sanity check.
    """
    con = duckdb.connect()
    q = """
    SELECT
        uid, source, period, seniority_final, seniority_3level,
        is_aggregator, company_name_canonical,
        description_core_llm, description, llm_extraction_coverage
    FROM 'data/unified.parquet'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND description IS NOT NULL
    """
    df = con.execute(q).df()
    df["period_label"] = df.apply(lambda r: period_label(r["source"], r["period"]), axis=1)
    df["is_llm"] = df["llm_extraction_coverage"] == "labeled"
    df["text_llm"] = np.where(df["is_llm"] & df["description_core_llm"].notna(),
                              df["description_core_llm"], None)
    return df


def compute_tone_metrics(text):
    """Return dict of raw counts for tone markers. Per-1K-chars done later."""
    t = text or ""
    tl = t.lower()
    n_chars = len(t)
    # Imperative / you-directed
    imperative = len(re.findall(r"\byou will\b|\byou'll\b|\byou\u2019ll\b|\bmust\b|\bshould\b", tl))
    # Inclusive
    inclusive = len(re.findall(r"\bwe\b|\bour\b|\bus\b|\bour team\b", tl))
    # Marketing
    marketing = len(re.findall(
        r"\bexciting\b|\binnovative\b|\bcutting[\- ]edge\b|\bworld[\- ]class\b|\bpassionate\b|\bdynamic\b|\bfast[\- ]paced\b|\bmission[\- ]driven\b|\bgame[\- ]changing\b",
        tl))
    # Passive constructions (rough): "is/are/was/were/been/be + past participle"
    passive = len(re.findall(r"\b(?:is|are|was|were|been|be|being)\b\s+\w+ed\b", tl))
    return {
        "n_chars": n_chars,
        "imperative_cnt": imperative,
        "inclusive_cnt": inclusive,
        "marketing_cnt": marketing,
        "passive_cnt": passive,
    }


def ttr_first_1000(text):
    s = (text or "")[:1000]
    toks = re.findall(r"[A-Za-z][A-Za-z0-9+#./\-]*", s.lower())
    if not toks:
        return np.nan
    return len(set(toks)) / len(toks)


def readability_row(text):
    t = text or ""
    if len(t) < 30:
        return None
    try:
        return {
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(t),
            "flesch_reading_ease": textstat.flesch_reading_ease(t),
            "gunning_fog": textstat.gunning_fog(t),
            "avg_sentence_length": textstat.avg_sentence_length(t),
            "lexicon_count": textstat.lexicon_count(t),
            "syllable_count": textstat.syllable_count(t),
            "ttr_first_1000": ttr_first_1000(t),
        }
    except Exception:
        return None


def run_readability(df):
    """Sample ~2000 per (period_label, seniority_3level) cell and compute metrics.
    Primary on text_source='llm'. Report by period and seniority."""
    base = df[df["is_llm"] & df["text_llm"].notna()].copy()
    print(f"readability base rows (llm): {len(base)}")
    cells = []
    for (period, sen), g in base.groupby(["period_label", "seniority_3level"]):
        n = min(2000, len(g))
        cells.append(g.sample(n=n, random_state=SEED))
    sampled = pd.concat(cells, ignore_index=True)
    print(f"readability sampled rows: {len(sampled)}")
    metrics = []
    for _, r in sampled.iterrows():
        m = readability_row(r["text_llm"])
        if m is None:
            continue
        m["uid"] = r["uid"]
        m["period_label"] = r["period_label"]
        m["seniority_3level"] = r["seniority_3level"]
        m["source"] = r["source"]
        m["is_aggregator"] = r["is_aggregator"]
        metrics.append(m)
    mdf = pd.DataFrame(metrics)
    # Summary by period x seniority
    metric_cols = ["flesch_kincaid_grade", "flesch_reading_ease", "gunning_fog",
                   "avg_sentence_length", "lexicon_count", "syllable_count", "ttr_first_1000"]
    summary = (
        mdf.groupby(["period_label", "seniority_3level"])[metric_cols]
        .median().reset_index()
    )
    summary.to_csv(TABLES_DIR / "readability_by_period_seniority.csv", index=False)
    # Aggregator-exclusion sensitivity
    ex = mdf[~mdf["is_aggregator"].astype(bool)]
    summary_ex = (
        ex.groupby(["period_label", "seniority_3level"])[metric_cols]
        .median().reset_index()
    )
    summary_ex.to_csv(TABLES_DIR / "readability_by_period_seniority_excl_aggregators.csv", index=False)
    return mdf, summary, summary_ex


def run_section_anatomy(df):
    """For every llm row, compute per-section char counts. Aggregate by period/seniority."""
    base = df[df["is_llm"] & df["text_llm"].notna()].copy()
    print(f"section anatomy base rows: {len(base)}")
    rows = []
    for _, r in base.iterrows():
        counts = section_char_proportions(r["text_llm"])
        total = len(r["text_llm"])
        rows.append({
            "uid": r["uid"],
            "source": r["source"],
            "period_label": r["period_label"],
            "seniority_3level": r["seniority_3level"],
            "is_aggregator": bool(r["is_aggregator"]),
            "total_chars": total,
            **{f"sec_{k}_chars": v for k, v in counts.items()},
            **{f"sec_{k}_prop": (v / total if total > 0 else 0.0) for k, v in counts.items()},
        })
    sdf = pd.DataFrame(rows)
    sdf.to_parquet(TABLES_DIR / "section_anatomy_per_posting.parquet", index=False)
    return sdf


def summarize_section_anatomy(sdf):
    prop_cols = [f"sec_{k}_prop" for k in SECTIONS]
    char_cols = [f"sec_{k}_chars" for k in SECTIONS]

    # Median proportions by period x seniority
    prop_summary = (
        sdf.groupby(["period_label", "seniority_3level"])[prop_cols + ["total_chars"]]
        .median().reset_index()
    )
    prop_summary.to_csv(TABLES_DIR / "section_proportions_median.csv", index=False)

    # Mean proportions — better for length-growth decomposition
    mean_summary = (
        sdf.groupby(["period_label", "seniority_3level"])[prop_cols + ["total_chars"]]
        .mean().reset_index()
    )
    mean_summary.to_csv(TABLES_DIR / "section_proportions_mean.csv", index=False)

    # All-SWE (not stratified by seniority)
    all_swe_mean = (
        sdf.groupby("period_label")[char_cols + prop_cols + ["total_chars"]]
        .mean().reset_index()
    )
    all_swe_mean.to_csv(TABLES_DIR / "section_mean_all_swe.csv", index=False)

    # Aggregator-exclusion sensitivity
    ex = sdf[~sdf["is_aggregator"]]
    all_swe_mean_ex = (
        ex.groupby("period_label")[char_cols + prop_cols + ["total_chars"]]
        .mean().reset_index()
    )
    all_swe_mean_ex.to_csv(TABLES_DIR / "section_mean_all_swe_excl_aggregators.csv", index=False)

    # Length-growth decomposition: what fraction of the mean length growth came from each section?
    row_2024 = all_swe_mean[all_swe_mean["period_label"] == "2024"].iloc[0]
    row_2026 = all_swe_mean[all_swe_mean["period_label"] == "2026"].iloc[0]
    decomp = []
    total_delta = row_2026["total_chars"] - row_2024["total_chars"]
    for s in SECTIONS:
        d = row_2026[f"sec_{s}_chars"] - row_2024[f"sec_{s}_chars"]
        decomp.append({
            "section": s,
            "mean_chars_2024": row_2024[f"sec_{s}_chars"],
            "mean_chars_2026": row_2026[f"sec_{s}_chars"],
            "delta_chars": d,
            "share_of_growth": (d / total_delta) if total_delta != 0 else np.nan,
        })
    decomp.append({
        "section": "TOTAL",
        "mean_chars_2024": row_2024["total_chars"],
        "mean_chars_2026": row_2026["total_chars"],
        "delta_chars": total_delta,
        "share_of_growth": 1.0,
    })
    decomp_df = pd.DataFrame(decomp)
    decomp_df.to_csv(TABLES_DIR / "length_growth_decomposition.csv", index=False)

    # Aggregator-excluded decomposition
    row_2024_ex = all_swe_mean_ex[all_swe_mean_ex["period_label"] == "2024"].iloc[0]
    row_2026_ex = all_swe_mean_ex[all_swe_mean_ex["period_label"] == "2026"].iloc[0]
    decomp_ex = []
    total_delta_ex = row_2026_ex["total_chars"] - row_2024_ex["total_chars"]
    for s in SECTIONS:
        d = row_2026_ex[f"sec_{s}_chars"] - row_2024_ex[f"sec_{s}_chars"]
        decomp_ex.append({
            "section": s,
            "mean_chars_2024": row_2024_ex[f"sec_{s}_chars"],
            "mean_chars_2026": row_2026_ex[f"sec_{s}_chars"],
            "delta_chars": d,
            "share_of_growth": (d / total_delta_ex) if total_delta_ex != 0 else np.nan,
        })
    pd.DataFrame(decomp_ex).to_csv(TABLES_DIR / "length_growth_decomposition_excl_aggregators.csv", index=False)

    return prop_summary, mean_summary, decomp_df, all_swe_mean


def run_tone(df):
    base = df[df["is_llm"] & df["text_llm"].notna()].copy()
    # sample cap
    cells = []
    for (period, sen), g in base.groupby(["period_label", "seniority_3level"]):
        n = min(2000, len(g))
        cells.append(g.sample(n=n, random_state=SEED + 1))
    sampled = pd.concat(cells, ignore_index=True)
    rows = []
    for _, r in sampled.iterrows():
        m = compute_tone_metrics(r["text_llm"])
        if m["n_chars"] < 50:
            continue
        per_1k = 1000.0 / m["n_chars"]
        rows.append({
            "uid": r["uid"],
            "period_label": r["period_label"],
            "seniority_3level": r["seniority_3level"],
            "is_aggregator": bool(r["is_aggregator"]),
            "imperative_per_1k": m["imperative_cnt"] * per_1k,
            "inclusive_per_1k": m["inclusive_cnt"] * per_1k,
            "marketing_per_1k": m["marketing_cnt"] * per_1k,
            "passive_per_1k": m["passive_cnt"] * per_1k,
        })
    tdf = pd.DataFrame(rows)
    cols = ["imperative_per_1k", "inclusive_per_1k", "marketing_per_1k", "passive_per_1k"]
    tone_summary = tdf.groupby(["period_label", "seniority_3level"])[cols].mean().reset_index()
    tone_summary.to_csv(TABLES_DIR / "tone_by_period_seniority.csv", index=False)
    # Aggregator-exclusion sensitivity
    tone_ex = tdf[~tdf["is_aggregator"]].groupby(["period_label", "seniority_3level"])[cols].mean().reset_index()
    tone_ex.to_csv(TABLES_DIR / "tone_by_period_seniority_excl_aggregators.csv", index=False)
    return tdf, tone_summary


def run_raw_text_sanity(df):
    """Sensitivity (d): did length grow in raw description too? This is a sanity check on
    the binary 'did length grow' question only."""
    raw = df.copy()
    raw["raw_chars"] = raw["description"].astype(str).str.len()
    raw["llm_chars"] = raw["text_llm"].apply(lambda x: len(x) if isinstance(x, str) else np.nan)
    s = (
        raw.groupby("period_label")
        .agg(n=("uid", "count"),
             mean_raw=("raw_chars", "mean"),
             median_raw=("raw_chars", "median"),
             mean_llm=("llm_chars", "mean"),
             median_llm=("llm_chars", "median"))
        .reset_index()
    )
    s.to_csv(TABLES_DIR / "raw_vs_llm_length.csv", index=False)
    return s


def audit_classifier_precision(df, per_cell=30):
    """Spot-check precision by sampling classified sections across period x source.
    Per-cell is 30, so 30 samples x 7 sections x 2 periods = 420 max, but we cap by
    availability. Save CSV with text snippets for manual review and compute
    'automatic' precision: a section label is considered correct if the segment text
    contains any of the header keywords that define that section.
    """
    base = df[df["is_llm"] & df["text_llm"].notna()].copy()
    # Collect a pool of (uid, source, period, section, snippet) across all postings
    samples = defaultdict(list)  # (section, source, period) -> list
    iter_rows = base.sample(n=min(5000, len(base)), random_state=SEED + 2).itertuples()
    target_sections = ["role_summary", "responsibilities", "requirements",
                       "preferred", "benefits", "about_company", "legal"]
    for r in iter_rows:
        segs = classify_sections(r.text_llm)
        for s in segs:
            if s["section"] not in target_sections:
                continue
            key = (s["section"], r.source, r.period_label)
            if len(samples[key]) >= per_cell:
                continue
            snippet = s["text"][:400].replace("\n", " / ")
            samples[key].append({
                "uid": r.uid,
                "source": r.source,
                "period_label": r.period_label,
                "section": s["section"],
                "char_len": s["char_len"],
                "snippet": snippet,
            })
        # Early stop when all cells filled
        if all(len(samples[(sec, src, per)]) >= per_cell
               for sec in target_sections
               for src in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
               for per in ["2024", "2026"]
               if not (src == "scraped" and per == "2024") and not (src.startswith("kaggle") and per == "2026")):
            break

    flat = []
    for key, items in samples.items():
        for it in items:
            flat.append(it)
    audit_df = pd.DataFrame(flat)

    # Automatic precision: does the first ~120 chars of the segment contain a
    # header-word consistent with the section label? This is a loose lower-bound
    # heuristic that will overcount false positives but gives an indicative number.
    # Liberal keyword heuristic for automatic precision audit. This is a
    # LOWER BOUND — the classifier may be correct on segments without these
    # exact keywords (e.g., role_summary is assigned to pre-header prose).
    header_kw = {
        "role_summary": r"summary|about|overview|position|role|the role|who we are|we are|seeking|is looking|we're looking|we have an opening|join our|join us|description|reporting to|as a |as the ",
        "responsibilities": r"responsibilit|duties|tasks|what you|you will|you'll|day[- ]to[- ]day|in this role|your role|your impact|your mission|develop|design|build|lead|implement",
        "requirements": r"qualification|requirement|skill|experience|must have|you have|who you are|what you'll need|what we're looking|education|bachelor|degree|years of",
        "preferred": r"preferred|nice[- ]to[- ]have|bonus|plus|desired|additional",
        "benefits": r"benefit|compensation|perk|pay|salary|rewards|offer|total rewards",
        "about_company": r"about us|about the|about our|company|who we are|our story|our mission|why join|our culture|our team|why work",
        "legal": r"equal opportunity|eeo|diversity|affirmative|accommodation|unable to provide.*sponsorship",
    }
    def auto_ok(row):
        kw_pat = header_kw.get(row["section"], "")
        if not kw_pat:
            return False
        head = row["snippet"][:120].lower()
        return bool(re.search(kw_pat, head))
    audit_df["auto_ok"] = audit_df.apply(auto_ok, axis=1)
    audit_df.to_csv(TABLES_DIR / "classifier_precision_audit.csv", index=False)

    # Precision summary: by section
    precision = audit_df.groupby("section").agg(
        n=("uid", "count"),
        auto_ok_share=("auto_ok", "mean")
    ).reset_index()
    precision.to_csv(TABLES_DIR / "classifier_precision_summary.csv", index=False)
    return audit_df, precision


def make_figures(sdf, decomp_df, tdf, mdf):
    # Stacked bar: mean section chars by period (all SWE)
    all_swe = sdf.groupby("period_label")[[f"sec_{s}_chars" for s in SECTIONS]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(all_swe))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SECTIONS)))
    for i, s in enumerate(SECTIONS):
        vals = all_swe[f"sec_{s}_chars"].values
        ax.bar(all_swe.index, vals, bottom=bottom, label=s, color=colors[i])
        bottom += vals
    ax.set_ylabel("Mean chars per posting")
    ax.set_title("Section composition of SWE job postings by period\n(llm cleaned text, all SWE)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "section_composition_stacked_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Proportion stacked bar
    all_swe_p = sdf.groupby("period_label")[[f"sec_{s}_prop" for s in SECTIONS]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(all_swe_p))
    for i, s in enumerate(SECTIONS):
        vals = all_swe_p[f"sec_{s}_prop"].values
        ax.bar(all_swe_p.index, vals, bottom=bottom, label=s, color=colors[i])
        bottom += vals
    ax.set_ylabel("Mean proportion of posting")
    ax.set_title("Section proportions of SWE postings by period\n(llm cleaned text)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "section_proportions_stacked_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Readability by period (boxplot Flesch-Kincaid grade)
    if mdf is not None and len(mdf) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [mdf[mdf["period_label"] == p]["flesch_kincaid_grade"].dropna().values for p in ["2024", "2026"]]
        ax.boxplot(data, labels=["2024", "2026"], showfliers=False)
        ax.set_ylabel("Flesch-Kincaid Grade Level")
        ax.set_title("Readability by period — SWE postings")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "readability_fk_grade.png", dpi=150)
        plt.close(fig)

    # Tone markers by period x seniority
    if tdf is not None and len(tdf) > 0:
        cols = ["imperative_per_1k", "inclusive_per_1k", "marketing_per_1k", "passive_per_1k"]
        agg = tdf.groupby(["period_label", "seniority_3level"])[cols].mean().reset_index()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for i, c in enumerate(cols):
            ax = axes.flat[i]
            pivot = agg.pivot(index="seniority_3level", columns="period_label", values=c)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(c)
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation=20)
        fig.suptitle("Tone markers by period × seniority — SWE postings")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "tone_markers.png", dpi=150)
        plt.close(fig)


def main():
    print("Loading data…")
    df = load_data()
    print(f"rows: {len(df)}")
    print("period_label distribution:")
    print(df.groupby(["period_label", "source"]).size())

    print("\n=== Raw vs LLM length sanity ===")
    raw_s = run_raw_text_sanity(df)
    print(raw_s)

    print("\n=== Section anatomy ===")
    sdf = run_section_anatomy(df)
    prop_summary, mean_summary, decomp_df, all_swe_mean = summarize_section_anatomy(sdf)
    print("Length growth decomposition (all SWE, llm):")
    print(decomp_df.to_string(index=False))

    print("\n=== Readability ===")
    mdf, read_summary, read_summary_ex = run_readability(df)
    print(read_summary)

    print("\n=== Tone markers ===")
    tdf, tone_summary = run_tone(df)
    print(tone_summary)

    print("\n=== Classifier precision audit ===")
    audit_df, precision = audit_classifier_precision(df, per_cell=30)
    print(precision)

    print("\n=== Figures ===")
    make_figures(sdf, decomp_df, tdf, mdf)

    print("\nDone. Tables -> exploration/tables/T13/, figures -> exploration/figures/T13/")


if __name__ == "__main__":
    main()
