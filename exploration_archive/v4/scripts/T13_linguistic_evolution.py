"""T13 — Linguistic and structural evolution.

Analyzes how the FORM of SWE job postings has evolved between 2024 and 2026:
readability, section anatomy, tone markers, imperative/inclusive density,
marketing language, under both combined best-available seniority and the
YOE-based entry proxy.

Outputs written under:
    exploration/tables/T13/
    exploration/figures/T13/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import textstat

warnings.filterwarnings("ignore")

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES = ROOT / "exploration" / "tables" / "T13"
OUT_FIGS = ROOT / "exploration" / "figures" / "T13"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# Make sure the section classifier is importable.
sys.path.insert(0, str(SHARED))
from section_classifier import classify_sections, SECTIONS  # noqa: E402

RANDOM_SEED = 42
SAMPLE_PER_CELL = 2000


def load_data() -> pd.DataFrame:
    """Load cleaned text (metadata only) and join combined seniority + raw
    un-stopworded text from unified.parquet.

    IMPORTANT: swe_cleaned_text.parquet has English stopwords removed, which
    breaks phrase-level section headers ("you will" -> "", "we are" -> "")
    and tone markers. For T13 we need the original text; we use
    COALESCE(description_core_llm, description_core, description) to match the
    same text source logic the cleaned artifact uses, minus the stopword step.
    """
    print("[load] reading swe_cleaned_text.parquet (metadata only) ...")
    df = pq.read_table(SHARED / "swe_cleaned_text.parquet").to_pandas()
    df = df.drop(columns=["description_cleaned"])  # will be replaced by raw
    df["period_bucket"] = df["period"].map(
        lambda p: "2024" if str(p).startswith("2024") else "2026"
    )

    print("[load] joining raw text + combined seniority from unified ...")
    con = duckdb.connect()
    q = """
        SELECT
            uid,
            seniority_llm,
            llm_classification_coverage,
            CASE
                WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
                WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
                ELSE NULL
            END AS seniority_best_available,
            COALESCE(description_core_llm, description_core, description) AS description_raw_text
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin' AND is_english = true
          AND date_flag = 'ok' AND is_swe = true
    """
    meta = con.execute(q, [str(ROOT / "data" / "unified.parquet")]).fetchdf()
    df = df.merge(meta, on="uid", how="left")
    # Rename so downstream code keeps working unchanged.
    df = df.rename(columns={"description_raw_text": "description_cleaned"})
    con.close()

    # Derive coarse seniority buckets for display.
    def bucket(s):
        if s in ("entry", "associate"):
            return "entry/associate"
        if s == "mid-senior":
            return "mid-senior"
        if s == "director":
            return "director"
        return "unknown"

    df["sen_best_bucket"] = df["seniority_best_available"].map(bucket)
    df["yoe_entry_proxy"] = (df["yoe_extracted"].fillna(-1) <= 2) & (df["yoe_extracted"].fillna(-1) >= 0)
    df["yoe_entry_label"] = np.where(
        df["yoe_entry_proxy"], "entry_yoe",
        np.where(df["yoe_extracted"].notna(), "nonentry_yoe", "yoe_missing"),
    )
    return df


def sample_for_readability(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified sample by period x best-seniority bucket for readability metrics."""
    frames = []
    rng = np.random.default_rng(RANDOM_SEED)
    for (p, b), grp in df.groupby(["period_bucket", "sen_best_bucket"]):
        if len(grp) == 0:
            continue
        take = min(len(grp), SAMPLE_PER_CELL)
        idx = rng.choice(grp.index.values, size=take, replace=False)
        frames.append(df.loc[idx])
    out = pd.concat(frames, ignore_index=True)
    print(f"[sample] readability sample size: {len(out):,}")
    return out


def compute_readability(sample: pd.DataFrame) -> pd.DataFrame:
    """Compute textstat metrics per row."""
    print("[readability] computing metrics ...")
    fk_grade = []
    flesch_ease = []
    fog = []
    avg_sent = []
    lexicon = []
    syllables = []
    ttr_1k = []

    for t in sample["description_cleaned"].fillna(""):
        if not t or len(t) < 20:
            fk_grade.append(np.nan)
            flesch_ease.append(np.nan)
            fog.append(np.nan)
            avg_sent.append(np.nan)
            lexicon.append(0)
            syllables.append(0)
            ttr_1k.append(np.nan)
            continue
        try:
            fk_grade.append(textstat.flesch_kincaid_grade(t))
            flesch_ease.append(textstat.flesch_reading_ease(t))
            fog.append(textstat.gunning_fog(t))
            avg_sent.append(textstat.avg_sentence_length(t))
            lexicon.append(textstat.lexicon_count(t, removepunct=True))
            syllables.append(textstat.syllable_count(t))
        except Exception:
            fk_grade.append(np.nan)
            flesch_ease.append(np.nan)
            fog.append(np.nan)
            avg_sent.append(np.nan)
            lexicon.append(0)
            syllables.append(0)
        # TTR on first 1000 chars
        head = t[:1000].lower()
        toks = [w for w in head.split() if w.isalpha()]
        if len(toks) >= 50:
            ttr_1k.append(len(set(toks)) / len(toks))
        else:
            ttr_1k.append(np.nan)

    sample = sample.copy()
    sample["fk_grade"] = fk_grade
    sample["flesch_ease"] = flesch_ease
    sample["gunning_fog"] = fog
    sample["avg_sentence_length"] = avg_sent
    sample["lexicon_count"] = lexicon
    sample["syllable_count"] = syllables
    sample["ttr_first_1k"] = ttr_1k
    return sample


def summarize_readability(sample: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fk_grade", "flesch_ease", "gunning_fog", "avg_sentence_length",
        "lexicon_count", "syllable_count", "ttr_first_1k",
    ]
    rows = []
    for (p, b), grp in sample.groupby(["period_bucket", "sen_best_bucket"]):
        row = {"period": p, "seniority": b, "n": len(grp)}
        for m in metrics:
            row[f"{m}_median"] = float(np.nanmedian(grp[m]))
            row[f"{m}_mean"] = float(np.nanmean(grp[m]))
        rows.append(row)
    # Also overall by period
    for p, grp in sample.groupby("period_bucket"):
        row = {"period": p, "seniority": "ALL", "n": len(grp)}
        for m in metrics:
            row[f"{m}_median"] = float(np.nanmedian(grp[m]))
            row[f"{m}_mean"] = float(np.nanmean(grp[m]))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["seniority", "period"])
    return summary


def compute_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Run section classifier on the full corpus."""
    print("[sections] classifying sections for all rows ...")
    n = len(df)
    recs = []
    for i, (uid, txt) in enumerate(zip(df["uid"].values, df["description_cleaned"].values)):
        if i % 10000 == 0:
            print(f"  [sections] {i}/{n}")
        out = classify_sections(txt or "")
        out["uid"] = uid
        recs.append(out)
    sec = pd.DataFrame(recs)
    # Add total and proportions
    sec["total"] = sec[SECTIONS].sum(axis=1)
    for s in SECTIONS:
        sec[f"{s}_prop"] = np.where(sec["total"] > 0, sec[s] / sec["total"], 0.0)
    return sec


def section_summary(df: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Median proportions and absolute character counts by period x seniority."""
    merged = df.merge(sec, on="uid", how="left")
    rows = []
    for (p, b), grp in merged.groupby(["period_bucket", "sen_best_bucket"]):
        row = {"period": p, "seniority": b, "n": len(grp)}
        for s in SECTIONS:
            row[f"{s}_median_prop"] = float(grp[f"{s}_prop"].median())
            row[f"{s}_median_chars"] = float(grp[s].median())
        rows.append(row)
    for p, grp in merged.groupby("period_bucket"):
        row = {"period": p, "seniority": "ALL", "n": len(grp)}
        for s in SECTIONS:
            row[f"{s}_median_prop"] = float(grp[f"{s}_prop"].median())
            row[f"{s}_median_chars"] = float(grp[s].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seniority", "period"]), merged


def plot_section_composition(sec_summary: pd.DataFrame, out_path: Path):
    """Stacked bar chart: median absolute section chars by period x seniority."""
    order = ["ALL", "entry/associate", "mid-senior", "unknown"]
    periods = ["2024", "2026"]
    cats = [
        "summary", "responsibilities", "requirements", "preferred",
        "benefits", "about_company", "legal", "unclassified",
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(cats)))

    fig, ax = plt.subplots(figsize=(10, 6))
    xticks = []
    xlabels = []
    x = 0
    for grp in order:
        for p in periods:
            row = sec_summary[(sec_summary["seniority"] == grp) & (sec_summary["period"] == p)]
            if row.empty:
                continue
            bottom = 0
            for cat, color in zip(cats, colors):
                val = float(row[f"{cat}_median_chars"].iloc[0])
                ax.bar(x, val, bottom=bottom, color=color,
                       label=cat if x == 0 else None, width=0.7)
                bottom += val
            xticks.append(x)
            xlabels.append(f"{grp}\n{p}")
            x += 1
        x += 0.5  # gap between seniority groups
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Median characters per posting")
    ax.set_title("T13 — SWE LinkedIn: median section anatomy (chars) by period x seniority")
    ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_path}")


def plot_section_composition_pct(sec_summary: pd.DataFrame, out_path: Path):
    """Stacked bar chart normalized to 100% (composition view)."""
    order = ["ALL", "entry/associate", "mid-senior", "unknown"]
    periods = ["2024", "2026"]
    cats = [
        "summary", "responsibilities", "requirements", "preferred",
        "benefits", "about_company", "legal", "unclassified",
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(cats)))

    fig, ax = plt.subplots(figsize=(10, 6))
    xticks = []
    xlabels = []
    x = 0
    for grp in order:
        for p in periods:
            row = sec_summary[(sec_summary["seniority"] == grp) & (sec_summary["period"] == p)]
            if row.empty:
                continue
            bottom = 0
            for cat, color in zip(cats, colors):
                val = float(row[f"{cat}_median_prop"].iloc[0]) * 100
                ax.bar(x, val, bottom=bottom, color=color,
                       label=cat if x == 0 else None, width=0.7)
                bottom += val
            xticks.append(x)
            xlabels.append(f"{grp}\n{p}")
            x += 1
        x += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Median % of posting")
    ax.set_title("T13 — SWE LinkedIn: median section share (%) by period x seniority")
    ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_path}")


# ----- Tone markers -----

IMPERATIVE_PATTERNS = [
    r"\byou will\b", r"\byou'?ll\b", r"\byou must\b", r"\byou should\b",
    r"\byou need\b", r"\bmust have\b", r"\bshould have\b", r"\brequired\b",
]
INCLUSIVE_PATTERNS = [
    r"\bwe\b", r"\bour team\b", r"\bour\b", r"\byou'?ll join\b",
    r"\bjoin us\b", r"\btogether\b", r"\bcollaborate\b", r"\bour company\b",
]
MARKETING_PATTERNS = [
    r"\bexciting\b", r"\binnovative\b", r"\bcutting[- ]edge\b",
    r"\bworld[- ]class\b", r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bpassionate\b", r"\bfast[- ]paced\b", r"\bdynamic\b",
    r"\bmission[- ]driven\b", r"\brockstar\b", r"\bninja\b",
    r"\brevolution(ary|ize)\b", r"\bbest[- ]in[- ]class\b",
    r"\bgame[- ]chang(er|ing)\b",
]
PASSIVE_PATTERNS = [
    r"\bis (being|required|expected)\b",
    r"\bare (being|required|expected)\b",
    r"\bwill be\b", r"\bhas been\b", r"\bhave been\b",
    r"\bbe (responsible|expected|required)\b",
]


def count_patterns(text: str, patterns) -> int:
    import re
    n = 0
    for p in patterns:
        n += len(re.findall(p, text, flags=re.I))
    return n


def compute_tone(sample: pd.DataFrame) -> pd.DataFrame:
    print("[tone] computing tone markers ...")
    rows = []
    for _, r in sample.iterrows():
        t = r["description_cleaned"] or ""
        L = max(len(t), 1)
        per_1k = 1000 / L
        rows.append({
            "uid": r["uid"],
            "imperative_per_1k": count_patterns(t, IMPERATIVE_PATTERNS) * per_1k,
            "inclusive_per_1k": count_patterns(t, INCLUSIVE_PATTERNS) * per_1k,
            "marketing_per_1k": count_patterns(t, MARKETING_PATTERNS) * per_1k,
            "passive_per_1k": count_patterns(t, PASSIVE_PATTERNS) * per_1k,
        })
    return pd.DataFrame(rows)


def tone_summary(sample: pd.DataFrame, tone: pd.DataFrame) -> pd.DataFrame:
    merged = sample.merge(tone, on="uid", how="left")
    metrics = ["imperative_per_1k", "inclusive_per_1k", "marketing_per_1k", "passive_per_1k"]
    rows = []
    for (p, b), grp in merged.groupby(["period_bucket", "sen_best_bucket"]):
        row = {"period": p, "seniority": b, "n": len(grp)}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(grp[m]))
            row[f"{m}_median"] = float(np.nanmedian(grp[m]))
        rows.append(row)
    for p, grp in merged.groupby("period_bucket"):
        row = {"period": p, "seniority": "ALL", "n": len(grp)}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(grp[m]))
            row[f"{m}_median"] = float(np.nanmedian(grp[m]))
        rows.append(row)
    # YOE-proxy split
    for (p, y), grp in merged.groupby(["period_bucket", "yoe_entry_label"]):
        row = {"period": p, "seniority": y, "n": len(grp)}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(grp[m]))
            row[f"{m}_median"] = float(np.nanmedian(grp[m]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["seniority", "period"])


# ----- Entry-level specifically -----

def entry_split_summary(df: pd.DataFrame, sec: pd.DataFrame, readab: pd.DataFrame) -> pd.DataFrame:
    """Compare entry definitions (combined column vs YOE proxy) on structure and tone."""
    merged = df.merge(sec, on="uid", how="left")
    # For the full corpus: how do best-seniority-entry vs yoe-entry postings differ on structure?
    rows = []
    defs = {
        "combined_entry_2024": (merged["period_bucket"] == "2024") & (merged["sen_best_bucket"] == "entry/associate"),
        "combined_entry_2026": (merged["period_bucket"] == "2026") & (merged["sen_best_bucket"] == "entry/associate"),
        "yoe_entry_2024": (merged["period_bucket"] == "2024") & (merged["yoe_entry_proxy"]),
        "yoe_entry_2026": (merged["period_bucket"] == "2026") & (merged["yoe_entry_proxy"]),
        "combined_mid_2024": (merged["period_bucket"] == "2024") & (merged["sen_best_bucket"] == "mid-senior"),
        "combined_mid_2026": (merged["period_bucket"] == "2026") & (merged["sen_best_bucket"] == "mid-senior"),
    }
    for name, mask in defs.items():
        grp = merged[mask]
        row = {"group": name, "n": int(mask.sum())}
        for s in SECTIONS:
            row[f"{s}_median_prop"] = float(grp[f"{s}_prop"].median()) if len(grp) else np.nan
            row[f"{s}_median_chars"] = float(grp[s].median()) if len(grp) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----- Entry-vs-mid drivers of length growth -----

def length_growth_decomposition(df: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Show median chars per section in 2024 vs 2026 across all SWE — this is the
    critical diagnostic: where did the 57% length growth come from?
    """
    merged = df.merge(sec, on="uid", how="left")
    rows = []
    for p, grp in merged.groupby("period_bucket"):
        row = {"period": p, "n": len(grp), "median_total": float(grp["total"].median())}
        for s in SECTIONS:
            row[f"{s}_median"] = float(grp[s].median())
            row[f"{s}_mean"] = float(grp[s].mean())
        rows.append(row)
    tab = pd.DataFrame(rows)
    # Compute deltas (2026 - 2024) and share-of-growth
    if len(tab) == 2:
        t24 = tab[tab["period"] == "2024"].iloc[0]
        t26 = tab[tab["period"] == "2026"].iloc[0]
        delta_rows = []
        for s in SECTIONS:
            d_median = t26[f"{s}_median"] - t24[f"{s}_median"]
            d_mean = t26[f"{s}_mean"] - t24[f"{s}_mean"]
            delta_rows.append({
                "section": s,
                "median_2024": t24[f"{s}_median"],
                "median_2026": t26[f"{s}_median"],
                "delta_median": d_median,
                "mean_2024": t24[f"{s}_mean"],
                "mean_2026": t26[f"{s}_mean"],
                "delta_mean": d_mean,
            })
        delta = pd.DataFrame(delta_rows)
        total_delta_mean = delta["delta_mean"].sum()
        if total_delta_mean != 0:
            delta["share_of_growth_mean"] = delta["delta_mean"] / total_delta_mean
        return tab, delta
    return tab, pd.DataFrame()


def main():
    df = load_data()
    print(f"[main] loaded {len(df):,} rows (2024={len(df[df.period_bucket=='2024']):,}, 2026={len(df[df.period_bucket=='2026']):,})")

    # 1. Readability on a stratified sample.
    sample = sample_for_readability(df)
    sample = compute_readability(sample)
    read_summary = summarize_readability(sample)
    read_summary.to_csv(OUT_TABLES / "readability_by_period_seniority.csv", index=False)
    print(f"[out] wrote readability_by_period_seniority.csv ({len(read_summary)} rows)")

    # 2. Section anatomy (full corpus).
    sec = compute_sections(df)
    sec.to_parquet(OUT_TABLES / "section_chars_per_posting.parquet", index=False)
    sec_summary, merged = section_summary(df, sec)
    sec_summary.to_csv(OUT_TABLES / "section_summary_by_period_seniority.csv", index=False)
    print(f"[out] wrote section_summary_by_period_seniority.csv ({len(sec_summary)} rows)")

    # 3. Length growth decomposition — the critical T13 diagnostic.
    full_tab, delta = length_growth_decomposition(df, sec)
    full_tab.to_csv(OUT_TABLES / "length_growth_full_table.csv", index=False)
    delta.to_csv(OUT_TABLES / "length_growth_decomposition.csv", index=False)
    print("[out] wrote length_growth_decomposition.csv")
    print(delta.to_string(index=False))

    # Figures.
    plot_section_composition(sec_summary, OUT_FIGS / "section_anatomy_chars.png")
    plot_section_composition_pct(sec_summary, OUT_FIGS / "section_anatomy_pct.png")

    # 4. Tone markers on same sample.
    tone = compute_tone(sample)
    tone.to_csv(OUT_TABLES / "tone_per_posting.csv", index=False)
    merged_sample = sample.merge(tone, on="uid", how="left")
    # merged_sample already contains yoe_entry_label; rebuild tone summary
    tone_sum = tone_summary(sample, tone)
    tone_sum.to_csv(OUT_TABLES / "tone_summary_by_period_seniority.csv", index=False)
    print(f"[out] wrote tone_summary_by_period_seniority.csv ({len(tone_sum)} rows)")

    # 5. Entry-level split (structure).
    entry_sum = entry_split_summary(df, sec, sample)
    entry_sum.to_csv(OUT_TABLES / "entry_split_structure.csv", index=False)
    print(f"[out] wrote entry_split_structure.csv ({len(entry_sum)} rows)")

    # 6. Print headline summary.
    print("\n========= T13 HEADLINE =========")
    all_24 = full_tab[full_tab["period"] == "2024"].iloc[0]
    all_26 = full_tab[full_tab["period"] == "2026"].iloc[0]
    print(f"Median total chars: 2024={all_24['median_total']:.0f}, 2026={all_26['median_total']:.0f}")
    print(f"Mean delta chars by section (2026 - 2024):")
    print(delta[["section", "delta_mean", "share_of_growth_mean"]].to_string(index=False))

    # Save a headline markdown fragment for the report to reuse.
    summary_md = OUT_TABLES / "headline_numbers.md"
    with open(summary_md, "w") as f:
        f.write("# T13 headline numbers\n\n")
        f.write(f"- Median total cleaned-text chars: 2024 = {all_24['median_total']:.0f}, 2026 = {all_26['median_total']:.0f}\n")
        f.write("\n## Section delta (mean chars)\n\n")
        f.write(delta[["section", "mean_2024", "mean_2026", "delta_mean", "share_of_growth_mean"]].to_string(index=False))
        f.write("\n")
    print(f"[out] wrote {summary_md}")

    print("\n[done] T13 complete")


if __name__ == "__main__":
    main()
