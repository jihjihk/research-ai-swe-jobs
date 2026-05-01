"""T29 — LLM-authored description detection.

Compute per-posting authorship-style features, aggregate to corpus-level
distributions, split by period/company, correlate with Wave-2 metrics,
and run the unifying-mechanism test (re-compute Gate 2 headlines on
low-authorship-score postings).

Critical caveats:
  - Text-source composition is a major confound. We restrict the PRIMARY
    analysis to rows with `llm_extraction_coverage='labeled'` (description_core_llm),
    because that text column is comparable across periods. We also run a
    sensitivity on raw `description` (binary and length-normalized).
  - AI-domain postings genuinely use AI vocabulary; that's not LLM authorship.
    We handle this by reporting the AI-vocabulary signal separately and by
    computing the composite score WITHOUT raw AI vocabulary tokens.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES = ROOT / "exploration" / "tables" / "T29"
OUT_FIGS = ROOT / "exploration" / "figures" / "T29"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "is_swe=true AND source_platform='linkedin' AND is_english=true AND date_flag='ok'"
)

# ---------------------------------------------------------------------------
# LLM authorship signature vocabulary.
# NOTE: we deliberately EXCLUDE AI-domain terms (ai, ml, llm, neural, model)
# so AI-topic postings don't get tagged as LLM-authored by content.
# ---------------------------------------------------------------------------

LLM_TELL_WORDS = [
    # Classic sparkle words
    r"delve(?:d|s|ing)?",
    r"tapestry",
    r"leverag(?:e|es|ed|ing)",
    r"robust",
    r"unleash(?:ed|es|ing)?",
    r"embark(?:ed|s|ing)?",
    r"navigat(?:e|es|ed|ing)",
    r"cutting[-\s]?edge",
    r"in the realm of",
    r"comprehensive",
    r"seamless(?:ly)?",
    r"furthermore",
    r"moreover",
    r"notably",
    r"align(?:s|ed|ing)?\s+with",
    r"at the forefront",
    r"pivotal",
    r"harness(?:es|ed|ing)?",
    r"dynamic",
    r"vibrant",
    r"intricate",
    r"meticulous(?:ly)?",
    # Additional LLM patterns
    r"plethora",
    r"foster(?:ing|ed|s)?",
    r"multifaceted",
    r"underscore(?:s|d|ing)?",
    r"paramount",
    r"testament",
    r"it['\u2019]s worth noting",
    r"it is worth noting",
    r"nestled",
    r"bespoke",
    r"realm",
    r"ever[-\s]?evolving",
    r"empower(?:s|ed|ing)?",
    r"orchestrat(?:e|es|ed|ing)",
    r"spearhead(?:s|ed|ing)?",
    r"revolutioniz(?:e|es|ed|ing)",
    r"game[-\s]?changer",
    r"transformative",
    r"synergy",
    r"holistic(?:ally)?",
]
LLM_TELL_PATTERN = re.compile(r"\b(?:" + "|".join(LLM_TELL_WORDS) + r")\b", re.IGNORECASE)

SENT_SPLIT = re.compile(r"[.!?]+(?=\s|$)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
# Em-dash counting: literal — plus " -- " surrogate
EMDASH_CHARS = ("\u2014", "\u2013")  # em-dash, en-dash
# Bullet markers
BULLET_RE = re.compile(r"(?m)^[\s]*(?:[-*\u2022\u25e6\u25cb\u2023\u2043]|\d+\.)\s+")
PARA_SPLIT = re.compile(r"\n{2,}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(texts: pd.Series) -> pd.DataFrame:
    """Compute per-posting style features. Input is a text series."""
    out = pd.DataFrame(index=texts.index)
    lens = texts.str.len().fillna(0).astype(int)
    out["char_len"] = lens

    safe_per_1k = np.where(lens > 0, 1000.0 / lens, 0.0)

    # 1. LLM tell density per 1K chars
    tell_counts = texts.str.count(LLM_TELL_PATTERN.pattern).fillna(0).astype(int)
    out["tell_count"] = tell_counts
    out["tell_density"] = tell_counts * safe_per_1k

    # 2. Em-dash density per 1K chars (counts em-dash OR en-dash OR " -- ")
    def emdash_count(s: str) -> int:
        if not isinstance(s, str):
            return 0
        return s.count("\u2014") + s.count("\u2013") + s.count(" -- ")

    emdash_counts = texts.fillna("").map(emdash_count).astype(int)
    out["emdash_count"] = emdash_counts
    out["emdash_density"] = emdash_counts * safe_per_1k

    # 3. Sentence length distribution
    def sent_stats(s: str):
        if not isinstance(s, str) or not s:
            return (0.0, 0.0, 0)
        parts = [p.strip() for p in SENT_SPLIT.split(s) if p.strip()]
        if not parts:
            return (0.0, 0.0, 0)
        word_counts = np.array([len(WORD_RE.findall(p)) for p in parts])
        word_counts = word_counts[word_counts > 0]
        if word_counts.size == 0:
            return (0.0, 0.0, 0)
        return (float(word_counts.mean()), float(word_counts.std(ddof=0)), int(word_counts.size))

    stats = texts.fillna("").map(sent_stats)
    out["sent_mean_words"] = stats.map(lambda t: t[0]).astype(float)
    out["sent_std_words"] = stats.map(lambda t: t[1]).astype(float)
    out["sent_count"] = stats.map(lambda t: t[2]).astype(int)

    # 4. Vocabulary diversity (type-token ratio)
    def ttr(s: str):
        if not isinstance(s, str) or not s:
            return 0.0
        tokens = [w.lower() for w in WORD_RE.findall(s)]
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    out["type_token_ratio"] = texts.fillna("").map(ttr).astype(float)

    # 5. Bullet density per 1K chars
    bullet_counts = texts.fillna("").map(lambda s: len(BULLET_RE.findall(s)) if isinstance(s, str) else 0).astype(int)
    out["bullet_count"] = bullet_counts
    out["bullet_density"] = bullet_counts * safe_per_1k

    # 6. Paragraph length / uniformity
    def para_stats(s: str):
        if not isinstance(s, str) or not s.strip():
            return (0.0, 0.0, 0)
        parts = [p for p in PARA_SPLIT.split(s) if p.strip()]
        if not parts:
            return (0.0, 0.0, 0)
        plen = np.array([len(p) for p in parts])
        return (float(plen.mean()), float(plen.std(ddof=0)), int(plen.size))

    pstats = texts.fillna("").map(para_stats)
    out["para_mean_chars"] = pstats.map(lambda t: t[0]).astype(float)
    out["para_std_chars"] = pstats.map(lambda t: t[1]).astype(float)
    out["para_count"] = pstats.map(lambda t: t[2]).astype(int)

    return out


# ---------------------------------------------------------------------------
# Composite authorship score
# ---------------------------------------------------------------------------

def compute_authorship_score(feat: pd.DataFrame) -> pd.Series:
    """Z-score each feature on its 2024 distribution then sum with signed weights.

    Higher score = more LLM-like. Signs:
      + tell_density, emdash_density, sent_mean_words, bullet_density, para_mean_chars
      − sent_std_words (low variance => uniform), type_token_ratio (low TTR is suggestive)

    We compute per-feature z on the full available sample so both periods
    contribute; this is a relative score within corpus.
    """
    # Signs (+1 => LLM-ish, -1 => LLM-ish means lower value)
    signs = {
        "tell_density": +1,
        "emdash_density": +1,
        "sent_mean_words": +1,
        "bullet_density": +1,
        "para_mean_chars": +1,
        "sent_std_words": -1,
        "type_token_ratio": -1,
    }
    z_cols = []
    for f, sign in signs.items():
        x = feat[f].astype(float)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if sd == 0 or np.isnan(sd):
            z = pd.Series(np.zeros(len(x)), index=x.index)
        else:
            z = (x - mu) / sd * sign
        z_cols.append(z.rename(f"z_{f}"))
    z_df = pd.concat(z_cols, axis=1)
    score = z_df.mean(axis=1)
    return score, z_df


# ---------------------------------------------------------------------------
# Load SWE corpus
# ---------------------------------------------------------------------------

def load_swe() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT uid, period, source, is_aggregator, company_name_canonical,
               seniority_final, yoe_extracted,
               description, description_core_llm,
               llm_extraction_coverage, description_length
        FROM read_parquet('{UNIFIED}')
        WHERE {DEFAULT_FILTER}
        """
    ).fetchdf()
    con.close()
    df["period2"] = df["period"].map(lambda p: "2024" if str(p).startswith("2024") else "2026")
    df["is_entry_final"] = df["seniority_final"].eq("entry")
    df["yoe_le2"] = df["yoe_extracted"].le(2).fillna(False)
    df["has_clean_text"] = df["llm_extraction_coverage"].eq("labeled") & df["description_core_llm"].notna()
    df["text_clean"] = np.where(df["has_clean_text"], df["description_core_llm"].fillna(""), "")
    df["text_raw"] = df["description"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[T29] Loading SWE corpus...")
    df = load_swe()
    print(f"[T29] Rows: {len(df):,}; clean-text rows: {df['has_clean_text'].sum():,}")

    # Restrict primary analysis to has_clean_text rows (text source comparable)
    clean = df[df["has_clean_text"]].copy().reset_index(drop=True)
    print(f"[T29] Primary sample (clean text): {len(clean):,}")
    print(clean.groupby("period2").size().to_string())

    print("[T29] Extracting features on cleaned text...")
    feat = extract_features(clean["text_clean"])
    feat["uid"] = clean["uid"].to_numpy()
    feat["period2"] = clean["period2"].to_numpy()
    feat["company_name_canonical"] = clean["company_name_canonical"].to_numpy()
    feat["is_aggregator"] = clean["is_aggregator"].to_numpy()
    feat["is_entry_final"] = clean["is_entry_final"].to_numpy()
    feat["yoe_le2"] = clean["yoe_le2"].to_numpy()

    print("[T29] Computing authorship score...")
    score, z_df = compute_authorship_score(feat)
    feat["authorship_score"] = score.to_numpy()
    for c in z_df.columns:
        feat[c] = z_df[c].to_numpy()

    feat.to_csv(OUT_TABLES / "authorship_scores.csv", index=False)
    print(f"[T29] Wrote authorship_scores.csv: {len(feat):,} rows")

    # ----- 4. Distribution by period -----
    print("\n[T29] Step 4: distribution by period")
    period_stats = (
        feat.groupby("period2")
        .agg(
            n=("uid", "count"),
            score_mean=("authorship_score", "mean"),
            score_median=("authorship_score", "median"),
            score_std=("authorship_score", "std"),
            tell_density_mean=("tell_density", "mean"),
            tell_density_median=("tell_density", "median"),
            emdash_density_mean=("emdash_density", "mean"),
            emdash_density_median=("emdash_density", "median"),
            sent_mean_mean=("sent_mean_words", "mean"),
            sent_std_mean=("sent_std_words", "mean"),
            ttr_mean=("type_token_ratio", "mean"),
            bullet_density_mean=("bullet_density", "mean"),
            para_mean_mean=("para_mean_chars", "mean"),
        )
        .reset_index()
    )
    period_stats.to_csv(OUT_TABLES / "period_stats.csv", index=False)
    print(period_stats.to_string())

    # Cross-posting variance = std of authorship_score
    # Is 2026 more uniform (lower std)?
    for p in ["2024", "2026"]:
        s = feat[feat["period2"] == p]["authorship_score"]
        print(f"  {p}: n={len(s)}, median={s.median():.3f}, std={s.std():.3f}, iqr={s.quantile(0.75) - s.quantile(0.25):.3f}")

    # ----- 5. Distribution by company -----
    print("\n[T29] Step 5: distribution by company")
    company_stats = (
        feat.groupby(["company_name_canonical", "period2"])
        .agg(
            n=("uid", "count"),
            score_mean=("authorship_score", "mean"),
            tell_density_mean=("tell_density", "mean"),
        )
        .reset_index()
    )
    # Filter companies with >=10 postings
    company_filt = company_stats[company_stats["n"] >= 10].copy()
    company_filt.to_csv(OUT_TABLES / "company_stats.csv", index=False)

    # Top 20 high-score companies in 2024 and 2026
    top_2024 = company_filt[company_filt["period2"] == "2024"].sort_values("score_mean", ascending=False).head(20)
    top_2026 = company_filt[company_filt["period2"] == "2026"].sort_values("score_mean", ascending=False).head(20)
    low_2026 = company_filt[company_filt["period2"] == "2026"].sort_values("score_mean").head(20)
    top_2024.to_csv(OUT_TABLES / "top_llm_companies_2024.csv", index=False)
    top_2026.to_csv(OUT_TABLES / "top_llm_companies_2026.csv", index=False)
    low_2026.to_csv(OUT_TABLES / "low_llm_companies_2026.csv", index=False)
    print("Top LLM-score companies 2024:")
    print(top_2024[["company_name_canonical", "n", "score_mean"]].to_string())
    print("Top LLM-score companies 2026:")
    print(top_2026[["company_name_canonical", "n", "score_mean"]].to_string())

    # ----- 6. Correlation with Wave 2 findings -----
    print("\n[T29] Step 6: correlation with Wave 2 content metrics")
    # Join to T28 per-row metrics
    per_row = pd.read_parquet(ROOT / "exploration" / "tables" / "T28" / "per_row_metrics.parquet")
    merged = feat.merge(
        per_row[["uid", "requirement_breadth", "tech_count", "credential_stack_depth",
                 "any_ai_narrow", "any_ai_broad", "scope_density", "clean_len"]],
        on="uid",
        how="left",
    )
    corr_cols = [
        "authorship_score",
        "tell_density",
        "emdash_density",
        "char_len",
        "requirement_breadth",
        "tech_count",
        "credential_stack_depth",
        "any_ai_narrow",
        "any_ai_broad",
        "scope_density",
    ]
    corr = merged[corr_cols].corr().round(3)
    corr.to_csv(OUT_TABLES / "correlation_matrix.csv")
    print(corr.to_string())

    # ----- 7. The unifying-mechanism test -----
    print("\n[T29] Step 7: unifying-mechanism test")
    # Low-LLM-score subset: below 25th percentile of 2024
    cutoff_2024 = feat[feat["period2"] == "2024"]["authorship_score"].quantile(0.25)
    print(f"Using score cutoff = 2024 p25 = {cutoff_2024:.3f}")
    feat["low_llm"] = feat["authorship_score"] <= cutoff_2024
    merged["low_llm"] = merged["authorship_score"] <= cutoff_2024

    rows = []
    for subset_label, subset in [("full", merged), ("low_llm", merged[merged["low_llm"]])]:
        for metric in [
            "char_len",
            "requirement_breadth",
            "tech_count",
            "credential_stack_depth",
            "any_ai_narrow",
            "any_ai_broad",
            "scope_density",
        ]:
            grp = subset.groupby("period2")[metric].mean()
            rows.append(
                {
                    "subset": subset_label,
                    "metric": metric,
                    "value_2024": grp.get("2024", np.nan),
                    "value_2026": grp.get("2026", np.nan),
                    "delta": grp.get("2026", np.nan) - grp.get("2024", np.nan),
                    "pct_change": (grp.get("2026", np.nan) / grp.get("2024", np.nan) - 1) * 100 if grp.get("2024", 0) else np.nan,
                }
            )
    headline_test = pd.DataFrame(rows)

    # Wide comparison: compute attenuation of delta between full and low_llm
    wide = headline_test.pivot_table(index="metric", columns="subset", values="delta")
    wide["attenuation_pct"] = (1 - wide["low_llm"] / wide["full"]) * 100
    wide.to_csv(OUT_TABLES / "unifying_mechanism_test.csv")
    print(wide.to_string())

    # Internal homogeneity test on low-LLM subset (std of authorship score within period)
    homog_rows = []
    for subset_label in ["full", "low_llm"]:
        for p in ["2024", "2026"]:
            sub = merged if subset_label == "full" else merged[merged["low_llm"]]
            sub = sub[sub["period2"] == p]
            # Use std of length and req_breadth as proxies for internal homogeneity
            homog_rows.append(
                {
                    "subset": subset_label,
                    "period": p,
                    "n": len(sub),
                    "score_std": sub["authorship_score"].std(),
                    "char_len_std": sub["char_len"].std(),
                    "req_breadth_std": sub["requirement_breadth"].std(),
                    "tech_count_std": sub["tech_count"].std(),
                }
            )
    homog = pd.DataFrame(homog_rows)
    homog.to_csv(OUT_TABLES / "homogeneity_test.csv", index=False)
    print("\nHomogeneity comparison:")
    print(homog.to_string())

    # ----- Sensitivity: raw description text re-run -----
    print("\n[T29] Sensitivity: raw-description re-run")
    # Use ALL rows with raw description text
    raw_sample = df[df["description"].notna() & (df["description"].str.len() > 100)].copy()
    # Subsample 10k for speed
    if len(raw_sample) > 15000:
        raw_sample = raw_sample.sample(15000, random_state=0).reset_index(drop=True)
    raw_feat = extract_features(raw_sample["text_raw"])
    raw_feat["period2"] = raw_sample["period2"].to_numpy()
    raw_score, _ = compute_authorship_score(raw_feat)
    raw_feat["authorship_score"] = raw_score.to_numpy()
    raw_stats = (
        raw_feat.groupby("period2")
        .agg(n=("char_len", "count"), score_mean=("authorship_score", "mean"),
             score_median=("authorship_score", "median"), score_std=("authorship_score", "std"))
        .reset_index()
    )
    raw_stats.to_csv(OUT_TABLES / "sensitivity_raw_text.csv", index=False)
    print(raw_stats.to_string())

    # Aggregator sensitivity
    print("\n[T29] Sensitivity: aggregator exclusion")
    for include_agg in [True, False]:
        sub = feat if include_agg else feat[~feat["is_aggregator"].fillna(False)]
        grp = (
            sub.groupby("period2")
            .agg(n=("uid", "count"), score_median=("authorship_score", "median"),
                 score_std=("authorship_score", "std"))
            .reset_index()
        )
        grp["include_aggregators"] = include_agg
        print(grp.to_string())

    # Save a light flag table for downstream
    flag = feat[["uid", "period2", "authorship_score", "tell_density", "emdash_density",
                 "sent_mean_words", "sent_std_words", "type_token_ratio", "bullet_density",
                 "para_mean_chars", "low_llm"]].copy()
    flag.to_parquet(OUT_TABLES / "authorship_flags.parquet", index=False)
    print("\n[T29] Done.")


if __name__ == "__main__":
    main()
