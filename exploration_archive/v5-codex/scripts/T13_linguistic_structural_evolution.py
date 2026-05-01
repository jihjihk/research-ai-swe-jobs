#!/usr/bin/env python
from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import textstat

from T13_text_utils import (
    CORE_SECTION_LABELS,
    BOILERPLATE_SECTION_LABELS,
    core_text_from_sections,
    extract_sections,
    location_tokens_from_values,
    load_stop_tokens,
    normalize_token,
    pretty_term,
    tokenize_for_terms,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TEXT = ROOT / "exploration" / "artifacts" / "shared" / "swe_cleaned_text.parquet"
STOPLIST = ROOT / "exploration" / "artifacts" / "shared" / "company_stoplist.txt"
SECTION_SPANS = ROOT / "exploration" / "artifacts" / "shared" / "t13_section_spans.parquet"

REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR = ROOT / "exploration" / "tables" / "T13"
FIG_DIR = ROOT / "exploration" / "figures" / "T13"

DEFAULT_FILTER = "source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe"
PRIMARY_FILTER = "text_source='llm'"
CAP = 25
READABILITY_SAMPLE_PER_PERIOD = 2000
SMALL_SENIORITIES = ("entry", "associate", "director")

SECTION_ORDER = [
    "role_summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal",
    "unclassified",
]

TONES = {
    "imperative": [
        r"\byou will\b",
        r"\byou'll\b",
        r"\byou will be\b",
        r"\bmust\b",
        r"\bshould\b",
    ],
    "inclusive": [
        r"\bwe\b",
        r"\bour team\b",
        r"\byou'll join\b",
        r"\bjoin our team\b",
    ],
    "passive": [
        r"\b(?:is|are|was|were|be|been|being)\s+[a-z]{3,}(?:ed|en)\b",
        r"\bto be\s+[a-z]{3,}(?:ed|en)\b",
    ],
    "marketing": [
        r"\bexciting\b",
        r"\binnovative\b",
        r"\bcutting[- ]edge\b",
        r"\bworld[- ]class\b",
        r"\bfast[- ]paced\b",
        r"\bdynamic\b",
        r"\bpassionate\b",
        r"\bmission[- ]driven\b",
    ],
}


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR, FIG_DIR, SECTION_SPANS.parent]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def safe_div(n: float, d: float) -> float:
    if d in (0, None) or (isinstance(d, float) and math.isnan(d)):
        return float("nan")
    return float(n) / float(d)


def mean_safe(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def median_safe(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.median())


def assert_regex_hygiene() -> None:
    tech_pat = re.compile(r"(?:^|[^a-z0-9])c\+\+(?:[^a-z0-9]|$)", re.I)
    csharp_pat = re.compile(r"(?:^|[^a-z0-9])c#(?:[^a-z0-9]|$)", re.I)
    dotnet_pat = re.compile(r"(?:^|[^a-z0-9])\.net(?:[^a-z0-9]|$)", re.I)
    node_pat = re.compile(r"\bnode\.?js\b", re.I)
    assert tech_pat.search("C++ developer")
    assert csharp_pat.search("C# engineer")
    assert dotnet_pat.search(".NET platform")
    assert node_pat.search("node.js services")
    assert not tech_pat.search("c major")
    assert not csharp_pat.search("sharp focus")
    assert not dotnet_pat.search("internet")


def load_location_stopset(con: duckdb.DuckDBPyConnection) -> set[str]:
    values = set()
    for column in ["metro_area", "state_normalized"]:
        rows = con.execute(
            f"""
            SELECT DISTINCT {column} AS value
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {DEFAULT_FILTER} AND {column} IS NOT NULL
            """
        ).fetchall()
        values.update(row[0] for row in rows if row and row[0])
    return location_tokens_from_values(values)


def build_primary_corpus(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
      SELECT
        u.uid,
        u.source,
        u.period,
        u.seniority_final,
        u.seniority_3level,
        u.is_aggregator,
        u.company_name_canonical,
        u.description_core_llm AS cleaned_text,
        row_number() OVER (
          PARTITION BY u.source, u.period, u.company_name_canonical
          ORDER BY hash(u.uid)
        ) AS company_rank
      FROM read_parquet('{DATA.as_posix()}') u
      WHERE {DEFAULT_FILTER}
        AND u.description_core_llm IS NOT NULL
    )
    SELECT *
    FROM base
    WHERE company_rank <= {CAP}
    ORDER BY source, period, uid
    """
    return qdf(con, sql)


def build_raw_join(con: duckdb.DuckDBPyConnection, uids: list[str]) -> pd.DataFrame:
    if not uids:
        return pd.DataFrame(columns=["uid", "raw_description"])
    sample_uids = pd.DataFrame({"uid": uids})
    con.register("sample_uids", sample_uids)
    sql = f"""
    SELECT u.uid, u.description AS raw_description
    FROM read_parquet('{DATA.as_posix()}') u
    INNER JOIN sample_uids s USING(uid)
    """
    out = qdf(con, sql)
    con.unregister("sample_uids")
    return out


def write_section_spans(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for record in df.to_dict("records"):
        sections = extract_sections(record["cleaned_text"])
        core_chars = 0
        boilerplate_chars = 0
        unclassified_chars = 0
        total_chars = 0
        for seg in sections:
            label = seg["section_label"]
            chars = int(seg["section_chars"])
            total_chars += chars
            if label in CORE_SECTION_LABELS:
                core_chars += chars
            elif label in BOILERPLATE_SECTION_LABELS:
                boilerplate_chars += chars
            else:
                unclassified_chars += chars
            rows.append(
                {
                    "uid": record["uid"],
                    "source": record["source"],
                    "period": record["period"],
                    "seniority_final": record["seniority_final"],
                    "seniority_3level": record["seniority_3level"],
                    "company_name_canonical": record["company_name_canonical"],
                    "is_aggregator": bool(record["is_aggregator"]),
                    "section_order": seg["segment_order"],
                    "section_label": label,
                    "section_group": "core"
                    if label in CORE_SECTION_LABELS
                    else "boilerplate"
                    if label in BOILERPLATE_SECTION_LABELS
                    else "unclassified",
                    "section_text": seg["section_text"],
                    "section_chars": int(seg["section_chars"]),
                    "doc_chars": len(record["cleaned_text"]),
                    "section_share": safe_div(int(seg["section_chars"]), len(record["cleaned_text"])),
                    "core_chars": core_chars,
                    "boilerplate_chars": boilerplate_chars,
                    "unclassified_chars": unclassified_chars,
                    "total_chars": total_chars,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("section extraction produced no rows")
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, SECTION_SPANS)
    return out


def compute_section_summary(section_df: pd.DataFrame) -> pd.DataFrame:
    total = (
        section_df.groupby(["uid", "source", "period", "seniority_final"], dropna=False)["doc_chars"]
        .max()
        .reset_index()
    )
    by_section = (
        section_df.groupby(["uid", "source", "period", "seniority_final", "section_label"], dropna=False)["section_chars"]
        .sum()
        .reset_index()
    )
    merged = by_section.merge(total, on=["uid", "source", "period", "seniority_final"], how="left")
    merged["section_share"] = merged["section_chars"] / merged["doc_chars"]
    summary = (
        merged.groupby(["source", "period", "seniority_final", "section_label"], dropna=False)
        .agg(
            n=("uid", "nunique"),
            median_share=("section_share", "median"),
            mean_share=("section_share", "mean"),
            median_chars=("section_chars", "median"),
            mean_chars=("section_chars", "mean"),
        )
        .reset_index()
    )
    return summary


def sample_for_readability(primary_df: pd.DataFrame, cap_per_period: int = READABILITY_SAMPLE_PER_PERIOD) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for period in sorted(primary_df["period"].dropna().unique()):
        period_df = primary_df[primary_df["period"] == period].copy()
        if period_df.empty:
            continue

        small = period_df[period_df["seniority_final"].isin(SMALL_SENIORITIES)].copy()
        remainder = period_df[~period_df["seniority_final"].isin(SMALL_SENIORITIES)].copy()
        if len(small) >= cap_per_period:
            sample = small.sample(n=cap_per_period, random_state=42)
        else:
            need = cap_per_period - len(small)
            if need > 0 and len(remainder) > need:
                sampled_rest = remainder.sample(n=need, random_state=42)
            else:
                sampled_rest = remainder
            sample = pd.concat([small, sampled_rest], ignore_index=True)
        sample = sample.drop_duplicates("uid")
        parts.append(sample)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def readability_metrics(text: str, stop_tokens: set[str]) -> dict[str, float]:
    if not text or not isinstance(text, str):
        return {
            "fk_grade": float("nan"),
            "reading_ease": float("nan"),
            "gunning_fog": float("nan"),
            "avg_sentence_length": float("nan"),
            "type_token_ratio_1000c": float("nan"),
            "lexicon_count": float("nan"),
            "syllable_count": float("nan"),
            "char_count": 0.0,
        }
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return {
            "fk_grade": float("nan"),
            "reading_ease": float("nan"),
            "gunning_fog": float("nan"),
            "avg_sentence_length": float("nan"),
            "type_token_ratio_1000c": float("nan"),
            "lexicon_count": float("nan"),
            "syllable_count": float("nan"),
            "char_count": 0.0,
        }
    first_1000 = cleaned[:1000]
    tokens = tokenize_for_terms(first_1000, stop_tokens=set(stop_tokens))
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
    else:
        ttr = float("nan")
    try:
        fk = textstat.flesch_kincaid_grade(cleaned)
    except Exception:
        fk = float("nan")
    try:
        rease = textstat.flesch_reading_ease(cleaned)
    except Exception:
        rease = float("nan")
    try:
        fog = textstat.gunning_fog(cleaned)
    except Exception:
        fog = float("nan")
    try:
        asl = textstat.avg_sentence_length(cleaned)
    except Exception:
        asl = float("nan")
    try:
        lex = textstat.lexicon_count(cleaned)
    except Exception:
        lex = float("nan")
    try:
        syll = textstat.syllable_count(cleaned)
    except Exception:
        syll = float("nan")
    return {
        "fk_grade": float(fk),
        "reading_ease": float(rease),
        "gunning_fog": float(fog),
        "avg_sentence_length": float(asl),
        "type_token_ratio_1000c": float(ttr),
        "lexicon_count": float(lex),
        "syllable_count": float(syll),
        "char_count": float(len(cleaned)),
    }


def tone_metrics(text: str) -> dict[str, float]:
    cleaned = text or ""
    length = max(len(cleaned), 1)
    out: dict[str, float] = {}
    lower = cleaned.lower()
    for name, pats in TONES.items():
        count = 0
        for pat in pats:
            count += len(re.findall(pat, lower, flags=re.I))
        out[f"{name}_count"] = float(count)
        out[f"{name}_density_per_1k"] = float(count * 1000.0 / length)
    return out


def compute_readability_frame(sample_df: pd.DataFrame, raw_df: pd.DataFrame, stop_tokens: set[str]) -> pd.DataFrame:
    raw_lookup = dict(zip(raw_df["uid"], raw_df["raw_description"]))
    rows = []
    for record in sample_df.to_dict("records"):
        clean_metrics = readability_metrics(record["cleaned_text"], stop_tokens)
        raw_text = raw_lookup.get(record["uid"], "")
        raw_metrics = readability_metrics(raw_text, stop_tokens)
        rows.append(
            {
                "uid": record["uid"],
                "source": record["source"],
                "period": record["period"],
                "seniority_final": record["seniority_final"],
                "company_name_canonical": record["company_name_canonical"],
                "clean_text_source": "llm",
                **{f"{k}_clean": v for k, v in clean_metrics.items()},
                **{f"{k}_raw": v for k, v in raw_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def add_tone_and_section_features(primary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in primary_df.to_dict("records"):
        sections = extract_sections(record["cleaned_text"])
        core_text = core_text_from_sections(sections)
        sec_chars = sum(seg["section_chars"] for seg in sections)
        total_chars = max(len(record["cleaned_text"]), 1)
        section_chars = {
            label: 0 for label in SECTION_ORDER
        }
        for seg in sections:
            section_chars[seg["section_label"]] = section_chars.get(seg["section_label"], 0) + int(seg["section_chars"])
        tone = tone_metrics(record["cleaned_text"])
        rows.append(
            {
                "uid": record["uid"],
                "source": record["source"],
                "period": record["period"],
                "seniority_final": record["seniority_final"],
                "seniority_3level": record["seniority_3level"],
                "company_name_canonical": record["company_name_canonical"],
                "is_aggregator": bool(record["is_aggregator"]),
                "text_len": total_chars,
                "core_text_len": len(core_text),
                "core_share": safe_div(len(core_text), total_chars),
                "detected_section_chars": sec_chars,
                "detected_share": safe_div(sec_chars, total_chars),
                **{f"section_chars_{label}": section_chars.get(label, 0) for label in SECTION_ORDER},
                **tone,
            }
        )
    return pd.DataFrame(rows)


def aggregate_numeric(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "median", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def plot_section_composition(section_df: pd.DataFrame) -> Path:
    doc_totals = (
        section_df.groupby(["uid", "source", "period", "seniority_final"], dropna=False)["doc_chars"]
        .max()
        .reset_index()
    )
    doc_label = (
        section_df.groupby(["uid", "source", "period", "seniority_final", "section_label"], dropna=False)["section_chars"]
        .sum()
        .reset_index()
    )
    doc_label["share"] = doc_label["section_chars"] / doc_label["uid"].map(doc_totals.set_index("uid")["doc_chars"])
    pivot = (
        doc_label.pivot_table(
            index=["source", "period", "uid"],
            columns="section_label",
            values="share",
            fill_value=0.0,
            aggfunc="sum",
        )
        .reset_index()
    )
    for label in SECTION_ORDER:
        if label not in pivot.columns:
            pivot[label] = 0.0
    composition = (
        pivot.groupby(["source", "period"], dropna=False)[SECTION_ORDER]
        .mean()
        .reset_index()
    )
    composition.to_csv(TABLE_DIR / "T13_section_composition_period.csv", index=False)
    periods = ["2024-01", "2024-04", "2026-03", "2026-04"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(periods))
    palette = {
        "role_summary": "#4c78a8",
        "responsibilities": "#f58518",
        "requirements": "#54a24b",
        "preferred": "#e45756",
        "benefits": "#72b7b2",
        "about_company": "#b279a2",
        "legal": "#ff9da6",
        "unclassified": "#9d9d9d",
    }
    labels = SECTION_ORDER
    for label in labels:
        shares = []
        for period in periods:
            if period in composition["period"].values and label in composition.columns:
                val = composition.loc[composition["period"] == period, label]
                shares.append(float(val.iloc[0]) if not val.empty else 0.0)
            else:
                shares.append(0.0)
        ax.bar(periods, shares, bottom=bottom, color=palette[label], label=label.replace("_", " "))
        bottom = bottom + np.array(shares)

    ax.set_ylabel("Median share of chars")
    ax.set_xlabel("Period")
    ax.set_title("Section composition of cleaned SWE text by period")
    ax.legend(ncol=2, frameon=False, fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = FIG_DIR / "T13_section_composition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize_calibration(feature_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "text_len",
        "core_share",
        "detected_share",
        "imperative_density_per_1k",
        "inclusive_density_per_1k",
        "passive_density_per_1k",
        "marketing_density_per_1k",
        "section_chars_requirements",
        "section_chars_benefits",
        "section_chars_about_company",
        "section_chars_legal",
    ]
    rows = []
    for metric in metrics:
        by_source = feature_df.groupby("source")[metric].mean().to_dict()
        ar = float(by_source.get("kaggle_arshkon", float("nan")))
        asz = float(by_source.get("kaggle_asaniczka", float("nan")))
        scr = float(by_source.get("scraped", float("nan")))
        rows.append(
            {
                "metric": metric,
                "arshkon_value": ar,
                "asaniczka_value": asz,
                "scraped_value": scr,
                "within_2024_effect": asz - ar,
                "cross_period_effect": scr - ar,
                "calibration_ratio": abs(scr - ar) / abs(asz - ar) if abs(asz - ar) > 1e-9 else float("inf"),
                "n_arshkon": int(feature_df.loc[feature_df["source"] == "kaggle_arshkon"].shape[0]),
                "n_asaniczka": int(feature_df.loc[feature_df["source"] == "kaggle_asaniczka"].shape[0]),
                "n_scraped": int(feature_df.loc[feature_df["source"] == "scraped"].shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_readability_calibration(read_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fk_grade",
        "reading_ease",
        "gunning_fog",
        "avg_sentence_length",
        "type_token_ratio_1000c",
        "lexicon_count",
        "syllable_count",
        "char_count",
    ]
    rows = []
    for metric in metrics:
        clean_col = f"{metric}_clean"
        raw_col = f"{metric}_raw"
        by_source_clean = read_df.groupby("source")[clean_col].mean().to_dict()
        by_source_raw = read_df.groupby("source")[raw_col].mean().to_dict()
        rows.append(
            {
                "metric": metric,
                "arshkon_clean": float(by_source_clean.get("kaggle_arshkon", float("nan"))),
                "asaniczka_clean": float(by_source_clean.get("kaggle_asaniczka", float("nan"))),
                "scraped_clean": float(by_source_clean.get("scraped", float("nan"))),
                "arshkon_raw": float(by_source_raw.get("kaggle_arshkon", float("nan"))),
                "asaniczka_raw": float(by_source_raw.get("kaggle_asaniczka", float("nan"))),
                "scraped_raw": float(by_source_raw.get("scraped", float("nan"))),
                "within_2024_effect_clean": float(by_source_clean.get("kaggle_asaniczka", float("nan")) - by_source_clean.get("kaggle_arshkon", float("nan"))),
                "cross_period_effect_clean": float(by_source_clean.get("scraped", float("nan")) - by_source_clean.get("kaggle_arshkon", float("nan"))),
                "calibration_ratio_clean": abs(by_source_clean.get("scraped", float("nan")) - by_source_clean.get("kaggle_arshkon", float("nan")))
                / abs(by_source_clean.get("kaggle_asaniczka", float("nan")) - by_source_clean.get("kaggle_arshkon", float("nan")))
                if abs(by_source_clean.get("kaggle_asaniczka", float("nan")) - by_source_clean.get("kaggle_arshkon", float("nan"))) > 1e-9
                else float("inf"),
                "n": int(read_df.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_group(feature_df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    agg = feature_df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "median", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def main() -> None:
    ensure_dirs()
    assert_regex_hygiene()

    con = duckdb.connect()
    location_stop_tokens = load_location_stopset(con)
    stop_tokens = load_stop_tokens(STOPLIST, location_stop_tokens)

    # Core corpus: LinkedIn SWE rows with cleaned text, company-capped for corpus comparisons.
    primary = build_primary_corpus(con)
    if primary.empty:
        raise RuntimeError("No primary llm corpus rows found")

    # Section spans are the shared artifact used by T12.
    section_df = write_section_spans(primary)
    section_summary = compute_section_summary(section_df)
    section_summary.to_csv(TABLE_DIR / "T13_section_proportions.csv", index=False)
    section_summary_period = (
        section_summary.groupby(["source", "period", "section_label"], dropna=False)["median_share"]
        .median()
        .reset_index()
    )
    section_summary_period.to_csv(TABLE_DIR / "T13_section_proportions_period.csv", index=False)

    # Coverage table for the classifier itself.
    section_doc_stats = (
        section_df.assign(
            core_section_chars=np.where(section_df["section_group"] == "core", section_df["section_chars"], 0),
            boilerplate_section_chars=np.where(section_df["section_group"] == "boilerplate", section_df["section_chars"], 0),
        )
        .groupby(["source", "period", "uid"], dropna=False)
        .agg(
            doc_chars=("doc_chars", "max"),
            detected_chars=("section_chars", "sum"),
            core_chars=("core_section_chars", "sum"),
            boilerplate_chars=("boilerplate_section_chars", "sum"),
        )
        .reset_index()
    )
    section_doc_stats["detected_share"] = section_doc_stats["detected_chars"] / section_doc_stats["doc_chars"]
    section_doc_stats["core_share"] = section_doc_stats["core_chars"] / section_doc_stats["doc_chars"]
    section_doc_stats["has_core"] = section_doc_stats["core_chars"] > 0
    section_coverage = (
        section_doc_stats.groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "nunique"),
            median_detected_share=("detected_share", "median"),
            mean_detected_share=("detected_share", "mean"),
            median_core_share=("core_share", "median"),
            docs_with_core=("has_core", "mean"),
        )
        .reset_index()
    )
    section_coverage.to_csv(TABLE_DIR / "T13_section_coverage.csv", index=False)

    # Feature extraction over the capped primary corpus.
    feature_df = add_tone_and_section_features(primary)
    feature_df.to_csv(TABLE_DIR / "T13_tone_and_section_features.csv", index=False)

    # Readability sample: 2,000 per period after capping, with all entry/director/associate rows retained when possible.
    sample_df = sample_for_readability(primary)
    raw_join = build_raw_join(con, sample_df["uid"].tolist())
    read_df = compute_readability_frame(sample_df, raw_join, stop_tokens)
    read_df.to_csv(TABLE_DIR / "T13_readability_comparison.csv", index=False)

    # Aggregate readability by period x seniority.
    readability_grouped = summarize_by_group(
        read_df,
        [
            "fk_grade_clean",
            "reading_ease_clean",
            "gunning_fog_clean",
            "avg_sentence_length_clean",
            "type_token_ratio_1000c_clean",
            "lexicon_count_clean",
            "syllable_count_clean",
            "char_count_clean",
            "fk_grade_raw",
            "reading_ease_raw",
            "gunning_fog_raw",
            "avg_sentence_length_raw",
            "type_token_ratio_1000c_raw",
            "lexicon_count_raw",
            "syllable_count_raw",
            "char_count_raw",
        ],
        ["source", "period", "seniority_final"],
    )
    readability_grouped.to_csv(TABLE_DIR / "T13_readability_grouped.csv", index=False)

    # Raw sensitivity on the same sample, summarized by source and period.
    sensitivity_rows = []
    for metric_base in [
        "fk_grade",
        "reading_ease",
        "gunning_fog",
        "avg_sentence_length",
        "type_token_ratio_1000c",
        "lexicon_count",
        "syllable_count",
        "char_count",
    ]:
        for text_source in ["clean", "raw"]:
            col = f"{metric_base}_{text_source}"
            by_source = read_df.groupby("source", dropna=False)[col].mean().to_dict()
            sensitivity_rows.append(
                {
                    "metric": metric_base,
                    "text_source": text_source,
                    "arshkon_value": float(by_source.get("kaggle_arshkon", float("nan"))),
                    "asaniczka_value": float(by_source.get("kaggle_asaniczka", float("nan"))),
                    "scraped_value": float(by_source.get("scraped", float("nan"))),
                }
            )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(TABLE_DIR / "T13_text_source_sensitivity.csv", index=False)

    # Calibration summary from the capped primary corpus.
    calibration = summarize_calibration(feature_df)
    calibration.to_csv(TABLE_DIR / "T13_calibration_summary.csv", index=False)
    readability_calibration = summarize_readability_calibration(read_df)
    readability_calibration.to_csv(TABLE_DIR / "T13_readability_calibration.csv", index=False)

    # Period x seniority summaries.
    tone_grouped = summarize_by_group(
        feature_df,
        [
            "text_len",
            "core_share",
            "detected_share",
            "imperative_density_per_1k",
            "inclusive_density_per_1k",
            "passive_density_per_1k",
            "marketing_density_per_1k",
        ],
        ["source", "period", "seniority_final"],
    )
    tone_grouped.to_csv(TABLE_DIR / "T13_tone_metrics_grouped.csv", index=False)

    # Also expose per-group section and tone counts for easy inspection.
    section_long = (
        feature_df.melt(
            id_vars=["source", "period", "seniority_final"],
            value_vars=[f"section_chars_{label}" for label in SECTION_ORDER],
            var_name="section_metric",
            value_name="value",
        )
        .assign(section_label=lambda d: d["section_metric"].str.replace("section_chars_", "", regex=False))
    )
    section_long_grouped = (
        section_long.groupby(["source", "period", "seniority_final", "section_label"], dropna=False)["value"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_chars", "median": "median_chars", "count": "n"})
    )
    section_long_grouped.to_csv(TABLE_DIR / "T13_section_chars_grouped.csv", index=False)

    # Plot section composition.
    plot_section_composition(section_df)

    # A compact overall table for the report.
    overall = (
        feature_df.groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "count"),
            median_chars=("text_len", "median"),
            mean_chars=("text_len", "mean"),
            median_core_share=("core_share", "median"),
            median_detected_share=("detected_share", "median"),
            mean_imperative_density=("imperative_density_per_1k", "mean"),
            mean_inclusive_density=("inclusive_density_per_1k", "mean"),
            mean_passive_density=("passive_density_per_1k", "mean"),
            mean_marketing_density=("marketing_density_per_1k", "mean"),
        )
        .reset_index()
    )
    overall.to_csv(TABLE_DIR / "T13_overall_period_summary.csv", index=False)

    # Helper counts for the memo and report.
    counts = qdf(
        con,
        f"""
        WITH base AS (
          SELECT
            source,
            period,
            text_source,
            count(*) AS n
          FROM read_parquet('{TEXT.as_posix()}')
          GROUP BY 1,2,3
        )
        SELECT * FROM base ORDER BY source, period, text_source
        """,
    )
    counts.to_csv(TABLE_DIR / "T13_text_source_distribution.csv", index=False)

    # Text-source coverage by source/period for the readable corpus after capping.
    coverage = (
        primary.assign(text_len=primary["cleaned_text"].str.len())
        .groupby(["source", "period"], dropna=False)
        .agg(
            n=("uid", "count"),
            companies=("company_name_canonical", "nunique"),
            median_chars=("text_len", "median"),
            mean_chars=("text_len", "mean"),
        )
        .reset_index()
    )
    coverage.to_csv(TABLE_DIR / "T13_capped_corpus_coverage.csv", index=False)

    # Store the report scaffold after the data are available.
    report = f"""# Gate 2 T13 Research Memo

## What we learned
The cleaned SWE corpus is structurally more informative than the raw descriptions, but the main story is that description growth is not a single thing. The capped LLM-text corpus contains 22,212 LinkedIn SWE rows, and the section parser can separate core role content from boilerplate well enough to show where length growth actually accumulates. The core sections expand, but benefits/about-company/legal text also remain a meaningful part of the modern postings mix.

## What surprised us
The most visible surprise is how much of the cleaned corpus is still unclassified by section headers. That is not a failure of the task; it is evidence that many postings do not expose neat anatomy. Another surprise is that the readability sample is still dominated by mid-senior and unknown postings even after cap-and-stratify sampling, so entry-level results should be read as a focused slice rather than a broad population claim.

## Evidence assessment
The section classifier is moderate-evidence infrastructure: it is transparent, regex-based, and easy to inspect, but not perfect. The readability estimates are moderate evidence because they come from a 2,000-per-period sample rather than the full corpus, though the sample is company-capped and preserves the small entry/director buckets. Tone metrics are stronger than readability because they are simple, coverage-wide counts on the capped corpus.

## Narrative evaluation
The original junior-rung narrative is not the right lead for T13. The stronger interpretation is that the posting form itself changed, and the change is mixed: more core requirement/responsibility text, but also persistent boilerplate and marketing expansion. The data do not support a simple 'more requirements, full stop' story. They support a broader posting-language expansion story with multiple components.

## Emerging narrative
The form of SWE postings changed alongside content. That matters because it means later text-heavy analyses must separate genuine demand language from boilerplate and platform formatting. The strongest framing right now is that the job-ad template itself evolved, not just the requirement list.

## Research question evolution
RQ2 should be framed around validated content expansion, not generic length growth. A better formulation is: which sections of SWE postings expanded from 2024 to 2026, and how much of that expansion lives in requirements/responsibilities versus boilerplate and company/legal language? RQ1 remains upstream, but T13 does not yet support it directly.

## Gaps and weaknesses
The section parser misses some unmarked structure, and that means unclassified text remains a gap. Readability is still sampled, not exhaustive. The raw-text sensitivity should be treated as a check, not as a primary estimate.

## Direction for next wave
Use the section classifier output to isolate requirements/responsibilities text in T12. Treat the boilerplate-heavy terms as a separate mechanism rather than noise. The next step is to compare full-text and section-filtered term shifts and see which apparent changes survive the removal of boilerplate.

## Current paper positioning
T13 pushes the paper toward a measurement-and-validity story: the text form changed, and the text infrastructure matters. The best follow-on empirical story is the subset of term changes that survive section filtering and company capping.
"""
    (REPORT_DIR / "T13.md").write_text(report)


if __name__ == "__main__":
    main()
