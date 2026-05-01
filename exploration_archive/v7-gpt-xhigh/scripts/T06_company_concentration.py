#!/usr/bin/env python3
"""T06 company concentration diagnostics.

All full-file reads use DuckDB. The script materializes the SWE subset with
derived binary/count metrics after filtering, then writes compact CSV outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T06"
FIG_DIR = ROOT / "exploration" / "figures" / "T06"
SHARED_DIR = ROOT / "exploration" / "artifacts" / "shared"
SPECIALIST_PATH = SHARED_DIR / "entry_specialist_employers.csv"
SUMMARY_PATH = TABLE_DIR / "summary.json"

SOURCE_ORDER = ["kaggle_arshkon", "kaggle_asaniczka", "scraped_linkedin", "scraped_indeed"]

AI_REGEX = (
    r"\b(ai|a\.i\.|artificial intelligence|machine learning|ml|llm|large language model|"
    r"generative ai|genai|gpt|chatgpt|openai|anthropic|claude|copilot|cursor|rag|"
    r"agentic|ai agent|prompt engineering)\b"
)

TECH_PATTERNS = {
    "python": r"\bpython\b",
    "java": r"\bjava\b",
    "javascript": r"\bjavascript\b|\bjs\b",
    "typescript": r"\btypescript\b|\bts\b",
    "react": r"\breact\b|\breactjs\b",
    "angular": r"\bangular\b",
    "vue": r"\bvue\b|\bvuejs\b",
    "node": r"\bnode\.?js\b",
    "csharp": r"(^|[^a-z0-9])c#([^a-z0-9]|$)|\bc sharp\b",
    "cplusplus": r"(^|[^a-z0-9])c\+\+([^a-z0-9]|$)",
    "golang": r"\bgolang\b",
    "rust": r"\brust\b",
    "ruby": r"\bruby\b",
    "php": r"\bphp\b",
    "swift": r"\bswift\b",
    "kotlin": r"\bkotlin\b",
    "sql": r"\bsql\b",
    "postgres": r"\bpostgres\b|\bpostgresql\b",
    "mysql": r"\bmysql\b",
    "mongodb": r"\bmongodb\b|\bmongo\b",
    "aws": r"\baws\b|amazon web services",
    "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|google cloud",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "terraform": r"\bterraform\b",
    "cicd": r"\bci/cd\b|\bcontinuous integration\b|\bcontinuous delivery\b",
    "git": r"\bgit\b|\bgithub\b|\bgitlab\b",
    "linux": r"\blinux\b",
    "spark": r"\bspark\b|\bapache spark\b",
    "kafka": r"\bkafka\b",
    "redis": r"\bredis\b",
    "graphql": r"\bgraphql\b",
    "rest": r"\brestful\b|\brest api\b|\brest\b",
    "microservices": r"\bmicroservices\b|\bmicroservice\b",
    "spring": r"\bspring boot\b|\bspring\b",
    "django": r"\bdjango\b",
    "flask": r"\bflask\b",
    "dotnet": r"(^|[^a-z0-9])\.net([^a-z0-9]|$)|\bdotnet\b",
}


def regex_asserts() -> None:
    tests = [
        (TECH_PATTERNS["cplusplus"], "C++ services", True),
        (TECH_PATTERNS["cplusplus"], "C plus plus", False),
        (TECH_PATTERNS["dotnet"], ".NET developer", True),
        (TECH_PATTERNS["java"], "JavaScript engineer", False),
        (TECH_PATTERNS["java"], "Java engineer", True),
        (TECH_PATTERNS["cicd"], "CI/CD pipelines", True),
        (AI_REGEX, "Build with Claude and LLM systems", True),
        (AI_REGEX, "This role supports retail availability", False),
    ]
    for pattern, text, expected in tests:
        got = bool(re.search(pattern, text.lower(), flags=re.IGNORECASE))
        assert got is expected, (pattern, text, got, expected)


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)


def sql_regex(pattern: str) -> str:
    return pattern.replace("'", "''")


def source_key_expr() -> str:
    return "CASE WHEN source = 'scraped' THEN source || '_' || source_platform ELSE source END"


def tech_count_sql(text_expr: str) -> str:
    parts = [
        f"CASE WHEN regexp_matches({text_expr}, '{sql_regex(pattern)}') THEN 1 ELSE 0 END"
        for pattern in TECH_PATTERNS.values()
    ]
    return " + ".join(parts)


def load_swe_subset() -> pd.DataFrame:
    con = duckdb.connect()
    lower_desc = "lower(coalesce(description, ''))"
    query = f"""
        SELECT
            uid,
            {source_key_expr()} AS source_key,
            source,
            source_platform,
            company_name_canonical,
            company_industry,
            is_aggregator,
            description_hash,
            description_length,
            seniority_final,
            yoe_extracted,
            title_normalized,
            CASE WHEN regexp_matches({lower_desc}, '{sql_regex(AI_REGEX)}') THEN true ELSE false END AS ai_mention,
            ({tech_count_sql(lower_desc)})::INTEGER AS tech_count
        FROM read_parquet('{DATA.as_posix()}')
        WHERE is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
    """
    df = con.execute(query).fetchdf()
    df["source_key"] = pd.Categorical(df["source_key"], categories=SOURCE_ORDER, ordered=True)
    for col in ["is_aggregator", "ai_mention"]:
        df[col] = df[col].fillna(False).astype(bool)
    df["company_name_canonical"] = df["company_name_canonical"].fillna("__missing_company__")
    df["company_known"] = df["company_name_canonical"] != "__missing_company__"
    df["known_seniority"] = df["seniority_final"].notna() & (df["seniority_final"] != "unknown")
    df["j1_entry"] = df["seniority_final"] == "entry"
    df["j2_entry_associate"] = df["seniority_final"].isin(["entry", "associate"])
    df["yoe_known"] = df["yoe_extracted"].notna()
    df["j3_yoe_le2"] = df["yoe_known"] & (df["yoe_extracted"] <= 2)
    df["j4_yoe_le3"] = df["yoe_known"] & (df["yoe_extracted"] <= 3)
    return df


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or np.sum(x) == 0:
        return float("nan")
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * np.sum(x))) - ((n + 1) / n))


def concentration_for_group(df: pd.DataFrame, exclude_aggregators: bool) -> pd.DataFrame:
    rows = []
    work = df[df["company_known"]].copy()
    if exclude_aggregators:
        work = work[~work["is_aggregator"]].copy()
    for source_key, sub in work.groupby("source_key", observed=False):
        counts = sub.groupby("company_name_canonical").size().sort_values(ascending=False)
        total = int(counts.sum())
        shares = counts / total if total else counts.astype(float)
        row = {
            "source_key": source_key,
            "aggregator_excluded": exclude_aggregators,
            "postings_with_known_company": total,
            "companies": int(len(counts)),
            "hhi": float(np.sum(np.square(shares))) if total else np.nan,
            "gini": gini(counts.to_numpy()),
        }
        for k in [1, 5, 10, 20, 50]:
            row[f"top_{k}_share"] = float(counts.head(k).sum() / total) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_concentration(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(
        [concentration_for_group(df, False), concentration_for_group(df, True)],
        ignore_index=True,
    )
    out.to_csv(TABLE_DIR / "concentration_metrics.csv", index=False)

    plot = out[~out["aggregator_excluded"]].set_index("source_key")[
        ["top_1_share", "top_5_share", "top_20_share", "top_50_share"]
    ]
    plot.loc[SOURCE_ORDER].plot(kind="bar", figsize=(9, 5))
    plt.ylabel("Share of SWE postings with known company")
    plt.title("T06: company concentration by source-platform")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_company_shares.png", dpi=150)
    plt.close()
    return out


def mode_nonnull(series: pd.Series) -> str | None:
    s = series.dropna()
    if s.empty:
        return None
    return str(s.value_counts().index[0])


def top20_employer_profile(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.groupby("source_key", observed=False).size().rename("source_total")
    rows = []
    known = df[df["company_known"]].copy()
    for source_key, sub in known.groupby("source_key", observed=False):
        company_counts = sub.groupby("company_name_canonical").size().sort_values(ascending=False).head(20)
        for company, n in company_counts.items():
            c = sub[sub["company_name_canonical"] == company]
            known_sen = c[c["known_seniority"]]
            yoe_known = c[c["yoe_known"]]
            rows.append(
                {
                    "source_key": source_key,
                    "company_name_canonical": company,
                    "postings": int(n),
                    "share_of_source_swe": float(n / totals.loc[source_key]),
                    "is_aggregator_any": bool(c["is_aggregator"].any()),
                    "aggregator_posting_share": float(c["is_aggregator"].mean()),
                    "industry_mode": mode_nonnull(c["company_industry"]),
                    "mean_yoe": float(c["yoe_extracted"].mean()) if c["yoe_known"].any() else np.nan,
                    "yoe_known_n": int(c["yoe_known"].sum()),
                    "mean_description_length": float(c["description_length"].mean()),
                    "known_seniority_n": int(c["known_seniority"].sum()),
                    "entry_share_of_known_seniority": float((known_sen["seniority_final"] == "entry").mean())
                    if len(known_sen)
                    else np.nan,
                    "entry_share_of_all_postings": float(c["j1_entry"].mean()),
                    "yoe_le2_share_of_yoe_known": float(c["j3_yoe_le2"].sum() / c["yoe_known"].sum())
                    if c["yoe_known"].sum()
                    else np.nan,
                    "ai_mention_share": float(c["ai_mention"].mean()),
                    "mean_tech_count": float(c["tech_count"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "top20_employer_profile.csv", index=False)
    return out


def duplicate_template_audit(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["company_known"] & df["description_hash"].notna()].copy()
    company_hash = (
        work.groupby(["source_key", "company_name_canonical", "description_hash"], observed=False)
        .size()
        .rename("hash_n")
        .reset_index()
    )
    rows = []
    for (source_key, company), sub in company_hash.groupby(["source_key", "company_name_canonical"], observed=False):
        postings = int(sub["hash_n"].sum())
        if postings < 5:
            continue
        distinct = int(sub["description_hash"].nunique())
        max_hash_n = int(sub["hash_n"].max())
        rows.append(
            {
                "source_key": source_key,
                "company_name_canonical": company,
                "postings": postings,
                "distinct_description_hashes": distinct,
                "postings_per_distinct_hash": float(postings / distinct) if distinct else np.nan,
                "largest_hash_n": max_hash_n,
                "largest_hash_share": float(max_hash_n / postings) if postings else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["postings_per_distinct_hash", "postings"], ascending=[False, False]
    )
    out.groupby("source_key", observed=False).head(10).to_csv(
        TABLE_DIR / "duplicate_template_top10_by_source.csv", index=False
    )
    return out


def company_entry_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["company_known"]].copy()
    grouped = work.groupby(["source_key", "company_name_canonical"], observed=False)
    out = grouped.agg(
        postings=("uid", "size"),
        aggregator_postings=("is_aggregator", "sum"),
        known_seniority_n=("known_seniority", "sum"),
        j1_entry_n=("j1_entry", "sum"),
        j2_entry_associate_n=("j2_entry_associate", "sum"),
        yoe_known_n=("yoe_known", "sum"),
        j3_yoe_le2_n=("j3_yoe_le2", "sum"),
        j4_yoe_le3_n=("j4_yoe_le3", "sum"),
        mean_description_length=("description_length", "mean"),
        ai_mention_share=("ai_mention", "mean"),
        mean_tech_count=("tech_count", "mean"),
    ).reset_index()
    out["aggregator_posting_share"] = out["aggregator_postings"] / out["postings"]
    out["any_aggregator"] = out["aggregator_postings"] > 0
    out["j1_share_all"] = out["j1_entry_n"] / out["postings"]
    out["j2_share_all"] = out["j2_entry_associate_n"] / out["postings"]
    out["j1_share_known_seniority"] = out["j1_entry_n"] / out["known_seniority_n"].replace(0, np.nan)
    out["j2_share_known_seniority"] = out["j2_entry_associate_n"] / out["known_seniority_n"].replace(0, np.nan)
    out["j3_share_yoe_known"] = out["j3_yoe_le2_n"] / out["yoe_known_n"].replace(0, np.nan)
    out["j4_share_yoe_known"] = out["j4_yoe_le3_n"] / out["yoe_known_n"].replace(0, np.nan)
    return out


def entry_posting_concentration(company_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_key, sub in company_df.groupby("source_key", observed=False):
        ge5 = sub[sub["postings"] >= 5]
        entry_posters = sub[sub["j1_entry_n"] > 0]
        yoe_entry_posters = sub[sub["j3_yoe_le2_n"] > 0]
        rows.append(
            {
                "source_key": source_key,
                "companies_total": int(len(sub)),
                "companies_with_any_j1_entry": int((sub["j1_entry_n"] > 0).sum()),
                "companies_with_any_j1_entry_share": float((sub["j1_entry_n"] > 0).mean())
                if len(sub)
                else np.nan,
                "companies_with_any_yoe_le2": int((sub["j3_yoe_le2_n"] > 0).sum()),
                "companies_with_any_yoe_le2_share": float((sub["j3_yoe_le2_n"] > 0).mean())
                if len(sub)
                else np.nan,
                "companies_ge5": int(len(ge5)),
                "ge5_zero_j1_entry": int((ge5["j1_entry_n"] == 0).sum()),
                "ge5_zero_j1_entry_share": float((ge5["j1_entry_n"] == 0).mean())
                if len(ge5)
                else np.nan,
                "ge5_zero_yoe_le2": int((ge5["j3_yoe_le2_n"] == 0).sum()),
                "ge5_zero_yoe_le2_share": float((ge5["j3_yoe_le2_n"] == 0).mean())
                if len(ge5)
                else np.nan,
                "j1_entry_poster_share_median": float(entry_posters["j1_share_all"].median())
                if len(entry_posters)
                else np.nan,
                "j1_entry_poster_share_p90": float(entry_posters["j1_share_all"].quantile(0.9))
                if len(entry_posters)
                else np.nan,
                "yoe_le2_poster_share_median": float(yoe_entry_posters["j3_share_yoe_known"].median())
                if len(yoe_entry_posters)
                else np.nan,
                "yoe_le2_poster_share_p90": float(yoe_entry_posters["j3_share_yoe_known"].quantile(0.9))
                if len(yoe_entry_posters)
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "entry_posting_concentration.csv", index=False)

    plot_rows = []
    for source_key, sub in company_df.groupby("source_key", observed=False):
        entry_posters = sub[sub["j1_entry_n"] > 0]
        for value in entry_posters["j1_share_all"].dropna():
            plot_rows.append({"source_key": source_key, "j1_share_all": value})
    plot = pd.DataFrame(plot_rows)
    if not plot.empty:
        plt.figure(figsize=(9, 5))
        bins = np.linspace(0, 1, 21)
        for source_key in SOURCE_ORDER:
            vals = plot.loc[plot["source_key"] == source_key, "j1_share_all"]
            if len(vals):
                plt.hist(vals, bins=bins, alpha=0.35, label=source_key)
        plt.xlabel("Company J1 entry share of all own SWE postings, entry posters only")
        plt.ylabel("Company count")
        plt.title("T06: entry share distribution among entry-posting companies")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "entry_share_distribution_entry_posters.png", dpi=150)
        plt.close()
    return out


def metric_company_values(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    work = df[
        df["source_key"].isin(["kaggle_arshkon", "scraped_linkedin"]) & df["company_known"]
    ].copy()
    for (source_key, company), sub in work.groupby(["source_key", "company_name_canonical"], observed=False):
        if metric == "entry_j1":
            denom = len(sub)
            value = sub["j1_entry"].sum() / denom if denom else np.nan
        elif metric == "entry_j2":
            denom = len(sub)
            value = sub["j2_entry_associate"].sum() / denom if denom else np.nan
        elif metric == "entry_j3":
            denom = int(sub["yoe_known"].sum())
            value = sub["j3_yoe_le2"].sum() / denom if denom else np.nan
        elif metric == "entry_j4":
            denom = int(sub["yoe_known"].sum())
            value = sub["j4_yoe_le3"].sum() / denom if denom else np.nan
        elif metric == "ai_mention_prevalence":
            denom = len(sub)
            value = sub["ai_mention"].mean()
        elif metric == "description_length_mean":
            denom = int(sub["description_length"].notna().sum())
            value = sub["description_length"].mean() if denom else np.nan
        elif metric == "tech_count_mean":
            denom = len(sub)
            value = sub["tech_count"].mean()
        else:
            raise ValueError(metric)
        rows.append(
            {
                "source_key": source_key,
                "company_name_canonical": company,
                "denominator": int(denom),
                "value": float(value) if not pd.isna(value) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def midpoint_decomposition(values: pd.DataFrame, common_companies: set[str]) -> dict[str, float | int]:
    wide = values[values["company_name_canonical"].isin(common_companies)].pivot(
        index="company_name_canonical", columns="source_key", values=["denominator", "value"]
    )
    needed = [
        ("denominator", "kaggle_arshkon"),
        ("denominator", "scraped_linkedin"),
        ("value", "kaggle_arshkon"),
        ("value", "scraped_linkedin"),
    ]
    for col in needed:
        if col not in wide.columns:
            return {
                "companies_used": 0,
                "arshkon_total": np.nan,
                "scraped_total": np.nan,
                "total_change": np.nan,
                "within_company_component": np.nan,
                "between_reweighting_component": np.nan,
            }
    wide = wide.dropna(subset=needed)
    wide = wide[(wide[("denominator", "kaggle_arshkon")] > 0) & (wide[("denominator", "scraped_linkedin")] > 0)]
    if wide.empty:
        return {
            "companies_used": 0,
            "arshkon_total": np.nan,
            "scraped_total": np.nan,
            "total_change": np.nan,
            "within_company_component": np.nan,
            "between_reweighting_component": np.nan,
        }
    n0 = wide[("denominator", "kaggle_arshkon")].astype(float)
    n1 = wide[("denominator", "scraped_linkedin")].astype(float)
    w0 = n0 / n0.sum()
    w1 = n1 / n1.sum()
    y0 = wide[("value", "kaggle_arshkon")].astype(float)
    y1 = wide[("value", "scraped_linkedin")].astype(float)
    total0 = float((w0 * y0).sum())
    total1 = float((w1 * y1).sum())
    wbar = (w0 + w1) / 2
    ybar = (y0 + y1) / 2
    within = float((wbar * (y1 - y0)).sum())
    between = float(((w1 - w0) * ybar).sum())
    return {
        "companies_used": int(len(wide)),
        "arshkon_total": total0,
        "scraped_total": total1,
        "total_change": total1 - total0,
        "within_company_component": within,
        "between_reweighting_component": between,
        "residual": (total1 - total0) - within - between,
    }


def decomposition(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df[
            df["source_key"].isin(["kaggle_arshkon", "scraped_linkedin"])
            & df["company_known"]
        ]
        .groupby(["source_key", "company_name_canonical"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    wide_counts = counts.pivot_table(
        index="company_name_canonical", columns="source_key", values="n", fill_value=0
    )
    common = set(
        wide_counts[
            (wide_counts.get("kaggle_arshkon", 0) >= 5)
            & (wide_counts.get("scraped_linkedin", 0) >= 5)
        ].index
    )
    metrics = [
        "entry_j1",
        "entry_j2",
        "entry_j3",
        "entry_j4",
        "ai_mention_prevalence",
        "description_length_mean",
        "tech_count_mean",
    ]
    rows = []
    for metric in metrics:
        values = metric_company_values(df, metric)
        result = midpoint_decomposition(values, common)
        result["metric"] = metric
        result["common_company_pool_n"] = len(common)
        rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "within_between_decomposition_common_arshkon_scraped_linkedin.csv", index=False)

    plot = out[out["metric"].isin(["entry_j1", "entry_j2", "entry_j3", "entry_j4"])]
    if not plot.empty:
        x = np.arange(len(plot))
        plt.figure(figsize=(8, 5))
        plt.bar(x - 0.18, plot["within_company_component"], width=0.36, label="within")
        plt.bar(x + 0.18, plot["between_reweighting_component"], width=0.36, label="between/reweighting")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, plot["metric"], rotation=20)
        plt.ylabel("Contribution to share change")
        plt.title("T06: entry-share decomposition, common arshkon/scraped LinkedIn companies")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "entry_decomposition_j1_j4.png", dpi=150)
        plt.close()
    return out


MANUAL_CATEGORY_OVERRIDES: dict[str, str] = {
    # Filled with known intermediary/consulting/large-pipeline names. The script
    # also applies keyword rules for any lower-ranked flagged companies.
    "revature": "bulk-posting consulting",
    "synergisticit": "bulk-posting consulting",
    "skillstorm": "bulk-posting consulting",
    "mthree": "bulk-posting consulting",
    "genesis10": "staffing firm",
    "dice": "staffing firm",
    "lensa": "staffing firm",
    "robert half": "staffing firm",
    "motion recruitment": "staffing firm",
    "tek systems": "staffing firm",
    "teksystems": "staffing firm",
    "insight global": "staffing firm",
    "kforce": "staffing firm",
    "apex systems": "staffing firm",
    "handshake": "college-jobsite intermediary",
    "ripplematch": "college-jobsite intermediary",
    "wayup": "college-jobsite intermediary",
    "amazon": "tech-giant intern pipeline",
    "amazon web services": "tech-giant intern pipeline",
    "google": "tech-giant intern pipeline",
    "microsoft": "tech-giant intern pipeline",
    "meta": "tech-giant intern pipeline",
    "apple": "tech-giant intern pipeline",
    "oracle": "tech-giant intern pipeline",
    "ibm": "tech-giant intern pipeline",
    "nvidia": "tech-giant intern pipeline",
    "accenture": "bulk-posting consulting",
    "deloitte": "bulk-posting consulting",
    "cognizant": "bulk-posting consulting",
    "infosys": "bulk-posting consulting",
    "wipro": "bulk-posting consulting",
    "tata consultancy services": "bulk-posting consulting",
    "capgemini": "bulk-posting consulting",
}


def categorize_company(name: str) -> str:
    n = name.lower()
    for key, category in MANUAL_CATEGORY_OVERRIDES.items():
        if key in n:
            return category
    if any(k in n for k in ["college", "university", "handshake", "wayup", "ripplematch"]):
        return "college-jobsite intermediary"
    if any(
        k in n
        for k in [
            "staff",
            "staffing",
            "recruit",
            "talent",
            "randstad",
            "cybercoders",
            "jobot",
            "experis",
            "judge group",
            "collabera",
            "hire",
        ]
    ):
        return "staffing firm"
    if any(k in n for k in ["consulting", "consultants", "solutions", "systems integrator"]):
        return "bulk-posting consulting"
    return "direct employer"


def entry_specialists(company_df: pd.DataFrame) -> pd.DataFrame:
    ge5 = company_df[company_df["postings"] >= 5].copy()
    share_cols = ["j1_share_all", "j2_share_all", "j3_share_yoe_known", "j4_share_yoe_known"]
    ge5["max_junior_share"] = ge5[share_cols].max(axis=1, skipna=True)
    ge5["max_junior_variant"] = ge5[share_cols].idxmax(axis=1)
    flagged = ge5[ge5["max_junior_share"] > 0.60].copy()
    flagged = flagged.sort_values(["max_junior_share", "postings"], ascending=[False, False]).reset_index(drop=True)
    flagged["rank"] = np.arange(1, len(flagged) + 1)
    flagged["manual_category"] = flagged["company_name_canonical"].apply(categorize_company)
    flagged["top20_manual_reviewed"] = flagged["rank"] <= 20
    keep = [
        "rank",
        "source_key",
        "company_name_canonical",
        "postings",
        "known_seniority_n",
        "yoe_known_n",
        "j1_entry_n",
        "j2_entry_associate_n",
        "j3_yoe_le2_n",
        "j4_yoe_le3_n",
        "j1_share_all",
        "j2_share_all",
        "j1_share_known_seniority",
        "j2_share_known_seniority",
        "j3_share_yoe_known",
        "j4_share_yoe_known",
        "max_junior_share",
        "max_junior_variant",
        "any_aggregator",
        "aggregator_posting_share",
        "manual_category",
        "top20_manual_reviewed",
    ]
    out = flagged[keep].copy()
    out.to_csv(SPECIALIST_PATH, index=False)
    crosstab = pd.crosstab(out["manual_category"], out["any_aggregator"], margins=True)
    crosstab.to_csv(TABLE_DIR / "entry_specialist_category_by_aggregator.csv")
    out.head(50).to_csv(TABLE_DIR / "entry_specialist_employers_top50.csv", index=False)
    return out


def aggregator_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (source_key, is_agg), sub in df.groupby(["source_key", "is_aggregator"], observed=False):
        known_sen = sub[sub["known_seniority"]]
        yoe_known = sub[sub["yoe_known"]]
        rows.append(
            {
                "source_key": source_key,
                "is_aggregator": is_agg,
                "postings": int(len(sub)),
                "share_of_source_swe": float(len(sub) / len(df[df["source_key"] == source_key])),
                "mean_description_length": float(sub["description_length"].mean()),
                "mean_yoe": float(sub["yoe_extracted"].mean()) if len(yoe_known) else np.nan,
                "yoe_known_n": int(len(yoe_known)),
                "known_seniority_n": int(len(known_sen)),
                "entry_share_of_all_postings": float(sub["j1_entry"].mean()),
                "entry_share_of_known_seniority": float((known_sen["seniority_final"] == "entry").mean())
                if len(known_sen)
                else np.nan,
                "yoe_le2_share_of_yoe_known": float((yoe_known["yoe_extracted"] <= 2).mean())
                if len(yoe_known)
                else np.nan,
                "ai_mention_share": float(sub["ai_mention"].mean()),
                "mean_tech_count": float(sub["tech_count"].mean()),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(TABLE_DIR / "aggregator_profile.csv", index=False)

    seniority = (
        df.groupby(["source_key", "is_aggregator", "seniority_final"], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    seniority["denominator"] = seniority.groupby(["source_key", "is_aggregator"], observed=False)[
        "n"
    ].transform("sum")
    seniority["share"] = seniority["n"] / seniority["denominator"]
    seniority.to_csv(TABLE_DIR / "aggregator_seniority_distribution.csv", index=False)

    agg_share = (
        df.groupby("source_key", observed=False)["is_aggregator"].mean().reset_index(name="aggregator_posting_share")
    )
    agg_share.set_index("source_key").loc[SOURCE_ORDER].plot(kind="bar", legend=False, figsize=(8, 4))
    plt.ylabel("Share of SWE postings")
    plt.title("T06: aggregator share by source-platform")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "aggregator_share.png", dpi=150)
    plt.close()
    return profile, seniority


def new_entrant_profile(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist_companies = set(
        df[
            df["source_key"].isin(["kaggle_arshkon", "kaggle_asaniczka"])
            & df["company_known"]
        ]["company_name_canonical"]
    )
    scraped = df[df["source_key"].isin(["scraped_linkedin", "scraped_indeed"]) & df["company_known"]].copy()
    scraped["new_entrant_vs_2024"] = ~scraped["company_name_canonical"].isin(hist_companies)
    rows = []
    for (source_key, is_new), sub in scraped.groupby(["source_key", "new_entrant_vs_2024"], observed=False):
        known_sen = sub[sub["known_seniority"]]
        yoe_known = sub[sub["yoe_known"]]
        rows.append(
            {
                "source_key": source_key,
                "new_entrant_vs_2024": is_new,
                "companies": int(sub["company_name_canonical"].nunique()),
                "postings": int(len(sub)),
                "share_of_scraped_source_postings": float(len(sub) / len(scraped[scraped["source_key"] == source_key])),
                "mean_description_length": float(sub["description_length"].mean()),
                "mean_yoe": float(sub["yoe_extracted"].mean()) if len(yoe_known) else np.nan,
                "entry_share_of_all_postings": float(sub["j1_entry"].mean()),
                "entry_share_of_known_seniority": float((known_sen["seniority_final"] == "entry").mean())
                if len(known_sen)
                else np.nan,
                "yoe_le2_share_of_yoe_known": float((yoe_known["yoe_extracted"] <= 2).mean())
                if len(yoe_known)
                else np.nan,
                "ai_mention_share": float(sub["ai_mention"].mean()),
                "mean_tech_count": float(sub["tech_count"].mean()),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(TABLE_DIR / "new_entrant_profile.csv", index=False)

    top = (
        scraped[scraped["new_entrant_vs_2024"]]
        .groupby(["source_key", "company_name_canonical"], observed=False)
        .agg(
            postings=("uid", "size"),
            industry_mode=("company_industry", mode_nonnull),
            ai_mention_share=("ai_mention", "mean"),
            mean_description_length=("description_length", "mean"),
        )
        .reset_index()
        .sort_values(["source_key", "postings"], ascending=[True, False])
    )
    top.groupby("source_key", observed=False).head(20).to_csv(
        TABLE_DIR / "new_entrant_top20_by_scraped_source.csv", index=False
    )
    return profile, top


def prediction_table(
    concentration: pd.DataFrame,
    entry_conc: pd.DataFrame,
    dupes: pd.DataFrame,
    specialists: pd.DataFrame,
) -> pd.DataFrame:
    linked = concentration[
        (concentration["source_key"] == "scraped_linkedin") & (~concentration["aggregator_excluded"])
    ].iloc[0]
    entry_linked = entry_conc[entry_conc["source_key"] == "scraped_linkedin"].iloc[0]
    dupe_linked = dupes[dupes["source_key"] == "scraped_linkedin"]
    specialist_linked = specialists[specialists["source_key"] == "scraped_linkedin"]
    evidence = {
        "top20_share": linked["top_20_share"],
        "top50_share": linked["top_50_share"],
        "ge5_zero_j1_share": entry_linked["ge5_zero_j1_entry_share"],
        "specialists": len(specialist_linked),
        "max_duplicate_largest_hash_share": float(dupe_linked["largest_hash_share"].max())
        if len(dupe_linked)
        else np.nan,
    }
    rows = [
        {
            "analysis_category": "entry share",
            "concentration_risk": "high",
            "evidence": (
                f"scraped_linkedin top-20 company share={evidence['top20_share']:.3f}; "
                f">=5-posting companies with zero J1 rows={evidence['ge5_zero_j1_share']:.3f}; "
                f"entry-specialist flagged companies={evidence['specialists']}"
            ),
            "recommended_default": "report J1-J4; add company-weighted estimate; exclude entry_specialist_employers as a required sensitivity",
        },
        {
            "analysis_category": "AI mention rate",
            "concentration_risk": "medium",
            "evidence": (
                f"scraped_linkedin top-50 company share={evidence['top50_share']:.3f}; "
                "AI is binary raw-description prevalence, so repeated templates can move rates"
            ),
            "recommended_default": "use row-level rate plus company-clustered/company-weighted sensitivity; cap prolific firms for corpus summaries",
        },
        {
            "analysis_category": "description length",
            "concentration_risk": "medium",
            "evidence": (
                f"top-20 company share={evidence['top20_share']:.3f}; "
                "length differs by aggregator status and source composition"
            ),
            "recommended_default": "use row-level distribution tests, then report company-weighted median/mean and aggregator-excluded sensitivity",
        },
        {
            "analysis_category": "term frequencies",
            "concentration_risk": "high",
            "evidence": (
                f"top-50 company share={evidence['top50_share']:.3f}; "
                f"largest residual exact-template share among scraped_linkedin companies={evidence['max_duplicate_largest_hash_share']:.3f}"
            ),
            "recommended_default": "cap at 20-50 postings per company and deduplicate exact description_hash within company before term ranking",
        },
        {
            "analysis_category": "topic models",
            "concentration_risk": "high",
            "evidence": "topic models over postings will learn employer/template clusters when prolific firms are uncapped",
            "recommended_default": "deduplicate exact templates; cap per company; inspect topic employer concentration before interpretation",
        },
        {
            "analysis_category": "co-occurrence networks",
            "concentration_risk": "high",
            "evidence": "skill co-occurrence edges are sensitive to repeated employer stacks and template duplication",
            "recommended_default": "construct company-capped network and require terms/edges to appear across >=20 distinct companies",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "concentration_prediction_table.csv", index=False)
    return out


def write_sample_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("source_key", observed=False)
        .agg(
            n_swe=("uid", "size"),
            companies_known=("company_known", "sum"),
            distinct_companies=("company_name_canonical", lambda s: int(s[s != "__missing_company__"].nunique())),
            aggregators=("is_aggregator", "sum"),
            yoe_known=("yoe_known", "sum"),
            known_seniority=("known_seniority", "sum"),
            ai_mentions=("ai_mention", "sum"),
            mean_tech_count=("tech_count", "mean"),
        )
        .reset_index()
    )
    out.to_csv(TABLE_DIR / "analysis_sample_counts.csv", index=False)
    return out


def main() -> None:
    regex_asserts()
    ensure_dirs()
    panel_exists = (SHARED_DIR / "seniority_definition_panel.csv").exists()
    df = load_swe_subset()
    sample = write_sample_counts(df)
    concentration = write_concentration(df)
    top20 = top20_employer_profile(df)
    dupes = duplicate_template_audit(df)
    company_df = company_entry_frame(df)
    company_df.to_csv(TABLE_DIR / "company_entry_metrics.csv", index=False)
    entry_conc = entry_posting_concentration(company_df)
    decomp = decomposition(df)
    specialists = entry_specialists(company_df)
    agg_profile, agg_seniority = aggregator_profile(df)
    new_profile, new_top = new_entrant_profile(df)
    predictions = prediction_table(concentration, entry_conc, dupes, specialists)

    summary = {
        "t30_panel_exists": panel_exists,
        "note": "T30 seniority panel absent; J1-J4 were computed locally from task-reference definitions.",
        "sample_counts": sample.to_dict(orient="records"),
        "concentration_scraped_linkedin": concentration[
            (concentration["source_key"] == "scraped_linkedin")
            & (~concentration["aggregator_excluded"])
        ].to_dict(orient="records"),
        "entry_concentration": entry_conc.to_dict(orient="records"),
        "decomposition": decomp.to_dict(orient="records"),
        "entry_specialist_count": int(len(specialists)),
        "entry_specialist_top20": specialists.head(20).to_dict(orient="records"),
        "aggregator_profile": agg_profile.to_dict(orient="records"),
        "new_entrant_profile": new_profile.to_dict(orient="records"),
        "prediction_table": predictions.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote T06 tables to {TABLE_DIR}")
    print(f"Wrote T06 figures to {FIG_DIR}")
    print(f"Wrote shared entry-specialist artifact to {SPECIALIST_PATH}")


if __name__ == "__main__":
    main()
