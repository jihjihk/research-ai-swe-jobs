"""Wave 1.5 Agent Prep — Step 6: within-2024 calibration table.

For ~30 metrics, compute:
  arshkon_value, asaniczka_value, within_2024_effect (Cohen's d / prop diff),
  scraped_value, cross_period_effect (arshkon vs scraped), calibration_ratio.

Scraped is collapsed across the 2026 windows (pool scraped rows).
Cross-period is arshkon vs scraped (per T05 convention — arshkon is the
cleaner 2024 baseline and matches the SNR table).

Output: exploration/artifacts/shared/calibration_table.csv
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

PARQUET = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
CLEANED = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
)
TECH = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet"
)
OUT = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/calibration_table.csv"
)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d for two samples. NaNs dropped."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def prop_diff(p1: float, p2: float) -> float:
    """Absolute proportion difference (|p1 - p2|)."""
    return float(abs(p1 - p2))


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # Load the cleaned text artifact (includes text_source, company_canonical,
    # seniority_final, yoe_extracted) and tech matrix keyed by uid.
    con.execute(f"CREATE VIEW ct AS SELECT * FROM '{CLEANED}'")
    con.execute(f"CREATE VIEW tm AS SELECT * FROM '{TECH}'")

    # We need additional columns (description_length, company_size, etc.) from unified.
    con.execute(
        f"""
        CREATE VIEW u AS
        SELECT uid, description_length, description, company_size, company_size_category
        FROM '{PARQUET}'
        """
    )

    # Full join view
    con.execute(
        """
        CREATE VIEW cal AS
        SELECT
            ct.uid,
            ct.source,
            ct.period,
            ct.seniority_final,
            ct.seniority_3level,
            ct.yoe_extracted,
            ct.is_aggregator,
            ct.swe_classification_tier,
            ct.text_source,
            ct.description_cleaned,
            u.description_length,
            u.description,
            tm.python, tm.java, tm.javascript, tm.typescript, tm.aws, tm.azure, tm.gcp,
            tm.kubernetes, tm.docker, tm.react, tm.nodejs, tm.sql_lang, tm.cicd,
            tm.postgres, tm.mongodb, tm.kafka, tm.spark, tm.snowflake, tm.databricks,
            tm.tensorflow, tm.pytorch, tm.scikit_learn, tm.pandas, tm.numpy,
            tm.llm, tm.langchain, tm.rag, tm.openai_api, tm.claude_api, tm.copilot,
            tm.cursor_tool, tm.chatgpt, tm.gpt, tm.agents_framework,
            tm.machine_learning, tm.deep_learning, tm.nlp, tm.computer_vision,
            tm.agile, tm.scrum, tm.tdd, tm.microservices
        FROM ct LEFT JOIN tm USING(uid) LEFT JOIN u USING(uid)
        """
    )

    # Build a reusable DataFrame (pandas for now, since ~64k rows is fine).
    df = con.execute("SELECT * FROM cal").fetch_df()
    print(f"Calibration dataframe: {len(df):,} rows")

    def split(df):
        ars = df[df["source"] == "kaggle_arshkon"]
        asa = df[df["source"] == "kaggle_asaniczka"]
        scr = df[df["source"] == "scraped"]
        return ars, asa, scr

    ars, asa, scr = split(df)
    print(f"  arshkon n={len(ars):,} | asaniczka n={len(asa):,} | scraped n={len(scr):,}")

    # Per-row keyword indicators that aren't in the tech matrix
    # (management, scope, soft skill, credential, education).
    # Build these by regex over description_cleaned and raw description.
    def build_indicator(column_series, patterns, regex_flags=0):
        rx = re.compile("|".join(patterns), regex_flags)
        return column_series.fillna("").map(lambda s: bool(rx.search(s)))

    df["mgmt_broad"] = build_indicator(
        df["description_cleaned"],
        [
            r"\blead\b", r"\bleading\b", r"\blead[s]?\b",
            r"\bmentor\b", r"\bmentoring\b",
            r"\bmanage\b", r"\bmanaging\b", r"\bmanager\b",
            r"\bteam\b", r"\bhire\b", r"\bhiring\b",
            r"\bcoach\b", r"\bcoaching\b",
        ],
    )
    df["mgmt_strict"] = build_indicator(
        df["description_cleaned"],
        [
            r"\bpeople.manage", r"\bmanage.people", r"\bdirect.report",
            r"\bperformance.review", r"\bhire", r"\bmentor.team",
            r"\blead.team", r"\blead.engineer", r"\bteam.lead",
            r"\bengineering.manager",
        ],
    )
    df["org_scope"] = build_indicator(
        df["description_cleaned"],
        [
            r"\bownership\b", r"\bend.to.end\b", r"\bcross.functional\b",
            r"\bstakeholder\b", r"\bstakeholders\b", r"\bimpact\b",
            r"\bscope\b", r"\bstrategy\b", r"\bstrategic\b",
            r"\bautonomy\b", r"\bindependently\b",
        ],
    )
    df["soft_skill"] = build_indicator(
        df["description_cleaned"],
        [
            r"\bcollaboration\b", r"\bcommunication\b", r"\bproblem.solving\b",
            r"\binterpersonal\b", r"\bteamwork\b", r"\badaptability\b",
        ],
    )
    df["credential"] = build_indicator(
        df["description_cleaned"],
        [
            r"\byears.experience\b", r"\b\d\+.years\b", r"\bphd\b", r"\bmaster",
            r"\bbachelor", r"\bcertification\b", r"\bdegree\b",
        ],
    )
    df["education_mention"] = build_indicator(
        df["description_cleaned"],
        [
            r"\bbachelor", r"\bmaster", r"\bphd\b", r"\bdegree\b",
            r"\buniversity\b", r"\bcollege\b",
        ],
    )
    df["ai_any"] = (
        df[["llm", "langchain", "rag", "openai_api", "claude_api", "copilot",
            "cursor_tool", "chatgpt", "gpt", "agents_framework",
            "machine_learning", "deep_learning"]].any(axis=1)
    )
    df["ai_tool_specific"] = (
        df[["copilot", "cursor_tool", "chatgpt", "claude_api", "openai_api",
            "gpt", "langchain", "rag", "agents_framework"]].any(axis=1)
    )
    df["ai_domain"] = (
        df[["machine_learning", "deep_learning", "nlp", "computer_vision",
            "tensorflow", "pytorch"]].any(axis=1)
    )
    # Per-row tech count
    tech_cols = [
        "python", "java", "javascript", "typescript", "aws", "azure", "gcp",
        "kubernetes", "docker", "react", "nodejs", "sql_lang", "cicd",
        "postgres", "mongodb", "kafka", "spark", "snowflake", "databricks",
        "tensorflow", "pytorch", "scikit_learn", "pandas", "numpy",
    ]
    df["tech_count"] = df[tech_cols].sum(axis=1)

    # Seniority entry shares
    df["is_entry_final"] = df["seniority_final"] == "entry"
    df["known_seniority"] = df["seniority_final"].isin(["entry", "mid_senior", "mid", "senior"]) | (
        df["seniority_final"] != "unknown"
    )
    df["known_seniority"] = df["seniority_final"] != "unknown"

    # description_core_llm labeled share
    df["is_llm_labeled"] = df["text_source"] == "llm"

    ars, asa, scr = split(df)

    def cont(metric: str, col: str) -> tuple[str, float, float, float, float, float, float]:
        a = ars[col].astype(float).to_numpy()
        b = asa[col].astype(float).to_numpy()
        c = scr[col].astype(float).to_numpy()
        v_ars = float(np.nanmean(a))
        v_asa = float(np.nanmean(b))
        v_scr = float(np.nanmean(c))
        w24 = abs(cohens_d(a, b))
        cross = abs(cohens_d(a, c))
        ratio = cross / w24 if w24 > 1e-9 else float("inf")
        return metric, v_ars, v_asa, w24, v_scr, cross, ratio

    def binary(metric: str, col: str, use_known_seniority: bool = False) -> tuple:
        if use_known_seniority:
            a_df = ars[ars["known_seniority"]]
            b_df = asa[asa["known_seniority"]]
            c_df = scr[scr["known_seniority"]]
        else:
            a_df, b_df, c_df = ars, asa, scr
        p_ars = float(a_df[col].mean()) if len(a_df) else float("nan")
        p_asa = float(b_df[col].mean()) if len(b_df) else float("nan")
        p_scr = float(c_df[col].mean()) if len(c_df) else float("nan")
        w24 = prop_diff(p_ars, p_asa)
        cross = prop_diff(p_ars, p_scr)
        ratio = cross / w24 if w24 > 1e-9 else float("inf")
        return metric, p_ars, p_asa, w24, p_scr, cross, ratio

    rows = []

    # Continuous
    rows.append(cont("description_length", "description_length"))
    rows.append(cont("yoe_extracted_mean", "yoe_extracted"))
    rows.append(cont("tech_count_mean", "tech_count"))

    # YOE median (special handling with medians)
    def cont_median(metric: str, col: str):
        a = ars[col].astype(float).dropna().to_numpy()
        b = asa[col].astype(float).dropna().to_numpy()
        c = scr[col].astype(float).dropna().to_numpy()
        v_ars = float(np.median(a)) if len(a) else float("nan")
        v_asa = float(np.median(b)) if len(b) else float("nan")
        v_scr = float(np.median(c)) if len(c) else float("nan")
        w24 = abs(cohens_d(a, b))
        cross = abs(cohens_d(a, c))
        ratio = cross / w24 if w24 > 1e-9 else float("inf")
        return metric, v_ars, v_asa, w24, v_scr, cross, ratio

    rows.append(cont_median("yoe_extracted_median", "yoe_extracted"))

    # Binary indicators (any mention)
    rows.append(binary("yoe_le2_share", "yoe_extracted"))  # placeholder
    # Manually compute yoe_le2: share of YOE <=2 out of YOE-known.
    def yoe_le2(dfsub):
        sub = dfsub[~dfsub["yoe_extracted"].isna()]
        if len(sub) == 0:
            return float("nan")
        return float((sub["yoe_extracted"] <= 2).mean())

    p_ars = yoe_le2(ars)
    p_asa = yoe_le2(asa)
    p_scr = yoe_le2(scr)
    w = prop_diff(p_ars, p_asa)
    x = prop_diff(p_ars, p_scr)
    rows[-1] = ("yoe_le2_share", p_ars, p_asa, w, p_scr, x, x / w if w > 1e-9 else float("inf"))

    # AI prevalences
    rows.append(binary("ai_keyword_prevalence_any", "ai_any"))
    rows.append(binary("ai_tool_specific_prevalence", "ai_tool_specific"))
    rows.append(binary("ai_domain_prevalence", "ai_domain"))

    # Management / scope / soft / credential / education
    rows.append(binary("management_indicator_rate_broad", "mgmt_broad"))
    rows.append(binary("management_indicator_rate_strict", "mgmt_strict"))
    rows.append(binary("org_scope_term_rate", "org_scope"))
    rows.append(binary("soft_skill_rate", "soft_skill"))
    rows.append(binary("credential_mention_rate", "credential"))
    rows.append(binary("education_mention_rate", "education_mention"))

    # Seniority
    # Entry share of known
    def entry_of_known(dfsub):
        known = dfsub[dfsub["known_seniority"]]
        if len(known) == 0:
            return float("nan")
        return float((known["seniority_final"] == "entry").mean())

    p_ars = entry_of_known(ars)
    p_asa = entry_of_known(asa)
    p_scr = entry_of_known(scr)
    w = prop_diff(p_ars, p_asa)
    x = prop_diff(p_ars, p_scr)
    rows.append(
        ("seniority_final_entry_share_of_known", p_ars, p_asa, w, p_scr, x,
         x / w if w > 1e-9 else float("inf"))
    )

    # Entry share of all (including unknown)
    p_ars = float((ars["seniority_final"] == "entry").mean())
    p_asa = float((asa["seniority_final"] == "entry").mean())
    p_scr = float((scr["seniority_final"] == "entry").mean())
    w = prop_diff(p_ars, p_asa)
    x = prop_diff(p_ars, p_scr)
    rows.append(
        ("seniority_final_entry_share_of_all", p_ars, p_asa, w, p_scr, x,
         x / w if w > 1e-9 else float("inf"))
    )

    # description_core_llm labeled share
    rows.append(binary("description_core_llm_labeled_share", "is_llm_labeled"))

    # Tech prevalence binaries (>=10 picks from taxonomy)
    for col in [
        "python", "java", "javascript", "typescript", "aws", "azure", "gcp",
        "kubernetes", "docker", "sql_lang", "react", "nodejs", "postgres",
        "kafka", "spark", "snowflake", "databricks", "llm", "rag", "copilot",
        "machine_learning", "deep_learning", "pytorch", "tensorflow",
        "agile", "scrum", "microservices", "cicd",
    ]:
        rows.append(binary(f"tech_{col}_prevalence", col))

    # Write CSV
    import csv

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "metric", "arshkon_value", "asaniczka_value", "within_2024_effect",
            "scraped_value", "cross_period_effect", "calibration_ratio",
        ])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.6f}" if isinstance(v, float) and np.isfinite(v) else (
                "inf" if isinstance(v, float) and np.isinf(v) else (
                    "nan" if isinstance(v, float) else v
                )
            ) for v in r[1:]])

    print(f"Wrote {OUT} with {len(rows)} metrics in {time.time() - t0:.1f}s")
    # Print top/bottom 5 by calibration_ratio
    ranked = [r for r in rows if np.isfinite(r[6])]
    ranked_sorted = sorted(ranked, key=lambda r: -r[6])
    print("\nTop 5 by calibration_ratio (strongest SNR):")
    for r in ranked_sorted[:5]:
        print(f"  {r[0]:45s} ratio={r[6]:.2f}")
    print("\nBottom 5 by calibration_ratio (noisiest):")
    for r in ranked_sorted[-5:]:
        print(f"  {r[0]:45s} ratio={r[6]:.2f}")


if __name__ == "__main__":
    main()
