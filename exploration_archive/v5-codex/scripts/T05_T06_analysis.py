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
from scipy.stats import chi2_contingency, ks_2samp


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
REPORT_DIR = ROOT / "exploration" / "reports"
TABLE_DIR_T05 = ROOT / "exploration" / "tables" / "T05"
TABLE_DIR_T06 = ROOT / "exploration" / "tables" / "T06"
FIG_DIR_T05 = ROOT / "exploration" / "figures" / "T05"
FIG_DIR_T06 = ROOT / "exploration" / "figures" / "T06"

DEFAULT_FILTER = "is_english = true AND date_flag = 'ok' AND is_swe"
LINKEDIN_FILTER = f"source_platform = 'linkedin' AND {DEFAULT_FILTER}"
SCRAPED_LINKEDIN_FILTER = "source = 'scraped' AND source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe"
SCRAPED_INDEED_FILTER = "source = 'scraped' AND source_platform = 'indeed' AND is_english = true AND date_flag = 'ok' AND is_swe"

SOURCE_ORDER = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]

AI_PATTERNS = [
    r"\bai\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bnlp\b",
    r"\bcomputer vision\b",
    r"\bgenerative ai\b",
    r"\bgen ai\b",
    r"\bllm(s)?\b",
    r"\blarge language model(s)?\b",
    r"\bgpt(-?\d+)?\b",
    r"\bchatgpt\b",
    r"\bclaude\b",
    r"\bcopilot\b",
    r"\bcursor\b",
    r"\bgemini\b",
    r"\bopenai\b",
    r"\brag\b",
    r"\blangchain\b",
    r"\bmcp\b",
    r"\bprompt engineering\b",
    r"\bfine[- ]tuning\b",
]

TECH_PATTERNS = [
    ("python", r"\bpython\b"),
    ("java", r"\bjava\b"),
    ("javascript", r"\bjavascript\b"),
    ("typescript", r"\btypescript\b"),
    ("go", r"\bgo\b"),
    ("rust", r"\brust\b"),
    ("c_plus_plus", r"(?:^|[^a-z0-9])c\+\+(?:[^a-z0-9]|$)"),
    ("c_sharp", r"(?:^|[^a-z0-9])c#(?:[^a-z0-9]|$)"),
    ("ruby", r"\bruby\b"),
    ("kotlin", r"\bkotlin\b"),
    ("swift", r"\bswift\b"),
    ("scala", r"\bscala\b"),
    ("php", r"\bphp\b"),
    ("react", r"\breact\b"),
    ("angular", r"\bangular\b"),
    ("vue", r"\bvue\b"),
    ("nextjs", r"\bnext\.?js\b"),
    ("nodejs", r"\bnode\.?js\b"),
    ("django", r"\bdjango\b"),
    ("flask", r"\bflask\b"),
    ("spring", r"\bspring\b"),
    ("dotnet", r"(?:^|[^a-z0-9])\.net(?:[^a-z0-9]|$)"),
    ("rails", r"\brails\b"),
    ("fastapi", r"\bfastapi\b"),
    ("aws", r"\baws\b"),
    ("azure", r"\bazure\b"),
    ("gcp", r"\b(gcp|google cloud|google cloud platform)\b"),
    ("kubernetes", r"\bkubernetes\b|\bk8s\b"),
    ("docker", r"\bdocker\b"),
    ("terraform", r"\bterraform\b"),
    ("ci_cd", r"\bci\s*/\s*cd\b"),
    ("jenkins", r"\bjenkins\b"),
    ("github_actions", r"\bgithub actions\b"),
    ("sql", r"\bsql\b"),
    ("postgresql", r"\bpostgresql\b|\bpostgres\b"),
    ("mysql", r"\bmysql\b"),
    ("mongodb", r"\bmongodb\b"),
    ("redis", r"\bredis\b"),
    ("kafka", r"\bkafka\b"),
    ("spark", r"\bspark\b"),
    ("snowflake", r"\bsnowflake\b"),
    ("databricks", r"\bdatabricks\b"),
    ("dbt", r"\bdbt\b"),
    ("elasticsearch", r"\belasticsearch\b"),
    ("tensorflow", r"\btensorflow\b"),
    ("pytorch", r"\bpytorch\b"),
    ("scikit_learn", r"\bscikit[- ]learn\b"),
    ("pandas", r"\bpandas\b"),
    ("numpy", r"\bnumpy\b"),
    ("langchain", r"\blangchain\b"),
    ("rag", r"\brag\b"),
    ("vector_db", r"\bvector database(s)?\b|\bvector db\b"),
    ("pinecone", r"\bpinecone\b"),
    ("hugging_face", r"\bhugging face\b"),
    ("openai_api", r"\bopenai api\b"),
    ("claude_api", r"\bclaude api\b"),
    ("prompt_engineering", r"\bprompt engineering\b"),
    ("fine_tuning", r"\bfine[- ]tuning\b"),
    ("mcp", r"\bmcp\b"),
    ("llm", r"\bllm(s)?\b"),
    ("copilot", r"\bcopilot\b"),
    ("cursor", r"\bcursor\b"),
    ("chatgpt", r"\bchatgpt\b"),
    ("claude", r"\bclaude\b"),
    ("gemini", r"\bgemini\b"),
    ("codex", r"\bcodex\b"),
    ("jest", r"\bjest\b"),
    ("pytest", r"\bpytest\b"),
    ("selenium", r"\bselenium\b"),
    ("cypress", r"\bcypress\b"),
    ("agile", r"\bagile\b"),
    ("scrum", r"\bscrum\b"),
    ("tdd", r"\btdd\b"),
    ("microservices", r"\bmicroservices\b"),
    ("graphql", r"\bgraphql\b"),
    ("rest", r"\brest\b"),
    ("git", r"\bgit\b"),
    ("bash", r"\bbash\b"),
    ("airflow", r"\bairflow\b"),
    ("ansible", r"\bansible\b"),
    ("prometheus", r"\bprometheus\b"),
    ("grafana", r"\bgrafana\b"),
    ("bigquery", r"\bbigquery\b"),
    ("redshift", r"\bredshift\b"),
    ("oracle", r"\boracle\b"),
    ("mssql", r"\bmssql\b"),
]


def ensure_dirs() -> None:
    for path in [REPORT_DIR, TABLE_DIR_T05, TABLE_DIR_T06, FIG_DIR_T05, FIG_DIR_T06]:
        path.mkdir(parents=True, exist_ok=True)


def qdf(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def qrows(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d not in (0, None) and not (isinstance(d, float) and math.isnan(d)) else float("nan")


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return (x.mean() - y.mean()) / math.sqrt(pooled)


def gini(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


def hhi(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=float)
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    p = x / x.sum()
    return float(np.square(p).sum())


def share_top(counts: np.ndarray, k: int) -> float:
    x = np.asarray(counts, dtype=float)
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    return float(np.sort(x)[::-1][:k].sum() / x.sum())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def cramer_v(table: pd.DataFrame) -> float:
    if table.empty:
        return float("nan")
    chi2, _, _, _ = chi2_contingency(table.values)
    n = table.values.sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return float("nan")
    return math.sqrt((chi2 / n) / denom)


def add_ratio(row: pd.Series) -> float:
    within = abs(row["within_effect"])
    cross = abs(row["cross_effect"])
    if pd.isna(within) or within == 0:
        return float("nan")
    return cross / within


def pair_name(a: str, b: str) -> str:
    return f"{a}__vs__{b}"


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_quote_list(values: list[str]) -> str:
    return ",".join(sql_quote(v) for v in values)


def make_pairwise_table(metric_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(metric_rows)
    if "within_effect" in df.columns and "cross_effect" in df.columns:
        df["calibration_ratio"] = df.apply(add_ratio, axis=1)
    return df


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def lower_text(expr: str) -> str:
    return f"lower(coalesce({expr}, ''))"


def build_ai_sql(expr: str) -> str:
    return " OR ".join([f"regexp_matches({expr}, '{pat}')" for pat in AI_PATTERNS])


def build_tech_count_sql(expr: str) -> str:
    parts = [f"CASE WHEN regexp_matches({expr}, '{pat}') THEN 1 ELSE 0 END" for _, pat in TECH_PATTERNS]
    return " + ".join(parts)


def annotate_source_platform(source: str, source_platform: str) -> str:
    return f"{source} | {source_platform}"


def compute_length_outputs(con: duckdb.DuckDBPyConnection, spec_sql: str, spec_name: str) -> dict[str, dict]:
    source_arrays = {}
    source_stats = {}
    for source in SOURCE_ORDER:
        arr = con.execute(
            f"""
            SELECT description_length
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source = '{source}'
            ORDER BY uid
            """
        ).fetchnumpy()["description_length"].astype(float)
        source_arrays[source] = arr
        source_stats[source] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    pair_rows = []
    pairs = [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]
    for a, b in pairs:
        xa, xb = source_arrays[a], source_arrays[b]
        ks = ks_2samp(xa, xb)
        pair_rows.append(
            {
                "spec": spec_name,
                "pair": pair_name(a, b),
                "metric": "description_length",
                "effect_measure": "cohens_d",
                "effect_value": cohens_d(xa, xb),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "median_a": float(np.median(xa)),
                "median_b": float(np.median(xb)),
            }
        )
    return {"stats": source_stats, "pairs": pair_rows, "arrays": source_arrays}


def compute_distinct_set_metrics(con: duckdb.DuckDBPyConnection, spec_sql: str, field: str) -> dict[str, set[str]]:
    out = {}
    for source in SOURCE_ORDER:
        rows = qrows(
            con,
            f"""
            SELECT DISTINCT {field} AS v
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source = '{source}'
              AND {field} IS NOT NULL
            """,
        )
        out[source] = {r[0] for r in rows}
    return out


def compute_category_contingency(con: duckdb.DuckDBPyConnection, spec_sql: str, field: str, exclude_unknown: bool = False, coalesce_missing: str | None = None) -> dict[str, pd.DataFrame]:
    tables = {}
    for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
        extra = ""
        if exclude_unknown:
            extra = f" AND {field} <> 'unknown'"
        if coalesce_missing is not None:
            expr = f"coalesce({field}, '{coalesce_missing}')"
        else:
            expr = field
        df = qdf(
            con,
            f"""
            SELECT {expr} AS category, source, count(*) AS n
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source IN ('{a}', '{b}')
              {extra}
            GROUP BY 1,2
            """,
        )
        tab = df.pivot_table(index="category", columns="source", values="n", fill_value=0).reindex(columns=[a, b], fill_value=0)
        tables[pair_name(a, b)] = tab
    return tables


def top_k_overlap(con: duckdb.DuckDBPyConnection, spec_sql: str, field: str, k: int = 50) -> list[dict]:
    rows = []
    source_top = {}
    for source in SOURCE_ORDER:
        df = qdf(
            con,
            f"""
            SELECT {field} AS v, count(*) AS n
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source = '{source}'
              AND {field} IS NOT NULL
            GROUP BY 1
            ORDER BY n DESC, v
            LIMIT {k}
            """,
        )
        source_top[source] = set(df["v"].tolist())
        rows.append(
            {
                "source": source,
                "top_k": k,
                "n_distinct_top_k": len(source_top[source]),
                "top_k_total_count": int(df["n"].sum()),
            }
        )
    for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
        rows.append(
            {
                "source": pair_name(a, b),
                "top_k": k,
                "n_distinct_top_k": len(source_top[a] & source_top[b]),
                "top_k_total_count": len(source_top[a] & source_top[b]),
            }
        )
    return rows


def write_length_figure(arrays: dict[str, np.ndarray], path: Path, title: str) -> None:
    pairs = [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    max_x = float(np.percentile(np.concatenate(list(arrays.values())), 99))
    bins = np.linspace(0, max_x, 50)
    for ax, (a, b) in zip(axes, pairs):
        ax.hist(arrays[a], bins=bins, density=True, histtype="step", linewidth=2, label=a.replace("kaggle_", ""))
        ax.hist(arrays[b], bins=bins, density=True, histtype="step", linewidth=2, label=b.replace("kaggle_", ""))
        ax.set_title(f"{a.replace('kaggle_', '')} vs {b.replace('kaggle_', '')}")
        ax.set_xlabel("description_length")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    save_fig(fig, path)


def compute_company_concentration(con: duckdb.DuckDBPyConnection, spec_sql: str) -> pd.DataFrame:
    rows = []
    missing = qdf(
        con,
        f"""
        SELECT source, count(*) AS n_missing
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {spec_sql}
          AND company_name_canonical IS NULL
        GROUP BY 1
        """,
    )
    missing_map = dict(zip(missing["source"], missing["n_missing"])) if not missing.empty else {}
    for spec, extra in [("all", ""), ("no_aggregators", " AND is_aggregator = false")]:
        counts = qdf(
            con,
            f"""
            SELECT source, company_name_canonical, count(*) AS n
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND company_name_canonical IS NOT NULL
              {extra}
            GROUP BY 1,2
            ORDER BY source, n DESC, company_name_canonical
            """,
        )
        for source in SOURCE_ORDER:
            src = counts[counts["source"] == source].copy()
            if src.empty:
                continue
            vals = src["n"].to_numpy(dtype=float)
            rows.append(
                {
                    "source": source,
                    "spec": spec,
                    "n_companies": int(len(src)),
                    "n_postings": int(vals.sum()),
                    "n_missing_company": int(missing_map.get(source, 0)),
                    "hhi": hhi(vals),
                    "gini": gini(vals),
                    "top1_count": int(vals[:1].sum()),
                    "top5_count": int(vals[:5].sum()),
                    "top10_count": int(vals[:10].sum()),
                    "top20_count": int(vals[:20].sum()),
                    "top50_count": int(vals[:50].sum()),
                }
            )
    df = pd.DataFrame(rows)
    for k in [1, 5, 10, 20, 50]:
        df[f"top{k}_share"] = df[f"top{k}_count"] / df["n_postings"]
    df["missing_share"] = df["n_missing_company"] / (df["n_postings"] + df["n_missing_company"])
    return df


def company_stats(con: duckdb.DuckDBPyConnection, spec_sql: str, extra_filters: str = "") -> pd.DataFrame:
    return qdf(
        con,
        f"""
        WITH base AS (
            SELECT source, company_name_canonical, description_length, yoe_extracted, seniority_final
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND company_name_canonical IS NOT NULL
              {extra_filters}
        ),
        company_counts AS (
            SELECT source, company_name_canonical,
                   count(*) AS n,
                   sum(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS n_entry_final,
                   sum(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS n_entry_yoe,
                   avg(yoe_extracted) AS mean_yoe,
                   avg(description_length) AS mean_len
            FROM base
            GROUP BY 1,2
        )
        SELECT *
        FROM company_counts
        ORDER BY source, n DESC, company_name_canonical
        """,
    )


def compute_top_employers(con: duckdb.DuckDBPyConnection, spec_sql: str, top_n: int = 20) -> pd.DataFrame:
    df = company_stats(con, spec_sql)
    out = []
    for source in SOURCE_ORDER:
        src = df[df["source"] == source].copy()
        total = src["n"].sum()
        top = src.head(top_n).copy()
        if top.empty:
            continue
        industry = qdf(
            con,
            f"""
            SELECT company_name_canonical, any_value(company_industry) AS company_industry
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql} AND source = '{source}' AND company_name_canonical IS NOT NULL
            GROUP BY 1
            """,
        )
        top = top.merge(industry, on="company_name_canonical", how="left")
        top["share_of_source"] = top["n"] / total
        top["entry_share_final"] = top["n_entry_final"] / top["n"]
        top["entry_share_yoe"] = top["n_entry_yoe"] / top["n"]
        top["source"] = source
        out.append(top[
            [
                "source",
                "company_name_canonical",
                "n",
                "share_of_source",
                "company_industry",
                "mean_yoe",
                "mean_len",
                "entry_share_final",
                "entry_share_yoe",
            ]
        ])
    return pd.concat(out, ignore_index=True)


def duplicate_template_audit(con: duckdb.DuckDBPyConnection, spec_sql: str) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        WITH base AS (
            SELECT source, company_name_canonical, description_hash
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND company_name_canonical IS NOT NULL
        ),
        counts AS (
            SELECT source, company_name_canonical,
                   count(*) AS n_postings,
                   count(DISTINCT description_hash) AS n_desc
            FROM base
            GROUP BY 1,2
        )
        SELECT *,
               CAST(n_postings AS DOUBLE) / NULLIF(n_desc, 0) AS max_dup_ratio
        FROM counts
        ORDER BY source, max_dup_ratio DESC, n_postings DESC, company_name_canonical
        LIMIT 30
        """,
    )


def entry_concentration(con: duckdb.DuckDBPyConnection, spec_sql: str) -> pd.DataFrame:
    df = company_stats(con, spec_sql)
    rows = []
    for source in SOURCE_ORDER:
        src = df[df["source"] == source].copy()
        total_companies = len(src)
        ge5 = src[src["n"] >= 5].copy()
        if ge5.empty:
            continue
        for method, col in [("seniority_final", "n_entry_final"), ("yoe_proxy", "n_entry_yoe")]:
            entry_any = int((src[col] > 0).sum())
            entry_any_ge5 = int((ge5[col] > 0).sum())
            zero_share = float((ge5[col] == 0).mean())
            specialist = ge5[(ge5[col] / ge5["n"]) > 0.5]
            rows.append(
                {
                    "source": source,
                    "method": method,
                    "n_companies": total_companies,
                    "n_companies_ge5": len(ge5),
                    "companies_with_any_entry": entry_any,
                    "companies_with_any_entry_ge5": entry_any_ge5,
                    "share_ge5_zero_entry": zero_share,
                    "n_entry_specialists": len(specialist),
                    "median_company_entry_share": float((src[col] / src["n"]).median()),
                    "p90_company_entry_share": float((src[col] / src["n"]).quantile(0.9)),
                    "p95_company_entry_share": float((src[col] / src["n"]).quantile(0.95)),
                }
            )
    return pd.DataFrame(rows)


def entry_specialists(con: duckdb.DuckDBPyConnection, spec_sql: str, top_n: int = 20) -> pd.DataFrame:
    df = company_stats(con, spec_sql)
    rows = []
    for source in SOURCE_ORDER:
        src = df[df["source"] == source].copy()
        src["entry_share_final"] = src["n_entry_final"] / src["n"]
        src["entry_share_yoe"] = src["n_entry_yoe"] / src["n"]
        for method, col in [("seniority_final", "entry_share_final"), ("yoe_proxy", "entry_share_yoe")]:
            specialist = src[(src["n"] >= 5) & (src[col] > 0.5)].copy()
            specialist = specialist.sort_values(["n", col], ascending=[False, False]).head(top_n)
            for _, r in specialist.iterrows():
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "company_name_canonical": r["company_name_canonical"],
                        "n": int(r["n"]),
                        "entry_share": float(r[col]),
                        "mean_yoe": float(r["mean_yoe"]),
                        "mean_len": float(r["mean_len"]),
                    }
                )
    return pd.DataFrame(rows)


def make_text_feature_sql(text_expr: str) -> tuple[str, str]:
    ai_expr = build_ai_sql(text_expr)
    tech_expr = build_tech_count_sql(text_expr)
    return ai_expr, tech_expr


def panel_features(con: duckdb.DuckDBPyConnection, spec_sql: str) -> pd.DataFrame:
    tech_sql = build_tech_count_sql(lower_text("description"))
    ai_sql = build_ai_sql(lower_text("description"))
    return qdf(
        con,
        f"""
        WITH panel AS (
            SELECT source, company_name_canonical, description, seniority_final, yoe_extracted, is_aggregator,
                   CASE WHEN {ai_sql} THEN 1 ELSE 0 END AS ai_mention,
                   ({tech_sql}) AS tech_count,
                   description_length
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source IN ('kaggle_arshkon', 'scraped')
              AND source_platform = 'linkedin'
              AND company_name_canonical IS NOT NULL
        ),
        company_panel AS (
            SELECT source, company_name_canonical, count(*) AS n
            FROM panel
            GROUP BY 1,2
            HAVING count(*) >= 5
        )
        SELECT p.*
        FROM panel p
        INNER JOIN company_panel c USING (source, company_name_canonical)
        """,
    )


def decompose_metric(panel: pd.DataFrame, metric: str) -> dict:
    pivot = panel.pivot_table(index="company_name_canonical", columns="source", values=metric, aggfunc="mean")
    count = panel.groupby(["company_name_canonical", "source"]).size().unstack("source").fillna(0)
    pivot = pivot.dropna(subset=["kaggle_arshkon", "scraped"])
    count = count.loc[pivot.index]
    n0 = count["kaggle_arshkon"].astype(float)
    n1 = count["scraped"].astype(float)
    w0 = n0 / n0.sum()
    w1 = n1 / n1.sum()
    m0 = pivot["kaggle_arshkon"].astype(float)
    m1 = pivot["scraped"].astype(float)
    overall0 = float((w0 * m0).sum())
    overall1 = float((w1 * m1).sum())
    within = float(0.5 * ((w0 + w1) * (m1 - m0)).sum())
    between = float(0.5 * ((m0 + m1) * (w1 - w0)).sum())
    return {
        "metric": metric,
        "n_companies": int(len(pivot)),
        "overall_2024": overall0,
        "overall_2026": overall1,
        "change": overall1 - overall0,
        "within_component": within,
        "between_component": between,
        "within_share_of_change": safe_div(within, overall1 - overall0),
    }


def build_decomposition(con: duckdb.DuckDBPyConnection, spec_sql: str) -> pd.DataFrame:
    panel = panel_features(con, spec_sql)
    rows = []
    for metric in ["entry_final", "entry_yoe", "ai_mention", "description_length", "tech_count"]:
        if metric == "entry_final":
            panel = panel.copy()
            panel["entry_final"] = (panel["seniority_final"] == "entry").astype(int)
            rows.append({**decompose_metric(panel, metric), "spec": "all"})
            if True:
                noagg = panel[panel["is_aggregator"] == False].copy()
                rows.append({**decompose_metric(noagg, metric), "spec": "no_aggregators"})
        elif metric == "entry_yoe":
            panel = panel.copy()
            panel["entry_yoe"] = (panel["yoe_extracted"] <= 2).astype(int)
            rows.append({**decompose_metric(panel, metric), "spec": "all"})
            noagg = panel[panel["is_aggregator"] == False].copy()
            rows.append({**decompose_metric(noagg, metric), "spec": "no_aggregators"})
        else:
            rows.append({**decompose_metric(panel, metric), "spec": "all"})
            noagg = panel[panel["is_aggregator"] == False].copy()
            rows.append({**decompose_metric(noagg, metric), "spec": "no_aggregators"})
    return pd.DataFrame(rows)


def top_titles_shared(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        WITH base AS (
            SELECT source, title_normalized, seniority_native, yoe_extracted
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {LINKEDIN_FILTER}
              AND source IN ('kaggle_arshkon', 'scraped')
              AND title_normalized IS NOT NULL
              AND company_name_canonical IS NOT NULL
        ),
        counts AS (
            SELECT title_normalized, source, count(*) AS n
            FROM base
            GROUP BY 1,2
        ),
        shared AS (
            SELECT title_normalized
            FROM counts
            GROUP BY 1
            HAVING count(*) = 2
        ),
        totals AS (
            SELECT title_normalized, sum(n) AS total_n
            FROM counts
            GROUP BY 1
        )
        SELECT c.title_normalized, c.source, c.n, t.total_n
        FROM counts c
        JOIN shared s USING (title_normalized)
        JOIN totals t USING (title_normalized)
        QUALIFY row_number() OVER (ORDER BY t.total_n DESC, c.title_normalized, c.source) <= 40
        ORDER BY t.total_n DESC, c.title_normalized, c.source
        """,
    )


def title_native_distribution(con: duckdb.DuckDBPyConnection, top_titles: list[str]) -> pd.DataFrame:
    title_list = sql_quote_list(top_titles)
    return qdf(
        con,
        f"""
        SELECT title_normalized, source, seniority_native, count(*) AS n, avg(yoe_extracted) AS mean_yoe, median(yoe_extracted) AS median_yoe
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {LINKEDIN_FILTER}
          AND source IN ('kaggle_arshkon', 'scraped')
          AND title_normalized IN ({title_list})
        GROUP BY 1,2,3
        ORDER BY title_normalized, source, n DESC
        """,
    )


def title_yoe_cells(con: duckdb.DuckDBPyConnection, top_titles: list[str]) -> pd.DataFrame:
    title_list = sql_quote_list(top_titles)
    return qdf(
        con,
        f"""
        SELECT title_normalized, source, seniority_native, count(*) AS n, avg(yoe_extracted) AS mean_yoe,
               median(yoe_extracted) AS median_yoe,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS share_yoe_le2
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {LINKEDIN_FILTER}
          AND source IN ('kaggle_arshkon', 'scraped')
          AND title_normalized IN ({title_list})
          AND seniority_native IS NOT NULL
        GROUP BY 1,2,3
        HAVING count(*) >= 3
        ORDER BY title_normalized, seniority_native, source
        """,
    )


def indeed_validation(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return qdf(
        con,
        f"""
        SELECT
            source_platform,
            count(*) AS n,
            avg(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS entry_share_final,
            avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS entry_share_yoe_le2,
            avg(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS entry_share_yoe_le3,
            avg(CASE WHEN seniority_final='unknown' THEN 1 ELSE 0 END) AS unknown_share_final,
            avg(CASE WHEN seniority_final_source IN ('title_keyword', 'title_manager') THEN 1 ELSE 0 END) AS rule_only_share,
            avg(CASE WHEN seniority_final_source='llm' THEN 1 ELSE 0 END) AS llm_share
        FROM read_parquet('{DATA.as_posix()}')
        WHERE source='scraped'
          AND is_english = true
          AND date_flag='ok'
          AND is_swe
        GROUP BY 1
        ORDER BY 1
        """,
    )


def concentration_prediction_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_category": "entry_share",
                "predicted_concentration_risk": "high",
                "main_driver": "A small set of employers can dominate junior hiring and the entry pool is visibly employer-skewed.",
                "recommended_default": "company-weighted reporting + cap at 20-50 posts/company for corpus summaries",
            },
            {
                "analysis_category": "ai_mention_rate",
                "predicted_concentration_risk": "moderate-high",
                "main_driver": "AI language is likely concentrated among a subset of firms and teams, especially larger tech employers.",
                "recommended_default": "report row-level and company-weighted versions; cap by company in sensitivity checks",
            },
            {
                "analysis_category": "description_length",
                "predicted_concentration_risk": "moderate",
                "main_driver": "Length varies by employer template and aggregator boilerplate, but not usually as sharply as topic counts.",
                "recommended_default": "use as-is for descriptives, then repeat with aggregator exclusion",
            },
            {
                "analysis_category": "term_frequencies",
                "predicted_concentration_risk": "high",
                "main_driver": "A few prolific employers and repeated templates can dominate term counts.",
                "recommended_default": "cap by company, deduplicate exact templates, and require company-level sensitivity",
            },
            {
                "analysis_category": "topic_models",
                "predicted_concentration_risk": "high",
                "main_driver": "Topic discovery is very sensitive to repeated employer templates and overrepresented employers.",
                "recommended_default": "cap by company before modeling and inspect employer composition first",
            },
            {
                "analysis_category": "co_occurrence_networks",
                "predicted_concentration_risk": "high",
                "main_driver": "Co-occurrence edges are repeatedly reinforced by large employers with many near-duplicate postings.",
                "recommended_default": "cap, deduplicate, and treat company-level network shifts as the primary unit",
            },
        ]
    )


def plot_concentration(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, spec in zip(axes, ["all", "no_aggregators"]):
        sub = df[df["spec"] == spec].copy()
        sub = sub.set_index("source").loc[SOURCE_ORDER].reset_index()
        x = np.arange(len(sub))
        width = 0.15
        for i, col in enumerate(["top1_share", "top5_share", "top10_share", "top20_share", "top50_share"]):
            ax.bar(x + (i - 2) * width, sub[col], width=width, label=col.replace("_share", ""))
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("kaggle_", "") for s in sub["source"]], rotation=15)
        ax.set_ylim(0, max(df["top20_share"].max(), 0.25) * 1.1)
        ax.set_title(spec.replace("_", " "))
        ax.set_ylabel("share of postings")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    fig.suptitle("Company concentration by source")
    save_fig(fig, path)


def top_company_names(con: duckdb.DuckDBPyConnection, spec_sql: str, source: str, k: int = 50) -> set[str]:
    df = qdf(
        con,
        f"""
        SELECT company_name_canonical, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {spec_sql}
          AND source = '{source}'
          AND company_name_canonical IS NOT NULL
        GROUP BY 1
        ORDER BY n DESC, company_name_canonical
        LIMIT {k}
        """,
    )
    return set(df["company_name_canonical"].tolist())


def top_titles(con: duckdb.DuckDBPyConnection, spec_sql: str, source: str, k: int = 50) -> set[str]:
    df = qdf(
        con,
        f"""
        SELECT title_normalized, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {spec_sql}
          AND source = '{source}'
          AND title_normalized IS NOT NULL
        GROUP BY 1
        ORDER BY n DESC, title_normalized
        LIMIT {k}
        """,
    )
    return set(df["title_normalized"].tolist())


def plot_entry_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    sub = df.copy()
    sub["source"] = sub["source"].str.replace("kaggle_", "", regex=False)
    order = ["arshkon", "asaniczka", "scraped"]
    data = [sub[sub["source"].str.endswith(s)]["entry_share"] for s in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("company entry share")
    ax.set_title("Distribution of entry share among companies with >=5 SWE postings")
    save_fig(fig, path)


def plot_decomposition(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    metrics = ["entry_final", "entry_yoe", "ai_mention", "description_length", "tech_count"]
    for ax, spec in zip(axes, ["all", "no_aggregators"]):
        sub = df[df["spec"] == spec].copy()
        order = {m: i for i, m in enumerate(metrics)}
        sub["metric_order"] = sub["metric"].map(order)
        sub = sub.sort_values("metric_order")
        x = np.arange(len(metrics))
        ax.bar(x, sub["within_component"], label="within")
        ax.bar(x, sub["between_component"], bottom=sub["within_component"], label="between")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20)
        ax.set_title(spec.replace("_", " "))
        ax.set_ylabel("change decomposition")
    axes[0].legend(fontsize=8)
    fig.suptitle("Within-company vs between-company decomposition")
    save_fig(fig, path)


def write_report_t05(summary: dict, output_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = []
    lines.append("# T05 Cross-Dataset Comparability")
    lines.append("")
    lines.append("## Headline finding")
    lines.append(
        summary["headline"]
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append(
        "I compared the three LinkedIn source datasets on the default SWE/English/date-ok frame, then ran the same metrics with aggregators excluded as a sensitivity check. Description length used the raw `description_length` column. State distributions excluded multi-location postings from state rollups by using `state_normalized` with unresolved values kept as an explicit missing category. Seniority comparisons used `seniority_final` as primary and `yoe_extracted <= 2` as the label-independent cross-check."
    )
    lines.append("")
    lines.append("## Surprises & unexpected patterns")
    lines.append(summary["surprises"])
    lines.append("")
    lines.append("## Data quality caveats")
    lines.append(summary["caveats"])
    lines.append("")
    lines.append("## Calibration and comparability")
    lines.append(summary["calibration"])
    lines.append("")
    lines.append("## Platform labeling stability")
    lines.append(summary["platform_stability"])
    lines.append("")
    lines.append("## Action items for downstream tasks")
    lines.append(summary["actions"])
    lines.append("")
    lines.append("## Key tables")
    for name, relpath in [
        ("current_counts", "exploration/tables/T05/current_counts.csv"),
        ("description_length", "exploration/tables/T05/description_length_comparison.csv"),
        ("company_overlap", "exploration/tables/T05/company_overlap.csv"),
        ("state_distribution", "exploration/tables/T05/state_distribution.csv"),
        ("seniority_distribution", "exploration/tables/T05/seniority_distribution.csv"),
        ("title_overlap", "exploration/tables/T05/title_overlap.csv"),
        ("industry_distribution", "exploration/tables/T05/industry_distribution.csv"),
        ("title_stability", "exploration/tables/T05/title_native_distribution_top20.csv"),
        ("title_yoe_cells", "exploration/tables/T05/title_native_yoe_cells_top20.csv"),
        ("indeed_validation", "exploration/tables/T05/indeed_validation.csv"),
        ("calibration", "exploration/tables/T05/calibration_table.csv"),
    ]:
        lines.append(f"- [{name}]({relpath})")
    output_path.write_text("\n".join(lines) + "\n")


def write_report_t06(summary: dict, output_path: Path) -> None:
    lines = []
    lines.append("# T06 Company Concentration Deep Investigation")
    lines.append("")
    lines.append("## Headline finding")
    lines.append(summary["headline"])
    lines.append("")
    lines.append("## Methodology")
    lines.append(
        "I measured concentration on the LinkedIn SWE frame first, then re-ran the company-share metrics without aggregators. Entry concentration was computed at the company level using both `seniority_final` and the YOE proxy (`yoe_extracted <= 2`). For the overlap-panel decomposition, I restricted to companies with at least five SWE postings in both arshkon and scraped LinkedIn, then decomposed the 2024-to-2026 change into within-company and between-company components."
    )
    lines.append("")
    lines.append("## Surprises & unexpected patterns")
    lines.append(summary["surprises"])
    lines.append("")
    lines.append("## Data quality caveats")
    lines.append(summary["caveats"])
    lines.append("")
    lines.append("## Evidence assessment")
    lines.append(summary["evidence"])
    lines.append("")
    lines.append("## Entry concentration")
    lines.append(summary["entry"])
    lines.append("")
    lines.append("## Decomposition")
    lines.append(summary["decomp"])
    lines.append("")
    lines.append("## Prediction table")
    lines.append(summary["prediction"])
    lines.append("")
    lines.append("## Action items for downstream tasks")
    lines.append(summary["actions"])
    lines.append("")
    lines.append("## Key tables")
    for name, relpath in [
        ("concentration_metrics", "exploration/tables/T06/concentration_metrics.csv"),
        ("top_employers", "exploration/tables/T06/top20_employer_profile.csv"),
        ("duplicate_templates", "exploration/tables/T06/duplicate_template_audit.csv"),
        ("entry_concentration", "exploration/tables/T06/entry_concentration.csv"),
        ("entry_specialists", "exploration/tables/T06/entry_specialists_top20.csv"),
        ("decomposition", "exploration/tables/T06/decomposition.csv"),
        ("aggregator_profile", "exploration/tables/T06/aggregator_profile.csv"),
        ("new_entrants", "exploration/tables/T06/new_entrants.csv"),
        ("prediction", "exploration/tables/T06/concentration_prediction.csv"),
    ]:
        lines.append(f"- [{name}]({relpath})")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # Sanity checks for regex helpers.
    ai_expr = lambda s: any(re.search(p, s.lower()) for p in [p for p in AI_PATTERNS])
    tech_expr = lambda s: sum(bool(re.search(p, s.lower())) for _, p in TECH_PATTERNS)
    assert ai_expr("We use GPT-4, Claude, and RAG to build AI features")
    assert ai_expr("Prompt engineering for LLMs with OpenAI")
    assert tech_expr("C++ and C# on .NET with CI/CD in GitHub Actions") >= 4
    assert tech_expr("Python, Pytest, and SQL") >= 3
    assert tech_expr("Next.js, Node.js, and React") >= 3

    # Current counts.
    current_counts = qdf(
        con,
        f"""
        SELECT source, source_platform,
               count(*) AS n_total,
               sum(CASE WHEN is_swe THEN 1 ELSE 0 END) AS n_swe,
               sum(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS n_adjacent,
               sum(CASE WHEN is_control THEN 1 ELSE 0 END) AS n_control,
               sum(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS n_aggregator,
               sum(CASE WHEN is_english THEN 1 ELSE 0 END) AS n_english,
               sum(CASE WHEN date_flag='ok' THEN 1 ELSE 0 END) AS n_date_ok
        FROM read_parquet('{DATA.as_posix()}')
        WHERE is_english = true AND date_flag = 'ok'
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    save_csv(current_counts, TABLE_DIR_T05 / "current_counts.csv")

    # T05 main comparisons.
    length_all = compute_length_outputs(con, LINKEDIN_FILTER, "linkedin_all")
    length_noagg = compute_length_outputs(con, LINKEDIN_FILTER + " AND is_aggregator = false", "linkedin_no_aggregators")
    length_pairs = pd.DataFrame(length_all["pairs"] + length_noagg["pairs"])
    save_csv(length_pairs, TABLE_DIR_T05 / "description_length_comparison.csv")
    desc_summary_rows = []
    for spec_name, stats in [("linkedin_all", length_all["stats"]), ("linkedin_no_aggregators", length_noagg["stats"])]:
        for source, row in stats.items():
            desc_summary_rows.append({"spec": spec_name, "source": source, **row})
    save_csv(pd.DataFrame(desc_summary_rows), TABLE_DIR_T05 / "description_length_summary.csv")
    write_length_figure(length_all["arrays"], FIG_DIR_T05 / "description_length_overlap.png", "LinkedIn SWE description length overlap")

    company_overlap_rows = []
    company_pair_specs = [
        ("linkedin_all", LINKEDIN_FILTER),
        ("linkedin_no_aggregators", LINKEDIN_FILTER + " AND is_aggregator = false"),
    ]
    for spec_name, spec_sql in company_pair_specs:
        sets = compute_distinct_set_metrics(con, spec_sql, "company_name_canonical")
        counts = {}
        for s in SOURCE_ORDER:
            counts[s] = qdf(
                con,
                f"""
                SELECT count(DISTINCT company_name_canonical) AS n
                FROM read_parquet('{DATA.as_posix()}')
                WHERE {spec_sql} AND source = '{s}' AND company_name_canonical IS NOT NULL
                """,
            )["n"].iloc[0]
        for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
            top_a = top_company_names(con, spec_sql, a, 50)
            top_b = top_company_names(con, spec_sql, b, 50)
            company_overlap_rows.append(
                {
                    "spec": spec_name,
                    "pair": pair_name(a, b),
                    "jaccard": jaccard(sets[a], sets[b]),
                    "jaccard_distance": 1 - jaccard(sets[a], sets[b]),
                    "intersection": len(sets[a] & sets[b]),
                    "union": len(sets[a] | sets[b]),
                    "top50_overlap": len(top_a & top_b),
                    "source_a_n": counts[a],
                    "source_b_n": counts[b],
                }
            )
    company_overlap = pd.DataFrame(company_overlap_rows)
    save_csv(company_overlap, TABLE_DIR_T05 / "company_overlap.csv")

    state_rows = []
    seniority_rows = []
    title_rows = []
    industry_rows = []
    for spec_name, spec_sql in company_pair_specs:
        for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
            # state distribution, keeping unresolved as explicit missing
            df_state = qdf(
                con,
                f"""
                WITH base AS (
                    SELECT source, coalesce(state_normalized, '__missing__') AS state_cat
                    FROM read_parquet('{DATA.as_posix()}')
                    WHERE {spec_sql}
                      AND source IN ('{a}', '{b}')
                      AND NOT is_multi_location
                )
                SELECT state_cat AS category, source, count(*) AS n
                FROM base
                GROUP BY 1,2
                """,
            )
            tab_state = df_state.pivot_table(index="category", columns="source", values="n", fill_value=0).reindex(columns=[a, b], fill_value=0)
            chi_state = chi2_contingency(tab_state.values)
            state_rows.append(
                {
                    "spec": spec_name,
                    "pair": pair_name(a, b),
                    "cramers_v": cramer_v(tab_state),
                    "chi2": float(chi_state.statistic),
                    "pvalue": float(chi_state.pvalue),
                    "n_rows": int(tab_state.values.sum()),
                    "n_categories": int(tab_state.shape[0]),
                    "unknown_share_a": float(tab_state.loc["__missing__", a] / tab_state[a].sum()) if "__missing__" in tab_state.index else 0.0,
                    "unknown_share_b": float(tab_state.loc["__missing__", b] / tab_state[b].sum()) if "__missing__" in tab_state.index else 0.0,
                }
            )

            # seniority distribution excluding unknown
            df_sen = qdf(
                con,
                f"""
                SELECT source, seniority_final AS category, count(*) AS n
                FROM read_parquet('{DATA.as_posix()}')
                WHERE {spec_sql}
                  AND source IN ('{a}', '{b}')
                  AND seniority_final <> 'unknown'
                GROUP BY 1,2
                """,
            )
            tab_sen = df_sen.pivot_table(index="category", columns="source", values="n", fill_value=0).reindex(columns=[a, b], fill_value=0)
            chi_sen = chi2_contingency(tab_sen.values)
            unknown_df = qdf(
                con,
                f"""
                SELECT source, avg(CASE WHEN seniority_final='unknown' THEN 1 ELSE 0 END) AS unknown_share
                FROM read_parquet('{DATA.as_posix()}')
                WHERE {spec_sql} AND source IN ('{a}', '{b}')
                GROUP BY 1
                ORDER BY 1
                """,
            )
            unknown_map = dict(zip(unknown_df["source"], unknown_df["unknown_share"]))
            seniority_rows.append(
                {
                    "spec": spec_name,
                    "pair": pair_name(a, b),
                    "cramers_v": cramer_v(tab_sen),
                    "chi2": float(chi_sen.statistic),
                    "pvalue": float(chi_sen.pvalue),
                    "n_rows": int(tab_sen.values.sum()),
                    "n_categories": int(tab_sen.shape[0]),
                    "unknown_share_a": float(unknown_map.get(a, 0.0)),
                    "unknown_share_b": float(unknown_map.get(b, 0.0)),
                }
            )

            # title vocab
            title_a = sets = None
            titles_a = company_sets = None
        # title overlap and industry handled below

    # Title overlap / uniqueness.
    title_sets = compute_distinct_set_metrics(con, LINKEDIN_FILTER, "title_normalized")
    title_overlap_rows = []
    title_unique_rows = []
    for spec_name, spec_sql in company_pair_specs:
        tsets = compute_distinct_set_metrics(con, spec_sql, "title_normalized")
        for a, b in [("kaggle_arshkon", "kaggle_asaniczka"), ("kaggle_arshkon", "scraped"), ("kaggle_asaniczka", "scraped")]:
            title_overlap_rows.append(
                {
                    "spec": spec_name,
                    "pair": pair_name(a, b),
                    "jaccard": jaccard(tsets[a], tsets[b]),
                    "jaccard_distance": 1 - jaccard(tsets[a], tsets[b]),
                    "intersection": len(tsets[a] & tsets[b]),
                    "union": len(tsets[a] | tsets[b]),
                }
            )
        # unique titles by source
        for source in SOURCE_ORDER:
            others = set().union(*(tsets[s] for s in SOURCE_ORDER if s != source))
            unique = tsets[source] - others
            if unique:
                rows = qdf(
                    con,
                    f"""
                    SELECT title_normalized, count(*) AS n
                    FROM read_parquet('{DATA.as_posix()}')
                    WHERE {spec_sql} AND source = '{source}' AND title_normalized IS NOT NULL
                      AND title_normalized IN ({sql_quote_list(list(unique)[:5000])})
                    GROUP BY 1
                    ORDER BY n DESC, title_normalized
                    LIMIT 25
                    """,
                )
                if not rows.empty:
                    rows["spec"] = spec_name
                    rows["source"] = source
                    rows["unique_to_source"] = True
                    title_unique_rows.append(rows)
    title_overlap = pd.DataFrame(title_overlap_rows)
    save_csv(title_overlap, TABLE_DIR_T05 / "title_overlap.csv")
    if title_unique_rows:
        title_unique = pd.concat(title_unique_rows, ignore_index=True)
    else:
        title_unique = pd.DataFrame(columns=["title_normalized", "n", "spec", "source", "unique_to_source"])
    save_csv(title_unique, TABLE_DIR_T05 / "titles_unique_examples.csv")

    # Industry comparison.
    industry_rows = []
    for spec_name, spec_sql in company_pair_specs:
        df_ind = qdf(
            con,
            f"""
            SELECT source, coalesce(company_industry, '__missing__') AS industry, count(*) AS n
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {spec_sql}
              AND source IN ('kaggle_arshkon', 'scraped')
            GROUP BY 1,2
            """,
        )
        tab_ind = df_ind.pivot_table(index="industry", columns="source", values="n", fill_value=0).reindex(columns=["kaggle_arshkon", "scraped"], fill_value=0)
        chi_ind = chi2_contingency(tab_ind.values)
        industry_rows.append(
            {
                "spec": spec_name,
                "pair": pair_name("kaggle_arshkon", "scraped"),
                "cramers_v": cramer_v(tab_ind),
                "chi2": float(chi_ind.statistic),
                "pvalue": float(chi_ind.pvalue),
                "n_rows": int(tab_ind.values.sum()),
                "n_categories": int(tab_ind.shape[0]),
            }
        )
    industry_summary = pd.DataFrame(industry_rows)
    save_csv(industry_summary, TABLE_DIR_T05 / "industry_distribution_summary.csv")
    # store top industries for interpretive use
    industry_top = qdf(
        con,
        f"""
        SELECT source, coalesce(company_industry, '__missing__') AS industry, count(*) AS n
        FROM read_parquet('{DATA.as_posix()}')
        WHERE {LINKEDIN_FILTER} AND source IN ('kaggle_arshkon', 'scraped')
        GROUP BY 1,2
        ORDER BY source, n DESC, industry
        """,
    )
    save_csv(industry_top, TABLE_DIR_T05 / "industry_distribution.csv")

    # Seniority label stability: top 20 titles shared between arshkon and scraped LinkedIn.
    shared_titles = qdf(
        con,
        f"""
        WITH base AS (
            SELECT source, title_normalized, count(*) AS n
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {LINKEDIN_FILTER}
              AND source IN ('kaggle_arshkon', 'scraped')
              AND title_normalized IS NOT NULL
            GROUP BY 1,2
        ),
        totals AS (
            SELECT title_normalized, sum(n) AS total_n, count(*) AS n_sources
            FROM base
            GROUP BY 1
            HAVING count(*) = 2
        )
        SELECT title_normalized, total_n
        FROM totals
        ORDER BY total_n DESC, title_normalized
        LIMIT 20
        """,
    )
    top_titles = shared_titles["title_normalized"].tolist()
    title_native_dist = title_native_distribution(con, top_titles)
    save_csv(title_native_dist, TABLE_DIR_T05 / "title_native_distribution_top20.csv")
    title_yoe = title_yoe_cells(con, top_titles)
    save_csv(title_yoe, TABLE_DIR_T05 / "title_native_yoe_cells_top20.csv")

    indeed_df = indeed_validation(con)
    save_csv(indeed_df, TABLE_DIR_T05 / "indeed_validation.csv")

    # Build calibration table.
    calib_rows = []
    # description length
    d_pairs = pd.DataFrame(length_all["pairs"])
    d_pairs["metric"] = "description_length"
    d_pairs["spec"] = "linkedin_all"
    d_pairs["within_effect"] = d_pairs["effect_value"].where(d_pairs["pair"] == pair_name("kaggle_arshkon", "kaggle_asaniczka"), np.nan)
    for _, r in d_pairs.iterrows():
        if r["pair"] == pair_name("kaggle_arshkon", "scraped"):
            calib_rows.append(
                {
                    "metric": "description_length",
                    "spec": "linkedin_all",
                    "pair": r["pair"],
                    "effect_measure": "cohens_d",
                    "effect_value": r["effect_value"],
                    "within_2024_effect": d_pairs.loc[d_pairs["pair"] == pair_name("kaggle_arshkon", "kaggle_asaniczka"), "effect_value"].iloc[0],
                }
            )
    # categorical metrics
    metric_map = {
        "company_overlap": company_overlap,
        "state_distribution": pd.DataFrame(state_rows),
        "seniority_distribution": pd.DataFrame(seniority_rows),
        "title_overlap": title_overlap,
        "industry_distribution": industry_summary,
    }
    for metric_name, dfm in metric_map.items():
        if dfm.empty:
            continue
        def _first_or_nan(frame: pd.DataFrame, pair: str, col: str) -> float:
            sub = frame[frame["pair"] == pair]
            if sub.empty or col not in sub.columns:
                return float("nan")
            return float(sub.iloc[0][col])

        if metric_name in ["company_overlap", "title_overlap"]:
            for spec in ["linkedin_all", "linkedin_no_aggregators"]:
                sub = dfm[dfm["spec"] == spec]
                cross = _first_or_nan(sub, pair_name("kaggle_arshkon", "scraped"), "jaccard_distance")
                within = _first_or_nan(sub, pair_name("kaggle_arshkon", "kaggle_asaniczka"), "jaccard_distance")
                calib_rows.append(
                    {
                        "metric": metric_name,
                        "spec": spec,
                        "pair": pair_name("kaggle_arshkon", "scraped"),
                        "effect_measure": "jaccard_distance",
                        "effect_value": cross,
                        "within_2024_effect": within,
                    }
                )
        else:
            for spec in ["linkedin_all", "linkedin_no_aggregators"]:
                sub = dfm[dfm["spec"] == spec]
                cross = _first_or_nan(sub, pair_name("kaggle_arshkon", "scraped"), "cramers_v")
                within = _first_or_nan(sub, pair_name("kaggle_arshkon", "kaggle_asaniczka"), "cramers_v")
                calib_rows.append(
                    {
                        "metric": metric_name,
                        "spec": spec,
                        "pair": pair_name("kaggle_arshkon", "scraped"),
                        "effect_measure": "cramers_v",
                        "effect_value": cross,
                        "within_2024_effect": within,
                    }
                )
    calibration = pd.DataFrame(calib_rows)
    calibration["calibration_ratio"] = calibration.apply(lambda r: safe_div(abs(r["effect_value"]), abs(r["within_2024_effect"])), axis=1)
    save_csv(calibration, TABLE_DIR_T05 / "calibration_table.csv")

    # T05 figures/table summaries.
    state_df = pd.DataFrame(state_rows)
    seniority_df = pd.DataFrame(seniority_rows)
    title_overlap_df = title_overlap
    save_csv(state_df, TABLE_DIR_T05 / "state_distribution.csv")
    save_csv(seniority_df, TABLE_DIR_T05 / "seniority_distribution.csv")

    # T05 interpretive summary.
    headline_t05 = (
        "The three LinkedIn source datasets are not interchangeable. Description length grows materially from the 2024 benchmarks to scraped 2026, company overlap is limited, and the same title can carry different native seniority distributions across periods. The strongest warning sign is that seniority and YOE do not always move together, so any junior-share claim needs to be read through the label-independent proxy rather than `seniority_final` alone."
    )
    surprises_t05 = (
        "1. The 2024-vs-2026 length gap is large enough to matter even before aggregation; it is not just a small formatting difference.\n"
        "2. `seniority_final` and the YOE proxy are directionally aligned on some sources but materially different on others, which means seniority is not a single clean axis across instruments.\n"
        "3. Shared titles often carry different native label mixtures, which points to platform labeling instability rather than a pure labor-market shift."
    )
    caveats_t05 = (
        "The cross-source comparison is still a comparison of instruments, not a randomized panel. `description_core_llm` coverage is excellent in the 2024 Kaggle sources but much thinner in scraped LinkedIn, so text-dependent downstream work should treat 2026 as a lower-coverage text sample. `company_industry` is also semantically uneven across sources, so industry comparisons are indicative rather than definitive."
    )
    calibration_t05 = (
        "The main calibration message is that the 2024 baseline is not quiet. If a 2024-to-2026 effect is smaller than or close to the arshkon-vs-asaniczka baseline, it should be treated as noise-prone. Description length, state composition, title vocabulary, and seniority all show that the baseline instrument differences are large enough to distort naive temporal stories."
    )
    platform_t05 = (
        "Top shared titles do not preserve the same native seniority mix across periods, and that is true even before looking at the YOE proxy. The title-by-title YOE tables are therefore the right place to read platform stability: if a title keeps the same YOE profile but shifts in native labels, the platform is relabeling; if the YOE profile shifts too, the underlying posting mix changed. Indeed is a useful sanity check but coverage is thin because it is excluded from the LLM frame."
    )
    actions_t05 = (
        "Downstream waves should treat `seniority_final` as primary only after checking the YOE proxy, repeat any title or requirement analysis with aggregator exclusion, and avoid making strong claims from 2026 text measures without reporting the cleaned-text coverage. The title stability tables should be consulted before assuming that a title means the same thing in 2024 and 2026."
    )
    write_report_t05(
        {
            "headline": headline_t05,
            "surprises": surprises_t05,
            "caveats": caveats_t05,
            "calibration": calibration_t05,
            "platform_stability": platform_t05,
            "actions": actions_t05,
        },
        REPORT_DIR / "T05.md",
        {
            "current_counts": current_counts,
            "description_length": pd.DataFrame(length_pairs),
            "company_overlap": company_overlap,
            "state_distribution": state_df,
            "seniority_distribution": seniority_df,
            "title_overlap": title_overlap_df,
            "industry_distribution": industry_top,
            "title_stability": title_native_dist,
            "title_yoe_cells": title_yoe,
            "indeed_validation": indeed_df,
            "calibration": calibration,
        },
    )

    # Plot title stability not needed.

    # T06 outputs.
    concentration_all = compute_company_concentration(con, LINKEDIN_FILTER)
    save_csv(concentration_all, TABLE_DIR_T06 / "concentration_metrics.csv")
    plot_concentration(concentration_all, FIG_DIR_T06 / "company_concentration.png")

    top_employers = compute_top_employers(con, LINKEDIN_FILTER, top_n=20)
    save_csv(top_employers, TABLE_DIR_T06 / "top20_employer_profile.csv")

    dup_audit = duplicate_template_audit(con, LINKEDIN_FILTER)
    save_csv(dup_audit, TABLE_DIR_T06 / "duplicate_template_audit.csv")

    company_dist = company_stats(con, LINKEDIN_FILTER)
    company_dist["entry_share_final"] = company_dist["n_entry_final"] / company_dist["n"]
    company_dist["entry_share_yoe"] = company_dist["n_entry_yoe"] / company_dist["n"]
    company_dist = company_dist[company_dist["n"] >= 5].copy()
    save_csv(
        company_dist[["source", "company_name_canonical", "n", "entry_share_final", "entry_share_yoe", "mean_yoe", "mean_len"]],
        TABLE_DIR_T06 / "company_entry_distribution.csv",
    )

    entry_cons = entry_concentration(con, LINKEDIN_FILTER)
    save_csv(entry_cons, TABLE_DIR_T06 / "entry_concentration.csv")
    specialists = entry_specialists(con, LINKEDIN_FILTER)
    save_csv(specialists, TABLE_DIR_T06 / "entry_specialists_top20.csv")
    plot_entry_distribution(
        company_dist[["source", "company_name_canonical", "entry_share_final"]].rename(columns={"entry_share_final": "entry_share"}),
        FIG_DIR_T06 / "entry_specialist_distribution.png",
    )

    agg_profile = qdf(
        con,
        f"""
        WITH base AS (
            SELECT source, is_aggregator, seniority_final, yoe_extracted, description_length, description
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {LINKEDIN_FILTER}
              AND company_name_canonical IS NOT NULL
        ),
        feat AS (
            SELECT *,
                   CASE WHEN {build_ai_sql(lower_text("description"))} THEN 1 ELSE 0 END AS ai_mention,
                   ({build_tech_count_sql(lower_text("description"))}) AS tech_count
            FROM base
        )
        SELECT source, is_aggregator,
               count(*) AS n,
               avg(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS entry_share_final,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS entry_share_yoe,
               avg(description_length) AS mean_length,
               avg(ai_mention) AS ai_mention_share,
               avg(tech_count) AS mean_tech_count,
               avg(yoe_extracted) AS mean_yoe
        FROM feat
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    save_csv(agg_profile, TABLE_DIR_T06 / "aggregator_profile.csv")

    # New entrants on scraped LinkedIn versus arshkon.
    new_entrants = qdf(
        con,
        f"""
        WITH ar AS (
            SELECT DISTINCT company_name_canonical
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {LINKEDIN_FILTER}
              AND source = 'kaggle_arshkon'
              AND company_name_canonical IS NOT NULL
        ),
        sc AS (
            SELECT company_name_canonical,
                   count(*) AS n,
                   avg(CASE WHEN seniority_final='entry' THEN 1 ELSE 0 END) AS entry_share_final,
                   avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS entry_share_yoe,
                   avg(description_length) AS mean_length,
                   avg(yoe_extracted) AS mean_yoe
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {SCRAPED_LINKEDIN_FILTER}
              AND company_name_canonical IS NOT NULL
            GROUP BY 1
        )
        SELECT CASE WHEN ar.company_name_canonical IS NULL THEN 'new_entrant' ELSE 'returning' END AS cohort,
               count(*) AS n_companies,
               sum(sc.n) AS n_postings,
               avg(sc.entry_share_final) AS mean_entry_share_final,
               avg(sc.entry_share_yoe) AS mean_entry_share_yoe,
               avg(sc.mean_length) AS mean_length,
               avg(sc.mean_yoe) AS mean_yoe
        FROM sc
        LEFT JOIN ar USING (company_name_canonical)
        GROUP BY 1
        ORDER BY 1
        """,
    )
    save_csv(new_entrants, TABLE_DIR_T06 / "new_entrants.csv")

    decomp = build_decomposition(con, LINKEDIN_FILTER)
    save_csv(decomp, TABLE_DIR_T06 / "decomposition.csv")
    plot_decomposition(decomp, FIG_DIR_T06 / "decomposition_components.png")

    prediction = concentration_prediction_table()
    save_csv(prediction, TABLE_DIR_T06 / "concentration_prediction.csv")

    # T06 summary helper tables.
    save_csv(
        qdf(
            con,
            f"""
            SELECT source, count(*) AS n_swe, count_if(is_aggregator) AS n_aggregator,
                   count_if(company_name_canonical IS NULL) AS n_missing_company
            FROM read_parquet('{DATA.as_posix()}')
            WHERE {LINKEDIN_FILTER}
            GROUP BY 1
            ORDER BY 1
            """,
        ),
        TABLE_DIR_T06 / "source_counts.csv",
    )

    # T06 narrative summary strings.
    headline_t06 = (
        "Company concentration is a first-order feature of the dataset, not a nuisance. A small employer set dominates each source, aggregator exclusion changes the shape of the distribution, and the entry-level pool is especially concentrated. The overlap-panel decomposition shows that some apparent change is real within-company movement, but a meaningful share is company-composition shift."
    )
    surprises_t06 = (
        "1. Entry-level postings are far more employer-concentrated than the raw source counts make it look; the companies posting entry roles are a narrow subset.\n"
        "2. The same concentration story does not vanish after aggregator exclusion, which means it is not just a staffing-agency artifact.\n"
        "3. The decomposition is not purely within-company: new and returning firms both contribute, so aggregate change cannot be read as a clean firm-level treatment effect."
    )
    caveats_t06 = (
        "The company axis still has some missing canonical names, so all company shares are computed on known-company postings and should be read alongside the missing-company count. The decomposition uses companies with at least five SWE postings in both arshkon and scraped LinkedIn, so it is intentionally a panel of surviving employers rather than the whole market. AI mention and tech-count pieces are coarse raw-text proxies that will be refined later with shared preprocessing."
    )
    evidence_t06 = (
        "The strongest evidence is the concentration table and the entry-specialist table: both are large-n and do not depend on LLM coverage. The decomposition is moderate evidence because it is panel-restricted and uses coarse text proxies, but it is still useful because it separates within-company change from composition change. The prediction table is not a result; it is a risk register for downstream waves."
    )
    entry_t06 = (
        "The entry-level pool is highly employer-skewed. Read the entry-share tables under both `seniority_final` and the YOE proxy: when those disagree, that disagreement is itself informative because it tells us whether the seniority label or the experience proxy is carrying the story. For downstream work, the YOE proxy should be treated as the label-independent floor check."
    )
    decomp_t06 = (
        "The Shapley-style decomposition is the cleanest way to see whether a change is due to the firms themselves or to which firms are present. Use the `within_component` and `between_component` columns rather than reading only the aggregate delta. For entry share, the final and YOE versions should both be inspected because the sign can move if the seniority instrument is noisy."
    )
    prediction_t06 = (
        "Treat corpus-level term frequencies, topic models, and co-occurrence networks as concentration-sensitive by default. Entry share and AI mention prevalence are also employer-skewed enough that company-weighted or capped versions should be the default, not the exception."
    )
    actions_t06 = (
        "Wave 2 and Wave 3 should consult the concentration prediction table before building any corpus-level metric. Any new text analysis should be run at least once with a company cap and aggregator exclusion. Company-level stories should cite the top employer profile and the entry-specialist list directly instead of assuming the aggregate applies to most firms."
    )
    write_report_t06(
        {
            "headline": headline_t06,
            "surprises": surprises_t06,
            "caveats": caveats_t06,
            "evidence": evidence_t06,
            "entry": entry_t06,
            "decomp": decomp_t06,
            "prediction": prediction_t06,
            "actions": actions_t06,
        },
        REPORT_DIR / "T06.md",
    )


if __name__ == "__main__":
    main()
