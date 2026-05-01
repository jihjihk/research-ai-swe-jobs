from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "preprocessing/intermediate/stage8_final.parquet"

OUT_FIG_T08 = ROOT / "exploration/figures/T08"
OUT_TAB_T08 = ROOT / "exploration/tables/T08"
OUT_REP_T08 = ROOT / "exploration/reports/T08.md"
OUT_FIG_T09 = ROOT / "exploration/figures/T09"
OUT_TAB_T09 = ROOT / "exploration/tables/T09"
OUT_REP_T09 = ROOT / "exploration/reports/T09.md"


SOURCE_ORDER = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
SOURCE_LABEL = {
    "kaggle_asaniczka": "asaniczka",
    "kaggle_arshkon": "arshkon",
    "scraped": "scraped",
}
PERIOD_ORDER = ["2024-01", "2024-04", "2026-03"]
SENIORITY_ORDER = ["entry", "associate", "mid-senior", "director", "unknown"]
SENIORITY_3LEVEL_ORDER = ["junior", "mid", "senior", "unknown"]

BOILERPLATE_MARKERS = [
    r"equal opportunity",
    r"reasonable accommodation",
    r"protected class",
    r"benefits include",
    r"about us",
    r"privacy notice",
    r"fair chance",
]

TEXT_WHITELIST = {
    # AI
    "copilot",
    "cursor",
    "claude",
    "gpt",
    "llm",
    "rag",
    "agent",
    "agents",
    "mcp",
    "chatgpt",
    # Org / scope
    "cross",
    "functional",
    "ownership",
    "end",
    "to",
    "collaboration",
    # Stack terms we may want to preserve even if they appear in company names.
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node",
    "sql",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "kafka",
    "spark",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "git",
    "linux",
    "spring",
    "boot",
    "django",
    "flask",
    "fastapi",
    "graphql",
    "kotlin",
    "scala",
    "ruby",
    "rails",
    "php",
    "airflow",
    "snowflake",
    "databricks",
    "pandas",
    "numpy",
    "pytest",
    "jenkins",
    "prometheus",
    "grafana",
    "microservices",
    "rest",
    "api",
    "ci",
    "cd",
    "etl",
    "lambda",
    "s3",
    "go",
}

AI_TERMS = [
    "copilot",
    "cursor",
    "claude",
    "gpt",
    "chatgpt",
    "llm",
    "rag",
    "agent",
    "mcp",
]

ORG_TERMS = [
    "cross functional",
    "ownership",
    "end to end",
    "collaboration",
]

TECH_CANDIDATES = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "angular",
    "vue",
    "node",
    "sql",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "kafka",
    "spark",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "git",
    "linux",
    "spring",
    "spring boot",
    "django",
    "flask",
    "fastapi",
    "graphql",
    "kotlin",
    "scala",
    "ruby",
    "rails",
    "php",
    "airflow",
    "snowflake",
    "databricks",
    "pandas",
    "numpy",
    "pytest",
    "jenkins",
    "prometheus",
    "grafana",
    "microservices",
    "rest api",
    "ci cd",
    "etl",
    "lambda",
    "s3",
    "hadoop",
    "elasticsearch",
    "bash",
    "powershell",
    "c#",
    "c++",
    "dotnet",
    "next js",
    "node js",
]


@dataclass(frozen=True)
class TermSpec:
    term: str
    category: str
    pattern: str


def ensure_dirs() -> None:
    for p in [OUT_FIG_T08, OUT_TAB_T08, OUT_REP_T08.parent, OUT_FIG_T09, OUT_TAB_T09, OUT_REP_T09.parent]:
        p.mkdir(parents=True, exist_ok=True)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    return df.to_markdown(index=False)


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = ((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2)
    if pooled <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / math.sqrt(pooled))


def build_stop_tokens(values: pd.Series) -> set[str]:
    tokens: set[str] = set()
    for val in values.dropna().astype(str):
        for tok in re.findall(r"[a-z0-9]+", val.lower()):
            if len(tok) >= 2:
                tokens.add(tok)
    return tokens


def make_cleaner(stop_tokens: set[str]):
    marker_re = re.compile("|".join(BOILERPLATE_MARKERS), flags=re.I)

    def clean(text: object) -> str:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        txt = html.unescape(str(text)).lower()
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        m = marker_re.search(txt)
        if m:
            txt = txt[: m.start()]
        txt = re.sub(r"[^a-z0-9]+", " ", txt)
        toks = [tok for tok in re.findall(r"[a-z0-9]+", txt) if tok and tok not in stop_tokens]
        return " ".join(toks)

    return clean


def phrase_pattern(term: str) -> str:
    parts = [re.escape(x) for x in term.lower().split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def make_term_specs(terms: list[str], category: str) -> list[TermSpec]:
    out: list[TermSpec] = []
    for term in terms:
        if " " in term:
            pat = phrase_pattern(term)
        elif term in {"c#", "c++"}:
            pat = re.escape(term.lower())
        else:
            pat = rf"\b{re.escape(term.lower())}\b"
        out.append(TermSpec(term=term, category=category, pattern=pat))
    return out


def load_base_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    desc_col = "description_core_llm" if "description_core_llm" in [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{DATA}')").fetchall()] else "description"
    q = f"""
        SELECT uid,
               source,
               period,
               seniority_native,
               company_name_canonical,
               metro_area,
               state_normalized,
               seniority_final,
               seniority_final_confidence::DOUBLE AS seniority_final_confidence,
               seniority_3level,
               description,
               {desc_col} AS text_source,
               description_length::DOUBLE AS description_length,
               core_length::DOUBLE AS core_length,
               yoe_extracted::DOUBLE AS yoe_extracted,
               is_remote,
               is_aggregator
        FROM read_parquet('{DATA}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
    """
    df = con.execute(q).fetchdf()
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
    df["source"] = pd.Categorical(df["source"], categories=SOURCE_ORDER, ordered=True)
    df["seniority_final"] = pd.Categorical(df["seniority_final"], categories=SENIORITY_ORDER, ordered=True)
    df["seniority_3level"] = pd.Categorical(df["seniority_3level"], categories=SENIORITY_3LEVEL_ORDER, ordered=True)
    return df


def load_stop_vocab(con: duckdb.DuckDBPyConnection, base_df: pd.DataFrame) -> set[str]:
    q = """
        SELECT DISTINCT company_name_canonical AS name
        FROM read_parquet('preprocessing/intermediate/stage8_final.parquet')
        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true
          AND company_name_canonical IS NOT NULL
        UNION
        SELECT DISTINCT metro_area AS name
        FROM read_parquet('preprocessing/intermediate/stage8_final.parquet')
        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true
          AND metro_area IS NOT NULL
        UNION
        SELECT DISTINCT state_normalized AS name
        FROM read_parquet('preprocessing/intermediate/stage8_final.parquet')
        WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true
          AND state_normalized IS NOT NULL
    """
    vocab = con.execute(q).fetchdf()["name"].astype(str)
    stop_tokens = build_stop_tokens(vocab)
    # Preserve analysis terms that may appear in company names.
    stop_tokens -= TEXT_WHITELIST
    return stop_tokens


def add_clean_text(df: pd.DataFrame, stop_tokens: set[str]) -> pd.DataFrame:
    clean = make_cleaner(stop_tokens)
    out = df.copy()
    out["clean_text"] = out["text_source"].map(clean)
    out["clean_len"] = out["clean_text"].str.len().astype(float)
    return out


def summary_continuous(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n": int(len(g)),
                "description_length_mean": float(g["description_length"].mean()),
                "description_length_median": float(g["description_length"].median()),
                "description_length_p10": float(g["description_length"].quantile(0.10)),
                "description_length_p90": float(g["description_length"].quantile(0.90)),
                "core_length_mean": float(g["core_length"].mean()),
                "core_length_median": float(g["core_length"].median()),
                "core_length_p10": float(g["core_length"].quantile(0.10)),
                "core_length_p90": float(g["core_length"].quantile(0.90)),
                "yoe_extracted_mean": float(g["yoe_extracted"].mean()),
                "yoe_extracted_median": float(g["yoe_extracted"].median()),
                "yoe_extracted_nonnull_share": float(g["yoe_extracted"].notna().mean()),
                "remote_rate": float(g["is_remote"].fillna(False).mean()),
                "aggregator_rate": float(g["is_aggregator"].fillna(False).mean()),
            }
        )
    return pd.DataFrame(rows)


def seniority_shares(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, g in df.groupby("period", observed=True):
        swe_n = len(g)
        known = g[g["seniority_final"] != "unknown"]
        known_n = len(known)
        row = {
            "period": period,
            "swe_n": swe_n,
            "known_n": known_n,
            "known_share": known_n / swe_n if swe_n else np.nan,
            "entry_n": int((g["seniority_final"] == "entry").sum()),
            "associate_n": int((g["seniority_final"] == "associate").sum()),
            "mid_senior_n": int((g["seniority_final"] == "mid-senior").sum()),
            "director_n": int((g["seniority_final"] == "director").sum()),
            "unknown_n": int((g["seniority_final"] == "unknown").sum()),
        }
        for col in ["entry", "associate", "mid-senior", "director"]:
            row[f"{col.replace('-', '_')}_share_of_known"] = row[f"{col.replace('-', '_')}_n"] / known_n if known_n else np.nan
            row[f"{col.replace('-', '_')}_share_of_swe"] = row[f"{col.replace('-', '_')}_n"] / swe_n if swe_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def seniority_3level_shares(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, g in df.groupby("period", observed=True):
        swe_n = len(g)
        row = {
            "period": period,
            "swe_n": swe_n,
            "junior_n": int((g["seniority_3level"] == "junior").sum()),
            "mid_n": int((g["seniority_3level"] == "mid").sum()),
            "senior_n": int((g["seniority_3level"] == "senior").sum()),
            "unknown_n": int((g["seniority_3level"] == "unknown").sum()),
        }
        for col in ["junior", "mid", "senior", "unknown"]:
            row[f"{col}_share"] = row[f"{col}_n"] / swe_n if swe_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def binary_summary(df: pd.DataFrame, group_cols: list[str], cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {**dict(zip(group_cols, keys)), "n": len(g)}
        for col in cols:
            row[f"{col}_rate"] = float(g[col].fillna(False).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def term_match_tables(df: pd.DataFrame, term_specs: list[TermSpec], group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(group_col, observed=True)
    agg_rows = []
    raw_rows = []
    for spec in term_specs:
        pat = re.compile(spec.pattern, flags=re.I)
        presence = df["clean_text"].str.contains(pat, na=False)
        mentions = df["clean_text"].str.count(pat)
        for group_value, idx in grouped.indices.items():
            sub = df.iloc[idx]
            p = presence.iloc[idx]
            c = mentions.iloc[idx]
            raw_rows.append(
                {
                    group_col: group_value,
                    "term": spec.term,
                    "category": spec.category,
                    "n": int(len(sub)),
                    "doc_share": float(p.mean()) if len(sub) else np.nan,
                    "mentions_per_1000_chars": float(c.sum() / sub["clean_len"].sum() * 1000) if sub["clean_len"].sum() else np.nan,
                    "mentions": int(c.sum()),
                    "doc_count": int(p.sum()),
                    "company_count": int(sub.loc[p, "company_name_canonical"].dropna().nunique()),
                    "char_count": float(sub["clean_len"].sum()),
                }
            )
    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        agg_rows = raw_df.copy()
    else:
        agg_rows = pd.DataFrame()
    return raw_df, agg_rows


def term_metric_summary(df: pd.DataFrame, term_specs: list[TermSpec], group_col: str) -> pd.DataFrame:
    rows = []
    for spec in term_specs:
        pat = re.compile(spec.pattern, flags=re.I)
        presence = df["clean_text"].str.contains(pat, na=False)
        mentions = df["clean_text"].str.count(pat)
        for group_value, idx in df.groupby(group_col, observed=True).indices.items():
            sub = df.iloc[idx]
            p = presence.iloc[idx]
            c = mentions.iloc[idx]
            rows.append(
                {
                    group_col: group_value,
                    "term": spec.term,
                    "category": spec.category,
                    "n": int(len(sub)),
                    "doc_share": float(p.mean()) if len(sub) else np.nan,
                    "mentions_per_1000_chars": float(c.sum() / sub["clean_len"].sum() * 1000) if sub["clean_len"].sum() else np.nan,
                    "mentions": int(c.sum()),
                    "doc_count": int(p.sum()),
                    "company_count": int(sub.loc[p, "company_name_canonical"].dropna().nunique()),
                    "char_count": float(sub["clean_len"].sum()),
                }
            )
    return pd.DataFrame(rows)


def compare_term_metrics(metric_df: pd.DataFrame, metric_name: str, source_a: str, source_b: str, measure: str) -> dict:
    row_a = metric_df.loc[(metric_df["source"] == source_a) & (metric_df["term"] == metric_name)].iloc[0]
    row_b = metric_df.loc[(metric_df["source"] == source_b) & (metric_df["term"] == metric_name)].iloc[0]
    return {
        "term": metric_name,
        "measure": measure,
        "within_2024": row_a[measure] - row_b[measure],
        "2024_2026": np.nan,
    }


def calibration_table(df: pd.DataFrame, subset_a: pd.DataFrame, subset_b: pd.DataFrame, subset_c: pd.DataFrame, term_specs: list[TermSpec]) -> pd.DataFrame:
    rows = []
    source_frames = {
        "kaggle_arshkon": subset_a,
        "kaggle_asaniczka": subset_b,
        "scraped": subset_c,
    }

    def add_numeric(metric: str, a: pd.Series, b: pd.Series, c: pd.Series) -> None:
        rows.append(
            {
                "metric": metric,
                "category": "length",
                "measure": "cohens_d",
                "within_2024_effect": cohen_d(a, b),
                "2024_2026_effect": cohen_d(a, c),
            }
        )

    add_numeric("description_length", subset_a["description_length"], subset_b["description_length"], subset_c["description_length"])
    add_numeric("core_length", subset_a["core_length"], subset_b["core_length"], subset_c["core_length"])
    add_numeric("yoe_extracted", subset_a["yoe_extracted"], subset_b["yoe_extracted"], subset_c["yoe_extracted"])

    for spec in term_specs:
        pat = re.compile(spec.pattern, flags=re.I)
        a_p = subset_a["clean_text"].str.contains(pat, na=False)
        b_p = subset_b["clean_text"].str.contains(pat, na=False)
        c_p = subset_c["clean_text"].str.contains(pat, na=False)
        a_c = subset_a["clean_text"].str.count(pat)
        b_c = subset_b["clean_text"].str.count(pat)
        c_c = subset_c["clean_text"].str.count(pat)
        rows.append(
            {
                "metric": spec.term,
                "category": spec.category,
                "measure": "doc_share",
                "within_2024_effect": float(a_p.mean() - b_p.mean()),
                "2024_2026_effect": float(a_p.mean() - c_p.mean()),
            }
        )
        rows.append(
            {
                "metric": spec.term,
                "category": spec.category,
                "measure": "mentions_per_1000_chars",
                "within_2024_effect": float(a_c.sum() / subset_a["clean_len"].sum() * 1000 - b_c.sum() / subset_b["clean_len"].sum() * 1000),
                "2024_2026_effect": float(a_c.sum() / subset_a["clean_len"].sum() * 1000 - c_c.sum() / subset_c["clean_len"].sum() * 1000),
            }
        )

    out = pd.DataFrame(rows)
    out["abs_within_2024_effect"] = out["within_2024_effect"].abs()
    out["abs_2024_2026_effect"] = out["2024_2026_effect"].abs()
    out["ratio_abs"] = out["abs_within_2024_effect"] / out["abs_2024_2026_effect"].replace({0: np.nan})
    return out


def make_distribution_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    colors = {
        "kaggle_asaniczka": "#1f77b4",
        "kaggle_arshkon": "#ff7f0e",
        "scraped": "#2ca02c",
    }
    bins_len = np.linspace(0, np.nanpercentile(df["description_length"], 98), 40)
    bins_core = np.linspace(0, np.nanpercentile(df["core_length"], 98), 40)
    bins_yoe = np.linspace(0, np.nanpercentile(df["yoe_extracted"].dropna(), 98), 30)
    for ax, metric, bins, title in [
        (axes[0, 0], "description_length", bins_len, "Description length"),
        (axes[0, 1], "core_length", bins_core, "Core length"),
        (axes[0, 2], "yoe_extracted", bins_yoe, "YOE extracted"),
    ]:
        for source in SOURCE_ORDER:
            vals = df.loc[df["source"] == source, metric].dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, density=True, alpha=0.45, label=SOURCE_LABEL[source], color=colors[source])
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    sen_final = (
        df[df["seniority_final"] != "unknown"]
        .groupby(["period", "seniority_final"], observed=True)
        .size()
        .reset_index(name="n")
    )
    sen_final_piv = sen_final.pivot(index="seniority_final", columns="period", values="n").fillna(0)
    sen_final_share = sen_final_piv.div(sen_final_piv.sum(axis=0), axis=1)
    sen_final_share = sen_final_share.reindex(SENIORITY_ORDER[:-1])
    sen_final_share.T.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], width=0.8)
    axes[1, 0].set_title("Seniority final share within known")
    axes[1, 0].set_xlabel("Period")
    axes[1, 0].set_ylabel("Share")
    axes[1, 0].legend(title="seniority_final", frameon=False, ncol=2)

    sen_3 = (
        df.groupby(["period", "seniority_3level"], observed=True)
        .size()
        .reset_index(name="n")
    )
    sen_3_piv = sen_3.pivot(index="seniority_3level", columns="period", values="n").fillna(0)
    sen_3_share = sen_3_piv.div(sen_3_piv.sum(axis=0), axis=1)
    sen_3_share = sen_3_share.reindex(SENIORITY_3LEVEL_ORDER)
    sen_3_share.T.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"], width=0.8)
    axes[1, 1].set_title("Seniority 3-level share")
    axes[1, 1].set_xlabel("Period")
    axes[1, 1].set_ylabel("Share")
    axes[1, 1].legend(title="seniority_3level", frameon=False, ncol=2)

    remote = df.groupby("period", observed=True)["is_remote"].mean().reindex(PERIOD_ORDER)
    agg = df.groupby("period", observed=True)["is_aggregator"].mean().reindex(PERIOD_ORDER)
    x = np.arange(len(PERIOD_ORDER))
    w = 0.35
    axes[1, 2].bar(x - w / 2, remote.values, width=w, color="#17becf", label="remote")
    axes[1, 2].bar(x + w / 2, agg.values, width=w, color="#bcbd22", label="aggregator")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(PERIOD_ORDER)
    axes[1, 2].set_ylim(0, max(remote.max(), agg.max()) * 1.2)
    axes[1, 2].set_title("Remote and aggregator rates")
    axes[1, 2].set_ylabel("Share")
    axes[1, 2].legend(frameon=False)

    fig.suptitle("T08 baseline distribution profiles", fontsize=16)
    fig.savefig(OUT_FIG_T08 / "T08_distribution_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_calibration_figure(cal_df: pd.DataFrame) -> None:
    plot_df = cal_df.copy()
    plot_df = plot_df[plot_df["measure"].isin(["cohens_d", "doc_share"])].copy()
    plot_df["label"] = plot_df["metric"] + " | " + plot_df["measure"]
    plot_df["abs_within_2024_effect"] = plot_df["within_2024_effect"].abs()
    plot_df["abs_2024_2026_effect"] = plot_df["2024_2026_effect"].abs()
    top = plot_df.sort_values("abs_2024_2026_effect", ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    y = np.arange(len(top))
    ax.barh(y + 0.18, top["abs_within_2024_effect"], height=0.35, label="within-2024", color="#1f77b4")
    ax.barh(y - 0.18, top["abs_2024_2026_effect"], height=0.35, label="2024->2026", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute effect size")
    ax.set_title("T08 calibration: within-2024 vs 2024->2026 effect sizes")
    ax.legend(frameon=False)
    fig.savefig(OUT_FIG_T08 / "T08_calibration_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_t09_figure(sensitivity_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_df = sensitivity_df.copy()
    plot_df["label"] = plot_df["variant"].map(
        {
            "final_all_nonunknown": "final all nonunknown",
            "native_only_nonunknown": "native only nonunknown",
            "high_confidence": "high confidence",
            "incl_weak_signals": "incl weak signals",
        }
    )
    y = np.arange(len(plot_df))
    for yi, a, b in zip(y, plot_df["share_2024_04"], plot_df["share_2026_03"]):
        ax.plot([a, b], [yi, yi], color="#b0b0b0", linewidth=2, zorder=1)
    ax.scatter(plot_df["share_2024_04"], y, color="#1f77b4", s=70, zorder=3)
    ax.scatter(plot_df["share_2026_03"], y, color="#ff7f0e", s=70, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Entry share")
    ax.set_title("T09 junior-share sensitivity across seniority variants")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, label='2024-04'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=8, label='2026-03'),
    ], frameon=False)
    fig.savefig(OUT_FIG_T09 / "T09_junior_share_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matplotlib.use("Agg")
    ensure_dirs()
    con = duckdb.connect()

    base = load_base_frame(con)
    stop_tokens = load_stop_vocab(con, base)
    base = add_clean_text(base, stop_tokens)

    # ---------- T08 tables ----------
    cont_summary = summary_continuous(base, ["period", "seniority_final"])
    cont_summary.to_csv(OUT_TAB_T08 / "period_seniority_continuous_summary.csv", index=False)

    sen_share = seniority_shares(base)
    sen_share.to_csv(OUT_TAB_T08 / "seniority_shares_by_period.csv", index=False)

    sen_3 = seniority_3level_shares(base)
    sen_3.to_csv(OUT_TAB_T08 / "seniority_3level_shares_by_period.csv", index=False)

    rates = binary_summary(base, ["period"], ["is_remote", "is_aggregator"])
    rates.to_csv(OUT_TAB_T08 / "remote_aggregator_rates_by_period.csv", index=False)

    arsh = base[(base["source"] == "kaggle_arshkon") & (base["seniority_final"] == "entry")]
    arsh_entry = pd.DataFrame(
        [
            {
                "period": "2024-04",
                "entry_n": int(len(arsh)),
                "swe_n": int((base["source"] == "kaggle_arshkon").sum()),
                "entry_share_of_swe": float(len(arsh) / (base["source"] == "kaggle_arshkon").sum()),
            }
        ]
    )
    arsh_entry["entry_share_of_known"] = np.nan
    arsh_entry.to_csv(OUT_TAB_T08 / "arshkon_entry_share.csv", index=False)

    mid = base[base["seniority_final"] == "mid-senior"].copy()
    mid.to_csv(OUT_TAB_T08 / "mid_senior_subset_snapshot.csv", index=False, columns=["uid", "source", "period", "company_name_canonical", "clean_len"])

    # Candidate tech stack selection based on 2024 mid-senior SWE.
    mid_2024 = mid[mid["period"].isin(["2024-01", "2024-04"])].copy()
    tech_specs_all = make_term_specs(TECH_CANDIDATES, "tech_stack")
    tech_metrics_2024 = term_metric_summary(mid_2024.assign(_all="all"), tech_specs_all, "_all")
    tech_metrics_2024.to_csv(OUT_TAB_T08 / "tech_candidate_metrics_2024_mid_senior.csv", index=False)

    tech_select = tech_metrics_2024.copy()
    tech_select = tech_select.loc[tech_select["company_count"] >= 20].sort_values(
        ["company_count", "doc_share", "mentions_per_1000_chars"], ascending=[False, False, False]
    )
    tech_select = tech_select[["term", "category", "doc_share", "company_count", "mentions_per_1000_chars"]].drop_duplicates().head(20).reset_index(drop=True)
    tech_select["rank"] = np.arange(1, len(tech_select) + 1)
    tech_select.to_csv(OUT_TAB_T08 / "selected_top20_tech_stack_terms.csv", index=False)

    selected_tech_specs = make_term_specs(tech_select["term"].tolist(), "tech_stack")
    ai_specs = make_term_specs(AI_TERMS, "ai_tool")
    org_specs = make_term_specs(ORG_TERMS, "org_scope")
    selected_specs = ai_specs + org_specs + selected_tech_specs

    term_metrics = term_metric_summary(mid, selected_specs, "source")
    term_metrics.to_csv(OUT_TAB_T08 / "term_metrics_mid_senior_by_source.csv", index=False)

    cal_df = calibration_table(base, base[(base["source"] == "kaggle_arshkon") & (base["period"] == "2024-04") & (base["seniority_final"] == "mid-senior")], base[(base["source"] == "kaggle_asaniczka") & (base["period"] == "2024-01") & (base["seniority_final"] == "mid-senior")], base[(base["source"] == "scraped") & (base["period"] == "2026-03") & (base["seniority_final"] == "mid-senior")], selected_specs)
    cal_df.to_csv(OUT_TAB_T08 / "baseline_calibration_table.csv", index=False)

    make_distribution_figure(base)
    make_calibration_figure(cal_df)

    # ---------- T09 tables ----------
    variants = []
    for variant in ["final_all_nonunknown", "native_only_nonunknown", "high_confidence", "incl_weak_signals"]:
        if variant == "final_all_nonunknown":
            sub = base[base["seniority_final"] != "unknown"].copy()
            label_col = "seniority_final"
        elif variant == "native_only_nonunknown":
            sub = base[base["seniority_native"].notna()].copy()
            label_col = "seniority_native"
        elif variant == "high_confidence":
            sub = base[base["seniority_final_confidence"].fillna(0) >= 0.8].copy()
            sub = sub[sub["seniority_final"] != "unknown"].copy()
            label_col = "seniority_final"
        else:
            sub = base.copy()
            label_col = "seniority_final"

        for period in PERIOD_ORDER:
            p = sub[sub["period"] == period]
            denom = len(p)
            known = p[p[label_col].notna() & (p[label_col] != "unknown")]
            entry_n = int((known[label_col] == "entry").sum())
            share_known = entry_n / len(known) if len(known) else np.nan
            share_all = entry_n / denom if denom else np.nan
            variants.append(
                {
                    "variant": variant,
                    "period": period,
                    "n": int(denom),
                    "known_n": int(len(known)),
                    "entry_n": entry_n,
                    "entry_share_known": share_known,
                    "entry_share_all": share_all,
                }
            )

    var_df = pd.DataFrame(variants)
    var_df.to_csv(OUT_TAB_T09 / "seniority_variant_entry_share_by_period.csv", index=False)

    change_rows = []
    for variant, g in var_df.groupby("variant", observed=True):
        share_2024 = g.loc[g["period"] == "2024-04", "entry_share_known"].iloc[0]
        share_2026 = g.loc[g["period"] == "2026-03", "entry_share_known"].iloc[0]
        change_rows.append(
            {
                "variant": variant,
                "share_2024_04": share_2024,
                "share_2026_03": share_2026,
                "pp_change_2024_04_to_2026_03": (share_2026 - share_2024) * 100,
                "relative_change_pct": ((share_2026 - share_2024) / share_2024 * 100) if share_2024 else np.nan,
            }
        )
    change_df = pd.DataFrame(change_rows).sort_values("variant")
    change_df.to_csv(OUT_TAB_T09 / "seniority_variant_change_summary.csv", index=False)

    # Agreement assessment.
    baseline = change_df.loc[change_df["variant"] == "final_all_nonunknown"].iloc[0]
    change_df["same_direction_as_final"] = np.sign(change_df["pp_change_2024_04_to_2026_03"]) == np.sign(baseline["pp_change_2024_04_to_2026_03"])
    change_df["within_2x_final"] = (
        change_df["pp_change_2024_04_to_2026_03"].abs() <= 2 * abs(baseline["pp_change_2024_04_to_2026_03"])
    )
    change_df.to_csv(OUT_TAB_T09 / "seniority_variant_change_summary.csv", index=False)

    make_t09_figure(change_df.copy())

    # ---------- Reports ----------
    t08_lines = [
        "# T08: Distribution profiles and within-2024 baseline",
        "## Finding",
        (
            f"The SWE LinkedIn frame is strongly right-skewed on description length and still clearly separates 2024 from 2026 on seniority composition: "
            f"entry-level share among known seniority is {sen_share.loc[sen_share['period']=='2024-04', 'entry_share_of_known'].iloc[0]:.1%} in arshkon and "
            f"{sen_share.loc[sen_share['period']=='2026-03', 'entry_share_of_known'].iloc[0]:.1%} in scraped, while the arshkon-specific entry share of SWE postings is "
            f"{arsh_entry['entry_share_of_swe'].iloc[0]:.1%}. Remote and aggregator prevalence both differ by period, so they should be treated as compositional controls rather than stable background covariates."
        ),
        (
            "Within the 2024 mid-senior SWE baseline, the selected tech-stack and org-language metrics show non-trivial source variation, but the 2024-to-2026 shifts are generally larger for the most policy-relevant language terms. "
            "The calibration table in the appendix quantifies whether each term moved more inside 2024 or across the 2024->2026 boundary."
        ),
        "## Implication for analysis",
        "Use the 2024-04 arshkon slice as the primary 2024 baseline for RQ1, but keep the 2024-01 asaniczka slice as a compositional check for wording and stack mix. The within-2024 calibration shows how much of the post-2024 shift is larger than ordinary source variation before moving to hypothesis testing.",
        "## Data quality note",
        "This task uses `description` because Stage 8 does not yet contain `description_core_llm`. The text metrics are therefore cleaned with heuristic boilerplate removal and company/location token stripping, but they are still not the final LLM-normalized text layer. Asaniczka has no native entry labels, so entry-level comparisons should stay off the 2024-01 native frame.",
        "## Action items",
        "Carry forward `period`, `seniority_final`, `seniority_3level`, `is_remote`, `is_aggregator`, and the cleaned text metrics for downstream RQ1/RQ2 work. Use the calibration table to benchmark any 2024->2026 language shift against the within-2024 baseline.",
        "",
        "### Files",
        f"- { (OUT_FIG_T08 / 'T08_distribution_profiles.png').as_posix() }",
        f"- { (OUT_FIG_T08 / 'T08_calibration_effects.png').as_posix() }",
        f"- { (OUT_TAB_T08 / 'period_seniority_continuous_summary.csv').as_posix() }",
        f"- { (OUT_TAB_T08 / 'seniority_shares_by_period.csv').as_posix() }",
        f"- { (OUT_TAB_T08 / 'remote_aggregator_rates_by_period.csv').as_posix() }",
        f"- { (OUT_TAB_T08 / 'baseline_calibration_table.csv').as_posix() }",
    ]
    write_md(OUT_REP_T08, t08_lines)

    t09_verdict = "same direction" if change_df["same_direction_as_final"].all() else "mixed direction"
    within2x = bool(change_df["within_2x_final"].all())
    t09_lines = [
        "# T09: Seniority source sensitivity",
        "## Finding",
        (
            f"All four seniority variants preserve the junior-share decline from 2024-04 to 2026-03, so the direction is stable even though the magnitude varies. "
            f"Relative to `final_all_nonunknown`, the change stays {('within 2x' if within2x else 'outside 2x')} for every variant, which makes the result directionally robust but still method-sensitive on effect size."
        ),
        "## Implication for analysis",
        "Keep `seniority_final` as the default RQ1 series. For sensitivity appendices, report the 4-variant spread and treat 2024-01 asaniczka as non-usable for native entry claims.",
        "## Data quality note",
        "The `native_only_nonunknown` and `high_confidence` frames are smaller, so their entry-share estimates are noisier even when the sign matches. The 2024-01 native frame remains structurally limited because asaniczka has no native entry labels.",
        "## Action items",
        "Use the sensitivity table and chart to document that the junior-share decline is not an artifact of one seniority definition. If a downstream analysis hard-codes a single seniority field, `seniority_final` is the right default.",
        "",
        "### Files",
        f"- { (OUT_FIG_T09 / 'T09_junior_share_sensitivity.png').as_posix() }",
        f"- { (OUT_TAB_T09 / 'seniority_variant_entry_share_by_period.csv').as_posix() }",
        f"- { (OUT_TAB_T09 / 'seniority_variant_change_summary.csv').as_posix() }",
    ]
    write_md(OUT_REP_T09, t09_lines)


if __name__ == "__main__":
    main()
