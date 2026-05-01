#!/usr/bin/env python3
"""T10 title taxonomy evolution.

Memory posture:
- reads only default-filtered LinkedIn SWE title/metadata columns from unified
- uses shared LLM-cleaned text for the bounded same-title TF-IDF check
- DuckDB is capped at 4GB / 1 thread
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "exploration" / "tables" / "T10"
FIG_DIR = ROOT / "exploration" / "figures" / "T10"
SHARED_DIR = ROOT / "exploration" / "artifacts" / "shared"

UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED_TEXT = SHARED_DIR / "swe_cleaned_text.parquet"
COMPANY_STOPLIST = SHARED_DIR / "company_stoplist.txt"
T30_PANEL = SHARED_DIR / "seniority_definition_panel.csv"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


TITLE_PARENS_RE = re.compile(
    r"\s*[\(\[][^)\]]*\b(?:remote|hybrid|on[- ]?site|wfh|contract|temporary)\b[^)\]]*[\)\]]\s*",
    re.I,
)
TITLE_SUFFIX_RE = re.compile(
    r"\s+(?:-|\u2013|\u2014|\||/)\s*(?:remote|hybrid|on[- ]?site|wfh|contract|temporary|"
    r"united states|usa|us|full[- ]time|part[- ]time).*$",
    re.I,
)
PUNCT_RE = re.compile(r"[^a-z0-9+#./ ]+")
SPACE_RE = re.compile(r"\s+")


def normalize_title(title: object) -> str:
    """Normalize raw titles while preserving seniority markers."""
    if title is None or (isinstance(title, float) and math.isnan(title)):
        return ""
    text = str(title).lower()
    text = text.replace("&amp;", "&").replace("\u2013", "-").replace("\u2014", "-")
    text = TITLE_PARENS_RE.sub(" ", text)
    text = TITLE_SUFFIX_RE.sub("", text)
    text = re.sub(r"\bsr\.?\b", "senior", text)
    text = re.sub(r"\bjr\.?\b", "junior", text)
    text = re.sub(r"\bs/w\b", "software", text)
    text = PUNCT_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def stable_rank(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


SENIOR_TITLE_RE = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|architect|distinguished)\b", re.I
)


def add_panel_flags(df: pd.DataFrame) -> pd.DataFrame:
    title = df["title"].fillna("")
    yoe = pd.to_numeric(df["yoe_extracted"], errors="coerce")
    seniority = df["seniority_final"].fillna("unknown")
    out = df.copy()
    out["J1"] = seniority.eq("entry")
    out["J2"] = seniority.isin(["entry", "associate"])
    out["J3"] = yoe.le(2)
    out["J4"] = yoe.le(3)
    out["S1"] = seniority.isin(["mid-senior", "director"])
    out["S2"] = seniority.eq("director")
    out["S3"] = title.map(lambda x: bool(SENIOR_TITLE_RE.search(str(x))))
    out["S4"] = yoe.ge(5)
    out["yoe_known"] = yoe.notna()
    out["seniority_known"] = ~seniority.eq("unknown")
    return out


def load_title_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = f"""
        SELECT
            uid,
            source,
            period,
            title,
            seniority_final,
            yoe_extracted,
            is_aggregator,
            company_name_canonical,
            swe_classification_tier
        FROM read_parquet('{UNIFIED.as_posix()}')
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
    """
    df = con.execute(query).fetchdf()
    df["year"] = df["period"].astype(str).str.slice(0, 4).astype(int)
    df["title_key"] = df["title"].map(normalize_title)
    df["title_key"] = df["title_key"].replace("", "(missing title)")
    df["source_group"] = np.select(
        [
            df["source"].eq("kaggle_arshkon"),
            df["source"].eq("kaggle_asaniczka"),
            df["source"].eq("scraped"),
        ],
        ["arshkon", "asaniczka", "scraped_2026"],
        default=df["source"],
    )
    df = add_panel_flags(df)
    return df


def most_common_title_display(df: pd.DataFrame) -> pd.DataFrame:
    display = (
        df.groupby(["title_key", "title"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["title_key", "n", "title"], ascending=[True, False, True])
        .drop_duplicates("title_key")
        .rename(columns={"title": "title_display"})
    )
    return display[["title_key", "title_display"]]


def title_counts(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby(["source_group", "year", "title_key"], dropna=False)
        .agg(n=("uid", "count"), aggregator_n=("is_aggregator", "sum"))
        .reset_index()
    )
    display = most_common_title_display(df)
    base = base.merge(display, on="title_key", how="left")
    base.to_csv(TABLE_DIR / "title_counts_by_source_year.csv", index=False)
    return base


def new_disappeared_titles(counts: pd.DataFrame) -> None:
    pivot = (
        counts.pivot_table(
            index="title_key",
            columns="source_group",
            values="n",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for col in ["arshkon", "asaniczka", "scraped_2026"]:
        if col not in pivot:
            pivot[col] = 0
    pivot["pooled_2024"] = pivot["arshkon"] + pivot["asaniczka"]
    display = counts[["title_key", "title_display"]].drop_duplicates("title_key")
    pivot = pivot.merge(display, on="title_key", how="left")
    cols = [
        "title_key",
        "title_display",
        "arshkon",
        "asaniczka",
        "pooled_2024",
        "scraped_2026",
    ]
    new = pivot.loc[(pivot["scraped_2026"] >= 3) & (pivot["arshkon"] == 0), cols].copy()
    new["new_vs_pooled_2024"] = new["pooled_2024"].eq(0)
    new.sort_values(["scraped_2026", "pooled_2024", "title_key"], ascending=[False, True, True]).head(
        200
    ).to_csv(TABLE_DIR / "new_2026_titles_vs_arshkon_top200.csv", index=False)

    disappeared = pivot.loc[
        (pivot["arshkon"] >= 2) & (pivot["scraped_2026"] == 0), cols
    ].copy()
    disappeared["also_absent_from_asaniczka"] = disappeared["asaniczka"].eq(0)
    disappeared.sort_values(["arshkon", "asaniczka", "title_key"], ascending=[False, True, True]).head(
        200
    ).to_csv(TABLE_DIR / "disappeared_2024_titles_vs_scraped_top200.csv", index=False)


def concentration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("all_rows", df),
        ("no_aggregators", df.loc[~df["is_aggregator"].fillna(False)]),
    ]
    groups = {
        "arshkon": lambda x: x["source_group"].eq("arshkon"),
        "asaniczka": lambda x: x["source_group"].eq("asaniczka"),
        "pooled_2024": lambda x: x["year"].eq(2024),
        "scraped_2026": lambda x: x["source_group"].eq("scraped_2026"),
    }
    for spec_name, data in specs:
        for source_group, mask_fn in groups.items():
            part = data.loc[mask_fn(data)]
            n = len(part)
            counts = part["title_key"].value_counts()
            if n == 0 or counts.empty:
                continue
            shares = counts / n
            rows.append(
                {
                    "spec": spec_name,
                    "source_group": source_group,
                    "rows": n,
                    "unique_titles": int(counts.size),
                    "unique_titles_per_1000": counts.size / n * 1000,
                    "singleton_title_share": float((counts == 1).sum() / counts.size),
                    "top10_title_share": float(shares.head(10).sum()),
                    "hhi": float((shares**2).sum()),
                    "shannon_entropy": float(-(shares * np.log(shares)).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "title_concentration.csv", index=False)
    return out


TITLE_HYBRID_PATTERNS = {
    "ai_general": re.compile(r"\b(?:ai|a\.i\.|artificial intelligence|genai|generative ai)\b", re.I),
    "ml": re.compile(r"\b(?:ml|machine learning|deep learning|nlp|computer vision|mlops)\b", re.I),
    "data": re.compile(r"\bdata\b", re.I),
    "llm": re.compile(r"\b(?:llm|llms|large language model|large language models|rag)\b", re.I),
    "agent": re.compile(r"\b(?:agent|agents|agentic)\b", re.I),
}

TITLE_MARKER_PATTERNS = {
    "senior": re.compile(r"\b(?:senior|sr\.?)\b", re.I),
    "staff": re.compile(r"\bstaff\b", re.I),
    "principal": re.compile(r"\bprincipal\b", re.I),
    "lead": re.compile(r"\blead\b", re.I),
    "architect": re.compile(r"\barchitect\b", re.I),
    "director_manager": re.compile(r"\b(?:director|manager|head|vp|vice president)\b", re.I),
    "junior_entry": re.compile(r"\b(?:junior|jr\.?|entry[- ]level|new[- ]grad|graduate|intern)\b", re.I),
    "associate": re.compile(r"\bassociate\b", re.I),
}


def title_pattern_prevalence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_hybrid = []
    rows_marker = []
    specs = [
        ("all_rows", df),
        ("no_aggregators", df.loc[~df["is_aggregator"].fillna(False)]),
    ]
    groups = {
        "arshkon": lambda x: x["source_group"].eq("arshkon"),
        "asaniczka": lambda x: x["source_group"].eq("asaniczka"),
        "pooled_2024": lambda x: x["year"].eq(2024),
        "scraped_2026": lambda x: x["source_group"].eq("scraped_2026"),
    }
    for spec_name, data in specs:
        title_text = data["title"].fillna("")
        hybrid_any = pd.Series(False, index=data.index)
        marker_flags = {}
        hybrid_flags = {}
        for name, pat in TITLE_HYBRID_PATTERNS.items():
            flag = title_text.map(lambda x, p=pat: bool(p.search(str(x))))
            hybrid_any |= flag
            hybrid_flags[name] = flag
        for name, pat in TITLE_MARKER_PATTERNS.items():
            marker_flags[name] = title_text.map(lambda x, p=pat: bool(p.search(str(x))))

        for source_group, mask_fn in groups.items():
            mask = mask_fn(data)
            denom = int(mask.sum())
            if denom == 0:
                continue
            row = {"spec": spec_name, "source_group": source_group, "denominator": denom}
            row["hybrid_any_n"] = int(hybrid_any.loc[mask].sum())
            row["hybrid_any_share"] = row["hybrid_any_n"] / denom
            for name, flag in hybrid_flags.items():
                row[f"{name}_share"] = float(flag.loc[mask].mean())
                row[f"{name}_n"] = int(flag.loc[mask].sum())
            rows_hybrid.append(row)

            for name, flag in marker_flags.items():
                rows_marker.append(
                    {
                        "spec": spec_name,
                        "source_group": source_group,
                        "marker": name,
                        "n": int(flag.loc[mask].sum()),
                        "denominator": denom,
                        "share": float(flag.loc[mask].mean()),
                    }
                )
    hybrid = pd.DataFrame(rows_hybrid)
    marker = pd.DataFrame(rows_marker)
    hybrid.to_csv(TABLE_DIR / "hybrid_ai_ml_data_title_prevalence.csv", index=False)
    marker.to_csv(TABLE_DIR / "raw_title_seniority_marker_prevalence.csv", index=False)
    return hybrid, marker


def shared_title_seniority_panel(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.loc[df["source_group"].isin(["arshkon", "scraped_2026"])]
        .groupby(["title_key", "source_group"])
        .size()
        .unstack(fill_value=0)
    )
    eligible = counts.loc[(counts.get("arshkon", 0) >= 10) & (counts.get("scraped_2026", 0) >= 10)].copy()
    eligible["min_count"] = eligible[["arshkon", "scraped_2026"]].min(axis=1)
    top_titles = eligible.sort_values("min_count", ascending=False).head(50).index.tolist()

    definition_meta = {
        "J1": ("junior", "seniority_final = entry", "seniority_known"),
        "J2": ("junior", "seniority_final in entry/associate", "seniority_known"),
        "J3": ("junior", "yoe_extracted <= 2", "yoe_known"),
        "J4": ("junior", "yoe_extracted <= 3", "yoe_known"),
        "S1": ("senior", "seniority_final in mid-senior/director", "seniority_known"),
        "S2": ("senior", "seniority_final = director", "seniority_known"),
        "S3": ("senior", "raw title senior regex", "all_title_rows"),
        "S4": ("senior", "yoe_extracted >= 5", "yoe_known"),
    }
    rows = []
    for title_key in top_titles:
        for definition, (side, label, denom_basis) in definition_meta.items():
            shares = {}
            ns = {}
            denoms = {}
            for source_group in ["arshkon", "scraped_2026"]:
                part = df.loc[(df["title_key"].eq(title_key)) & (df["source_group"].eq(source_group))]
                if denom_basis == "seniority_known":
                    denom_mask = part["seniority_known"]
                elif denom_basis == "yoe_known":
                    denom_mask = part["yoe_known"]
                else:
                    denom_mask = pd.Series(True, index=part.index)
                denom = int(denom_mask.sum())
                n = int((part[definition] & denom_mask).sum())
                shares[source_group] = n / denom if denom else np.nan
                ns[source_group] = n
                denoms[source_group] = denom
            rows.append(
                {
                    "title_key": title_key,
                    "definition": definition,
                    "side": side,
                    "definition_label": label,
                    "arshkon_n": ns["arshkon"],
                    "arshkon_denominator": denoms["arshkon"],
                    "arshkon_share": shares["arshkon"],
                    "scraped_2026_n": ns["scraped_2026"],
                    "scraped_2026_denominator": denoms["scraped_2026"],
                    "scraped_2026_share": shares["scraped_2026"],
                    "delta_scraped_minus_arshkon": shares["scraped_2026"] - shares["arshkon"],
                }
            )
    out = pd.DataFrame(rows).merge(most_common_title_display(df), on="title_key", how="left")
    out.to_csv(TABLE_DIR / "shared_title_seniority_panel.csv", index=False)
    return out


EMERGING_CATEGORY_PATTERNS = [
    ("AI/ML/Data", re.compile(r"\b(ai|artificial intelligence|genai|ml|machine learning|deep learning|data|llm|rag|nlp|computer vision|mle)\b", re.I)),
    ("Platform/Infrastructure", re.compile(r"\b(platform|infrastructure|cloud|kubernetes|systems engineer|site reliability|sre|devops|reliability)\b", re.I)),
    ("Security", re.compile(r"\b(security|devsecops|application security|cyber)\b", re.I)),
    ("Mobile", re.compile(r"\b(ios|android|mobile)\b", re.I)),
    ("Embedded/Firmware", re.compile(r"\b(embedded|firmware|robotics|autonomous|controls)\b", re.I)),
    ("Full-stack/Web", re.compile(r"\b(full[- ]stack|frontend|front[- ]end|web|react|javascript|typescript)\b", re.I)),
    ("Backend/API", re.compile(r"\b(backend|back[- ]end|api|server|services)\b", re.I)),
    ("QA/Test", re.compile(r"\b(qa|quality|test|automation)\b", re.I)),
    ("Management/Leadership", re.compile(r"\b(manager|director|lead|principal|staff|architect)\b", re.I)),
]


def emerging_role_categories() -> pd.DataFrame:
    new_path = TABLE_DIR / "new_2026_titles_vs_arshkon_top200.csv"
    new_titles = pd.read_csv(new_path)
    rows = []
    for _, row in new_titles.iterrows():
        title = str(row["title_key"])
        category = "Other"
        for name, pat in EMERGING_CATEGORY_PATTERNS:
            if pat.search(title):
                category = name
                break
        rows.append({**row.to_dict(), "emerging_category": category})
    tagged = pd.DataFrame(rows)
    tagged.to_csv(TABLE_DIR / "new_2026_titles_tagged.csv", index=False)
    summary = (
        tagged.groupby("emerging_category", dropna=False)
        .agg(
            title_count=("title_key", "count"),
            scraped_2026_postings=("scraped_2026", "sum"),
            pooled_2024_postings=("pooled_2024", "sum"),
            examples=("title_display", lambda x: "; ".join(map(str, x.head(8)))),
        )
        .reset_index()
        .sort_values("scraped_2026_postings", ascending=False)
    )
    summary.to_csv(TABLE_DIR / "emerging_role_categories.csv", index=False)
    return summary


def load_stopwords() -> set[str]:
    if COMPANY_STOPLIST.exists():
        company_words = {line.strip() for line in COMPANY_STOPLIST.read_text().splitlines() if line.strip()}
    else:
        company_words = set()
    generic = {"ll", "job", "role", "work", "team", "teams", "experience", "skills"}
    return set(ENGLISH_STOP_WORDS) | company_words | generic


def same_title_similarity(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    text_query = f"""
        SELECT
            u.uid,
            u.source,
            u.period,
            u.title,
            t.description_cleaned,
            t.text_source,
            u.company_name_canonical
        FROM read_parquet('{UNIFIED.as_posix()}') u
        JOIN read_parquet('{CLEANED_TEXT.as_posix()}') t USING (uid)
        WHERE u.source_platform = 'linkedin'
          AND u.is_english = true
          AND u.date_flag = 'ok'
          AND u.is_swe = true
          AND u.source IN ('kaggle_arshkon', 'scraped')
          AND t.text_source = 'llm'
          AND length(coalesce(t.description_cleaned, '')) >= 100
    """
    text_df = con.execute(text_query).fetchdf()
    text_df["title_key"] = text_df["title"].map(normalize_title).replace("", "(missing title)")
    text_df["source_group"] = np.where(text_df["source"].eq("kaggle_arshkon"), "arshkon", "scraped_2026")

    counts = text_df.groupby(["title_key", "source_group"]).size().unstack(fill_value=0)
    counts = counts.loc[(counts.get("arshkon", 0) >= 20) & (counts.get("scraped_2026", 0) >= 20)]
    counts["min_count"] = counts[["arshkon", "scraped_2026"]].min(axis=1)
    top_titles = counts.sort_values("min_count", ascending=False).head(10).index.tolist()

    stopwords = load_stopwords()
    rows = []
    for title_key in top_titles:
        part = text_df.loc[text_df["title_key"].eq(title_key)].copy()
        part["rank"] = part["uid"].map(stable_rank)
        sampled = (
            part.sort_values("rank")
            .groupby("source_group", group_keys=False)
            .head(300)
            .reset_index(drop=True)
        )
        if sampled["source_group"].nunique() < 2:
            continue
        vectorizer = TfidfVectorizer(
            stop_words=list(stopwords),
            min_df=2,
            max_df=0.85,
            ngram_range=(1, 2),
            max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+#./-]{1,}\b",
        )
        matrix = vectorizer.fit_transform(sampled["description_cleaned"].fillna(""))
        centroids = {}
        for source_group in ["arshkon", "scraped_2026"]:
            idx = np.flatnonzero(sampled["source_group"].eq(source_group).to_numpy())
            centroids[source_group] = np.asarray(matrix[idx].mean(axis=0))
        sim = float(cosine_similarity(centroids["arshkon"], centroids["scraped_2026"])[0, 0])
        rows.append(
            {
                "title_key": title_key,
                "title_display": most_common_title_display(df.loc[df["title_key"].eq(title_key)])[
                    "title_display"
                ].iloc[0],
                "arshkon_labeled_text_n": int((part["source_group"] == "arshkon").sum()),
                "scraped_2026_labeled_text_n": int((part["source_group"] == "scraped_2026").sum()),
                "sampled_arshkon_n": int((sampled["source_group"] == "arshkon").sum()),
                "sampled_scraped_2026_n": int((sampled["source_group"] == "scraped_2026").sum()),
                "tfidf_centroid_cosine": sim,
            }
        )
    out = pd.DataFrame(rows).sort_values("tfidf_centroid_cosine")
    out.to_csv(TABLE_DIR / "same_title_tfidf_similarity.csv", index=False)
    return out


def write_figures(concentration: pd.DataFrame, hybrid: pd.DataFrame, markers: pd.DataFrame, similarity: pd.DataFrame) -> None:
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = concentration.loc[concentration["spec"].isin(["all_rows", "no_aggregators"])]
    for spec, data in plot.groupby("spec"):
        order = ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]
        data = data.set_index("source_group").reindex(order).reset_index()
        ax.plot(data["source_group"], data["unique_titles_per_1000"], marker="o", label=spec)
    ax.set_ylabel("Unique raw-title keys per 1,000 postings")
    ax.set_xlabel("")
    ax.set_title("Title Fragmentation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "title_concentration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    h = hybrid.loc[hybrid["spec"].eq("all_rows")].set_index("source_group")
    order = ["arshkon", "asaniczka", "pooled_2024", "scraped_2026"]
    h = h.reindex(order)
    ax.bar(h.index, h["hybrid_any_share"] * 100, color="#2f7f6f")
    ax.set_ylabel("Share of postings (%)")
    ax.set_title("AI / ML / Data / LLM / Agent Terms In Titles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hybrid_title_share.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    keep = ["senior", "staff", "principal", "lead", "junior_entry", "associate"]
    m = markers.loc[(markers["spec"].eq("all_rows")) & (markers["marker"].isin(keep))]
    pivot = m.pivot(index="marker", columns="source_group", values="share").reindex(keep)
    pivot[["arshkon", "scraped_2026"]].mul(100).plot(kind="bar", ax=ax, color=["#6278a5", "#c46f4f"])
    ax.set_ylabel("Share of postings (%)")
    ax.set_xlabel("")
    ax.set_title("Raw Title Seniority Marker Shares")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_title_marker_share.png", dpi=150)
    plt.close(fig)

    if not similarity.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        s = similarity.sort_values("tfidf_centroid_cosine")
        ax.barh(s["title_display"], s["tfidf_centroid_cosine"], color="#5f8fbd")
        ax.set_xlabel("2024 vs 2026 TF-IDF centroid cosine")
        ax.set_title("Same-Title Content Similarity")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "same_title_tfidf_similarity.png", dpi=150)
        plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Load the T30 panel for downstream traceability; row-level flags are defined
    # from the same published definitions because the panel CSV is aggregate-only.
    pd.read_csv(T30_PANEL).to_csv(TABLE_DIR / "t30_panel_loaded_for_reference.csv", index=False)

    con = connect()
    df = load_title_frame(con)
    sample_counts = (
        df.groupby(["source_group", "source", "period", "year"], dropna=False)
        .agg(rows=("uid", "count"), aggregators=("is_aggregator", "sum"), unique_titles=("title_key", "nunique"))
        .reset_index()
    )
    sample_counts.to_csv(TABLE_DIR / "analysis_sample_counts.csv", index=False)

    counts = title_counts(df)
    new_disappeared_titles(counts)
    concentration = concentration_metrics(df)
    hybrid, markers = title_pattern_prevalence(df)
    shared_title_seniority_panel(df)
    emerging_role_categories()
    similarity = same_title_similarity(con, df)
    write_figures(concentration, hybrid, markers, similarity)
    print(f"Wrote T10 outputs under {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
