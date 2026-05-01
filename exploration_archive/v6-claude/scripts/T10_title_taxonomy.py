"""T10. Title taxonomy evolution.

Builds new/disappeared/shared title lists, computes title concentration,
AI compound-title shares, seniority marker trends, TF-IDF cosine of shared
titles between periods, and an emerging-role thematic grouping.

Primary filter: SWE LinkedIn is_english date_flag ok (via shared artifact).
Title source: unified.parquet `title_normalized` joined to shared cleaned text.
Description text for cosine: `text_source='llm'` rows only.
Essential sensitivities: (a) aggregator exclusion.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES = ROOT / "exploration" / "tables" / "T10"
OUT_FIGS = ROOT / "exploration" / "figures" / "T10"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

UNIFIED = ROOT / "data" / "unified.parquet"
CLEANED = SHARED / "swe_cleaned_text.parquet"

# ---------- Load title + metadata via DuckDB ----------
con = duckdb.connect()
q = f"""
SELECT
  uid,
  title_normalized,
  title,
  CASE WHEN period LIKE '2024%' THEN '2024' ELSE '2026' END AS period_group,
  source,
  seniority_final,
  seniority_3level,
  is_aggregator,
  company_name_canonical,
  yoe_extracted
FROM read_parquet('{UNIFIED}')
WHERE is_swe = true
  AND source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
"""
meta = con.execute(q).df()
# sanity check against shared artifact count
assert len(meta) == 63701, f"expected 63701, got {len(meta)}"

# Normalize null titles
meta["title_normalized"] = meta["title_normalized"].fillna("").str.lower().str.strip()
meta = meta[meta["title_normalized"] != ""].copy()
print(f"rows with title_normalized: {len(meta)}")
print(meta["period_group"].value_counts())

# ---------- 1. Title vocabulary comparison ----------
def title_set(df: pd.DataFrame, period: str, min_count: int = 1) -> pd.Series:
    sub = df[df["period_group"] == period]
    return sub["title_normalized"].value_counts()

titles_2024 = title_set(meta, "2024")
titles_2026 = title_set(meta, "2026")
set_2024 = set(titles_2024.index)
set_2026 = set(titles_2026.index)

new_titles = titles_2026[titles_2026.index.isin(set_2026 - set_2024)].reset_index()
new_titles.columns = ["title_normalized", "count_2026"]
new_titles = new_titles.sort_values("count_2026", ascending=False)

disappeared = titles_2024[titles_2024.index.isin(set_2024 - set_2026)].reset_index()
disappeared.columns = ["title_normalized", "count_2024"]
disappeared = disappeared.sort_values("count_2024", ascending=False)

shared_titles_df = pd.DataFrame({
    "title_normalized": list(set_2024 & set_2026),
})
shared_titles_df["count_2024"] = shared_titles_df["title_normalized"].map(titles_2024)
shared_titles_df["count_2026"] = shared_titles_df["title_normalized"].map(titles_2026)
shared_titles_df = shared_titles_df.sort_values("count_2026", ascending=False)

new_titles.head(100).to_csv(OUT_TABLES / "new_titles_top100.csv", index=False)
disappeared.head(100).to_csv(OUT_TABLES / "disappeared_titles_top100.csv", index=False)
shared_titles_df.head(200).to_csv(OUT_TABLES / "shared_titles_top200.csv", index=False)

print(f"\nNew titles (in 2026 but not 2024): {len(new_titles)} (of {len(set_2026)} 2026 uniques)")
print(f"Disappeared titles (in 2024 but not 2026): {len(disappeared)}")
print(f"Shared titles: {len(shared_titles_df)}")

print("\nTop 20 NEW 2026 titles:")
print(new_titles.head(20).to_string(index=False))

print("\nTop 20 DISAPPEARED 2024 titles:")
print(disappeared.head(20).to_string(index=False))

# Sensitivity (a): aggregator exclusion on new/disappeared lists
meta_na = meta[~meta["is_aggregator"].fillna(False)].copy()
t24_na = meta_na[meta_na["period_group"] == "2024"]["title_normalized"].value_counts()
t26_na = meta_na[meta_na["period_group"] == "2026"]["title_normalized"].value_counts()
new_titles_na = t26_na[~t26_na.index.isin(set(t24_na.index))].reset_index()
new_titles_na.columns = ["title_normalized", "count_2026_no_agg"]
new_titles_na = new_titles_na.sort_values("count_2026_no_agg", ascending=False)
new_titles_na.head(100).to_csv(OUT_TABLES / "new_titles_top100_no_aggregator.csv", index=False)

# ---------- 2. Title concentration ----------
def concentration_stats(df: pd.DataFrame, period: str) -> dict:
    sub = df[df["period_group"] == period]
    n = len(sub)
    uniq = sub["title_normalized"].nunique()
    vc = sub["title_normalized"].value_counts()
    hhi = float(((vc / n) ** 2).sum())
    top10_share = float(vc.head(10).sum() / n)
    top50_share = float(vc.head(50).sum() / n)
    return {
        "period": period,
        "n_postings": n,
        "n_unique_titles": uniq,
        "unique_per_1000_postings": uniq / n * 1000,
        "hhi": hhi,
        "top10_share": top10_share,
        "top50_share": top50_share,
    }

conc = pd.DataFrame([concentration_stats(meta, "2024"), concentration_stats(meta, "2026")])
conc.to_csv(OUT_TABLES / "title_concentration.csv", index=False)
print("\nTitle concentration:")
print(conc.to_string(index=False))

# ---------- 3. Compound/hybrid titles with AI terms ----------
AI_PATTERNS = {
    "ai_generic": r"\b(ai|a\.i\.)\b",
    "ml_ml_engineer": r"\b(ml|machine learning)\b",
    "llm": r"\bllm(s)?\b",
    "agent": r"\bagent(s|ic)?\b",
    "data": r"\bdata\b",
    "genai": r"\b(gen ?ai|generative ai)\b",
    "nlp": r"\bnlp\b",
    "deep_learning": r"\bdeep learning\b",
}
ALL_AI_RE = re.compile("|".join(f"({p})" for p in AI_PATTERNS.values()), flags=re.IGNORECASE)

def tag_ai(t: str) -> str | None:
    hits = [name for name, pat in AI_PATTERNS.items() if re.search(pat, t, flags=re.IGNORECASE)]
    return ",".join(hits) if hits else None

meta["ai_tag"] = meta["title_normalized"].apply(tag_ai)
ai_share = (
    meta.assign(has_ai=meta["ai_tag"].notna())
    .groupby("period_group")["has_ai"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_ai_titles", "count": "n_total", "mean": "share"})
)
ai_share.to_csv(OUT_TABLES / "ai_compound_title_share.csv")
print("\nAI compound title share:")
print(ai_share)

# per-pattern breakdown
rows = []
for name, pat in AI_PATTERNS.items():
    p24 = meta[meta["period_group"] == "2024"]["title_normalized"].str.contains(pat, case=False, regex=True).mean()
    p26 = meta[meta["period_group"] == "2026"]["title_normalized"].str.contains(pat, case=False, regex=True).mean()
    rows.append({"pattern": name, "regex": pat, "share_2024": p24, "share_2026": p26, "ratio": p26 / p24 if p24 > 0 else np.inf})
ai_breakdown = pd.DataFrame(rows).sort_values("share_2026", ascending=False)
ai_breakdown.to_csv(OUT_TABLES / "ai_compound_title_patterns.csv", index=False)
print(ai_breakdown.to_string(index=False))

# Top AI-tagged new titles
ai_new = new_titles[new_titles["title_normalized"].apply(lambda t: tag_ai(t) is not None)].head(25)
ai_new.to_csv(OUT_TABLES / "new_ai_titles.csv", index=False)
print("\nTop new AI-tagged 2026 titles:")
print(ai_new.to_string(index=False))

# ---------- 4. Title-to-content alignment (cosine on top shared titles) ----------
# Need description text for each row. Load from shared cleaned text and filter to text_source='llm'.
cleaned = pq.read_table(CLEANED, columns=["uid", "description_cleaned", "text_source"]).to_pandas()
text_llm = cleaned[cleaned["text_source"] == "llm"].copy()
print(f"\ncleaned text_source='llm' rows: {len(text_llm)}")

meta_txt = meta.merge(text_llm, on="uid", how="inner")
print(f"meta joined with text: {len(meta_txt)}")

# Top 10 shared titles (both periods have at least 20 text-labeled rows)
shared_with_text = (
    meta_txt.groupby(["title_normalized", "period_group"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={"2024": "n_2024", "2026": "n_2026"})
)
shared_with_text["total"] = shared_with_text.get("n_2024", 0) + shared_with_text.get("n_2026", 0)
shared_with_text = shared_with_text[(shared_with_text["n_2024"] >= 20) & (shared_with_text["n_2026"] >= 20)]
shared_with_text = shared_with_text.sort_values("total", ascending=False)
top10 = shared_with_text.head(10).index.tolist()
print(f"\nTop 10 shared titles with >=20 text-labeled in each period:")
for t in top10:
    print(f"  {t}: 2024={shared_with_text.loc[t,'n_2024']}, 2026={shared_with_text.loc[t,'n_2026']}")

cosine_rows = []
for t in top10:
    d24 = meta_txt[(meta_txt["title_normalized"] == t) & (meta_txt["period_group"] == "2024")]["description_cleaned"].dropna().tolist()
    d26 = meta_txt[(meta_txt["title_normalized"] == t) & (meta_txt["period_group"] == "2026")]["description_cleaned"].dropna().tolist()
    if len(d24) < 20 or len(d26) < 20:
        continue
    docs = d24 + d26
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), sublinear_tf=True)
    M = vec.fit_transform(docs)
    v24 = np.asarray(M[: len(d24)].mean(axis=0)).ravel()
    v26 = np.asarray(M[len(d24) :].mean(axis=0)).ravel()
    denom = (np.linalg.norm(v24) * np.linalg.norm(v26)) or 1.0
    cos = float(v24 @ v26 / denom)
    cosine_rows.append({
        "title_normalized": t,
        "n_2024": len(d24),
        "n_2026": len(d26),
        "cosine_similarity": cos,
    })

cos_df = pd.DataFrame(cosine_rows).sort_values("cosine_similarity")
cos_df.to_csv(OUT_TABLES / "top10_shared_title_cosine.csv", index=False)
print("\nTop 10 shared titles TF-IDF cosine 2024↔2026 (lowest = most drift):")
print(cos_df.to_string(index=False))

# ---------- 5. Seniority marker shares in RAW title ----------
# NOTE: title_normalized has seniority prefixes stripped by pipeline, so we use the raw
# `title` column for marker detection. Shares are therefore slightly non-orthogonal
# across markers (one title may contain multiple).
SENIORITY_MARKERS = {
    "junior": r"\bjunior\b|\bjr\.?\b",
    "associate": r"\bassociate\b",
    "entry_level": r"\bentry[- ]level\b|\bentry\b",
    "mid_ii_iii": r"\b(ii|iii|mid[- ]level|mid)\b",
    "senior": r"\bsenior\b|\bsr\.?\b",
    "staff": r"\bstaff\b",
    "principal": r"\bprincipal\b",
    "lead": r"\blead\b",
    "architect": r"\barchitect\b",
    "manager": r"\bmanager\b|\bmgr\.?\b",
    "director": r"\bdirector\b",
    "head_of": r"\bhead of\b",
}
meta["title_raw_lower"] = meta["title"].fillna("").str.lower()
rows = []
for name, pat in SENIORITY_MARKERS.items():
    s24 = meta[meta["period_group"] == "2024"]["title_raw_lower"].str.contains(pat, case=False, regex=True).mean()
    s26 = meta[meta["period_group"] == "2026"]["title_raw_lower"].str.contains(pat, case=False, regex=True).mean()
    rows.append({"marker": name, "regex": pat, "share_2024": s24, "share_2026": s26, "delta_pp": (s26 - s24) * 100})
sen_markers = pd.DataFrame(rows).sort_values("delta_pp", ascending=False)
sen_markers.to_csv(OUT_TABLES / "title_seniority_markers.csv", index=False)
print("\nTitle seniority markers:")
print(sen_markers.to_string(index=False))

# ---------- 6. Emerging role categories (thematic grouping of new titles) ----------
THEMES = {
    "ai_ml_engineering": r"\b(ai|ml|llm|agent|genai|generative|machine learning|nlp|deep learning|applied ai)\b",
    "platform_infra": r"\b(platform|infrastructure|devops|site reliability|sre|cloud)\b",
    "data_eng_analytics": r"\b(data engineer|analytics|analyst|bi|business intelligence|data scientist)\b",
    "security": r"\b(security|appsec|infosec|cyber)\b",
    "frontend": r"\b(frontend|front-end|ui|ux|react|angular|vue)\b",
    "backend": r"\b(backend|back-end|api|server|microservic)\b",
    "fullstack": r"\b(full[- ]?stack|fullstack)\b",
    "mobile": r"\b(mobile|ios|android|swift|kotlin)\b",
    "embedded_systems": r"\b(embedded|firmware|hardware|systems|kernel|driver)\b",
    "qa_test": r"\b(qa|quality assurance|sdet|test|automation engineer)\b",
    "game_dev": r"\b(game|unity|unreal|gameplay)\b",
    "blockchain": r"\b(blockchain|crypto|web3|solidity)\b",
}
def theme_of(title: str) -> str:
    for name, pat in THEMES.items():
        if re.search(pat, title, flags=re.IGNORECASE):
            return name
    return "other"

new_titles["theme"] = new_titles["title_normalized"].apply(theme_of)
theme_summary = (
    new_titles.groupby("theme")
    .agg(n_new_titles=("title_normalized", "count"), postings_2026=("count_2026", "sum"))
    .sort_values("postings_2026", ascending=False)
)
theme_summary.to_csv(OUT_TABLES / "new_title_themes.csv")
print("\nNew 2026 title themes (by postings contributed):")
print(theme_summary)

# Example top 5 new titles per theme
theme_examples = {}
for t in theme_summary.index:
    examples = new_titles[new_titles["theme"] == t].head(5)["title_normalized"].tolist()
    theme_examples[t] = examples
import json
(OUT_TABLES / "new_title_theme_examples.json").write_text(json.dumps(theme_examples, indent=2))

# ---------- 7. Figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: seniority marker share
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(sen_markers))
w = 0.38
ax.bar(x - w / 2, sen_markers["share_2024"] * 100, w, label="2024")
ax.bar(x + w / 2, sen_markers["share_2026"] * 100, w, label="2026")
ax.set_xticks(x)
ax.set_xticklabels(sen_markers["marker"], rotation=45, ha="right")
ax.set_ylabel("Share of postings (%)")
ax.set_title("Title seniority marker share, 2024 vs 2026")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_FIGS / "title_seniority_markers.png", dpi=150)
plt.close()

# Figure 2: AI compound share
fig, ax = plt.subplots(figsize=(7, 4))
ai_share[["share"]].plot(kind="bar", ax=ax, legend=False)
ax.set_ylabel("Share of postings")
ax.set_title("Titles containing AI/ML/LLM/agent/data terms")
plt.tight_layout()
plt.savefig(OUT_FIGS / "ai_compound_title_share.png", dpi=150)
plt.close()

# Figure 3: top-10 shared title cosine
if len(cos_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cos_df.plot.barh(x="title_normalized", y="cosine_similarity", ax=ax, legend=False)
    ax.set_xlabel("TF-IDF cosine (2024 vs 2026)")
    ax.set_title("Title-to-content drift: top 10 shared titles")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "top10_title_cosine.png", dpi=150)
    plt.close()

# Figure 4: new title themes (postings contributed)
fig, ax = plt.subplots(figsize=(8, 4.5))
theme_summary.head(12)["postings_2026"].plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("2026 postings contributed by new titles in theme")
ax.set_title("Emerging title themes (new-in-2026 titles only)")
plt.tight_layout()
plt.savefig(OUT_FIGS / "new_title_themes.png", dpi=150)
plt.close()

print("\nT10 done.")
