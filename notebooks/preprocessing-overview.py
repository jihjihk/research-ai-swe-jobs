# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessing pipeline overview
#
# This notebook documents the 8-stage preprocessing pipeline that transforms
# 1.57M raw job postings from four data sources into a unified, deduplicated,
# classified dataset of 1.2M rows (44,975 SWE postings).
#
# **Data sources:**
# - `kaggle_arshkon_2024` -- LinkedIn postings, Apr 2024 (124K rows)
# - `kaggle_asaniczka_2024` -- LinkedIn postings, Jan 2024 (1.35M rows)
# - `scraped_linkedin_2026` -- Our scraper, Mar 5-18 2026 (60K rows)
# - `scraped_indeed_2026` -- Our scraper, Mar 5-18 2026 (41K rows)
#
# **Pipeline stages:**
#
# | Stage | Name | Rows after | Key operation |
# |-------|------|------------|---------------|
# | 1 | Ingest & schema unification | 1,573,042 | Load CSVs, harmonize columns |
# | 2 | Aggregator flagging | 1,573,042 | Flag staffing agencies (Jobot, CyberCoders, etc.) |
# | 3 | Boilerplate removal | 1,573,042 | Extract `description_core` from full descriptions |
# | 4 | Deduplication | 1,203,817 | Exact + fuzzy dedup removes 369,225 rows (23.5%) |
# | 5 | SWE classification | 1,203,817 | 3-tier classifier: regex, embedding, unresolved |
# | 6 | Normalization | 1,203,817 | Company names, location, salary, remote, dates |
# | 7 | Temporal alignment | 1,203,817 | Period labels, scrape_week |
# | 8 | Quality flags | 1,203,817 | Ghost jobs, description quality, language, date range |
#
# All visualizations use streaming reads from the 6.5GB parquet file to stay
# within the 31GB RAM constraint.

# %%
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

# -- Paths ----------------------------------------------------------------
PROJECT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = PROJECT / "data" / "unified.parquet"
QUALITY = PROJECT / "data" / "quality_report.json"
FIG_DIR = PROJECT / "preprocessing"
FIG_DIR.mkdir(exist_ok=True)

# -- Style ----------------------------------------------------------------
SOURCE_COLORS = {
    "kaggle_arshkon_2024":    "#4C72B0",
    "kaggle_asaniczka_2024":  "#55A868",
    "scraped_linkedin_2026":  "#DD8452",
    "scraped_indeed_2026":    "#C44E52",
}
SOURCE_SHORT = {
    "kaggle_arshkon_2024":    "Arshkon\n(Apr 2024)",
    "kaggle_asaniczka_2024":  "Asaniczka\n(Jan 2024)",
    "scraped_linkedin_2026":  "LinkedIn\n(Mar 2026)",
    "scraped_indeed_2026":    "Indeed\n(Mar 2026)",
}
SOURCE_ORDER = list(SOURCE_COLORS.keys())

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 150

print("Setup complete.")

# %%
# -- Memory-safe parquet reader -------------------------------------------

def read_columns(path, columns, filter_col=None, filter_val=None, filter_func=None):
    """Read specific columns from parquet, optionally filtered.

    filter_col + filter_val: simple equality filter.
    filter_func: callable(df) -> bool mask for complex filters.
    """
    pf = pq.ParquetFile(path)
    chunks = []
    for batch in pf.iter_batches(batch_size=200_000, columns=columns):
        df = batch.to_pandas()
        if filter_col is not None and filter_val is not None:
            df = df[df[filter_col] == filter_val]
        if filter_func is not None:
            df = df[filter_func(df)]
        chunks.append(df)
        del batch
    result = pd.concat(chunks, ignore_index=True)
    gc.collect()
    return result


def aggregate_counts(path, group_cols, filter_col=None, filter_val=None):
    """Stream through parquet and accumulate value_counts for group_cols."""
    pf = pq.ParquetFile(path)
    acc = defaultdict(int)
    for batch in pf.iter_batches(batch_size=200_000, columns=group_cols +
                                 ([filter_col] if filter_col else [])):
        df = batch.to_pandas()
        if filter_col is not None and filter_val is not None:
            df = df[df[filter_col] == filter_val]
        if len(group_cols) == 1:
            for val, cnt in df[group_cols[0]].value_counts().items():
                acc[val] += cnt
        else:
            for vals, cnt in df.groupby(group_cols).size().items():
                acc[vals] += cnt
        del batch, df
    gc.collect()
    return dict(acc)


print("Utility functions defined.")

# %% [markdown]
# ---
# ## 1. Pipeline overview (from log data)
#
# The table below comes directly from `preprocessing_log.txt` and
# `quality_report.json`. No data loading required.

# %%
# Load the quality report
with open(QUALITY) as f:
    qr = json.load(f)

funnel = qr["funnel"]
print(f"Total raw rows:     {funnel['kaggle_arshkon_raw'] + funnel['kaggle_asaniczka_raw'] + funnel['scraped_raw']:>12,}")
print(f"After dedup:        {funnel['after_dedup']:>12,}")
print(f"Final SWE:          {funnel['final_swe']:>12,}")
print(f"Final control:      {funnel['final_control']:>12,}")

# %% [markdown]
# ---
# ## 2. Data funnel visualization

# %%
# Waterfall-style horizontal bar chart of the pipeline funnel
stages = [
    ("1. Ingest",               1_573_042),
    ("2. Aggregator flagging",  1_573_042),
    ("3. Boilerplate removal",  1_573_042),
    ("4. Deduplication",        1_203_817),
    ("5. SWE classification",   1_203_817),
    ("6. Normalization",        1_203_817),
    ("7. Temporal alignment",   1_203_817),
    ("8. Quality flags",        1_203_817),
]

fig, ax = plt.subplots(figsize=(12, 5))

labels = [s[0] for s in stages]
values = [s[1] for s in stages]
max_val = max(values)

# Draw bars
bars = ax.barh(labels[::-1], [v for v in values[::-1]], color="#4C72B0", edgecolor="white", height=0.65)

# Add the removed rows as a lighter bar for stage 4
removed = 1_573_042 - 1_203_817
dedup_idx = len(stages) - 1 - 3  # reversed index of stage 4
ax.barh(labels[::-1][dedup_idx], removed, left=values[3], color="#E8B4B4", edgecolor="white", height=0.65)
ax.text(values[3] + removed / 2, dedup_idx, f"-{removed:,}\nduplicates",
        ha="center", va="center", fontsize=8.5, color="#8B0000", fontweight="bold")

# Annotate row counts
for i, (bar, val) in enumerate(zip(bars, values[::-1])):
    ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)

ax.set_xlabel("Rows")
ax.set_title("Data funnel: rows at each pipeline stage")
ax.set_xlim(0, max_val * 1.18)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

plt.tight_layout()
fig.savefig(FIG_DIR / "fig01_data_funnel.png")
plt.show()
print("Saved: preprocessing/fig01_data_funnel.png")

# %% [markdown]
# ---
# ## 3. Source composition
#
# How many rows does each source contribute, before and after deduplication?

# %%
# Pre-dedup counts (from stage 1 ingest log)
pre_dedup = {
    "kaggle_arshkon_2024":   123_849,
    "kaggle_asaniczka_2024": 1_348_454,
    "scraped_linkedin_2026": 60_009,
    "scraped_indeed_2026":   40_730,
}

# Post-dedup counts (from stage 4 dedup log)
post_dedup = {
    "kaggle_arshkon_2024":   107_078,
    "kaggle_asaniczka_2024": 1_022_768,
    "scraped_linkedin_2026": 48_549,
    "scraped_indeed_2026":   25_422,
}

# SWE counts by source (from quality report)
swe_by_source = qr["swe_by_source"]

# --- 3a. Stacked bar chart: pre-dedup vs post-dedup ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})

x = np.arange(len(SOURCE_ORDER))
width = 0.35

pre_vals = [pre_dedup[s] for s in SOURCE_ORDER]
post_vals = [post_dedup[s] for s in SOURCE_ORDER]
colors = [SOURCE_COLORS[s] for s in SOURCE_ORDER]
short_labels = [SOURCE_SHORT[s] for s in SOURCE_ORDER]

bars_pre = ax1.bar(x - width / 2, pre_vals, width, color=colors, alpha=0.5,
                   edgecolor="white", label="Pre-dedup")
bars_post = ax1.bar(x + width / 2, post_vals, width, color=colors,
                    edgecolor="white", label="Post-dedup")

# Annotate removal percentages
for i, s in enumerate(SOURCE_ORDER):
    pct = (1 - post_dedup[s] / pre_dedup[s]) * 100
    ax1.text(x[i] + width / 2, post_vals[i] + max(pre_vals) * 0.02,
             f"-{pct:.0f}%", ha="center", va="bottom", fontsize=8, color="#8B0000")

ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, fontsize=9)
ax1.set_ylabel("Rows")
ax1.set_title("Rows by source: pre- vs post-deduplication")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e6:.1f}M" if y >= 1e6 else f"{y/1e3:.0f}K"))
ax1.legend(fontsize=9)

# --- 3b. Pie chart: SWE postings by source ---
swe_vals = [swe_by_source.get(s, 0) for s in SOURCE_ORDER]
swe_labels = [f"{SOURCE_SHORT[s].split(chr(10))[0]}\n{v:,} ({v/sum(swe_vals):.0%})" for s, v in zip(SOURCE_ORDER, swe_vals)]

wedges, texts = ax2.pie(swe_vals, labels=swe_labels, colors=colors,
                        startangle=90, textprops={"fontsize": 8.5})
ax2.set_title("SWE postings by source (n=44,975)")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig02_source_composition.png")
plt.show()
print("Saved: preprocessing/fig02_source_composition.png")

# %% [markdown]
# ---
# ## 4. SWE classification tiers
#
# The classifier runs in two tiers:
# - **Tier 1 (regex):** Pattern matching on normalized titles (141,514 rows matched)
# - **Tier 2 (embedding):** JobBERT-v2 cosine similarity > 0.70 for ambiguous titles (15,773 rows)
# - **Unresolved:** 1,046,530 rows -- mostly genuinely non-SWE titles

# %%
# Classification tier breakdown (from quality report)
cls = qr["classification_rates"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 4a. Overall tier breakdown ---
tiers = ["Regex (Tier 1)", "Embedding (Tier 2)", "Unresolved"]
tier_vals = [cls["swe_regex"], cls["swe_embedding"], cls["swe_unresolved"]]
tier_colors = ["#4C72B0", "#DD8452", "#CCCCCC"]

bars = ax1.barh(tiers[::-1], tier_vals[::-1], color=tier_colors[::-1], edgecolor="white", height=0.5)
for bar, val in zip(bars, tier_vals[::-1]):
    label = f"{val:,} ({val/sum(tier_vals):.1%})"
    ax1.text(val + sum(tier_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
             label, va="center", fontsize=9)
ax1.set_xlabel("Postings classified")
ax1.set_title("SWE classification tiers (all postings)")
ax1.set_xlim(0, max(tier_vals) * 1.25)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

# --- 4b. SWE by tier and source (from stage 5 log) ---
tier_by_source = {
    "kaggle_arshkon_2024":   {"Regex": 3043,  "Embedding": 2025},
    "kaggle_asaniczka_2024": {"Regex": 15770, "Embedding": 10641},
    "scraped_linkedin_2026": {"Regex": 7581,  "Embedding": 2388},
    "scraped_indeed_2026":   {"Regex": 2808,  "Embedding": 719},
}

x = np.arange(len(SOURCE_ORDER))
width = 0.35
regex_vals = [tier_by_source[s]["Regex"] for s in SOURCE_ORDER]
embed_vals = [tier_by_source[s]["Embedding"] for s in SOURCE_ORDER]
colors_src = [SOURCE_COLORS[s] for s in SOURCE_ORDER]

ax2.bar(x - width / 2, regex_vals, width, color=colors_src, alpha=0.7,
        edgecolor="white", label="Regex (Tier 1)")
ax2.bar(x + width / 2, embed_vals, width, color=colors_src,
        edgecolor="white", hatch="///", label="Embedding (Tier 2)")

ax2.set_xticks(x)
ax2.set_xticklabels([SOURCE_SHORT[s] for s in SOURCE_ORDER], fontsize=9)
ax2.set_ylabel("SWE postings")
ax2.set_title("SWE classifications by tier and source")
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e3:.0f}K" if y >= 1000 else str(int(y))))

plt.tight_layout()
fig.savefig(FIG_DIR / "fig03_swe_classification_tiers.png")
plt.show()
print("Saved: preprocessing/fig03_swe_classification_tiers.png")

# %%
# Sample titles caught by each tier (read a small batch from parquet)
print("=== Sample SWE titles by classification tier ===\n")

tier_samples = read_columns(
    PARQUET,
    columns=["title", "swe_classification_tier", "is_swe"],
    filter_col="is_swe",
    filter_val=True,
)

for tier_name in ["regex", "embedding"]:
    subset = tier_samples[tier_samples["swe_classification_tier"] == tier_name]
    unique_titles = subset["title"].value_counts().head(15)
    print(f"--- Tier: {tier_name} (top 15 most common titles) ---")
    for title, count in unique_titles.items():
        print(f"  {title} ({count:,})")
    print()

del tier_samples
gc.collect()

# %% [markdown]
# ---
# ## 5. Seniority distribution
#
# The pipeline creates `seniority_final` by preferring the native LinkedIn label
# (where available) and falling back to title/description imputation. This reduced
# the SWE unknown rate from 55.1% to 7.3%.

# %%
# Read seniority columns for SWE postings
sen_df = read_columns(
    PARQUET,
    columns=["source", "is_swe", "seniority_native", "seniority_imputed",
             "seniority_final", "seniority_source"],
    filter_col="is_swe",
    filter_val=True,
)

# --- 5a. Before/after comparison: imputed-only vs seniority_final ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

seniority_order = ["entry level", "internship", "associate", "mid-senior level", "director", "executive", "unknown"]
seniority_labels = ["Entry level", "Internship", "Associate", "Mid-senior", "Director", "Executive", "Unknown"]

# Before: seniority_imputed only
imputed_counts = sen_df["seniority_imputed"].value_counts()
imputed_vals = [imputed_counts.get(s, 0) for s in seniority_order]

# After: seniority_final (native preferred, imputed fallback)
final_counts = sen_df["seniority_final"].value_counts()
final_vals = [final_counts.get(s, 0) for s in seniority_order]

y = np.arange(len(seniority_order))
height = 0.35

ax1.barh(y, imputed_vals, height, color="#DD8452", edgecolor="white")
for i, v in enumerate(imputed_vals):
    pct = v / len(sen_df) * 100
    ax1.text(v + len(sen_df) * 0.01, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=8)
ax1.set_yticks(y)
ax1.set_yticklabels(seniority_labels, fontsize=9)
ax1.set_title("Imputed seniority only (SWE)")
ax1.set_xlabel("Postings")

ax2.barh(y, final_vals, height, color="#4C72B0", edgecolor="white")
for i, v in enumerate(final_vals):
    pct = v / len(sen_df) * 100
    ax2.text(v + len(sen_df) * 0.01, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=8)
ax2.set_title("Final seniority (native + imputed fallback)")
ax2.set_xlabel("Postings")

max_x = max(max(imputed_vals), max(final_vals)) * 1.3
ax1.set_xlim(0, max_x)
ax2.set_xlim(0, max_x)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

fig.suptitle("SWE seniority: imputed-only vs final (unknown rate: 55.1% -> 7.3%)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig04_seniority_before_after.png")
plt.show()
print("Saved: preprocessing/fig04_seniority_before_after.png")

# %%
# --- 5b. Seniority by source ---
fig, ax = plt.subplots(figsize=(12, 5))

cross = pd.crosstab(sen_df["source"], sen_df["seniority_final"])
# Reorder columns
plot_cols = [c for c in seniority_order if c in cross.columns]
cross = cross[plot_cols]
cross = cross.loc[[s for s in SOURCE_ORDER if s in cross.index]]

cross.plot(kind="bar", stacked=True, ax=ax,
           color=sns.color_palette("Set2", len(plot_cols)),
           edgecolor="white", width=0.7)

ax.set_xticklabels([SOURCE_SHORT[s] for s in cross.index], rotation=0, fontsize=9)
ax.set_ylabel("SWE postings")
ax.set_title("SWE seniority_final distribution by source")
ax.legend(title="Seniority", fontsize=8, title_fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e3:.0f}K" if y >= 1000 else str(int(y))))

plt.tight_layout()
fig.savefig(FIG_DIR / "fig05_seniority_by_source.png")
plt.show()
print("Saved: preprocessing/fig05_seniority_by_source.png")

del sen_df
gc.collect()

# %% [markdown]
# ---
# ## 6. Boilerplate removal impact
#
# Stage 3 extracts `description_core` from full descriptions by removing
# section boilerplate (EEO statements, company info blocks, etc.).
# The median reduction varies by source.

# %%
# Read description lengths for SWE postings
bp_df = read_columns(
    PARQUET,
    columns=["source", "is_swe", "description_length", "core_length", "boilerplate_flag"],
    filter_col="is_swe",
    filter_val=True,
)

bp_df["removal_pct"] = ((bp_df["description_length"] - bp_df["core_length"]) /
                         bp_df["description_length"].clip(lower=1) * 100)

# --- 6a. Box plots: full vs core length by source ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Melt for box plot
bp_melt = bp_df[["source", "description_length", "core_length"]].melt(
    id_vars=["source"], var_name="Measure", value_name="Characters"
)
bp_melt["Measure"] = bp_melt["Measure"].map({
    "description_length": "Full description",
    "core_length": "Core (boilerplate removed)",
})

order = [s for s in SOURCE_ORDER if s in bp_df["source"].unique()]
short_order = [SOURCE_SHORT[s] for s in order]
bp_melt["source_short"] = bp_melt["source"].map(SOURCE_SHORT)

sns.boxplot(data=bp_melt, x="source_short", y="Characters", hue="Measure",
            order=short_order, ax=ax1, showfliers=False, width=0.6)
ax1.set_xlabel("")
ax1.set_ylabel("Characters")
ax1.set_title("Description length: full vs core (SWE only)")
ax1.legend(fontsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e3:.0f}K" if y >= 1000 else str(int(y))))

# --- 6b. Median chars removed by source ---
medians = bp_df.groupby("source").agg(
    median_full=("description_length", "median"),
    median_core=("core_length", "median"),
).loc[order]
medians["removed"] = medians["median_full"] - medians["median_core"]

bars = ax2.bar([SOURCE_SHORT[s] for s in order], medians["removed"],
               color=[SOURCE_COLORS[s] for s in order], edgecolor="white", width=0.6)
for bar, val in zip(bars, medians["removed"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 20,
             f"{val:.0f}", ha="center", va="bottom", fontsize=9)

ax2.set_ylabel("Median characters removed")
ax2.set_title("Median boilerplate removed by source (SWE only)")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig06_boilerplate_removal.png")
plt.show()
print("Saved: preprocessing/fig06_boilerplate_removal.png")

# %%
# --- 6c. Distribution of removal percentages ---
fig, ax = plt.subplots(figsize=(10, 5))

# Filter to rows with valid descriptions (description_length > 0)
valid = bp_df[bp_df["description_length"] > 0].copy()

for src in order:
    subset = valid[valid["source"] == src]["removal_pct"]
    ax.hist(subset, bins=50, range=(0, 100), alpha=0.5,
            color=SOURCE_COLORS[src], label=SOURCE_SHORT[src].replace("\n", " "),
            density=True)

ax.set_xlabel("Boilerplate removal (%)")
ax.set_ylabel("Density")
ax.set_title("Distribution of boilerplate removal percentage (SWE only)")
ax.legend(fontsize=9)
ax.set_xlim(0, 100)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig06b_removal_pct_dist.png")
plt.show()
print("Saved: preprocessing/fig06b_removal_pct_dist.png")

del bp_df, bp_melt, valid
gc.collect()

# %% [markdown]
# ---
# ## 7. Description length comparison
#
# After boilerplate removal, how comparable are the core descriptions across sources?
# This is the key "apples to apples" check.

# %%
# Read core_length for SWE postings
dl_df = read_columns(
    PARQUET,
    columns=["source", "is_swe", "core_length"],
    filter_col="is_swe",
    filter_val=True,
)

fig, ax = plt.subplots(figsize=(10, 5))

for src in SOURCE_ORDER:
    subset = dl_df[dl_df["source"] == src]["core_length"].dropna()
    if len(subset) == 0:
        continue
    ax.hist(subset, bins=80, range=(0, 12000), alpha=0.45,
            color=SOURCE_COLORS[src],
            label=f"{SOURCE_SHORT[src].replace(chr(10), ' ')} (med={subset.median():.0f})",
            density=True)

ax.set_xlabel("Core description length (characters)")
ax.set_ylabel("Density")
ax.set_title("Distribution of core description length by source (SWE only)")
ax.legend(fontsize=9)
ax.set_xlim(0, 12000)
ax.axvline(x=2000, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")
ax.text(2050, ax.get_ylim()[1] * 0.9, "2K chars", fontsize=8, color="gray")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig07_core_length_comparison.png")
plt.show()
print("Saved: preprocessing/fig07_core_length_comparison.png")

# Print summary stats
print("\nCore description length summary (SWE only):")
print(dl_df.groupby("source")["core_length"].describe().loc[
    SOURCE_ORDER, ["count", "mean", "50%", "75%"]
].rename(columns={"50%": "median", "75%": "p75"}).to_string())

del dl_df
gc.collect()

# %% [markdown]
# ---
# ## 8. Quality flags summary
#
# Stage 8 adds several quality flags. The most important ones:
# - `ghost_job_risk`: HIGH / MEDIUM / LOW based on junior seniority plus YOE-based contradiction signals
# - `boilerplate_flag`: ok / under_removed / over_removed / empty_core
# - `description_quality_flag`: ok / empty / too_short / non_english

# %%
# Stream and count quality flags
qf_cols = ["source", "ghost_job_risk", "boilerplate_flag", "description_quality_flag"]
qf_counts = defaultdict(lambda: defaultdict(int))

pf = pq.ParquetFile(PARQUET)
for batch in pf.iter_batches(batch_size=200_000, columns=qf_cols):
    df = batch.to_pandas()
    for col in ["ghost_job_risk", "boilerplate_flag", "description_quality_flag"]:
        for (src, val), cnt in df.groupby(["source", col]).size().items():
            qf_counts[(col, src)][val] += cnt
    del batch, df
gc.collect()

# Reshape into DataFrames
flag_dfs = {}
for col in ["ghost_job_risk", "boilerplate_flag", "description_quality_flag"]:
    rows = []
    for src in SOURCE_ORDER:
        vals = qf_counts.get((col, src), {})
        for val, cnt in vals.items():
            rows.append({"source": src, "flag_value": str(val), "count": cnt})
    flag_dfs[col] = pd.DataFrame(rows)

# --- 8a. Ghost job risk ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (col, title) in zip(axes, [
    ("ghost_job_risk", "Ghost job risk"),
    ("boilerplate_flag", "Boilerplate flag"),
    ("description_quality_flag", "Description quality"),
]):
    df_flag = flag_dfs[col]
    if df_flag.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    pivot = df_flag.pivot_table(index="source", columns="flag_value", values="count", fill_value=0)
    pivot = pivot.loc[[s for s in SOURCE_ORDER if s in pivot.index]]
    pivot.index = [SOURCE_SHORT[s] for s in pivot.index]

    # Normalize to percentages per source
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    pivot_pct.plot(kind="bar", stacked=True, ax=ax, edgecolor="white",
                   colormap="Set2", width=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_ylabel("% of postings")
    ax.set_title(title)
    ax.legend(fontsize=7, title="Value", title_fontsize=8,
              bbox_to_anchor=(1.0, 1), loc="upper left")
    ax.set_ylim(0, 105)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig08_quality_flags.png")
plt.show()
print("Saved: preprocessing/fig08_quality_flags.png")

# Print absolute counts for ghost job risk
print("\nGhost job risk counts:")
ghost_df = flag_dfs["ghost_job_risk"]
ghost_pivot = ghost_df.pivot_table(index="source", columns="flag_value", values="count", fill_value=0)
ghost_pivot = ghost_pivot.loc[[s for s in SOURCE_ORDER if s in ghost_pivot.index]]
print(ghost_pivot.to_string())

# %% [markdown]
# ---
# ## 9. Temporal coverage
#
# The four data sources span different time periods:
# - **arshkon**: LinkedIn postings from ~Apr 2024
# - **asaniczka**: LinkedIn snapshot from Jan 12-17, 2024
# - **scraped_linkedin + scraped_indeed**: Daily scrapes, Mar 5-18, 2026

# %%
# Read date and source columns -- only need scrape_date and date_posted
temp_df = read_columns(
    PARQUET,
    columns=["source", "scrape_date", "date_posted", "period"],
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 2]})

# --- 9a. Timeline overview ---
source_periods = {
    "kaggle_arshkon_2024":   ("2024-04-01", "2024-04-30"),
    "kaggle_asaniczka_2024": ("2024-01-12", "2024-01-17"),
    "scraped_linkedin_2026": ("2026-03-05", "2026-03-18"),
    "scraped_indeed_2026":   ("2026-03-05", "2026-03-18"),
}

for i, src in enumerate(SOURCE_ORDER):
    start, end = source_periods[src]
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ax1.barh(i, (end_dt - start_dt).days + 1, left=start_dt.toordinal(),
             color=SOURCE_COLORS[src], height=0.6, edgecolor="white")
    ax1.text(start_dt.toordinal() + (end_dt - start_dt).days / 2, i,
             f"{start} to {end}", ha="center", va="center", fontsize=8, color="white",
             fontweight="bold")

ax1.set_yticks(range(len(SOURCE_ORDER)))
ax1.set_yticklabels([SOURCE_SHORT[s].replace("\n", " ") for s in SOURCE_ORDER], fontsize=9)
ax1.set_title("Data source time coverage")

# Format x-axis as dates
x_min = pd.to_datetime("2024-01-01").toordinal()
x_max = pd.to_datetime("2026-04-01").toordinal()
ax1.set_xlim(x_min, x_max)
tick_dates = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01",
                              "2025-01-01", "2025-07-01", "2026-01-01", "2026-03-18"])
ax1.set_xticks([d.toordinal() for d in tick_dates])
ax1.set_xticklabels([d.strftime("%b %Y") for d in tick_dates], fontsize=8, rotation=30)
ax1.grid(axis="x", alpha=0.3)

# --- 9b. Scrape date distribution for scraped data (daily counts) ---
scraped = temp_df[temp_df["source"].str.startswith("scraped_")].copy()
scraped["scrape_dt"] = pd.to_datetime(scraped["scrape_date"], errors="coerce")
scraped = scraped.dropna(subset=["scrape_dt"])

daily = scraped.groupby(["source", scraped["scrape_dt"].dt.date]).size().reset_index(name="count")
daily.columns = ["source", "date", "count"]

for src in ["scraped_linkedin_2026", "scraped_indeed_2026"]:
    subset = daily[daily["source"] == src].sort_values("date")
    if len(subset) > 0:
        ax2.plot(pd.to_datetime(subset["date"]), subset["count"],
                 marker="o", markersize=5, color=SOURCE_COLORS[src],
                 label=SOURCE_SHORT[src].replace("\n", " "), linewidth=2)
        ax2.fill_between(pd.to_datetime(subset["date"]), subset["count"],
                         alpha=0.15, color=SOURCE_COLORS[src])

ax2.set_xlabel("Scrape date")
ax2.set_ylabel("Postings collected")
ax2.set_title("Daily scrape volume (Mar 5-18, 2026)")
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y/1e3:.1f}K" if y >= 1000 else str(int(y))))
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig09_temporal_coverage.png")
plt.show()
print("Saved: preprocessing/fig09_temporal_coverage.png")

del temp_df, scraped, daily
gc.collect()

# %% [markdown]
# ---
# ## 10. Geographic distribution
#
# Top 15 US states by posting count, broken out by source. Also remote work rates.

# %%
# Read location columns
geo_df = read_columns(
    PARQUET,
    columns=["source", "state_normalized", "is_remote", "is_swe"],
)

# --- 10a. Top 15 states (all postings) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]})

state_counts = geo_df.groupby(["state_normalized", "source"]).size().reset_index(name="count")
top_states = (state_counts.groupby("state_normalized")["count"].sum()
              .sort_values(ascending=False).head(15).index.tolist())

# Filter and pivot
state_top = state_counts[state_counts["state_normalized"].isin(top_states)]
state_pivot = state_top.pivot_table(index="state_normalized", columns="source",
                                     values="count", fill_value=0)
state_pivot = state_pivot[[s for s in SOURCE_ORDER if s in state_pivot.columns]]
state_pivot = state_pivot.loc[top_states]

state_pivot.plot(kind="barh", stacked=True, ax=ax1,
                 color=[SOURCE_COLORS[s] for s in state_pivot.columns],
                 edgecolor="white", width=0.7)
ax1.set_xlabel("Postings")
ax1.set_title("Top 15 US states by posting count")
ax1.legend([SOURCE_SHORT[s].replace("\n", " ") for s in state_pivot.columns],
           fontsize=8, loc="lower right")
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))))
ax1.invert_yaxis()

# --- 10b. Remote work rate by source ---
remote_rates = geo_df.groupby("source")["is_remote"].mean() * 100
remote_rates = remote_rates.loc[[s for s in SOURCE_ORDER if s in remote_rates.index]]

bars = ax2.bar([SOURCE_SHORT[s] for s in remote_rates.index], remote_rates.values,
               color=[SOURCE_COLORS[s] for s in remote_rates.index],
               edgecolor="white", width=0.6)
for bar, val in zip(bars, remote_rates.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

ax2.set_ylabel("% remote postings")
ax2.set_title("Remote work rate by source")
ax2.set_ylim(0, max(remote_rates.values) * 1.25 if len(remote_rates) > 0 else 50)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig10_geographic_distribution.png")
plt.show()
print("Saved: preprocessing/fig10_geographic_distribution.png")

# Print SWE remote rates
print("\nRemote work rate (SWE only):")
swe_remote = geo_df[geo_df["is_swe"]].groupby("source")["is_remote"].mean() * 100
for src in SOURCE_ORDER:
    if src in swe_remote.index:
        print(f"  {src}: {swe_remote[src]:.1f}%")

del geo_df
gc.collect()

# %% [markdown]
# ---
# ## Summary
#
# The preprocessing pipeline transforms 1.57M raw postings into a clean dataset
# of 1.2M rows:
#
# - **Deduplication** removed 23.5% of rows (369K), mostly from asaniczka and indeed
# - **SWE classification** identified 44,975 SWE postings using regex + embedding
# - **Seniority normalization** reduced the SWE unknown rate from 55.1% to 7.3%
#   by combining native LinkedIn labels with title/description imputation
# - **Boilerplate removal** extracted core descriptions, narrowing cross-source
#   length gaps (though scraped data is still longer)
# - **Quality flags** identified 3,030 high-risk ghost jobs and 180K postings
#   with empty/short descriptions
#
# The dataset is ready for the main analysis in the research notebooks.

# %%
print("Notebook complete. All figures saved to preprocessing/")
