#!/usr/bin/env python3
"""T18: Cross-occupation boundary analysis.

Tests whether scope inflation and restructuring patterns are SWE-specific
or field-wide by comparing SWE, SWE-adjacent, and control occupations.

Uses `description` (full text) for keyword searches to avoid
boilerplate-removal recall loss from `description_core`.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DATA = 'data/unified.parquet'
FIG_DIR = Path('exploration/figures/T18')
TBL_DIR = Path('exploration/tables/T18')
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILTER = "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"

con = duckdb.connect()
con.execute("PRAGMA enable_progress_bar=false")

# ========================================
# AI keyword SQL fragments (on `description`)
# ========================================

AI_BROAD = """(
    LOWER(description) LIKE '%artificial intelligence%'
    OR LOWER(description) LIKE '%machine learning%'
    OR LOWER(description) LIKE '%deep learning%'
    OR LOWER(description) LIKE '%neural network%'
    OR LOWER(description) LIKE '% llm%'
    OR LOWER(description) LIKE '%large language model%'
    OR LOWER(description) LIKE '% gpt%'
    OR LOWER(description) LIKE '%openai%'
    OR LOWER(description) LIKE '%chatgpt%'
    OR LOWER(description) LIKE '%generative ai%'
    OR LOWER(description) LIKE '%copilot%'
    OR LOWER(description) LIKE '%prompt engineer%'
    OR LOWER(description) LIKE '%langchain%'
    OR LOWER(description) LIKE '%retrieval augmented%'
    OR LOWER(description) LIKE '%ai/ml%'
    OR LOWER(description) LIKE '%ml/ai%'
    OR LOWER(description) LIKE '%computer vision%'
    OR LOWER(description) LIKE '%natural language processing%'
    OR LOWER(description) LIKE '% nlp %'
    OR LOWER(description) LIKE '%tensorflow%'
    OR LOWER(description) LIKE '%pytorch%'
)"""

GENAI = """(
    LOWER(description) LIKE '% llm%'
    OR LOWER(description) LIKE '%large language model%'
    OR LOWER(description) LIKE '% gpt%'
    OR LOWER(description) LIKE '%openai%'
    OR LOWER(description) LIKE '%chatgpt%'
    OR LOWER(description) LIKE '%generative ai%'
    OR LOWER(description) LIKE '%gen ai%'
    OR LOWER(description) LIKE '%copilot%'
    OR LOWER(description) LIKE '%prompt engineer%'
    OR LOWER(description) LIKE '%langchain%'
    OR LOWER(description) LIKE '%retrieval augmented%'
    OR LOWER(description) LIKE '%ai agent%'
    OR LOWER(description) LIKE '%agentic%'
)"""

TRAD_ML = """(
    LOWER(description) LIKE '%machine learning%'
    OR LOWER(description) LIKE '%deep learning%'
    OR LOWER(description) LIKE '%neural network%'
    OR LOWER(description) LIKE '%tensorflow%'
    OR LOWER(description) LIKE '%pytorch%'
    OR LOWER(description) LIKE '%computer vision%'
    OR LOWER(description) LIKE '%natural language processing%'
)"""

AI_TOOLS = """(
    LOWER(description) LIKE '%ai tool%'
    OR LOWER(description) LIKE '%ai-powered%'
    OR LOWER(description) LIKE '%ai powered%'
    OR LOWER(description) LIKE '%ai-assisted%'
    OR LOWER(description) LIKE '%ai assisted%'
    OR LOWER(description) LIKE '%leverage ai%'
    OR LOWER(description) LIKE '%ai-driven%'
    OR LOWER(description) LIKE '%ai driven%'
    OR LOWER(description) LIKE '%ai solution%'
)"""

SCOPE_COLLAB = """(
    LOWER(description) LIKE '%cross-functional%'
    OR LOWER(description) LIKE '%cross functional%'
    OR LOWER(description) LIKE '%stakeholder%'
)"""

SCOPE_MENTOR = """(
    LOWER(description) LIKE '%mentor%'
    OR LOWER(description) LIKE '%coach %'
    OR LOWER(description) LIKE '%coaching%'
)"""

SCOPE_LEADERSHIP = """(
    LOWER(description) LIKE '%leadership%'
    OR LOWER(description) LIKE '%team lead%'
)"""

SCOPE_SOFT = """(
    LOWER(description) LIKE '%communication skill%'
    OR LOWER(description) LIKE '%interpersonal%'
    OR LOWER(description) LIKE '%presentation skill%'
)"""

SCOPE_STRATEGY = """(
    LOWER(description) LIKE '%strategy%'
    OR LOWER(description) LIKE '%roadmap%'
)"""

OCC_CASE = """CASE WHEN is_swe THEN 'SWE'
         WHEN is_swe_adjacent THEN 'SWE-adjacent'
         WHEN is_control THEN 'Control' END"""

print("=" * 60)
print("T18: Cross-Occupation Boundary Analysis")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 1: Parallel trends — core metrics
# ─────────────────────────────────────────────

print("\n--- STEP 1: Parallel trends ---\n")

# 1a. Seniority distribution
seniority_df = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group, seniority_3level,
    COUNT(*) as n
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control)
GROUP BY period, occ_group, seniority_3level
ORDER BY period, occ_group, seniority_3level
""").fetchdf()

seniority_totals = seniority_df.groupby(['period', 'occ_group'])['n'].sum().reset_index()
seniority_totals.columns = ['period', 'occ_group', 'total']
seniority_df = seniority_df.merge(seniority_totals)
seniority_df['share'] = seniority_df['n'] / seniority_df['total']
seniority_df.to_csv(TBL_DIR / 'seniority_by_group_period.csv', index=False)

print("Seniority shares:")
for _, row in seniority_df.iterrows():
    print(f"  {row['period']}  {row['occ_group']:15s}  {row['seniority_3level']:8s}  {row['share']:.3f}  (n={row['n']:,})")

# 1b. AI keyword prevalence (broad, on full description)
ai_kw_df = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as total,
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END) as ai_broad,
    SUM(CASE WHEN {GENAI} THEN 1 ELSE 0 END) as gen_ai,
    SUM(CASE WHEN {TRAD_ML} THEN 1 ELSE 0 END) as trad_ml,
    SUM(CASE WHEN {AI_TOOLS} THEN 1 ELSE 0 END) as ai_tools
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control)
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

for col in ['ai_broad', 'gen_ai', 'trad_ml', 'ai_tools']:
    ai_kw_df[f'{col}_share'] = ai_kw_df[col] / ai_kw_df['total']

print("\nAI keyword prevalence:")
for _, r in ai_kw_df.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  broad={r['ai_broad_share']:.3f}  genAI={r['gen_ai_share']:.3f}  tradML={r['trad_ml_share']:.3f}  tools={r['ai_tools_share']:.3f}")

ai_kw_df.to_csv(TBL_DIR / 'ai_prevalence_by_group_period.csv', index=False)

# 1c. Description length
desc_len_df = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as n,
    AVG(core_length) as avg_core_len,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY core_length) as median_core_len,
    AVG(description_length) as avg_desc_len,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY description_length) as median_desc_len
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control) AND description_core IS NOT NULL
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

print("\nDescription length:")
for _, r in desc_len_df.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  core_med={r['median_core_len']:,.0f}  desc_med={r['median_desc_len']:,.0f}")

desc_len_df.to_csv(TBL_DIR / 'description_length_by_group_period.csv', index=False)

# 1d. Scope / organizational language
scope_df = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as total,
    SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as collab_share,
    SUM(CASE WHEN {SCOPE_MENTOR} THEN 1 ELSE 0 END)*1.0/COUNT(*) as mentor_share,
    SUM(CASE WHEN {SCOPE_LEADERSHIP} THEN 1 ELSE 0 END)*1.0/COUNT(*) as leadership_share,
    SUM(CASE WHEN {SCOPE_SOFT} THEN 1 ELSE 0 END)*1.0/COUNT(*) as soft_share,
    SUM(CASE WHEN {SCOPE_STRATEGY} THEN 1 ELSE 0 END)*1.0/COUNT(*) as strategy_share
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control)
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

print("\nScope language:")
for _, r in scope_df.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  collab={r['collab_share']:.3f}  mentor={r['mentor_share']:.3f}  lead={r['leadership_share']:.3f}  soft={r['soft_share']:.3f}  strat={r['strategy_share']:.3f}")

scope_df.to_csv(TBL_DIR / 'scope_language_by_group_period.csv', index=False)

# 1e. Tech mention count
tech_df = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as total,
    AVG(
        (CASE WHEN LOWER(description) LIKE '%python%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%java %' OR LOWER(description) LIKE '%java,%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%javascript%' OR LOWER(description) LIKE '%typescript%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%react%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%angular%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '% c++%' OR LOWER(description) LIKE '%c++ %' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '% c#%' OR LOWER(description) LIKE '%c# %' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%golang%' OR LOWER(description) LIKE '% go %' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%rust %' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%kubernetes%' OR LOWER(description) LIKE '% k8s%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%docker%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%terraform%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%aws%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%azure%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%gcp%' OR LOWER(description) LIKE '%google cloud%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%sql%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%node.js%' OR LOWER(description) LIKE '%nodejs%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%ruby%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%scala%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%spark%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%kafka%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%redis%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%mongodb%' OR LOWER(description) LIKE '%postgres%' OR LOWER(description) LIKE '%mysql%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%ci/cd%' OR LOWER(description) LIKE '%cicd%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '% git %' OR LOWER(description) LIKE '%github%' OR LOWER(description) LIKE '%gitlab%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%linux%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%agile%' OR LOWER(description) LIKE '%scrum%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%rest api%' OR LOWER(description) LIKE '%restful%' OR LOWER(description) LIKE '%graphql%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%microservice%' THEN 1 ELSE 0 END) +
        (CASE WHEN LOWER(description) LIKE '%swift%' OR LOWER(description) LIKE '%kotlin%' THEN 1 ELSE 0 END)
    ) as avg_tech_count
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control)
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

print("\nTech mention count:")
for _, r in tech_df.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  avg_tech={r['avg_tech_count']:.2f}")

tech_df.to_csv(TBL_DIR / 'tech_count_by_group_period.csv', index=False)


# ─────────────────────────────────────────────
# STEP 2: Difference-in-differences
# ─────────────────────────────────────────────

print("\n--- STEP 2: Difference-in-differences ---\n")

def get_val(df, grp, period, col):
    mask = (df['occ_group'] == grp) & (df['period'] == period)
    v = df.loc[mask, col]
    return v.values[0] if len(v) > 0 else np.nan

did_rows = []
metrics_for_did = [
    ('AI prevalence (broad)', ai_kw_df, 'ai_broad_share'),
    ('GenAI prevalence', ai_kw_df, 'gen_ai_share'),
    ('Trad ML prevalence', ai_kw_df, 'trad_ml_share'),
    ('AI tools language', ai_kw_df, 'ai_tools_share'),
    ('Description length (median core)', desc_len_df, 'median_core_len'),
    ('Cross-functional/stakeholder', scope_df, 'collab_share'),
    ('Mentoring language', scope_df, 'mentor_share'),
    ('Leadership language', scope_df, 'leadership_share'),
    ('Soft skills language', scope_df, 'soft_share'),
    ('Strategy/roadmap', scope_df, 'strategy_share'),
    ('Tech mention count', tech_df, 'avg_tech_count'),
]

for name, df, col in metrics_for_did:
    for grp in ['SWE', 'SWE-adjacent', 'Control']:
        v01 = get_val(df, grp, '2024-01', col)
        v04 = get_val(df, grp, '2024-04', col)
        v26 = get_val(df, grp, '2026-03', col)
        did_rows.append({
            'metric': name, 'group': grp,
            'val_2024_01': v01, 'val_2024_04': v04, 'val_2026_03': v26,
            'change_04_26': v26 - v04 if not np.isnan(v04) and not np.isnan(v26) else np.nan,
            'change_01_26': v26 - v01 if not np.isnan(v01) and not np.isnan(v26) else np.nan,
        })

did_df = pd.DataFrame(did_rows)

# Add DiD (SWE minus Control)
for metric in did_df['metric'].unique():
    swe_row = did_df[(did_df['metric'] == metric) & (did_df['group'] == 'SWE')]
    ctrl_row = did_df[(did_df['metric'] == metric) & (did_df['group'] == 'Control')]
    adj_row = did_df[(did_df['metric'] == metric) & (did_df['group'] == 'SWE-adjacent')]

    for change_col, did_col_suffix in [('change_04_26', '04_26'), ('change_01_26', '01_26')]:
        if len(swe_row) and len(ctrl_row):
            s = swe_row[change_col].values[0]
            c = ctrl_row[change_col].values[0]
            did_df.loc[(did_df['metric'] == metric) & (did_df['group'] == 'SWE'),
                       f'did_swe_ctrl_{did_col_suffix}'] = s - c if not np.isnan(s) and not np.isnan(c) else np.nan
        if len(swe_row) and len(adj_row):
            s = swe_row[change_col].values[0]
            a = adj_row[change_col].values[0]
            did_df.loc[(did_df['metric'] == metric) & (did_df['group'] == 'SWE'),
                       f'did_swe_adj_{did_col_suffix}'] = s - a if not np.isnan(s) and not np.isnan(a) else np.nan

did_df.to_csv(TBL_DIR / 'did_full_table.csv', index=False)

# Print DiD summary (SWE rows only, with 04->26 DiDs)
print("DiD Summary (2024-04 to 2026-03, SWE vs Control):")
swe_did = did_df[did_df['group'] == 'SWE'].copy()
for _, r in swe_did.iterrows():
    did_sc = r.get('did_swe_ctrl_04_26', np.nan)
    did_sa = r.get('did_swe_adj_04_26', np.nan)
    ch = r.get('change_04_26', np.nan)
    did_sc_str = f"{did_sc:+.4f}" if not np.isnan(did_sc) else "N/A"
    did_sa_str = f"{did_sa:+.4f}" if not np.isnan(did_sa) else "N/A"
    ch_str = f"{ch:+.4f}" if not np.isnan(ch) else "N/A"
    print(f"  {r['metric']:40s}  SWE_change={ch_str:>10s}  DiD(SWE-Ctrl)={did_sc_str:>10s}  DiD(SWE-Adj)={did_sa_str:>10s}")


# ─────────────────────────────────────────────
# STEP 3: Boundary shift analysis (TF-IDF similarity)
# ─────────────────────────────────────────────

print("\n--- STEP 3: Boundary shift analysis ---\n")

boundary_results = []
term_migration = {}

for period in ['2024-01', '2024-04', '2026-03']:
    print(f"  Processing period: {period}")

    swe_texts = con.execute(f"""
    SELECT description_core FROM '{DATA}'
    WHERE {BASE_FILTER} AND is_swe = true AND period = '{period}'
      AND description_core IS NOT NULL AND LENGTH(description_core) > 100
    ORDER BY MD5(uid) LIMIT 500
    """).fetchdf()['description_core'].tolist()

    adj_texts = con.execute(f"""
    SELECT description_core FROM '{DATA}'
    WHERE {BASE_FILTER} AND is_swe_adjacent = true AND period = '{period}'
      AND description_core IS NOT NULL AND LENGTH(description_core) > 100
    ORDER BY MD5(uid) LIMIT 500
    """).fetchdf()['description_core'].tolist()

    ctrl_texts = con.execute(f"""
    SELECT description_core FROM '{DATA}'
    WHERE {BASE_FILTER} AND is_control = true AND period = '{period}'
      AND description_core IS NOT NULL AND LENGTH(description_core) > 100
    ORDER BY MD5(uid) LIMIT 500
    """).fetchdf()['description_core'].tolist()

    print(f"    SWE: {len(swe_texts)}, Adj: {len(adj_texts)}, Ctrl: {len(ctrl_texts)}")

    all_texts = swe_texts + adj_texts + ctrl_texts
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                             min_df=5, max_df=0.8, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(all_texts)

    n_swe = len(swe_texts)
    n_adj = len(adj_texts)

    swe_centroid = np.asarray(tfidf_matrix[:n_swe].mean(axis=0))
    adj_centroid = np.asarray(tfidf_matrix[n_swe:n_swe+n_adj].mean(axis=0))
    ctrl_centroid = np.asarray(tfidf_matrix[n_swe+n_adj:].mean(axis=0))

    sim_swe_adj = cosine_similarity(swe_centroid, adj_centroid)[0, 0]
    sim_swe_ctrl = cosine_similarity(swe_centroid, ctrl_centroid)[0, 0]
    sim_adj_ctrl = cosine_similarity(adj_centroid, ctrl_centroid)[0, 0]

    print(f"    SWE<->Adj: {sim_swe_adj:.4f}  SWE<->Ctrl: {sim_swe_ctrl:.4f}  Adj<->Ctrl: {sim_adj_ctrl:.4f}")

    boundary_results.append({
        'period': period,
        'sim_swe_adj': sim_swe_adj,
        'sim_swe_ctrl': sim_swe_ctrl,
        'sim_adj_ctrl': sim_adj_ctrl,
    })

    # Term analysis for this period
    feature_names = tfidf.get_feature_names_out()
    swe_mean = np.asarray(tfidf_matrix[:n_swe].mean(axis=0)).flatten()
    adj_mean = np.asarray(tfidf_matrix[n_swe:n_swe+n_adj].mean(axis=0)).flatten()

    # Top shared terms (high in both, small difference)
    joint = swe_mean + adj_mean
    diff = np.abs(swe_mean - adj_mean)
    shared_score = joint - diff * 2
    top_shared = np.argsort(shared_score)[-15:][::-1]

    # SWE-distinctive
    top_swe_distinct = np.argsort(swe_mean - adj_mean)[-10:][::-1]
    # Adj-distinctive
    top_adj_distinct = np.argsort(adj_mean - swe_mean)[-10:][::-1]

    term_migration[period] = {
        'shared': [(feature_names[i], swe_mean[i], adj_mean[i]) for i in top_shared],
        'swe_distinct': [(feature_names[i], swe_mean[i], adj_mean[i]) for i in top_swe_distinct],
        'adj_distinct': [(feature_names[i], swe_mean[i], adj_mean[i]) for i in top_adj_distinct],
    }

boundary_df = pd.DataFrame(boundary_results)
boundary_df.to_csv(TBL_DIR / 'boundary_similarity.csv', index=False)

# Print term migration
for period in ['2024-04', '2026-03']:
    print(f"\n  === Term migration ({period}) ===")
    print(f"  Top shared (SWE & SWE-adj):")
    for term, sv, av in term_migration[period]['shared']:
        print(f"    {term:30s}  SWE={sv:.4f}  Adj={av:.4f}")
    print(f"  SWE-distinctive:")
    for term, sv, av in term_migration[period]['swe_distinct']:
        print(f"    {term:30s}  SWE={sv:.4f}  Adj={av:.4f}")
    print(f"  SWE-adj distinctive:")
    for term, sv, av in term_migration[period]['adj_distinct']:
        print(f"    {term:30s}  SWE={sv:.4f}  Adj={av:.4f}")


# ─────────────────────────────────────────────
# STEP 4: Specific adjacent roles
# ─────────────────────────────────────────────

print("\n--- STEP 4: Specific adjacent role changes ---\n")

adjacent_roles = [
    ('data engineer', "LOWER(title_normalized) LIKE '%data engineer%'"),
    ('data scientist', "LOWER(title_normalized) LIKE '%data scientist%'"),
    ('network engineer', "LOWER(title_normalized) LIKE '%network engineer%'"),
    ('security engineer', "(LOWER(title_normalized) LIKE '%security engineer%' OR LOWER(title_normalized) LIKE '%cybersecurity%')"),
    ('quality engineer', "LOWER(title_normalized) LIKE '%quality engineer%'"),
]

adj_role_results = []
for role_name, role_filter in adjacent_roles:
    metrics = con.execute(f"""
    SELECT
        period, COUNT(*) as n,
        AVG(core_length) as avg_core_len,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY core_length) as med_core_len,
        SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_share,
        SUM(CASE WHEN {GENAI} THEN 1 ELSE 0 END)*1.0/COUNT(*) as genai_share,
        SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as collab_share,
        SUM(CASE WHEN {SCOPE_STRATEGY} THEN 1 ELSE 0 END)*1.0/COUNT(*) as strategy_share
    FROM '{DATA}'
    WHERE {BASE_FILTER} AND is_swe_adjacent = true AND ({role_filter})
    GROUP BY period ORDER BY period
    """).fetchdf()

    print(f"  {role_name}:")
    for _, r in metrics.iterrows():
        print(f"    {r['period']}  n={r['n']:5,}  core_med={r['med_core_len']:,.0f}  AI={r['ai_share']:.3f}  genAI={r['genai_share']:.3f}  collab={r['collab_share']:.3f}  strat={r['strategy_share']:.3f}")
        adj_role_results.append({
            'role': role_name, 'period': r['period'], 'n': r['n'],
            'med_core_len': r['med_core_len'], 'ai_share': r['ai_share'],
            'genai_share': r['genai_share'], 'collab_share': r['collab_share'],
            'strategy_share': r['strategy_share']
        })

adj_role_df = pd.DataFrame(adj_role_results)
adj_role_df.to_csv(TBL_DIR / 'adjacent_role_metrics.csv', index=False)


# ─────────────────────────────────────────────
# STEP 5: AI adoption gradient
# ─────────────────────────────────────────────

print("\n--- STEP 5: AI adoption gradient ---\n")

# Reuse ai_kw_df from Step 1 which already has the gradient
print("AI adoption gradient (from Step 1):")
for _, r in ai_kw_df.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  broad={r['ai_broad_share']:.3f}  genAI={r['gen_ai_share']:.3f}")

# Check if gradient is widening or narrowing
for period in ['2024-01', '2024-04', '2026-03']:
    swe_v = get_val(ai_kw_df, 'SWE', period, 'ai_broad_share')
    ctrl_v = get_val(ai_kw_df, 'Control', period, 'ai_broad_share')
    adj_v = get_val(ai_kw_df, 'SWE-adjacent', period, 'ai_broad_share')
    print(f"  {period}  SWE-Ctrl gap: {swe_v-ctrl_v:.3f}  SWE-Adj gap: {swe_v-adj_v:.3f}")


# ─────────────────────────────────────────────
# SENSITIVITY: (a) Aggregator exclusion
# ─────────────────────────────────────────────

print("\n--- SENSITIVITY (a): Aggregator exclusion ---\n")

ai_noagg = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as total,
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_share,
    AVG(core_length) as avg_core_len,
    SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as collab_share
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent OR is_control) AND is_aggregator = false
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

print("Aggregator-excluded:")
for _, r in ai_noagg.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  n={r['total']:6,}  AI={r['ai_share']:.3f}  core={r['avg_core_len']:,.0f}  collab={r['collab_share']:.3f}")

ai_noagg.to_csv(TBL_DIR / 'sensitivity_no_aggregators.csv', index=False)


# ─────────────────────────────────────────────
# SENSITIVITY: (g) SWE classification tier
# ─────────────────────────────────────────────

print("\n--- SENSITIVITY (g): Regex-tier SWE only ---\n")

ai_regex = con.execute(f"""
SELECT
    period, {OCC_CASE} as occ_group,
    COUNT(*) as total,
    SUM(CASE WHEN {AI_BROAD} THEN 1 ELSE 0 END)*1.0/COUNT(*) as ai_share,
    AVG(core_length) as avg_core_len,
    SUM(CASE WHEN {SCOPE_COLLAB} THEN 1 ELSE 0 END)*1.0/COUNT(*) as collab_share
FROM '{DATA}'
WHERE {BASE_FILTER} AND (is_swe OR is_swe_adjacent) AND swe_classification_tier = 'regex'
GROUP BY period, occ_group ORDER BY period, occ_group
""").fetchdf()

print("Regex-tier only:")
for _, r in ai_regex.iterrows():
    print(f"  {r['period']}  {r['occ_group']:15s}  n={r['total']:6,}  AI={r['ai_share']:.3f}  core={r['avg_core_len']:,.0f}  collab={r['collab_share']:.3f}")

ai_regex.to_csv(TBL_DIR / 'sensitivity_regex_tier.csv', index=False)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

print("\n--- Generating figures ---\n")

colors = {'SWE': '#2196F3', 'SWE-adjacent': '#FF9800', 'Control': '#4CAF50'}
period_order = ['2024-01', '2024-04', '2026-03']
period_labels = ['Jan 2024', 'Apr 2024', 'Mar 2026']

def plot_trend(ax, df, col, title, ylabel, groups=None):
    if groups is None:
        groups = ['SWE', 'SWE-adjacent', 'Control']
    for grp in groups:
        vals = [get_val(df, grp, p, col) for p in period_order]
        ax.plot(period_labels, vals, 'o-', color=colors[grp], label=grp, linewidth=2, markersize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# FIGURE 1: 4-panel parallel trends
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('T18: Parallel Trends -- SWE vs SWE-adjacent vs Control', fontsize=14, fontweight='bold')

plot_trend(axes[0,0], ai_kw_df, 'ai_broad_share', 'AI/ML Keyword Prevalence', 'Share of postings')
plot_trend(axes[0,1], desc_len_df, 'median_core_len', 'Median Description Length (core)', 'Characters')
plot_trend(axes[1,0], scope_df, 'collab_share', 'Cross-functional/Stakeholder Language', 'Share of postings')
plot_trend(axes[1,1], tech_df, 'avg_tech_count', 'Mean Tech Mention Count (of 30)', 'Count')

plt.tight_layout()
plt.savefig(FIG_DIR / 'parallel_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved parallel_trends.png")

# FIGURE 2: AI adoption gradient with sub-categories
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('T18: AI Adoption Gradient by Category', fontsize=14, fontweight='bold')

plot_trend(axes[0], ai_kw_df, 'ai_broad_share', 'Any AI/ML Mention', 'Share')
plot_trend(axes[1], ai_kw_df, 'gen_ai_share', 'Generative AI Only', 'Share')
plot_trend(axes[2], ai_kw_df, 'ai_tools_share', 'AI Tool Usage Language', 'Share')

plt.tight_layout()
plt.savefig(FIG_DIR / 'ai_adoption_gradient.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved ai_adoption_gradient.png")

# FIGURE 3: Boundary similarity over time
fig, ax = plt.subplots(figsize=(8, 5))
pairs = [
    ('sim_swe_adj', 'SWE <-> SWE-adjacent', '#9C27B0'),
    ('sim_swe_ctrl', 'SWE <-> Control', '#F44336'),
    ('sim_adj_ctrl', 'SWE-adj <-> Control', '#607D8B'),
]
for col, label, color in pairs:
    vals = boundary_df[col].tolist()
    ax.plot(period_labels, vals, 'o-', label=label, color=color, linewidth=2, markersize=8)
ax.set_title('T18: Cross-Occupation Boundary Similarity (TF-IDF Cosine)', fontsize=13, fontweight='bold')
ax.set_ylabel('Centroid Cosine Similarity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'boundary_similarity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved boundary_similarity.png")

# FIGURE 4: Adjacent role trends
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('T18: SWE-Adjacent Role Trends', fontsize=13, fontweight='bold')

role_colors = {
    'data engineer': '#E91E63', 'data scientist': '#9C27B0',
    'network engineer': '#3F51B5', 'security engineer': '#009688',
    'quality engineer': '#795548',
}

for ax, col, title in [(axes[0], 'ai_share', 'AI Prevalence by Role'),
                        (axes[1], 'med_core_len', 'Description Length (Median Core)')]:
    for role_name in role_colors:
        rdf = adj_role_df[adj_role_df['role'] == role_name]
        if len(rdf) >= 2:
            ax.plot(rdf['period'], rdf[col], 'o-', color=role_colors[role_name],
                    label=role_name, linewidth=2, markersize=7)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Share' if 'share' in col else 'Characters')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'adjacent_role_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved adjacent_role_trends.png")

print("\nT18 analysis complete.")
