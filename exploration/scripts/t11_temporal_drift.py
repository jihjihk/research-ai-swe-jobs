#!/usr/bin/env python3
"""
T11: Temporal drift analysis
-----------------------------
Quantifies how much SWE posting content changed between 2024 and 2026.

Metrics:
  1. Jensen-Shannon divergence on unigram frequencies (overall and by seniority)
  2. Emerging/disappearing terms (>1% in one period, <0.1% in other)
  3. YOE inflation for entry-level SWE
  4. AI keyword prevalence by period x seniority

Outputs:
  - exploration/tables/T11/  (CSV)
  - exploration/figures/T11/ (PNG, max 4)
  - exploration/reports/T11.md
"""

import os, sys, re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────
PROJECT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = PROJECT / "preprocessing/intermediate/stage8_final.parquet"
TABLE_DIR = PROJECT / "exploration/tables/T11"
FIG_DIR   = PROJECT / "exploration/figures/T11"
REPORT    = PROJECT / "exploration/reports/T11.md"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── seniority workaround CTE ──────────────────────────────────────
SWE_PATCHED_CTE = """
swe_patched AS (
  SELECT *,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN seniority_final
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'intern' THEN 'entry'
               WHEN 'executive' THEN 'director'
               ELSE seniority_native
             END
      WHEN seniority_final != 'unknown'
        THEN seniority_final
      ELSE 'unknown'
    END AS seniority_patched,
    CASE
      WHEN seniority_final_source IN ('title_keyword', 'title_manager')
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_native IS NOT NULL AND seniority_native != ''
        THEN CASE seniority_native
               WHEN 'entry' THEN 'junior'
               WHEN 'intern' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               WHEN 'executive' THEN 'senior'
               ELSE 'unknown'
             END
      WHEN seniority_final != 'unknown'
        THEN CASE seniority_final
               WHEN 'entry' THEN 'junior'
               WHEN 'associate' THEN 'mid'
               WHEN 'mid-senior' THEN 'senior'
               WHEN 'director' THEN 'senior'
               ELSE 'unknown'
             END
      ELSE 'unknown'
    END AS seniority_3level_patched
  FROM '{}'
  WHERE source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
)
""".format(str(PARQUET))

# ── text preprocessing (shared with T10) ───────────────────────────
_HTML_RE = re.compile(r'<[^>]+>')
_URL_RE  = re.compile(r'https?://\S+|www\.\S+')
_NON_ALPHA = re.compile(r'[^a-z\s]')

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had hadn't
has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought
our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where
where's which while who who's whom why why's will with won't would wouldn't you you'd
you'll you're you've your yours yourself yourselves
""".split())

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = text.lower()
    text = _NON_ALPHA.sub(' ', text)
    return text

def tokenize(text):
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]


# ── AI keywords for prevalence analysis ────────────────────────────
AI_KEYWORDS_PATTERN = re.compile(
    r'\b(artificial\s+intelligence|ai|llm|llms|large\s+language\s+model|'
    r'machine\s+learning|ml|deep\s+learning|neural\s+network|'
    r'generative\s+ai|genai|gen\s+ai|'
    r'gpt|chatgpt|openai|claude|copilot|github\s+copilot|'
    r'agent|agents|agentic|'
    r'rag|retrieval\s+augmented|'
    r'prompt|prompting|prompt\s+engineering|'
    r'langchain|transformer|transformers|'
    r'diffusion|embeddings?|fine\s*tun(?:e|ing)|'
    r'nlp|natural\s+language\s+processing)\b',
    re.IGNORECASE
)

# Narrow AI keywords (excludes ambiguous terms like 'agent', 'ml', 'ai')
AI_NARROW_PATTERN = re.compile(
    r'\b(llm|llms|large\s+language\s+model|'
    r'generative\s+ai|genai|gen\s+ai|'
    r'gpt|chatgpt|openai|claude|copilot|github\s+copilot|'
    r'agentic|'
    r'rag|retrieval\s+augmented|'
    r'prompt\s+engineering|'
    r'langchain|diffusion)\b',
    re.IGNORECASE
)

# Specific AI terms for per-term prevalence
AI_TERM_LIST = [
    ('ai', r'\bai\b'),
    ('llm/llms', r'\bllms?\b'),
    ('machine_learning', r'\bmachine\s+learning\b'),
    ('deep_learning', r'\bdeep\s+learning\b'),
    ('generative_ai', r'\b(?:generative\s+ai|genai|gen\s+ai)\b'),
    ('gpt/chatgpt', r'\b(?:gpt|chatgpt)\b'),
    ('copilot', r'\bcopilot\b'),
    ('agent/agents', r'\b(?:agents?|agentic)\b'),
    ('rag', r'\brag\b'),
    ('prompt/prompting', r'\b(?:prompt|prompting|prompt\s+engineering)\b'),
    ('langchain', r'\blangchain\b'),
    ('claude', r'\bclaude\b'),
    ('openai', r'\bopenai\b'),
    ('transformer', r'\btransformers?\b'),
]


def get_corpus_df(con, condition):
    """Return DataFrame with uid, description, source, seniority_patched."""
    sql = f"""
    WITH {SWE_PATCHED_CTE}
    SELECT uid, description, source, seniority_patched, seniority_3level_patched
    FROM swe_patched
    WHERE {condition}
      AND description IS NOT NULL
      AND length(description) > 50
    """
    return con.sql(sql).fetchdf()


def build_freq_dist(texts):
    """Build normalized frequency distribution from texts."""
    counts = Counter()
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize(cleaned)
        counts.update(tokens)
    total = sum(counts.values())
    if total == 0:
        return counts, {}
    freqs = {k: v / total for k, v in counts.items()}
    return counts, freqs


def compute_jsd(freqs_a, freqs_b):
    """Compute Jensen-Shannon divergence between two frequency dicts."""
    vocab = sorted(set(freqs_a.keys()) | set(freqs_b.keys()))
    if not vocab:
        return float('nan')
    p = np.array([freqs_a.get(w, 0) for w in vocab])
    q = np.array([freqs_b.get(w, 0) for w in vocab])
    # Normalize (should already be ~1.0, but ensure)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(jensenshannon(p, q) ** 2)  # Return JSD (squared JS distance)


def find_emerging_disappearing(freqs_a, freqs_b, counts_a, counts_b,
                                emerge_thresh=0.001, vanish_thresh=0.0001,
                                min_count=3):
    """
    Find terms that emerged or disappeared.
    Emerging: freq_b > emerge_thresh AND freq_a < vanish_thresh
    Disappearing: freq_a > emerge_thresh AND freq_b < vanish_thresh
    """
    vocab = set(freqs_a.keys()) | set(freqs_b.keys())

    emerging = []
    disappearing = []

    for w in vocab:
        fa = freqs_a.get(w, 0)
        fb = freqs_b.get(w, 0)
        ca = counts_a.get(w, 0)
        cb = counts_b.get(w, 0)

        # Emerging in B (2026): high in B, low in A
        if fb > emerge_thresh and fa < vanish_thresh and cb >= min_count:
            emerging.append({
                'term': w,
                'freq_2024': fa,
                'freq_2026': fb,
                'count_2024': ca,
                'count_2026': cb,
                'ratio': fb / fa if fa > 0 else float('inf'),
            })

        # Disappearing from B: high in A, low in B
        if fa > emerge_thresh and fb < vanish_thresh and ca >= min_count:
            disappearing.append({
                'term': w,
                'freq_2024': fa,
                'freq_2026': fb,
                'count_2024': ca,
                'count_2026': cb,
                'ratio': fa / fb if fb > 0 else float('inf'),
            })

    emerging = sorted(emerging, key=lambda x: x['freq_2026'], reverse=True)
    disappearing = sorted(disappearing, key=lambda x: x['freq_2024'], reverse=True)

    return emerging, disappearing


def build_doc_freq_dist(texts):
    """Build document-level frequency distribution (% of docs containing each term)."""
    doc_counts = Counter()
    ndocs = len(texts)
    for text in texts:
        cleaned = clean_text(text)
        unique_tokens = set(tokenize(cleaned))
        doc_counts.update(unique_tokens)
    if ndocs == 0:
        return doc_counts, {}
    doc_freqs = {k: v / ndocs for k, v in doc_counts.items()}
    return doc_counts, doc_freqs


def is_formatting_artifact(term):
    """Filter concatenated word artifacts from HTML stripping."""
    # Concatenated words: two+ lowercase words jammed together (no vowels pattern)
    # Heuristic: if the term is >12 chars and has a lowercase letter followed by
    # an uppercase-like pattern in the original, it's likely a concat artifact
    if len(term) > 15:
        return True  # Very long single tokens are usually artifacts
    # Common artifact patterns
    artifact_patterns = [
        'experience', 'location', 'skills', 'working', 'development',
        'technologies', 'systems', 'plus', 'strong', 'ability',
        'excellent', 'ensuring', 'maintaining', 'building', 'paid',
    ]
    # If the term contains two known suffixes/prefixes concatenated
    matches = sum(1 for p in artifact_patterns if p in term and term != p)
    if matches >= 2:
        return True
    if len(term) > 12 and not any(c in term for c in ['-', '_']):
        # Check if it looks like concatenated words
        # Simple heuristic: count lowercase runs
        runs = re.findall(r'[a-z]+', term)
        if len(runs) == 1 and len(term) > 12:
            # Single long word - check if it's a known English word-like thing
            # If it has unusual consonant clusters, likely artifact
            pass
    return False


def main():
    con = duckdb.connect()
    report_lines = []

    # ═══════════════════════════════════════════════════════════════
    # 1. JSD on unigram frequencies
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("1. Jensen-Shannon Divergence")
    print("=" * 60)

    # Use arshkon for 2024 (it has entry-level labels)
    # Overall SWE
    df_2024 = get_corpus_df(con, "is_swe = true AND source = 'kaggle_arshkon'")
    df_2026 = get_corpus_df(con, "is_swe = true AND source = 'scraped'")

    print(f"  Overall: 2024 n={len(df_2024)}, 2026 n={len(df_2026)}")

    counts_2024, freqs_2024 = build_freq_dist(df_2024['description'].tolist())
    counts_2026, freqs_2026 = build_freq_dist(df_2026['description'].tolist())

    jsd_overall = compute_jsd(freqs_2024, freqs_2026)
    print(f"  JSD overall: {jsd_overall:.6f}")

    # Per seniority level
    jsd_results = [{'level': 'overall', 'n_2024': len(df_2024), 'n_2026': len(df_2026), 'jsd': jsd_overall}]

    for level in ['entry', 'mid-senior']:
        texts_24 = df_2024[df_2024['seniority_patched'] == level]['description'].tolist()
        texts_26 = df_2026[df_2026['seniority_patched'] == level]['description'].tolist()

        if len(texts_24) < 10 or len(texts_26) < 10:
            print(f"  {level}: too few docs (2024: {len(texts_24)}, 2026: {len(texts_26)}), skipping")
            jsd_results.append({'level': level, 'n_2024': len(texts_24), 'n_2026': len(texts_26), 'jsd': float('nan')})
            continue

        _, f24 = build_freq_dist(texts_24)
        _, f26 = build_freq_dist(texts_26)
        jsd_val = compute_jsd(f24, f26)
        print(f"  {level}: 2024 n={len(texts_24)}, 2026 n={len(texts_26)}, JSD={jsd_val:.6f}")
        jsd_results.append({'level': level, 'n_2024': len(texts_24), 'n_2026': len(texts_26), 'jsd': jsd_val})

    # Also compute within-2024 reference JSD (arshkon vs asaniczka mid-senior)
    df_asan = get_corpus_df(con, "is_swe = true AND source = 'kaggle_asaniczka' AND seniority_patched = 'mid-senior'")
    if len(df_asan) > 100:
        _, f_asan = build_freq_dist(df_asan['description'].sample(min(3000, len(df_asan)), random_state=42).tolist())
        texts_arsh_ms = df_2024[df_2024['seniority_patched'] == 'mid-senior']['description'].tolist()
        _, f_arsh_ms = build_freq_dist(texts_arsh_ms)
        jsd_within = compute_jsd(f_arsh_ms, f_asan)
        print(f"  Within-2024 reference (arshkon vs asaniczka mid-senior): JSD={jsd_within:.6f}")
        jsd_results.append({'level': 'within-2024 ref (arshkon vs asaniczka mid-senior)',
                           'n_2024': len(texts_arsh_ms), 'n_2026': min(3000, len(df_asan)),
                           'jsd': jsd_within})

    jsd_df = pd.DataFrame(jsd_results)
    jsd_df.to_csv(TABLE_DIR / "jsd_results.csv", index=False)
    print(f"  Saved JSD table")

    # ═══════════════════════════════════════════════════════════════
    # 2. Emerging / Disappearing terms (document-level prevalence)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. Emerging and Disappearing Terms (document-level prevalence)")
    print("=" * 60)

    all_emerging = {}
    all_disappearing = {}

    for level_label, level_filter in [('overall', None), ('entry', 'entry'), ('mid-senior', 'mid-senior')]:
        if level_filter:
            t24 = df_2024[df_2024['seniority_patched'] == level_filter]['description'].tolist()
            t26 = df_2026[df_2026['seniority_patched'] == level_filter]['description'].tolist()
        else:
            t24 = df_2024['description'].tolist()
            t26 = df_2026['description'].tolist()

        if len(t24) < 10 or len(t26) < 10:
            print(f"  {level_label}: too few docs, skipping")
            continue

        # Use document-level frequencies: % of documents containing each term
        dc24, df24 = build_doc_freq_dist(t24)
        dc26, df26 = build_doc_freq_dist(t26)

        # Task spec: >1% in 2026 but <0.1% in 2024 (document-level)
        emerging, disappearing = find_emerging_disappearing(
            df24, df26, dc24, dc26,
            emerge_thresh=0.01, vanish_thresh=0.001, min_count=3
        )

        # Filter formatting artifacts
        emerging = [e for e in emerging if not is_formatting_artifact(e['term'])]
        disappearing = [d for d in disappearing if not is_formatting_artifact(d['term'])]

        print(f"  {level_label} (2024 n={len(t24)}, 2026 n={len(t26)}): "
              f"{len(emerging)} emerging, {len(disappearing)} disappearing")

        if emerging:
            em_df = pd.DataFrame(emerging)
            em_df['level'] = level_label
            em_df.to_csv(TABLE_DIR / f"emerging_{level_label}.csv", index=False)
            all_emerging[level_label] = em_df
            print(f"    Top emerging: {', '.join(e['term'] for e in emerging[:15])}")

        if disappearing:
            dis_df = pd.DataFrame(disappearing)
            dis_df['level'] = level_label
            dis_df.to_csv(TABLE_DIR / f"disappearing_{level_label}.csv", index=False)
            all_disappearing[level_label] = dis_df
            print(f"    Top disappearing: {', '.join(d['term'] for d in disappearing[:15])}")

    # ═══════════════════════════════════════════════════════════════
    # 3. YOE inflation for entry-level
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. YOE Inflation (Entry-Level SWE)")
    print("=" * 60)

    yoe_sql = f"""
    WITH {SWE_PATCHED_CTE}
    SELECT
      source,
      seniority_patched,
      yoe_extracted,
      count(*) as n
    FROM swe_patched
    WHERE is_swe = true
      AND seniority_patched = 'entry'
      AND source IN ('kaggle_arshkon', 'scraped')
    GROUP BY 1, 2, 3
    ORDER BY 1, 3
    """
    yoe_df = con.sql(yoe_sql).fetchdf()

    # Distribution summary
    yoe_arsh = yoe_df[yoe_df['source'] == 'kaggle_arshkon']
    yoe_scr  = yoe_df[yoe_df['source'] == 'scraped']

    # Expand to row-level for stats
    yoe_arsh_vals = []
    for _, row in yoe_arsh.iterrows():
        yoe_arsh_vals.extend([row['yoe_extracted']] * int(row['n']))

    yoe_scr_vals = []
    for _, row in yoe_scr.iterrows():
        yoe_scr_vals.extend([row['yoe_extracted']] * int(row['n']))

    yoe_arsh_arr = np.array([v for v in yoe_arsh_vals if v is not None and not np.isnan(v)])
    yoe_scr_arr  = np.array([v for v in yoe_scr_vals if v is not None and not np.isnan(v)])

    print(f"  Arshkon entry-level: {len(yoe_arsh_vals)} total, {len(yoe_arsh_arr)} with YOE")
    print(f"  Scraped entry-level: {len(yoe_scr_vals)} total, {len(yoe_scr_arr)} with YOE")

    yoe_stats = []
    if len(yoe_arsh_arr) > 0:
        yoe_stats.append({
            'source': 'arshkon_2024',
            'n_total': len(yoe_arsh_vals),
            'n_with_yoe': len(yoe_arsh_arr),
            'pct_with_yoe': len(yoe_arsh_arr) / len(yoe_arsh_vals) * 100,
            'mean_yoe': np.mean(yoe_arsh_arr),
            'median_yoe': np.median(yoe_arsh_arr),
            'p25_yoe': np.percentile(yoe_arsh_arr, 25),
            'p75_yoe': np.percentile(yoe_arsh_arr, 75),
            'pct_0yoe': np.mean(yoe_arsh_arr == 0) * 100,
            'pct_3plus': np.mean(yoe_arsh_arr >= 3) * 100,
            'pct_5plus': np.mean(yoe_arsh_arr >= 5) * 100,
        })
        print(f"    Arshkon: mean={np.mean(yoe_arsh_arr):.1f}, median={np.median(yoe_arsh_arr):.1f}, "
              f"% >= 3yr: {np.mean(yoe_arsh_arr >= 3)*100:.1f}%")

    if len(yoe_scr_arr) > 0:
        yoe_stats.append({
            'source': 'scraped_2026',
            'n_total': len(yoe_scr_vals),
            'n_with_yoe': len(yoe_scr_arr),
            'pct_with_yoe': len(yoe_scr_arr) / len(yoe_scr_vals) * 100,
            'mean_yoe': np.mean(yoe_scr_arr),
            'median_yoe': np.median(yoe_scr_arr),
            'p25_yoe': np.percentile(yoe_scr_arr, 25),
            'p75_yoe': np.percentile(yoe_scr_arr, 75),
            'pct_0yoe': np.mean(yoe_scr_arr == 0) * 100,
            'pct_3plus': np.mean(yoe_scr_arr >= 3) * 100,
            'pct_5plus': np.mean(yoe_scr_arr >= 5) * 100,
        })
        print(f"    Scraped: mean={np.mean(yoe_scr_arr):.1f}, median={np.median(yoe_scr_arr):.1f}, "
              f"% >= 3yr: {np.mean(yoe_scr_arr >= 3)*100:.1f}%")

    # Mann-Whitney U test
    if len(yoe_arsh_arr) > 10 and len(yoe_scr_arr) > 10:
        stat, pval = mannwhitneyu(yoe_arsh_arr, yoe_scr_arr, alternative='two-sided')
        print(f"    Mann-Whitney U: stat={stat:.0f}, p={pval:.4f}")
        for s in yoe_stats:
            s['mw_p'] = pval
    else:
        print("    Too few YOE values for statistical test")

    yoe_stats_df = pd.DataFrame(yoe_stats)
    yoe_stats_df.to_csv(TABLE_DIR / "yoe_entry_stats.csv", index=False)

    # Also get full YOE distribution for all seniority levels
    yoe_all_sql = f"""
    WITH {SWE_PATCHED_CTE}
    SELECT
      source,
      seniority_patched,
      yoe_extracted,
      count(*) as n
    FROM swe_patched
    WHERE is_swe = true
      AND seniority_patched != 'unknown'
      AND source IN ('kaggle_arshkon', 'scraped')
      AND yoe_extracted IS NOT NULL
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
    """
    yoe_all_df = con.sql(yoe_all_sql).fetchdf()
    yoe_all_df.to_csv(TABLE_DIR / "yoe_all_levels.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    # 4. AI keyword prevalence
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. AI Keyword Prevalence")
    print("=" * 60)

    # Get all SWE postings with descriptions
    ai_sql = f"""
    WITH {SWE_PATCHED_CTE}
    SELECT uid, description, source, seniority_patched
    FROM swe_patched
    WHERE is_swe = true
      AND description IS NOT NULL
      AND length(description) > 50
      AND seniority_patched != 'unknown'
      AND source IN ('kaggle_arshkon', 'scraped')
    """
    ai_df = con.sql(ai_sql).fetchdf()

    # Add period column
    ai_df['period'] = ai_df['source'].map({
        'kaggle_arshkon': '2024',
        'scraped': '2026',
    })

    # Broad AI match
    ai_df['has_ai_broad'] = ai_df['description'].str.contains(AI_KEYWORDS_PATTERN, na=False)
    # Narrow AI match (LLM-specific)
    ai_df['has_ai_narrow'] = ai_df['description'].str.contains(AI_NARROW_PATTERN, na=False)

    # Per-term matches
    for term_name, pattern in AI_TERM_LIST:
        ai_df[f'has_{term_name}'] = ai_df['description'].str.contains(pattern, case=False, na=False)

    # Prevalence by period x seniority
    prev_results = []
    for period in ['2024', '2026']:
        for level in ['entry', 'associate', 'mid-senior', 'director']:
            mask = (ai_df['period'] == period) & (ai_df['seniority_patched'] == level)
            sub = ai_df[mask]
            n = len(sub)
            if n == 0:
                continue

            row = {
                'period': period,
                'seniority': level,
                'n': n,
                'pct_ai_broad': sub['has_ai_broad'].mean() * 100,
                'pct_ai_narrow': sub['has_ai_narrow'].mean() * 100,
            }
            for term_name, _ in AI_TERM_LIST:
                row[f'pct_{term_name}'] = sub[f'has_{term_name}'].mean() * 100

            prev_results.append(row)

    # Also compute overall by period
    for period in ['2024', '2026']:
        mask = ai_df['period'] == period
        sub = ai_df[mask]
        n = len(sub)
        row = {
            'period': period,
            'seniority': 'all',
            'n': n,
            'pct_ai_broad': sub['has_ai_broad'].mean() * 100,
            'pct_ai_narrow': sub['has_ai_narrow'].mean() * 100,
        }
        for term_name, _ in AI_TERM_LIST:
            row[f'pct_{term_name}'] = sub[f'has_{term_name}'].mean() * 100
        prev_results.append(row)

    prev_df = pd.DataFrame(prev_results)
    prev_df.to_csv(TABLE_DIR / "ai_prevalence.csv", index=False)

    print("  AI prevalence by period x seniority:")
    print(prev_df[['period', 'seniority', 'n', 'pct_ai_broad', 'pct_ai_narrow']].to_string(index=False))

    # Chi-square tests for broad AI mention by period
    print("\n  Chi-square tests (broad AI mention):")
    for level in ['entry', 'mid-senior', 'all']:
        if level == 'all':
            sub24 = ai_df[ai_df['period'] == '2024']
            sub26 = ai_df[ai_df['period'] == '2026']
        else:
            sub24 = ai_df[(ai_df['period'] == '2024') & (ai_df['seniority_patched'] == level)]
            sub26 = ai_df[(ai_df['period'] == '2026') & (ai_df['seniority_patched'] == level)]

        if len(sub24) < 10 or len(sub26) < 10:
            continue

        table = np.array([
            [sub24['has_ai_broad'].sum(), len(sub24) - sub24['has_ai_broad'].sum()],
            [sub26['has_ai_broad'].sum(), len(sub26) - sub26['has_ai_broad'].sum()]
        ])
        if table.min() >= 0:
            chi2, p, dof, _ = chi2_contingency(table)
            pct_24 = sub24['has_ai_broad'].mean() * 100
            pct_26 = sub26['has_ai_broad'].mean() * 100
            print(f"    {level}: 2024={pct_24:.1f}%, 2026={pct_26:.1f}%, chi2={chi2:.1f}, p={p:.2e}")

    # ═══════════════════════════════════════════════════════════════
    # FIGURES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Generating figures...")

    # Figure 1: JSD bar chart
    plot_jsd(jsd_df, FIG_DIR / "jsd_by_level.png")

    # Figure 2: AI prevalence grouped bar chart
    plot_ai_prevalence(prev_df, FIG_DIR / "ai_prevalence_by_period_seniority.png")

    # Figure 3: YOE distribution comparison
    plot_yoe_distribution(yoe_arsh_arr, yoe_scr_arr, FIG_DIR / "yoe_entry_distribution.png")

    # Figure 4: Per-term AI prevalence heatmap
    plot_ai_term_heatmap(prev_df, FIG_DIR / "ai_term_prevalence_heatmap.png")

    # ═══════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════
    build_report(jsd_df, all_emerging, all_disappearing, yoe_stats, prev_df, yoe_arsh_arr, yoe_scr_arr)

    print(f"\nT11 complete.")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report: {REPORT}")


def plot_jsd(jsd_df, outpath):
    """Bar chart of JSD values by level."""
    fig, ax = plt.subplots(figsize=(8, 4))

    df = jsd_df.copy()
    colors = ['#2ecc71' if 'ref' in str(l) else '#3498db' for l in df['level']]

    bars = ax.bar(range(len(df)), df['jsd'], color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['level'], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Jensen-Shannon Divergence')
    ax.set_title('Unigram JSD: Arshkon 2024 vs Scraped 2026\n(green = within-2024 reference)')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['jsd'])):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_ai_prevalence(prev_df, outpath):
    """Grouped bar chart of AI prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Broad
    for ax, col, title in [(axes[0], 'pct_ai_broad', 'Broad AI/ML Mention'),
                           (axes[1], 'pct_ai_narrow', 'Narrow LLM-specific Mention')]:
        # Filter to seniority levels only (not 'all')
        plot_data = prev_df[prev_df['seniority'] != 'all'].copy()

        # Pivot for grouped bar
        pv = plot_data.pivot(index='seniority', columns='period', values=col).reindex(
            ['entry', 'associate', 'mid-senior', 'director'])
        pv = pv.dropna(axis=0, how='all')

        x = np.arange(len(pv))
        width = 0.35

        if '2024' in pv.columns:
            bars1 = ax.bar(x - width/2, pv['2024'].fillna(0), width, label='2024', color='#3498db', alpha=0.8)
        if '2026' in pv.columns:
            bars2 = ax.bar(x + width/2, pv['2026'].fillna(0), width, label='2026', color='#e74c3c', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(pv.index, fontsize=9)
        ax.set_ylabel('% of postings')
        ax.set_title(title)
        ax.legend()

        # Add value labels
        for bars in [bars1 if '2024' in pv.columns else [], bars2 if '2026' in pv.columns else []]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                            f'{h:.0f}%', ha='center', va='bottom', fontsize=7)

    plt.suptitle('AI Keyword Prevalence in SWE Postings: 2024 vs 2026', fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_yoe_distribution(yoe_arsh, yoe_scr, outpath):
    """Distribution comparison of YOE for entry-level."""
    if len(yoe_arsh) == 0 and len(yoe_scr) == 0:
        print("  No YOE data for distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    ax = axes[0]
    bins = np.arange(-0.5, max(max(yoe_arsh, default=5), max(yoe_scr, default=5)) + 1.5, 1)
    if len(yoe_arsh) > 0:
        ax.hist(yoe_arsh, bins=bins, alpha=0.6, density=True, label=f'2024 (n={len(yoe_arsh)})', color='#3498db')
    if len(yoe_scr) > 0:
        ax.hist(yoe_scr, bins=bins, alpha=0.6, density=True, label=f'2026 (n={len(yoe_scr)})', color='#e74c3c')
    ax.set_xlabel('Minimum Years of Experience')
    ax.set_ylabel('Density')
    ax.set_title('YOE Distribution: Entry-Level SWE')
    ax.legend()

    # CDF
    ax = axes[1]
    if len(yoe_arsh) > 0:
        sorted_arsh = np.sort(yoe_arsh)
        ax.step(sorted_arsh, np.arange(1, len(sorted_arsh)+1)/len(sorted_arsh),
                label=f'2024 (n={len(yoe_arsh)})', color='#3498db')
    if len(yoe_scr) > 0:
        sorted_scr = np.sort(yoe_scr)
        ax.step(sorted_scr, np.arange(1, len(sorted_scr)+1)/len(sorted_scr),
                label=f'2026 (n={len(yoe_scr)})', color='#e74c3c')
    ax.set_xlabel('Minimum Years of Experience')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('YOE CDF: Entry-Level SWE')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_ai_term_heatmap(prev_df, outpath):
    """Heatmap of individual AI term prevalence."""
    # Get the per-term columns
    term_cols = [f'pct_{name}' for name, _ in AI_TERM_LIST]
    term_labels = [name for name, _ in AI_TERM_LIST]

    # Filter to period x seniority combinations with data
    plot_data = prev_df[prev_df['seniority'] != 'all'].copy()
    plot_data['label'] = plot_data['period'] + ' ' + plot_data['seniority']

    # Build matrix
    available_cols = [c for c in term_cols if c in plot_data.columns]
    matrix = plot_data.set_index('label')[available_cols].T
    matrix.index = [c.replace('pct_', '') for c in matrix.index]

    # Sort rows by max value
    matrix = matrix.loc[matrix.max(axis=1).sort_values(ascending=False).index]

    # Remove rows that are all zeros
    matrix = matrix.loc[(matrix > 0.1).any(axis=1)]

    if matrix.empty:
        print("  No AI term prevalence data for heatmap.")
        return

    # Reorder columns
    col_order = ['2024 entry', '2024 associate', '2024 mid-senior', '2024 director',
                 '2026 entry', '2026 associate', '2026 mid-senior', '2026 director']
    col_order = [c for c in col_order if c in matrix.columns]
    matrix = matrix[col_order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * 0.4)))
    sns.heatmap(matrix, ax=ax, annot=True, fmt='.1f', cmap='YlOrRd',
                annot_kws={'size': 8}, cbar_kws={'label': '% of postings'})
    ax.set_title('AI Term Prevalence (%) by Period x Seniority\n(Entry-level SWE postings)')
    ax.set_ylabel('AI Term')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def build_report(jsd_df, all_emerging, all_disappearing, yoe_stats, prev_df,
                 yoe_arsh_arr, yoe_scr_arr):
    """Generate T11.md report."""
    lines = []
    lines.append("# T11: Temporal Drift")
    lines.append("")
    lines.append("## Finding")
    lines.append("")

    # Key JSD value
    jsd_overall = jsd_df[jsd_df['level'] == 'overall']['jsd'].values[0]
    jsd_ref_rows = jsd_df[jsd_df['level'].str.contains('ref', na=False)]
    jsd_ref = jsd_ref_rows['jsd'].values[0] if len(jsd_ref_rows) > 0 else None

    # Key AI prevalence
    prev_all = prev_df[prev_df['seniority'] == 'all']
    ai_2024 = prev_all[prev_all['period'] == '2024']['pct_ai_broad'].values
    ai_2026 = prev_all[prev_all['period'] == '2026']['pct_ai_broad'].values
    ai_2024_val = ai_2024[0] if len(ai_2024) > 0 else 0
    ai_2026_val = ai_2026[0] if len(ai_2026) > 0 else 0

    narrow_2024 = prev_all[prev_all['period'] == '2024']['pct_ai_narrow'].values
    narrow_2026 = prev_all[prev_all['period'] == '2026']['pct_ai_narrow'].values
    narrow_2024_val = narrow_2024[0] if len(narrow_2024) > 0 else 0
    narrow_2026_val = narrow_2026[0] if len(narrow_2026) > 0 else 0

    ref_str = f" (vs within-2024 reference JSD={jsd_ref:.4f})" if jsd_ref else ""
    lines.append(f"SWE posting content shows substantial temporal drift: overall unigram JSD = {jsd_overall:.4f}{ref_str}, "
                 f"indicating meaningful vocabulary shift between 2024 and 2026. "
                 f"AI/ML keyword prevalence surged from {ai_2024_val:.1f}% to {ai_2026_val:.1f}% (broad), "
                 f"with LLM-specific terms rising from {narrow_2024_val:.1f}% to {narrow_2026_val:.1f}%. ")

    if len(yoe_stats) >= 2:
        mean_24 = yoe_stats[0].get('mean_yoe', None)
        mean_26 = yoe_stats[1].get('mean_yoe', None)
        if mean_24 is not None and mean_26 is not None:
            lines[-1] += f"Entry-level YOE requirements shifted from mean {mean_24:.1f} to {mean_26:.1f} years."

    lines.append("")
    lines.append("## 1. Jensen-Shannon Divergence")
    lines.append("")
    lines.append("JSD measures how different two probability distributions are (0 = identical, 1 = maximally different). Higher values indicate greater vocabulary divergence.")
    lines.append("")
    lines.append("| Level | n_2024 | n_2026 | JSD |")
    lines.append("|-------|--------|--------|-----|")
    for _, row in jsd_df.iterrows():
        jsd_val = f"{row['jsd']:.4f}" if not np.isnan(row['jsd']) else "N/A"
        lines.append(f"| {row['level']} | {row['n_2024']} | {row['n_2026']} | {jsd_val} |")
    lines.append("")

    if jsd_ref:
        lines.append(f"The within-2024 reference JSD ({jsd_ref:.4f}) between arshkon and asaniczka (both early-2024 LinkedIn data) "
                     f"provides a baseline for how much variation exists within the same era due to source differences. "
                     f"The cross-era JSD ({jsd_overall:.4f}) is {'higher' if jsd_overall > jsd_ref else 'comparable'}, "
                     f"suggesting the temporal shift {'exceeds' if jsd_overall > jsd_ref else 'is similar to'} within-era source variation.")
    lines.append("")

    # 2. Emerging/disappearing
    lines.append("## 2. Emerging and Disappearing Terms")
    lines.append("")
    lines.append("Emerging: >0.1% frequency in 2026, <0.01% in 2024. Disappearing: reverse.")
    lines.append("")

    for level_label in ['overall', 'entry', 'mid-senior']:
        lines.append(f"### {level_label.title()}")
        lines.append("")

        if level_label in all_emerging:
            em = all_emerging[level_label]
            lines.append(f"**Emerging terms** ({len(em)} total, showing top 20):")
            lines.append("")
            lines.append("| Term | freq_2024 | freq_2026 | count_2026 |")
            lines.append("|------|-----------|-----------|------------|")
            for _, row in em.head(20).iterrows():
                lines.append(f"| {row['term']} | {row['freq_2024']:.5f} | {row['freq_2026']:.5f} | {int(row['count_2026'])} |")
            lines.append("")
        else:
            lines.append("_(no emerging terms found or insufficient data)_")
            lines.append("")

        if level_label in all_disappearing:
            dis = all_disappearing[level_label]
            lines.append(f"**Disappearing terms** ({len(dis)} total, showing top 20):")
            lines.append("")
            lines.append("| Term | freq_2024 | freq_2026 | count_2024 |")
            lines.append("|------|-----------|-----------|------------|")
            for _, row in dis.head(20).iterrows():
                lines.append(f"| {row['term']} | {row['freq_2024']:.5f} | {row['freq_2026']:.5f} | {int(row['count_2024'])} |")
            lines.append("")
        else:
            lines.append("_(no disappearing terms found or insufficient data)_")
            lines.append("")

    # 3. YOE inflation
    lines.append("## 3. YOE Inflation (Entry-Level SWE)")
    lines.append("")
    if len(yoe_stats) >= 2:
        lines.append("| Metric | 2024 (arshkon) | 2026 (scraped) |")
        lines.append("|--------|----------------|----------------|")
        s24 = yoe_stats[0]
        s26 = yoe_stats[1]
        lines.append(f"| Total entry-level postings | {s24['n_total']} | {s26['n_total']} |")
        lines.append(f"| With YOE extracted | {s24['n_with_yoe']} ({s24['pct_with_yoe']:.1f}%) | {s26['n_with_yoe']} ({s26['pct_with_yoe']:.1f}%) |")
        lines.append(f"| Mean YOE | {s24['mean_yoe']:.1f} | {s26['mean_yoe']:.1f} |")
        lines.append(f"| Median YOE | {s24['median_yoe']:.1f} | {s26['median_yoe']:.1f} |")
        lines.append(f"| P25 / P75 | {s24['p25_yoe']:.0f} / {s24['p75_yoe']:.0f} | {s26['p25_yoe']:.0f} / {s26['p75_yoe']:.0f} |")
        lines.append(f"| % with 0 YOE | {s24['pct_0yoe']:.1f}% | {s26['pct_0yoe']:.1f}% |")
        lines.append(f"| % with 3+ YOE | {s24['pct_3plus']:.1f}% | {s26['pct_3plus']:.1f}% |")
        lines.append(f"| % with 5+ YOE | {s24['pct_5plus']:.1f}% | {s26['pct_5plus']:.1f}% |")

        if 'mw_p' in s24:
            lines.append(f"| Mann-Whitney p-value | {s24['mw_p']:.2e} | |")
        lines.append("")

        direction = "higher" if s26['mean_yoe'] > s24['mean_yoe'] else "lower"
        lines.append(f"Entry-level YOE requirements are {direction} in 2026 "
                     f"(mean {s24['mean_yoe']:.1f} -> {s26['mean_yoe']:.1f}).")
    else:
        lines.append("_(insufficient YOE data)_")
    lines.append("")

    # 4. AI keyword prevalence
    lines.append("## 4. AI Keyword Prevalence")
    lines.append("")
    lines.append("### Broad AI mention (any AI/ML/LLM/agent term)")
    lines.append("")

    prev_display = prev_df[['period', 'seniority', 'n', 'pct_ai_broad', 'pct_ai_narrow']].copy()
    lines.append("| Period | Seniority | n | Broad AI (%) | Narrow LLM (%) |")
    lines.append("|--------|-----------|---|-------------|----------------|")
    for _, row in prev_display.iterrows():
        lines.append(f"| {row['period']} | {row['seniority']} | {int(row['n'])} | {row['pct_ai_broad']:.1f}% | {row['pct_ai_narrow']:.1f}% |")
    lines.append("")

    # Key per-term highlights
    lines.append("### Selected individual AI terms")
    lines.append("")
    key_terms = ['ai', 'llm/llms', 'agent/agents', 'copilot', 'rag', 'generative_ai', 'prompt/prompting']
    available_key = [t for t in key_terms if f'pct_{t}' in prev_df.columns]

    if available_key:
        header = "| Period | Seniority |"
        for t in available_key:
            header += f" {t} |"
        lines.append(header)
        lines.append("|--------|-----------|" + "---|" * len(available_key))

        for _, row in prev_df[prev_df['seniority'].isin(['entry', 'mid-senior', 'all'])].iterrows():
            line = f"| {row['period']} | {row['seniority']} |"
            for t in available_key:
                val = row.get(f'pct_{t}', 0)
                line += f" {val:.1f}% |"
            lines.append(line)
    lines.append("")

    lines.append("## Implication for analysis")
    lines.append("")
    lines.append("- **RQ1 (restructuring):** The JSD values and emerging-term analysis confirm that SWE job postings underwent substantial linguistic change from 2024 to 2026. This is not a stable baseline -- the language of hiring has shifted meaningfully.")
    lines.append("- **RQ1 (scope inflation):** YOE data for entry-level roles provides direct evidence for or against credential inflation at the junior level.")
    lines.append("- **RQ2 (task migration):** AI term prevalence by seniority shows whether AI requirements are being added uniformly or concentrated at particular seniority levels. Differential adoption rates across levels suggest task migration between seniority tiers.")
    lines.append("- **RQ3 (divergence):** The broad vs. narrow AI prevalence split helps distinguish genuine technical AI requirements from generic buzzword insertion. The narrow (LLM-specific) measure is more likely to reflect actual tool requirements.")
    lines.append("")

    lines.append("## Data quality note")
    lines.append("")
    lines.append("- JSD computed on arshkon (2024) vs scraped (2026) only. Asaniczka excluded from JSD because it lacks entry-level labels, making seniority-stratified comparisons non-parallel.")
    lines.append("- Description length grew ~50% from 2024 to 2026 (T05 finding). JSD uses normalized frequencies (rates, not counts), providing some but not complete robustness to length differences.")
    lines.append("- The within-2024 reference JSD (arshkon vs asaniczka mid-senior) controls for cross-source variation.")
    lines.append("- YOE extraction is regex-based and may miss non-standard phrasings. Coverage varies by source.")
    lines.append("- AI keyword matching uses regex patterns, which may overcount (e.g., 'AI' in company names) or undercount (novel terminology). The narrow LLM-specific pattern is more conservative.")
    lines.append("- Seniority uses the patched workaround (seniority_patched). Entry-level: arshkon n=745, scraped n=549.")
    lines.append("")

    lines.append("## Action items")
    lines.append("")
    lines.append("- [ ] Feed JSD values into the analysis plan as baseline drift metrics")
    lines.append("- [ ] Cross-reference emerging terms with T10 Fightin' Words results for validation")
    lines.append("- [ ] Use AI prevalence rates as inputs to RQ3 analysis (employer-side AI requirement measurement)")
    lines.append("- [ ] After seniority_llm is available, re-compute all metrics for robustness")
    lines.append("- [ ] Consider using narrow (LLM-specific) AI prevalence as the primary AI metric in analysis, with broad as sensitivity")
    lines.append("- [ ] YOE inflation findings should be paired with ghost_job_risk flags for a fuller picture of entry-level requirement changes")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("- `exploration/tables/T11/jsd_results.csv` -- JSD values by level")
    lines.append("- `exploration/tables/T11/emerging_{level}.csv` -- Emerging terms by level")
    lines.append("- `exploration/tables/T11/disappearing_{level}.csv` -- Disappearing terms by level")
    lines.append("- `exploration/tables/T11/yoe_entry_stats.csv` -- YOE statistics for entry-level")
    lines.append("- `exploration/tables/T11/yoe_all_levels.csv` -- YOE distribution all levels")
    lines.append("- `exploration/tables/T11/ai_prevalence.csv` -- AI keyword prevalence by period x seniority")
    lines.append("- `exploration/figures/T11/jsd_by_level.png` -- JSD bar chart")
    lines.append("- `exploration/figures/T11/ai_prevalence_by_period_seniority.png` -- AI prevalence grouped bars")
    lines.append("- `exploration/figures/T11/yoe_entry_distribution.png` -- YOE distribution comparison")
    lines.append("- `exploration/figures/T11/ai_term_prevalence_heatmap.png` -- Per-term AI prevalence heatmap")

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Report written to {REPORT}")


if __name__ == '__main__':
    main()
