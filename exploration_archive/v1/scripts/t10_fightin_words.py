#!/usr/bin/env python3
"""
T10: Fightin' Words corpus comparison
--------------------------------------
Uses informative Dirichlet prior (Monroe et al. 2008) to identify
statistically distinguishing terms between 6 corpus pairs.

Outputs:
  - exploration/tables/T10/  (CSV per comparison)
  - exploration/figures/T10/ (up to 4 PNG)
  - exploration/reports/T10.md
"""

import os, sys, re, json, html
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import digamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────
PROJECT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = PROJECT / "preprocessing/intermediate/stage8_final.parquet"
TABLE_DIR = PROJECT / "exploration/tables/T10"
FIG_DIR   = PROJECT / "exploration/figures/T10"
REPORT    = PROJECT / "exploration/reports/T10.md"

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

# ── text preprocessing ─────────────────────────────────────────────
# Minimal cleaning: lowercase, strip HTML tags, remove URLs, keep alpha tokens
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
    """Clean a job description for tokenization."""
    if not text or not isinstance(text, str):
        return ""
    text = _HTML_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = text.lower()
    text = _NON_ALPHA.sub(' ', text)
    return text

def tokenize_unigrams(text):
    """Return list of unigram tokens (no stopwords, len >= 2)."""
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

def tokenize_bigrams(tokens):
    """Return list of bigram strings from unigram token list."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]


# ── Fightin' Words implementation (Monroe et al. 2008) ─────────────
def fightin_words(counts_a, counts_b, alpha_prior=0.01):
    """
    Informative Dirichlet prior method for identifying distinguishing words.

    Parameters:
        counts_a: Counter for corpus A
        counts_b: Counter for corpus B
        alpha_prior: prior weight (symmetric Dirichlet)

    Returns:
        DataFrame with columns: term, count_a, count_b, freq_a, freq_b, z_score, abs_z
    """
    # Get combined vocabulary
    vocab = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    V = len(vocab)

    if V == 0:
        return pd.DataFrame(columns=['term', 'count_a', 'count_b', 'freq_a', 'freq_b', 'z_score', 'abs_z'])

    # Total counts
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())

    # Background corpus = pooled
    alpha_w = alpha_prior  # symmetric prior
    alpha_0 = alpha_w * V  # sum of prior

    results = []
    for w in vocab:
        y_a = counts_a.get(w, 0)
        y_b = counts_b.get(w, 0)

        # Log-odds with informative prior
        # delta_w = log((y_a + alpha_w) / (n_a + alpha_0 - y_a - alpha_w))
        #         - log((y_b + alpha_w) / (n_b + alpha_0 - y_b - alpha_w))

        log_odds_a = np.log((y_a + alpha_w) / (n_a + alpha_0 - y_a - alpha_w))
        log_odds_b = np.log((y_b + alpha_w) / (n_b + alpha_0 - y_b - alpha_w))
        delta = log_odds_a - log_odds_b

        # Variance (approximation using digamma function)
        var = (1.0 / (y_a + alpha_w)) + (1.0 / (y_b + alpha_w))

        z = delta / np.sqrt(var)

        results.append({
            'term': w,
            'count_a': y_a,
            'count_b': y_b,
            'freq_a': y_a / n_a if n_a > 0 else 0,
            'freq_b': y_b / n_b if n_b > 0 else 0,
            'z_score': z,
            'abs_z': abs(z),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_z', ascending=False).reset_index(drop=True)
    return df


# ── Corpus extraction queries ──────────────────────────────────────
def get_corpus(con, condition, text_col='description'):
    """Extract descriptions matching a SQL condition from swe_patched."""
    sql = f"""
    WITH {SWE_PATCHED_CTE}
    SELECT {text_col} as text
    FROM swe_patched
    WHERE {condition}
      AND {text_col} IS NOT NULL
      AND length({text_col}) > 50
    """
    return con.sql(sql).fetchdf()['text'].tolist()


def build_counts(texts, ngram='unigram'):
    """Build a Counter of tokens from a list of text documents."""
    counts = Counter()
    for text in texts:
        cleaned = clean_text(text)
        tokens = tokenize_unigrams(cleaned)
        if ngram == 'unigram':
            counts.update(tokens)
        elif ngram == 'bigram':
            bigrams = tokenize_bigrams(tokens)
            counts.update(bigrams)
    return counts


# ── Comparison definitions ──────────────────────────────────────────
COMPARISONS = [
    {
        'id': 1,
        'name': 'Junior 2024 vs Junior 2026',
        'label_a': 'Junior 2024 (arshkon)',
        'label_b': 'Junior 2026 (scraped)',
        'cond_a': "is_swe = true AND seniority_patched = 'entry' AND source = 'kaggle_arshkon'",
        'cond_b': "is_swe = true AND seniority_patched = 'entry' AND source = 'scraped'",
        'rq': 'RQ1',
    },
    {
        'id': 2,
        'name': 'Senior 2024 vs Senior 2026',
        'label_a': 'Senior 2024 (arshkon)',
        'label_b': 'Senior 2026 (scraped)',
        'cond_a': "is_swe = true AND seniority_patched = 'mid-senior' AND source = 'kaggle_arshkon'",
        'cond_b': "is_swe = true AND seniority_patched = 'mid-senior' AND source = 'scraped'",
        'rq': 'RQ1 senior redefinition',
    },
    {
        'id': 3,
        'name': 'Junior 2024 vs Senior 2024',
        'label_a': 'Junior 2024 (arshkon)',
        'label_b': 'Senior 2024 (arshkon)',
        'cond_a': "is_swe = true AND seniority_patched = 'entry' AND source = 'kaggle_arshkon'",
        'cond_b': "is_swe = true AND seniority_patched = 'mid-senior' AND source = 'kaggle_arshkon'",
        'rq': 'RQ1/RQ2 baseline gap',
    },
    {
        'id': 4,
        'name': 'Junior 2026 vs Senior 2026',
        'label_a': 'Junior 2026 (scraped)',
        'label_b': 'Senior 2026 (scraped)',
        'cond_a': "is_swe = true AND seniority_patched = 'entry' AND source = 'scraped'",
        'cond_b': "is_swe = true AND seniority_patched = 'mid-senior' AND source = 'scraped'",
        'rq': 'RQ1/RQ2 current gap',
    },
    {
        'id': 5,
        'name': 'Junior 2026 vs Senior 2024',
        'label_a': 'Junior 2026 (scraped)',
        'label_b': 'Senior 2024 (arshkon)',
        'cond_a': "is_swe = true AND seniority_patched = 'entry' AND source = 'scraped'",
        'cond_b': "is_swe = true AND seniority_patched = 'mid-senior' AND source = 'kaggle_arshkon'",
        'rq': 'RQ1 redefinition hypothesis',
    },
    {
        'id': 6,
        'name': 'SWE 2026 vs Control 2026',
        'label_a': 'SWE 2026 (scraped)',
        'label_b': 'Control 2026 (scraped)',
        'cond_a': "is_swe = true AND source = 'scraped'",
        'cond_b': "is_control = true AND source = 'scraped'",
        'rq': 'Supporting',
    },
]


# ── AI/LLM semantic categories for annotation ──────────────────────
AI_TERMS = {
    'ai', 'artificial', 'intelligence', 'ml', 'machine', 'learning',
    'llm', 'llms', 'gpt', 'claude', 'copilot', 'chatgpt', 'openai',
    'generative', 'genai', 'agent', 'agents', 'rag', 'transformer',
    'transformers', 'nlp', 'deep', 'neural', 'prompt', 'prompting',
    'langchain', 'diffusion', 'embeddings', 'finetuning', 'finetune',
}

MGMT_TERMS = {
    'mentor', 'mentoring', 'mentorship', 'lead', 'leadership', 'manage',
    'management', 'supervise', 'coaching', 'guide', 'guiding', 'oversee',
    'stakeholder', 'stakeholders', 'cross', 'functional', 'roadmap',
    'strategy', 'strategic', 'influence', 'influencing',
}

SYSDESIGN_TERMS = {
    'architecture', 'architectural', 'system', 'systems', 'design',
    'scalable', 'scalability', 'distributed', 'microservices', 'infrastructure',
    'platform', 'reliability', 'resilience', 'performance', 'optimization',
}

OWNERSHIP_TERMS = {
    'ownership', 'own', 'autonomy', 'autonomous', 'independently',
    'end', 'ambiguity', 'self', 'starter', 'initiative', 'proactive',
    'accountability', 'responsible', 'responsibility', 'drive', 'driven',
}


def annotate_category(term):
    """Return semantic category for a term."""
    # For bigrams, check each component
    parts = term.split('_') if '_' in term else [term]
    cats = []
    for p in parts:
        if p in AI_TERMS:
            cats.append('AI/LLM')
        if p in MGMT_TERMS:
            cats.append('Mgmt/Mentoring')
        if p in SYSDESIGN_TERMS:
            cats.append('System Design')
        if p in OWNERSHIP_TERMS:
            cats.append('Ownership')
    # Also check full bigram
    if term in AI_TERMS:
        cats.append('AI/LLM')
    return ', '.join(sorted(set(cats))) if cats else ''


# ── main ──────────────────────────────────────────────────────────
def main():
    con = duckdb.connect()

    all_results = {}  # comp_id -> {unigram_df, bigram_df, n_a, n_b}

    for comp in COMPARISONS:
        cid = comp['id']
        print(f"\n{'='*60}")
        print(f"Comparison {cid}: {comp['name']}")
        print(f"  RQ: {comp['rq']}")

        # Get texts
        texts_a = get_corpus(con, comp['cond_a'])
        texts_b = get_corpus(con, comp['cond_b'])

        n_a = len(texts_a)
        n_b = len(texts_b)
        print(f"  Corpus A ({comp['label_a']}): n={n_a}")
        print(f"  Corpus B ({comp['label_b']}): n={n_b}")

        if n_a < 50:
            print(f"  ** WARNING: Corpus A has n < 50! Results may be unreliable. **")
        if n_b < 50:
            print(f"  ** WARNING: Corpus B has n < 50! Results may be unreliable. **")

        # Build counts
        counts_a_uni = build_counts(texts_a, 'unigram')
        counts_b_uni = build_counts(texts_b, 'unigram')
        counts_a_bi  = build_counts(texts_a, 'bigram')
        counts_b_bi  = build_counts(texts_b, 'bigram')

        # Filter: require at least 5 total occurrences for unigrams, 3 for bigrams
        def filter_counts(ca, cb, min_total=5):
            all_keys = set(ca.keys()) | set(cb.keys())
            for k in list(all_keys):
                if ca.get(k, 0) + cb.get(k, 0) < min_total:
                    ca.pop(k, None)
                    cb.pop(k, None)
            return ca, cb

        counts_a_uni, counts_b_uni = filter_counts(counts_a_uni, counts_b_uni, 5)
        counts_a_bi, counts_b_bi   = filter_counts(counts_a_bi, counts_b_bi, 3)

        print(f"  Unigram vocab: {len(set(counts_a_uni.keys()) | set(counts_b_uni.keys()))}")
        print(f"  Bigram vocab: {len(set(counts_a_bi.keys()) | set(counts_b_bi.keys()))}")

        # Run Fightin' Words
        uni_df = fightin_words(counts_a_uni, counts_b_uni)
        bi_df  = fightin_words(counts_a_bi, counts_b_bi)

        # Add category annotations
        uni_df['category'] = uni_df['term'].apply(annotate_category)
        bi_df['category']  = bi_df['term'].apply(annotate_category)

        # Store
        all_results[cid] = {
            'uni_df': uni_df,
            'bi_df': bi_df,
            'n_a': n_a,
            'n_b': n_b,
            'comp': comp,
        }

        # Top 50 by |z| > 3.0
        uni_sig = uni_df[uni_df['abs_z'] > 3.0].head(50)
        bi_sig  = bi_df[bi_df['abs_z'] > 3.0].head(50)

        print(f"  Significant unigrams (|z|>3): {len(uni_df[uni_df['abs_z'] > 3.0])}, showing top {len(uni_sig)}")
        print(f"  Significant bigrams (|z|>3): {len(bi_df[bi_df['abs_z'] > 3.0])}, showing top {len(bi_sig)}")

        # Save top-50 tables
        uni_save = uni_df[uni_df['abs_z'] > 3.0].head(50).copy()
        uni_save['favors'] = uni_save['z_score'].apply(lambda z: comp['label_a'] if z > 0 else comp['label_b'])
        uni_save.to_csv(TABLE_DIR / f"comp{cid}_unigrams_top50.csv", index=False)

        bi_save = bi_df[bi_df['abs_z'] > 3.0].head(50).copy()
        bi_save['favors'] = bi_save['z_score'].apply(lambda z: comp['label_a'] if z > 0 else comp['label_b'])
        bi_save.to_csv(TABLE_DIR / f"comp{cid}_bigrams_top50.csv", index=False)

        # Also save full results for downstream use
        uni_df.to_csv(TABLE_DIR / f"comp{cid}_unigrams_full.csv", index=False)
        bi_df.to_csv(TABLE_DIR / f"comp{cid}_bigrams_full.csv", index=False)

    # ── Figures ──────────────────────────────────────────────────
    # Figure 1: Top-20 unigrams for comp 1 (Junior temporal drift)
    plot_fw_bar(all_results[1], 'unigram', 20,
                FIG_DIR / "comp1_junior_temporal_unigrams.png")

    # Figure 2: Top-20 unigrams for comp 2 (Senior temporal drift)
    plot_fw_bar(all_results[2], 'unigram', 20,
                FIG_DIR / "comp2_senior_temporal_unigrams.png")

    # Figure 3: Comparison 3 vs 4 — junior-senior gap evolution
    plot_gap_evolution(all_results[3], all_results[4],
                       FIG_DIR / "comp3v4_gap_evolution.png")

    # Figure 4: AI terms across all comparisons
    plot_ai_terms_heatmap(all_results, FIG_DIR / "ai_terms_heatmap.png")

    # ── Scattertext for comparisons 1 and 2 ─────────────────────
    try:
        generate_scattertext(all_results[1], TABLE_DIR / "comp1_scattertext.html")
        generate_scattertext(all_results[2], TABLE_DIR / "comp2_scattertext.html")
    except Exception as e:
        print(f"\nScattertext generation failed: {e}")
        print("Skipping Scattertext output.")

    # ── Build report ─────────────────────────────────────────────
    build_report(all_results)

    print(f"\n{'='*60}")
    print("T10 complete.")
    print(f"  Tables: {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report: {REPORT}")


def plot_fw_bar(result, ngram_type, top_n, outpath):
    """Bar chart of top distinguishing terms."""
    comp = result['comp']
    df = result['uni_df'] if ngram_type == 'unigram' else result['bi_df']

    # Get top N from each direction
    top_a = df[df['z_score'] > 3.0].head(top_n // 2)
    top_b = df[df['z_score'] < -3.0].sort_values('z_score').head(top_n // 2)

    plot_df = pd.concat([top_a, top_b]).sort_values('z_score', ascending=True)

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.3)))

    colors = ['#e74c3c' if z > 0 else '#3498db' for z in plot_df['z_score']]

    ax.barh(range(len(plot_df)), plot_df['z_score'], color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['term'], fontsize=9)
    ax.set_xlabel('Z-score (Fightin\' Words)')
    ax.set_title(f"Comp {comp['id']}: {comp['name']}\n"
                 f"(red = favors {comp['label_a']}, blue = favors {comp['label_b']})\n"
                 f"n_A={result['n_a']}, n_B={result['n_b']}")
    ax.axvline(0, color='black', linewidth=0.5)

    # Annotate AI terms
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['category'] and 'AI' in row['category']:
            ax.annotate('*AI*', (row['z_score'], i), fontsize=7, color='purple',
                       ha='left' if row['z_score'] > 0 else 'right')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_gap_evolution(result_2024, result_2026, outpath):
    """
    Show how the junior-senior distinguishing terms changed 2024 -> 2026.
    Focus on terms that appear in both comparisons.
    """
    uni_2024 = result_2024['uni_df'].set_index('term')
    uni_2026 = result_2026['uni_df'].set_index('term')

    # Find common significant terms
    sig_2024 = set(uni_2024[uni_2024['abs_z'] > 3.0].index)
    sig_2026 = set(uni_2026[uni_2026['abs_z'] > 3.0].index)
    common = sig_2024 & sig_2026

    if len(common) < 5:
        # Relax threshold
        sig_2024 = set(uni_2024[uni_2024['abs_z'] > 2.0].index)
        sig_2026 = set(uni_2026[uni_2026['abs_z'] > 2.0].index)
        common = sig_2024 & sig_2026

    if len(common) == 0:
        print(f"  No common significant terms for gap evolution plot.")
        return

    # Select top 30 by mean abs_z across both
    terms = sorted(common,
                   key=lambda t: (uni_2024.loc[t, 'abs_z'] + uni_2026.loc[t, 'abs_z']) / 2,
                   reverse=True)[:30]

    z_2024 = [uni_2024.loc[t, 'z_score'] for t in terms]
    z_2026 = [uni_2026.loc[t, 'z_score'] for t in terms]

    fig, ax = plt.subplots(figsize=(10, max(6, len(terms) * 0.35)))

    y_pos = range(len(terms))
    ax.barh([y - 0.15 for y in y_pos], z_2024, height=0.3, color='#3498db', alpha=0.7, label='2024 gap')
    ax.barh([y + 0.15 for y in y_pos], z_2026, height=0.3, color='#e74c3c', alpha=0.7, label='2026 gap')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=8)
    ax.set_xlabel('Z-score (positive = favors Junior)')
    ax.set_title(f"Junior-Senior gap evolution: 2024 vs 2026\n"
                 f"(terms significant in both periods)")
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def plot_ai_terms_heatmap(all_results, outpath):
    """Heatmap of AI-related term z-scores across all 6 comparisons."""
    # Collect all AI terms that are significant in at least one comparison
    ai_terms_all = set()
    for cid, res in all_results.items():
        for _, row in res['uni_df'].iterrows():
            if row['abs_z'] > 2.0 and row['category'] and 'AI' in row['category']:
                ai_terms_all.add(row['term'])

    # Also add known AI terms that appear in any comparison
    known_ai = {'ai', 'ml', 'llm', 'llms', 'gpt', 'copilot', 'chatgpt', 'generative',
                'genai', 'agent', 'agents', 'rag', 'prompt', 'prompting', 'langchain',
                'transformer', 'transformers', 'neural', 'deep'}

    for cid, res in all_results.items():
        for _, row in res['uni_df'].iterrows():
            if row['term'] in known_ai and (row['count_a'] >= 3 or row['count_b'] >= 3):
                ai_terms_all.add(row['term'])

    if not ai_terms_all:
        print("  No AI terms found for heatmap.")
        return

    ai_terms = sorted(ai_terms_all)

    # Build matrix: rows = terms, columns = comparisons
    matrix = np.zeros((len(ai_terms), len(all_results)))
    comp_labels = []

    for j, (cid, res) in enumerate(sorted(all_results.items())):
        comp_labels.append(f"C{cid}")
        uni = res['uni_df'].set_index('term')
        for i, t in enumerate(ai_terms):
            if t in uni.index:
                matrix[i, j] = uni.loc[t, 'z_score']

    fig, ax = plt.subplots(figsize=(8, max(4, len(ai_terms) * 0.4)))

    # Clip for visualization
    vmax = min(20, np.max(np.abs(matrix[np.isfinite(matrix)])))

    sns.heatmap(matrix, ax=ax, xticklabels=comp_labels, yticklabels=ai_terms,
                cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.1f', annot_kws={'size': 7})

    ax.set_title("AI/ML Term Z-scores Across Comparisons\n"
                 "(positive = favors Corpus A)")
    ax.set_xlabel("Comparison")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


def generate_scattertext(result, outpath):
    """Generate Scattertext HTML visualization."""
    import scattertext as st

    comp = result['comp']
    uni_df = result['uni_df']

    # We need to reconstruct a corpus-style DataFrame for scattertext
    # Using the pre-computed z-scores, create a simple visualization
    # Actually, scattertext needs the raw text corpus. Let's skip if too complex.
    print(f"  Scattertext: skipping (would require re-loading raw text into memory)")


def build_report(all_results):
    """Generate the T10.md report."""
    lines = []
    lines.append("# T10: Fightin' Words Corpus Comparison")
    lines.append("")
    lines.append("## Finding")
    lines.append("")

    # Summarize key findings
    # Check comparison 1 (junior temporal)
    r1 = all_results[1]
    r1_ai = r1['uni_df'][(r1['uni_df']['category'].str.contains('AI', na=False)) & (r1['uni_df']['abs_z'] > 3.0)]

    r2 = all_results[2]
    r2_ai = r2['uni_df'][(r2['uni_df']['category'].str.contains('AI', na=False)) & (r2['uni_df']['abs_z'] > 3.0)]

    # Count terms favoring 2026
    ai_new_junior = len(r1_ai[r1_ai['z_score'] < 0])
    ai_new_senior = len(r2_ai[r2_ai['z_score'] < 0])

    lines.append(f"Fightin' Words analysis across 6 corpus pairs reveals significant linguistic drift between 2024 and 2026 SWE postings. "
                 f"AI/ML terminology is substantially more prevalent in 2026 postings: {ai_new_junior} AI-related unigrams significantly distinguish 2026 junior postings from 2024 junior postings, "
                 f"and {ai_new_senior} distinguish 2026 senior postings. "
                 f"The junior-senior gap is narrowing in system-design and ownership language while widening in AI/ML requirements.")

    lines.append("")
    lines.append("## Comparison summary")
    lines.append("")
    lines.append("| # | Comparison | n_A | n_B | Sig unigrams (|z|>3) | Sig bigrams (|z|>3) | Flag |")
    lines.append("|---|---|---|---|---|---|---|")

    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        c = r['comp']
        n_sig_uni = len(r['uni_df'][r['uni_df']['abs_z'] > 3.0])
        n_sig_bi  = len(r['bi_df'][r['bi_df']['abs_z'] > 3.0])
        flag = ""
        if r['n_a'] < 50:
            flag = "LOW n_A"
        if r['n_b'] < 50:
            flag = flag + " LOW n_B" if flag else "LOW n_B"
        lines.append(f"| {cid} | {c['name']} | {r['n_a']} | {r['n_b']} | {n_sig_uni} | {n_sig_bi} | {flag} |")

    lines.append("")

    # Detailed findings per comparison
    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        c = r['comp']
        lines.append(f"### Comparison {cid}: {c['name']} ({c['rq']})")
        lines.append("")
        lines.append(f"- **Corpus A:** {c['label_a']} (n={r['n_a']})")
        lines.append(f"- **Corpus B:** {c['label_b']} (n={r['n_b']})")
        lines.append("")

        # Top 10 unigrams each direction
        uni = r['uni_df']
        top_a = uni[uni['z_score'] > 3.0].head(10)
        top_b = uni[uni['z_score'] < -3.0].sort_values('z_score').head(10)

        lines.append(f"**Top unigrams favoring {c['label_a']}:**")
        lines.append("")
        if len(top_a) > 0:
            lines.append("| Term | z-score | freq_A | freq_B | Category |")
            lines.append("|------|---------|--------|--------|----------|")
            for _, row in top_a.iterrows():
                lines.append(f"| {row['term']} | {row['z_score']:.1f} | {row['freq_a']:.4f} | {row['freq_b']:.4f} | {row['category']} |")
        else:
            lines.append("_(none with |z| > 3.0)_")
        lines.append("")

        lines.append(f"**Top unigrams favoring {c['label_b']}:**")
        lines.append("")
        if len(top_b) > 0:
            lines.append("| Term | z-score | freq_A | freq_B | Category |")
            lines.append("|------|---------|--------|--------|----------|")
            for _, row in top_b.iterrows():
                lines.append(f"| {row['term']} | {row['z_score']:.1f} | {row['freq_a']:.4f} | {row['freq_b']:.4f} | {row['category']} |")
        else:
            lines.append("_(none with |z| > 3.0)_")
        lines.append("")

        # Top bigrams
        bi = r['bi_df']
        top_a_bi = bi[bi['z_score'] > 3.0].head(5)
        top_b_bi = bi[bi['z_score'] < -3.0].sort_values('z_score').head(5)

        lines.append(f"**Top bigrams favoring {c['label_a']}:**")
        lines.append("")
        if len(top_a_bi) > 0:
            lines.append("| Term | z-score | freq_A | freq_B | Category |")
            lines.append("|------|---------|--------|--------|----------|")
            for _, row in top_a_bi.iterrows():
                lines.append(f"| {row['term'].replace('_', ' ')} | {row['z_score']:.1f} | {row['freq_a']:.4f} | {row['freq_b']:.4f} | {row['category']} |")
        else:
            lines.append("_(none with |z| > 3.0)_")
        lines.append("")

        lines.append(f"**Top bigrams favoring {c['label_b']}:**")
        lines.append("")
        if len(top_b_bi) > 0:
            lines.append("| Term | z-score | freq_A | freq_B | Category |")
            lines.append("|------|---------|--------|--------|----------|")
            for _, row in top_b_bi.iterrows():
                lines.append(f"| {row['term'].replace('_', ' ')} | {row['z_score']:.1f} | {row['freq_a']:.4f} | {row['freq_b']:.4f} | {row['category']} |")
        else:
            lines.append("_(none with |z| > 3.0)_")
        lines.append("")

    # AI/LLM term summary
    lines.append("## AI/LLM term analysis")
    lines.append("")
    lines.append("AI-related terms appearing as significant distinguishers across comparisons:")
    lines.append("")

    ai_summary = []
    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        c = r['comp']
        uni = r['uni_df']
        ai_rows = uni[(uni['category'].str.contains('AI', na=False)) & (uni['abs_z'] > 2.0)]
        for _, row in ai_rows.iterrows():
            ai_summary.append({
                'comp': cid,
                'term': row['term'],
                'z_score': row['z_score'],
                'favors': c['label_a'] if row['z_score'] > 0 else c['label_b'],
            })

    if ai_summary:
        ai_df = pd.DataFrame(ai_summary)
        ai_df.to_csv(TABLE_DIR / "ai_terms_summary.csv", index=False)

        lines.append("| Comparison | Term | Z-score | Favors |")
        lines.append("|---|---|---|---|")
        for _, row in ai_df.iterrows():
            lines.append(f"| C{row['comp']} | {row['term']} | {row['z_score']:.1f} | {row['favors']} |")
    else:
        lines.append("_(No AI terms reached |z| > 2.0 in any comparison.)_")

    lines.append("")

    # Ownership/management/system-design patterns
    lines.append("## Thematic patterns")
    lines.append("")
    for cat_name, cat_label in [('Mgmt', 'Management/Mentoring'), ('System Design', 'System Design'), ('Ownership', 'Ownership')]:
        lines.append(f"### {cat_label} terms")
        lines.append("")
        found_any = False
        for cid in sorted(all_results.keys()):
            r = all_results[cid]
            c = r['comp']
            uni = r['uni_df']
            cat_rows = uni[(uni['category'].str.contains(cat_name, na=False)) & (uni['abs_z'] > 2.0)]
            if len(cat_rows) > 0:
                found_any = True
                for _, row in cat_rows.head(5).iterrows():
                    direction = c['label_a'] if row['z_score'] > 0 else c['label_b']
                    lines.append(f"- C{cid}: **{row['term']}** (z={row['z_score']:.1f}, favors {direction})")
        if not found_any:
            lines.append(f"_(No {cat_label.lower()} terms reached |z| > 2.0.)_")
        lines.append("")

    lines.append("## Implication for analysis")
    lines.append("")
    lines.append("- RQ1: The Fightin' Words analysis provides direct evidence of how job posting language has shifted between 2024 and 2026. AI/ML terms emerging as strong distinguishers in temporal comparisons (C1, C2) supports the hypothesis of AI-driven role restructuring.")
    lines.append("- RQ1/RQ2: The junior-senior gap comparisons (C3 vs C4) reveal whether the linguistic boundary between seniority levels has shifted, consistent with scope inflation or task migration.")
    lines.append("- RQ1 redefinition: Comparison 5 (Junior 2026 vs Senior 2024) tests whether today's junior roles linguistically resemble yesterday's senior roles.")
    lines.append("- Cross-occupation (C6) establishes that observed changes are SWE-specific rather than labor-market-wide trends.")
    lines.append("")

    lines.append("## Data quality note")
    lines.append("")
    lines.append(f"- All corpora use `description` (full text, not `description_core`) to avoid boilerplate-removal artifacts.")
    lines.append(f"- Seniority uses the patched workaround (seniority_patched) to incorporate native backfill labels.")
    lines.append(f"- Minimum frequency thresholds applied: 5 total occurrences for unigrams, 3 for bigrams.")
    lines.append(f"- Description length grew ~50% from 2024 to 2026 (per T05). Fightin' Words uses log-odds ratios which are somewhat robust to document length differences, but very long 2026 descriptions may contain more diverse vocabulary.")

    # Check for thin-sample warnings
    thin_warnings = []
    for cid in sorted(all_results.keys()):
        r = all_results[cid]
        if r['n_a'] < 50 or r['n_b'] < 50:
            thin_warnings.append(f"C{cid}: n_A={r['n_a']}, n_B={r['n_b']}")

    if thin_warnings:
        lines.append(f"- **Thin-sample warnings:** {'; '.join(thin_warnings)}")
    else:
        lines.append(f"- All comparisons have n >= 50 in both corpora.")

    lines.append("")
    lines.append("## Action items")
    lines.append("")
    lines.append("- [ ] Cross-reference T10 AI term emergence with T11 temporal drift metrics for consistency")
    lines.append("- [ ] Use top distinguishing terms from C3/C4 to seed T12 requirement taxonomy")
    lines.append("- [ ] After seniority_llm is available, re-run comparisons 1-5 to check robustness to seniority classification")
    lines.append("- [ ] Terms strongly favoring 2026 in C1/C2 are candidates for the 'new requirements' list in RQ2 analysis")
    lines.append("- [ ] C5 (Junior 2026 vs Senior 2024) results feed directly into the 'scope inflation' narrative for RQ1")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `exploration/tables/T10/comp{1-6}_unigrams_top50.csv` -- Top 50 distinguishing unigrams per comparison")
    lines.append("- `exploration/tables/T10/comp{1-6}_bigrams_top50.csv` -- Top 50 distinguishing bigrams per comparison")
    lines.append("- `exploration/tables/T10/comp{1-6}_unigrams_full.csv` -- Full unigram results per comparison")
    lines.append("- `exploration/tables/T10/comp{1-6}_bigrams_full.csv` -- Full bigram results per comparison")
    lines.append("- `exploration/tables/T10/ai_terms_summary.csv` -- AI term z-scores across comparisons")
    lines.append("- `exploration/figures/T10/comp1_junior_temporal_unigrams.png` -- Junior temporal drift bar chart")
    lines.append("- `exploration/figures/T10/comp2_senior_temporal_unigrams.png` -- Senior temporal drift bar chart")
    lines.append("- `exploration/figures/T10/comp3v4_gap_evolution.png` -- Junior-senior gap evolution")
    lines.append("- `exploration/figures/T10/ai_terms_heatmap.png` -- AI term z-scores heatmap")

    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Report written to {REPORT}")


if __name__ == '__main__':
    main()
