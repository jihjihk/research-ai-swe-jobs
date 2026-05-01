"""V1 Phase A — Headline H5 re-derivation.

Independently computes per-period share for RAG, multimodal, MCP, multi-agent.
Claims: RAG 75x, multimodal 31x, MCP 29x, multi-agent 24x.

Approach:
- Load swe_cleaned_text.parquet (cleaned text).
- Compute per-period binary share and count.
- Compare growth ratios to T12 claims.
- Sensitivity: also compute on raw description.
- Sensitivity: examine if growth is concentrated in a handful of companies.
"""

import re
import duckdb
import numpy as np
import pandas as pd

print("[V1_H5] Loading text data", flush=True)
con = duckdb.connect()

df = con.execute("""
    SELECT c.uid,
           c.description_cleaned,
           c.text_source,
           u.description,
           u.period,
           u.company_name_canonical,
           u.is_aggregator,
           u.source
    FROM read_parquet('exploration/artifacts/shared/swe_cleaned_text.parquet') c
    LEFT JOIN read_parquet('data/unified.parquet') u
      ON c.uid = u.uid
    WHERE u.source_platform = 'linkedin'
      AND u.is_english = TRUE
      AND u.date_flag = 'ok'
      AND u.is_swe = TRUE
""").fetchdf()
print(f"[V1_H5] Loaded {len(df)} SWE LinkedIn rows", flush=True)

df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
# Use cleaned text if LLM, else raw description as fallback for the scraped not_selected rows
df['text_for_clean'] = df['description_cleaned'].fillna('').astype(str).str.lower()
df['text_for_raw']   = df['description'].fillna('').astype(str).str.lower()

print(f"[V1_H5] Period distribution:", flush=True)
print(df['period2'].value_counts(), flush=True)

# Also split cleaned-LLM vs not, so we use only llm for the primary comparison
df_llm = df[df['text_source'] == 'llm'].copy()
print(f"[V1_H5] LLM-only subset: {len(df_llm)}", flush=True)

# Patterns
TERMS = {
    "rag":        r"\brag\b",
    "multimodal": r"\bmulti[\s\-]?modal\b",
    "mcp":        r"\bmcp\b",
    "multi-agent": r"\bmulti[\s\-]?agent(s|ic)?\b",
    "llm":        r"\bllm(s)?\b",
    "agent":      r"\bagent(s|ic)?\b",
    "embeddings": r"\bembedding(s)?\b",
    "fine-tuning": r"\bfine[\s\-]?tun(e|ing|ed)\b",
}

# Assert patterns work
assert re.search(TERMS['rag'], "we use rag for retrieval") is not None
assert re.search(TERMS['rag'], "dragged the data") is None  # rag should not match "dragged"
assert re.search(TERMS['rag'], "forage") is None
assert re.search(TERMS['multi-agent'], "multi-agent systems") is not None
# multi[\s\-]? already matches "multiagent" because ? makes separator optional
# Let's add a more permissive multi-agent check that explicitly handles bare "multiagent":
TERMS['multi-agent'] = r"\bmulti[\s\-]?agent(s|ic)?\b|\bmultiagent\b"
assert re.search(TERMS['multi-agent'], "multiagent") is not None
assert re.search(TERMS['multi-agent'], "multi agent architectures") is not None
assert re.search(TERMS['multi-agent'], "multi-agentic") is not None
assert re.search(TERMS['mcp'], "model context protocol (mcp)") is not None  # lowercased input

def bin_share(series, pat):
    m = series.str.contains(pat, regex=True, na=False)
    return m.mean(), int(m.sum())

print("\n[V1_H5] === AI TERM ACCELERATION (LLM-cleaned only, POOLED 2024) ===", flush=True)
print(f"{'term':<15} {'2024_share':>12} {'2024_n':>8} {'2026_share':>12} {'2026_n':>8} {'ratio_share':>12} {'ratio_count':>12}", flush=True)
rows = []
for term, pat in TERMS.items():
    s24, n24 = bin_share(df_llm[df_llm['period2']=='2024']['text_for_clean'], pat)
    s26, n26 = bin_share(df_llm[df_llm['period2']=='2026']['text_for_clean'], pat)
    # Denominators
    d24 = (df_llm['period2']=='2024').sum()
    d26 = (df_llm['period2']=='2026').sum()
    ratio_share = s26 / max(s24, 1e-9)
    ratio_count = n26 / max(n24, 1)
    rows.append({"term": term, "cleaned_2024_share": s24, "cleaned_2024_n": n24,
                 "cleaned_2026_share": s26, "cleaned_2026_n": n26,
                 "denom_2024": d24, "denom_2026": d26,
                 "ratio_share": ratio_share, "ratio_count": ratio_count})
    print(f"{term:<15} {s24:>12.5f} {n24:>8} {s26:>12.5f} {n26:>8} "
          f"{ratio_share:>12.1f} {ratio_count:>12.1f}", flush=True)

# T12 primary uses ARSHKON-ONLY for 2024 baseline (per report section 1)
print("\n[V1_H5] === T12-ALIGNED: ARSHKON-only 2024 baseline vs scraped 2026 ===", flush=True)
print(f"{'term':<15} {'2024_share':>12} {'2024_n':>8} {'2026_share':>12} {'2026_n':>8} {'ratio_share':>12} {'ratio_count':>12}", flush=True)
d24_ark = df_llm[(df_llm['period2']=='2024') & (df_llm['source']=='kaggle_arshkon')]
d26_all = df_llm[df_llm['period2']=='2026']
print(f"[V1_H5] n_arshkon_llm={len(d24_ark)} n_scraped_llm={len(d26_all)}", flush=True)
for term, pat in TERMS.items():
    s24, n24 = bin_share(d24_ark['text_for_clean'], pat)
    s26, n26 = bin_share(d26_all['text_for_clean'], pat)
    ratio_share = s26 / max(s24, 1e-9)
    ratio_count = n26 / max(n24, 1)
    print(f"{term:<15} {s24:>12.5f} {n24:>8} {s26:>12.5f} {n26:>8} "
          f"{ratio_share:>12.1f} {ratio_count:>12.1f}", flush=True)

# Do the same on RAW description (text_source = raw falls back to raw descr)
print("\n[V1_H5] === SAME TERMS on RAW description (denominator = ALL SWE LinkedIn) ===", flush=True)
print(f"{'term':<15} {'2024_share':>12} {'2026_share':>12} {'ratio':>8}", flush=True)
for term, pat in TERMS.items():
    s24, n24 = bin_share(df[df['period2']=='2024']['text_for_raw'], pat)
    s26, n26 = bin_share(df[df['period2']=='2026']['text_for_raw'], pat)
    print(f"{term:<15} {s24:>12.5f} {s26:>12.5f} {s26/max(s24,1e-9):>8.1f}", flush=True)

# Concentration check: for RAG, multimodal, MCP, multi-agent — how much is in top 10 companies?
print("\n[V1_H5] === CONCENTRATION CHECK (2026 mentions top-10 companies) ===", flush=True)
for term, pat in TERMS.items():
    if term not in ('rag','multimodal','mcp','multi-agent'):
        continue
    d2026 = df_llm[df_llm['period2']=='2026'].copy()
    d2026['match'] = d2026['text_for_clean'].str.contains(pat, regex=True, na=False)
    total = d2026['match'].sum()
    top = d2026[d2026['match']].groupby('company_name_canonical').size().sort_values(ascending=False).head(10)
    concentration = top.sum() / max(total, 1)
    top_list = ", ".join([f"{c}({n})" for c, n in top.head(5).items()])
    print(f"  {term}: total {total} mentions, top-10 companies = {top.sum()} ({concentration*100:.1f}%).", flush=True)
    print(f"      top-5: {top_list}", flush=True)

pd.DataFrame(rows).to_csv("exploration/tables/V1/H5_ai_terms.csv", index=False)
print("\n[V1_H5] Wrote exploration/tables/V1/H5_ai_terms.csv", flush=True)
print("\n[V1_H5] Wave 2 T12 claim: RAG 75x, multimodal 31x, MCP 29x, multi-agent 24x", flush=True)
