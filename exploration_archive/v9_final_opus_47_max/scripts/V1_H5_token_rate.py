"""V1 — H5 token-rate sanity check aligned with T12 methodology.

T12 accelerating_terms uses TOKEN COUNT / TOTAL TOKENS by period (pooled 2024
of LLM-cleaned text). This replicates that computation on our filtered frame.
"""

import re
import duckdb
import pandas as pd
import numpy as np

print("[V1_H5t] Loading LLM-cleaned text", flush=True)
con = duckdb.connect()
df = con.execute("""
    SELECT c.uid,
           c.description_cleaned,
           u.period,
           u.source
    FROM read_parquet('exploration/artifacts/shared/swe_cleaned_text.parquet') c
    LEFT JOIN read_parquet('data/unified.parquet') u
      ON c.uid = u.uid
    WHERE u.source_platform = 'linkedin'
      AND u.is_english = TRUE
      AND u.date_flag = 'ok'
      AND u.is_swe = TRUE
      AND c.text_source = 'llm'
""").fetchdf()
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
print(f"[V1_H5t] n 2024 LLM = {(df.period2=='2024').sum()}  n 2026 LLM = {(df.period2=='2026').sum()}", flush=True)

# Simple tokenizer: lowercase, split on non-[a-z0-9+\-] chars, keep compounds.
TOK = re.compile(r"[a-z][a-z0-9+\-\.#]+")
def tokens(s):
    if s is None or isinstance(s, float):
        return []
    return TOK.findall(str(s).lower())

def count_term(ser, term_pat):
    pat = re.compile(term_pat)
    total_tokens = 0
    total_matches = 0
    for s in ser:
        toks = tokens(s)
        total_tokens += len(toks)
        for t in toks:
            if pat.search(t):
                total_matches += 1
    return total_matches, total_tokens

TERMS = {
    "rag":         r"^rag$",
    "multimodal":  r"^multimodal$",
    "multi-modal": r"^multi-modal$",
    "mcp":         r"^mcp$",
    "multi-agent": r"^multi-agent$",
    "multiagent":  r"^multiagent$",
    "llm":         r"^llms?$",
    "agent":       r"^agent(s|ic)?$",
    "embeddings":  r"^embeddings?$",
    "fine-tuning": r"^fine-?tun(e|ing|ed)$",
    "ai-powered":  r"^ai-powered$",
    "genai":       r"^genai$",
}

d24 = df[df['period2']=='2024']['description_cleaned']
d26 = df[df['period2']=='2026']['description_cleaned']

print("\n[V1_H5t] === TOKEN-RATE COMPUTATION (POOLED 2024) ===", flush=True)
print(f"{'term':<15} {'count_2024':>11} {'rate_2024':>12} {'count_2026':>11} {'rate_2026':>12} {'ratio':>10}", flush=True)
results = []
total_24 = None
total_26 = None
for term, pat in TERMS.items():
    n24, tot24 = count_term(d24, pat)
    n26, tot26 = count_term(d26, pat)
    total_24 = tot24  # same across terms
    total_26 = tot26
    r24 = n24 / max(tot24, 1)
    r26 = n26 / max(tot26, 1)
    ratio = r26 / max(r24, 1e-15)
    results.append({"term": term, "count_2024": n24, "count_2026": n26,
                    "rate_2024": r24, "rate_2026": r26, "ratio": ratio,
                    "total_tokens_2024": tot24, "total_tokens_2026": tot26})
    print(f"{term:<15} {n24:>11} {r24:>12.6e} {n26:>11} {r26:>12.6e} {ratio:>10.1f}", flush=True)

pd.DataFrame(results).to_csv("exploration/tables/V1/H5_token_rate.csv", index=False)
print(f"\n[V1_H5t] Wrote V1/H5_token_rate.csv", flush=True)
print(f"[V1_H5t] total_tokens 2024={total_24:,}  2026={total_26:,}", flush=True)

print("\n[V1_H5t] === Wave 2 T12 Table (reference) ===", flush=True)
print("  rag          22 -> 2128  = 75x", flush=True)
print("  multimodal   11 -> 441   = 31x", flush=True)
print("  mcp          20 -> 740   = 29x", flush=True)
print("  multi-agent  20 -> 606   = 24x", flush=True)
