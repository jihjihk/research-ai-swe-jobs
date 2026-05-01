"""V1 Phase A — Headline H3 alternative.

A simpler independent header-regex classifier to cross-check T13_section_classifier.py
on the J3 subset. Tests whether 'requirements' section chars shrank in J3 from
2024 to 2026 (T13 claim: -57 chars / -5%).
"""

import re
import duckdb
import pandas as pd
import numpy as np

print("[V1_H3a] Loading raw descriptions for J3 subset", flush=True)
con = duckdb.connect()

df = con.execute("""
    SELECT uid,
           description,
           description_length,
           period,
           yoe_min_years_llm
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = TRUE
      AND date_flag = 'ok'
      AND is_swe = TRUE
      AND yoe_min_years_llm IS NOT NULL
      AND yoe_min_years_llm <= 2
""").fetchdf()
df['period2'] = df['period'].astype(str).apply(
    lambda p: '2024' if p.startswith('2024') else ('2026' if p.startswith('2026') else 'other')
)
df = df[df['period2'].isin(['2024','2026'])].copy()
print(f"[V1_H3a] J3 subset: {len(df)} rows. Period distribution:", flush=True)
print(df['period2'].value_counts(), flush=True)

# Independent header regexes (looser than T13's full module)
HEADERS = {
    "requirements": re.compile(
        r"(?im)^\s*(?:\*\*|##*\s*)?(required\s+(?:qualifications|skills|experience)|requirements?|qualifications?|(?:basic|minimum)\s+qualifications|what\s+you(?:'ll|\s+will)?\s+need|must\s+have(?:s)?|skills\s+(?:&|and)\s+qualifications|you(?:r|\s+should\s+have)|(?:technical\s+)?skills?|we(?:'re|\s+are)\s+looking\s+for)\s*[:\*\*]*\s*$"
    ),
    "responsibilities": re.compile(
        r"(?im)^\s*(?:\*\*|##*\s*)?(responsibilities?|what\s+you(?:'ll|\s+will)?\s+do|duties|your\s+role|role\s+responsibilities|day[\s\-]to[\s\-]day|impact|key\s+responsibilities)\s*[:\*\*]*\s*$"
    ),
    "benefits": re.compile(
        r"(?im)^\s*(?:\*\*|##*\s*)?(benefits|perks|compensation|what\s+we\s+offer|our\s+benefits|package|salary|pay|we\s+offer)\s*[:\*\*]*\s*$"
    ),
}

def classify_sections_alt(text):
    """Very simple section-header classifier.

    Scans line-by-line. Each line either is a header (triggering section switch) or
    adds to the currently active section's char count. 'lead' section is 'unclassified'.
    """
    if not text:
        return {'requirements': 0, 'responsibilities': 0, 'benefits': 0, 'other': 0, 'total': 0}
    section = 'other'
    counts = {'requirements': 0, 'responsibilities': 0, 'benefits': 0, 'other': 0}
    for line in text.split("\n"):
        line_clean = line.strip()
        matched = False
        for sec_name, pat in HEADERS.items():
            if pat.match(line):
                section = sec_name
                matched = True
                break
        if matched:
            continue
        counts[section] += len(line) + 1  # +1 for newline
    counts['total'] = sum(counts.values())
    return counts

# Assertion tests
t1 = "About\nWe build cool stuff.\n\nRequirements:\n- Python\n- 5 yoe\n\nResponsibilities:\n- Write code\n"
r = classify_sections_alt(t1)
print(f"[V1_H3a] Assertion test: {r}", flush=True)
assert r['requirements'] > 0
assert r['responsibilities'] > 0

# Apply
print("[V1_H3a] Applying alt classifier ...", flush=True)
descs = df['description'].fillna('').astype(str).tolist()
results = [classify_sections_alt(d) for d in descs]
for k in ('requirements', 'responsibilities', 'benefits', 'other', 'total'):
    df[f'alt_{k}'] = [r[k] for r in results]

g = df.groupby('period2')[[f'alt_{k}' for k in ('requirements','responsibilities','benefits','other','total')]].mean().round(1)
print("\n[V1_H3a] === Alt classifier on J3 subset (mean chars per section) ===", flush=True)
print(g, flush=True)

for col in ['alt_requirements', 'alt_responsibilities', 'alt_benefits', 'alt_total']:
    d24 = g.loc['2024', col]
    d26 = g.loc['2026', col]
    pct = 100 * (d26 - d24) / max(d24, 1e-9)
    print(f"  {col}: {d24:.0f} -> {d26:.0f}  Delta {d26-d24:+.0f} ({pct:+.1f}%)", flush=True)

g.to_csv("exploration/tables/V1/H3_alt_classifier_J3.csv")
print("\n[V1_H3a] Wrote exploration/tables/V1/H3_alt_classifier_J3.csv", flush=True)
print("\n[V1_H3a] Wave 2 T13 claim: J3 requirements 1057 -> 1001 (-57 chars, -5%), benefits +92%", flush=True)
