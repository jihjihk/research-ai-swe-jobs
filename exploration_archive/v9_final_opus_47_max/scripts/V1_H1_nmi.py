"""V1 Phase A — Headline H1 re-derivation.

Independently computes NMI(clusters, title_archetype) vs NMI(clusters, seniority_3level).
Claim: NMI ratio ~8.8x.

Approach:
- Load swe_archetype_labels.parquet (8000 rows with BERTopic labels).
- Join to unified.parquet to get title and seniority_3level.
- Build own title_archetype regex (mutually exclusive classes).
- Compute NMI with sklearn normalized_mutual_info_score.
"""

import re
import sys
import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

print("[V1_H1] Loading archetype labels + titles + seniority", flush=True)
con = duckdb.connect()

labels = con.execute("""
    SELECT a.uid,
           a.archetype,
           a.archetype_name,
           u.title,
           u.seniority_3level,
           u.seniority_final,
           u.yoe_min_years_llm,
           u.period
    FROM read_parquet('exploration/artifacts/shared/swe_archetype_labels.parquet') a
    LEFT JOIN read_parquet('data/unified.parquet') u
      ON a.uid = u.uid
""").fetchdf()
print(f"[V1_H1] Loaded {len(labels)} archetype rows. archetype distribution:", flush=True)
print(labels['archetype'].value_counts().head(5), flush=True)

# ---- Build title archetype regex (independent) ----
# Order matters: first match wins. 11 content classes + other_swe.
TITLE_PATTERNS = [
    ("ml_ai",    r"\b(ml|machine[\s\-]*learning|ai|a\.i\.|artificial[\s\-]*intelligence|llm|deep[\s\-]*learning|data[\s\-]*scien|applied[\s\-]*scien|research[\s\-]*scien|nlp|computer[\s\-]*vision|cv|mlops|gen\s*ai|agentic)\b"),
    ("data",     r"\b(data[\s\-]*engineer|etl|analytics[\s\-]*engineer|bi\s*engineer|data[\s\-]*platform)\b"),
    ("frontend", r"\b(front[\s\-]*end|front-end|frontend|ui[\s\-]*engineer|ux\s*engineer|react|angular)\b"),
    ("backend",  r"\b(back[\s\-]*end|back-end|backend|api[\s\-]*engineer|platform[\s\-]*engineer|server[\s\-]*engineer)\b"),
    ("fullstack", r"\b(full[\s\-]*stack|fullstack|full-stack)\b"),
    ("mobile",   r"\b(ios|android|mobile[\s\-]*engineer|mobile[\s\-]*developer|mobile[\s\-]*software)\b"),
    ("devops",   r"\b(devops|sre|site[\s\-]*reliability|platform[\s\-]*reliability|infrastructure[\s\-]*engineer|cloud[\s\-]*engineer|kubernetes[\s\-]*engineer)\b"),
    ("security", r"\b(security[\s\-]*engineer|appsec|application[\s\-]*security|cyber|infosec)\b"),
    ("embedded", r"\b(firmware|embedded|hardware[\s\-]*engineer|iot[\s\-]*engineer|robotics)\b"),
    ("qa_test",  r"\b(qa[\s\-]*engineer|test[\s\-]*engineer|sdet|quality[\s\-]*assur|automation[\s\-]*test)\b"),
    ("game",     r"\b(game[\s\-]*dev|game[\s\-]*engineer|gameplay)\b"),
]

def classify_title(title):
    if not title or pd.isna(title):
        return "other_swe"
    s = str(title).lower()
    for name, pat in TITLE_PATTERNS:
        if re.search(pat, s):
            return name
    return "other_swe"

# Assertion tests
assert classify_title("Senior ML Engineer") == "ml_ai"
assert classify_title("Data Engineer") == "data"
assert classify_title("Front End Engineer") == "frontend"
assert classify_title("Fullstack Developer") == "fullstack"
assert classify_title("iOS Developer") == "mobile"
assert classify_title("DevOps Engineer") == "devops"
assert classify_title("Senior Software Engineer") == "other_swe"
assert classify_title("Staff Engineer") == "other_swe"
assert classify_title("Security Engineer") == "security"
assert classify_title("Firmware Engineer") == "embedded"
assert classify_title("QA Engineer") == "qa_test"
assert classify_title(None) == "other_swe"

labels['title_archetype_v1'] = labels['title'].apply(classify_title)
print("[V1_H1] title_archetype_v1 distribution:", flush=True)
print(labels['title_archetype_v1'].value_counts(), flush=True)

# Build seniority_3level-equivalent column. Wave 2 used the T30 mapping.
# Gate 1 states: junior = J1 or J3; senior = S1 or S4-ish; use seniority_3level column if present.
print("[V1_H1] seniority_3level distribution:", flush=True)
print(labels['seniority_3level'].value_counts(dropna=False), flush=True)

# Use seniority_3level directly; map NaN to 'unknown'
labels['sen3'] = labels['seniority_3level'].fillna('unknown').astype(str)
labels['arch'] = labels['archetype'].astype(str)
labels['title_arch'] = labels['title_archetype_v1'].astype(str)

# Compute NMI. Note: sklearn normalized_mutual_info_score with 'arithmetic' by default.
nmi_title = normalized_mutual_info_score(labels['arch'], labels['title_arch'])
nmi_sen   = normalized_mutual_info_score(labels['arch'], labels['sen3'])

# Also compute "noise_reassigned" version. Reassign -1 cluster: assign to
# modal archetype of its nearest content cluster... easier approach: just drop -1.
mask_content = labels['arch'] != '-1'
nmi_title_content = normalized_mutual_info_score(
    labels.loc[mask_content, 'arch'], labels.loc[mask_content, 'title_arch']
)
nmi_sen_content = normalized_mutual_info_score(
    labels.loc[mask_content, 'arch'], labels.loc[mask_content, 'sen3']
)

print("\n[V1_H1] === NMI RESULTS ===", flush=True)
print(f"  Full sample (n={len(labels)}):", flush=True)
print(f"    NMI(arch, title_archetype_v1) = {nmi_title:.4f}", flush=True)
print(f"    NMI(arch, seniority_3level)   = {nmi_sen:.4f}", flush=True)
print(f"    Ratio = {nmi_title / max(nmi_sen, 1e-9):.2f}x", flush=True)
print(f"  Content-only (noise removed, n={mask_content.sum()}):", flush=True)
print(f"    NMI(arch, title_archetype_v1) = {nmi_title_content:.4f}", flush=True)
print(f"    NMI(arch, seniority_3level)   = {nmi_sen_content:.4f}", flush=True)
print(f"    Ratio = {nmi_title_content / max(nmi_sen_content, 1e-9):.2f}x", flush=True)

print("\n[V1_H1] Wave 2 T09 claim: NMI(clusters, title_arch)=0.216, NMI(clusters, sen3)=0.025, ratio ~8.8x", flush=True)
print(f"[V1_H1] V1 verdict:", flush=True)
print(f"  Title NMI   {'MATCHES' if abs(nmi_title - 0.216) / 0.216 < 0.05 else 'DIFFERS'} (5% tol): V1 {nmi_title:.4f} vs T09 0.216 (delta {nmi_title - 0.216:+.4f})", flush=True)
print(f"  Senior NMI  {'MATCHES' if abs(nmi_sen - 0.025) / 0.025 < 0.05 else 'DIFFERS'}  (5% tol): V1 {nmi_sen:.4f} vs T09 0.025 (delta {nmi_sen - 0.025:+.4f})", flush=True)
print(f"  Ratio:      V1 {nmi_title/max(nmi_sen,1e-9):.2f}x vs T09 8.8x", flush=True)

# Save table
out = pd.DataFrame([
    {"subset": "full (incl noise -1)", "n": len(labels),
     "nmi_title_arch": nmi_title, "nmi_seniority_3level": nmi_sen,
     "ratio": nmi_title / max(nmi_sen, 1e-9)},
    {"subset": "content-only (noise removed)", "n": int(mask_content.sum()),
     "nmi_title_arch": nmi_title_content, "nmi_seniority_3level": nmi_sen_content,
     "ratio": nmi_title_content / max(nmi_sen_content, 1e-9)},
])
out.to_csv("exploration/tables/V1/H1_nmi_verification.csv", index=False)
print("\n[V1_H1] Wrote exploration/tables/V1/H1_nmi_verification.csv", flush=True)
