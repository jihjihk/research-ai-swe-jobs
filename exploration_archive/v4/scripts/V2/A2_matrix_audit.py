"""V2 Part A step 3+4: audit the broken tech matrix.

1. Load matrix. Join to unified parquet for period info.
2. Compute tech_count per row from matrix; mean/median by period.
3. For each tech column, sample 20 '1' rows & 20 '0' rows; spot check against raw description.
4. Identify silently broken regexes (beyond known c++/c#).
"""

import duckdb
import re
import json
import random

random.seed(42)
con = duckdb.connect()

MATRIX = "exploration/artifacts/shared/swe_tech_matrix.parquet"
UNI = "data/unified.parquet"

# Step 1+2: overall tech_count from the matrix
mcols = con.execute(f"DESCRIBE SELECT * FROM '{MATRIX}' LIMIT 1").fetchdf()
TECHS = [c for c in mcols['column_name'].tolist() if c != 'uid']
print(f"n_techs in matrix: {len(TECHS)}")

sum_expr = " + ".join([f"CAST({t} AS INT)" for t in TECHS])
q = f"""
WITH m AS (
  SELECT uid, ({sum_expr}) AS tech_count
  FROM '{MATRIX}'
),
u AS (
  SELECT uid, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period
  FROM '{UNI}'
  WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
)
SELECT u.period, count(*) AS n,
       avg(m.tech_count) AS mean_tc,
       median(m.tech_count) AS median_tc,
       quantile(m.tech_count, 0.25) AS p25,
       quantile(m.tech_count, 0.75) AS p75
FROM u JOIN m USING (uid)
GROUP BY 1 ORDER BY 1
"""
print("\n=== Matrix-derived tech_count distribution ===")
print(con.execute(q).fetchdf().to_string())

# Step 3: per-tech prevalence from matrix
q2 = f"""
WITH m AS (
  SELECT uid, {", ".join([f"CAST({t} AS INT) AS {t}" for t in TECHS])}
  FROM '{MATRIX}'
),
u AS (
  SELECT uid, CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period
  FROM '{UNI}'
  WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
)
SELECT u.period, {", ".join([f"avg(m.{t})*100 AS {t}" for t in TECHS])}
FROM u JOIN m USING (uid)
GROUP BY 1 ORDER BY 1
"""
prev = con.execute(q2).fetchdf()
pr = prev.set_index('period').T
pr.columns = ['y2024', 'y2026']
pr['delta'] = pr['y2026'] - pr['y2024']
pr['ratio'] = pr['y2026'] / pr['y2024'].clip(lower=0.001)
pr = pr.sort_values('y2026', ascending=False)
pr.to_csv("exploration/tables/V2/A2_matrix_prevalence.csv")
print("\n=== Matrix per-tech prevalence (top 20 by 2026 share) ===")
print(pr.head(20).to_string())
print("\n=== Suspicious low-prevalence techs (ones you'd expect > 5%) ===")
suspect = ['c_cpp', 'csharp', 'nodejs', 'dotnet', 'react', 'angular', 'go_lang', 'rust', 'kotlin', 'swift']
print(pr.loc[[t for t in suspect if t in pr.index]].to_string())

# Step 4: audit ~30 techs — sample 20 positive and 20 negative rows, check via LIKE
# A "broken" regex will show positive-labeled rows that truly do contain the tech in raw description,
# but also (more importantly) negative-labeled rows that contain it in raw description.

# Define simple LIKE-based truth for each tech
TRUTH = {
    # literal lowercase substrings (just requires presence; simple but enough for FN detection)
    "python": ["python"],
    "java": ["java "],  # crude — will overcount; use as sanity
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "c_cpp": ["c++", "c\\+\\+"],  # raw parquet has markdown escapes
    "csharp": ["c#", "c\\#"],
    "go_lang": ["golang", "go programming", " go/", " go,", " go;"],
    "rust": [" rust "],
    "ruby": [" ruby "],
    "php": [" php "],
    "scala": [" scala "],
    "kotlin": ["kotlin"],
    "swift": [" swift "],
    "react": ["react"],
    "angular": ["angular"],
    "vue": ["vue.js", "vuejs", " vue "],
    "nodejs": ["node.js", "nodejs", "node\\.js"],
    "django": ["django"],
    "flask": [" flask "],
    "spring": ["spring "],
    "dotnet": [".net", "dotnet"],
    "aws": ["aws"],
    "azure": ["azure"],
    "gcp": ["gcp"],
    "kubernetes": ["kubernetes"],
    "docker": ["docker"],
    "terraform": ["terraform"],
    "linux": ["linux"],
    "postgresql": ["postgres"],
    "mysql": ["mysql"],
    "mongodb": ["mongo"],
    "redis": ["redis"],
    "sql": ["sql"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "langchain": ["langchain"],
    "mcp": ["mcp"],
    "rag": [" rag ", "rag "],
    "copilot": ["copilot"],
    "llm": ["llm"],
}

# For each audit tech, pick 2026 SWE rows, sample 20 matrix=1 and 20 matrix=0, classify
print("\n=== Per-tech audit: matrix label vs raw description substring ===")
print(f"{'tech':<14} {'m_pos':>6} {'tl_pos':>6} {'prec':>6}  {'m_neg':>6} {'raw_pos':>7} {'fn_rate':>8}")
audit_results = []
for tech in TRUTH:
    if tech not in TECHS:
        continue
    likes = TRUTH[tech]
    like_or = " OR ".join([f"position('{lit}' in lower(u.description)) > 0" for lit in likes])
    q_pos = f"""
    WITH m AS (SELECT uid, {tech} AS lbl FROM '{MATRIX}'),
         u AS (SELECT uid, description FROM '{UNI}'
               WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE AND source='scraped')
    SELECT sum(CAST(m.lbl AS INT)) AS m_pos,
           sum(CASE WHEN ({like_or}) THEN 1 ELSE 0 END) AS raw_pos,
           sum(CASE WHEN m.lbl AND ({like_or}) THEN 1 ELSE 0 END) AS both,
           count(*) AS n
    FROM u JOIN m USING (uid)
    """
    r = con.execute(q_pos).fetchdf().iloc[0]
    m_pos = int(r['m_pos']); raw_pos = int(r['raw_pos']); both = int(r['both']); n = int(r['n'])
    prec = both / m_pos if m_pos > 0 else 0.0  # true positive rate among matrix positives
    # of raw-pos, how many did the matrix miss?
    fn = raw_pos - both
    fn_rate = fn / raw_pos if raw_pos > 0 else 0.0
    m_neg = n - m_pos
    print(f"{tech:<14} {m_pos:>6d} {both:>6d} {prec:>6.2f}  {m_neg:>6d} {raw_pos:>7d} {fn_rate:>8.2f}")
    audit_results.append({
        "tech": tech, "m_pos": m_pos, "true_pos_of_m_pos": both,
        "precision": prec, "raw_pos": raw_pos, "fn_rate": fn_rate
    })

# Save audit
with open("exploration/tables/V2/A2_matrix_audit.json", "w") as f:
    json.dump(audit_results, f, indent=2)
print("\nSaved audit to exploration/tables/V2/A2_matrix_audit.json")

# Highlight broken techs (fn_rate > 0.30 or precision < 0.70)
print("\n=== BROKEN TECHS (fn_rate > 0.30 or precision < 0.70) ===")
for r in audit_results:
    if r["fn_rate"] > 0.30 or r["precision"] < 0.70:
        print(f"  {r['tech']:<14} prec={r['precision']:.2f} fn_rate={r['fn_rate']:.2f}  m_pos={r['m_pos']} raw_pos={r['raw_pos']}")
