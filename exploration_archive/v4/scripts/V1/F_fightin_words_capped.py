"""V1 Part F — capped Fightin' Words comparison.

Samples 5,000 capped postings per period (max 20 per company_name_canonical),
computes log-odds ratio with Dirichlet prior, and reports the top 30
distinguishing terms per direction. Compares against T12's top 30.
"""
import duckdb
import re
import numpy as np
import pandas as pd
from collections import Counter

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 40)

DB = duckdb.connect()
DB.execute(
    "CREATE VIEW u AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')"
)

DB.execute(
    """
CREATE VIEW swe AS
SELECT *,
       CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
            WHEN source='scraped' THEN '2026' END AS period2,
       COALESCE(NULLIF(description_core_llm, ''), description_core, description) AS text_best
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

# Cap at 20 per company_name_canonical per period, then reservoir sample 5000
DB.execute("SELECT setseed(0.42)")
df_2024 = DB.execute("""
WITH capped AS (
  SELECT uid, text_best, company_name_canonical, period2,
         ROW_NUMBER() OVER (PARTITION BY period2, company_name_canonical ORDER BY random()) AS rn
  FROM swe WHERE period2='2024'
)
SELECT * FROM capped WHERE rn <= 20 ORDER BY random() LIMIT 5000
""").fetchdf()
df_2026 = DB.execute("""
WITH capped AS (
  SELECT uid, text_best, company_name_canonical, period2,
         ROW_NUMBER() OVER (PARTITION BY period2, company_name_canonical ORDER BY random()) AS rn
  FROM swe WHERE period2='2026'
)
SELECT * FROM capped WHERE rn <= 20 ORDER BY random() LIMIT 5000
""").fetchdf()

print(f"2024 capped sample: {len(df_2024)}, unique companies: {df_2024['company_name_canonical'].nunique()}")
print(f"2026 capped sample: {len(df_2026)}, unique companies: {df_2026['company_name_canonical'].nunique()}")


_TOK_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")
_STOPWORDS = set("a an the and or but if in on at to for from of by with as is are was were be been being will would could should have has had do does did this that these those you your we our they them their i me my it its not no so such also other into over under about more most many any all some what which who whom when where how than then them why".split())


def tokenize(text):
    if not isinstance(text, str):
        return []
    toks = _TOK_RE.findall(text.lower())
    return [t for t in toks if len(t) > 2 and t not in _STOPWORDS]


def vocab_counts(texts):
    c = Counter()
    for t in texts:
        c.update(tokenize(t))
    return c


c_2024 = vocab_counts(df_2024["text_best"])
c_2026 = vocab_counts(df_2026["text_best"])

# Build vocabulary (terms appearing at least 5 times total, and in both periods or
# at the top contributed by one)
vocab = set(t for t, c in c_2024.items() if c >= 5) | set(t for t, c in c_2026.items() if c >= 5)
print(f"Vocabulary size (count>=5 in at least one period): {len(vocab)}")

# Dirichlet-prior log-odds with uniform prior alpha_w = 0.01
n_a = sum(c_2024[w] for w in vocab)  # 2024 counts
n_b = sum(c_2026[w] for w in vocab)  # 2026 counts
alpha = 0.01
a_0 = alpha * len(vocab)

rows = []
for w in vocab:
    y_iw_a = c_2024[w]
    y_iw_b = c_2026[w]
    if y_iw_a + y_iw_b == 0:
        continue
    # Monroe et al. log-odds ratio informative Dirichlet prior
    log_odds_a = np.log((y_iw_a + alpha) / (n_a + a_0 - y_iw_a - alpha))
    log_odds_b = np.log((y_iw_b + alpha) / (n_b + a_0 - y_iw_b - alpha))
    delta = log_odds_b - log_odds_a  # positive = 2026-favored
    var = 1.0 / (y_iw_a + alpha) + 1.0 / (y_iw_b + alpha)
    z = delta / np.sqrt(var)
    rows.append((w, y_iw_a, y_iw_b, delta, z))

fw = pd.DataFrame(rows, columns=["term", "c_2024", "c_2026", "delta", "z"])
fw["abs_z"] = fw["z"].abs()

print("\n=== Top 30 2026-favored terms (capped FW) ===")
top2026 = fw.sort_values("z", ascending=False).head(30)
print(top2026.to_string(index=False))
top2026[["term", "z", "c_2024", "c_2026"]].to_csv(
    "/home/jihgaboot/gabor/job-research/exploration/tables/V1/F_capped_top30_2026.csv", index=False
)

print("\n=== Top 30 2024-favored terms (capped FW) ===")
top2024 = fw.sort_values("z", ascending=True).head(30)
print(top2024.to_string(index=False))
top2024[["term", "z", "c_2024", "c_2026"]].to_csv(
    "/home/jihgaboot/gabor/job-research/exploration/tables/V1/F_capped_top30_2024.csv", index=False
)

# Compare against T12 top 100 uncapped
t12_2026 = pd.read_csv("/home/jihgaboot/gabor/job-research/exploration/tables/T12/fw_primary_core_top100_2026.csv")
t12_2024 = pd.read_csv("/home/jihgaboot/gabor/job-research/exploration/tables/T12/fw_primary_core_top100_2024.csv")

# T12's top_100 2026 column ordering: higher z = more 2026-favored. We want top 30.
t12_2026_top30 = t12_2026.head(30)["term"].tolist()
t12_2024_top30 = t12_2024.head(30)["term"].tolist()

capped_2026_top30 = top2026["term"].tolist()
capped_2024_top30 = top2024["term"].tolist()

print("\n=== Overlap 2026-favored: T12 uncapped top30 vs V1 capped top30 ===")
overlap_2026 = set(t12_2026_top30) & set(capped_2026_top30)
print(f"Overlap size: {len(overlap_2026)}/30 = {len(overlap_2026)/30*100:.0f}%")
print(f"Robust (in both): {sorted(overlap_2026)}")
print(f"Only in T12 (uncapped-only = potentially artifact): {sorted(set(t12_2026_top30) - set(capped_2026_top30))}")
print(f"Only in V1 capped: {sorted(set(capped_2026_top30) - set(t12_2026_top30))}")

print("\n=== Overlap 2024-favored: T12 uncapped top30 vs V1 capped top30 ===")
overlap_2024 = set(t12_2024_top30) & set(capped_2024_top30)
print(f"Overlap size: {len(overlap_2024)}/30 = {len(overlap_2024)/30*100:.0f}%")
print(f"Robust (in both): {sorted(overlap_2024)}")
print(f"Only in T12 (uncapped-only): {sorted(set(t12_2024_top30) - set(capped_2024_top30))}")
print(f"Only in V1 capped: {sorted(set(capped_2024_top30) - set(t12_2024_top30))}")
