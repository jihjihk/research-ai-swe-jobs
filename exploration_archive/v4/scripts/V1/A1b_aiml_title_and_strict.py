"""V1 Part A1 companion — stricter AI/ML domain proxies.

T09 reports an *exclusive* archetype share (+10.96pp). 'Any mention' measures
a different, broader concept. We also compute:

  * AI/ML in title (a post is in the AI/ML 'domain' if its title indicates it)
  * 'Strong' body signal: at least 2 distinct AI/ML terms present

These are stricter proxies for the archetype share.
"""
import duckdb

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
       COALESCE(NULLIF(description_core_llm, ''), description_core, description) AS text_best,
       COALESCE(title_normalized, title) AS title_best
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

# Title-based AI/ML marker (narrower, less prone to mentions of AI in benefits)
TITLE_RE = r"\b(ai|ml|machine[- ]learning|deep[- ]learning|nlp|mlops|llm|applied scientist|research scientist|data scientist|ai engineer|ml engineer|genai|gen[- ]ai)\b"

print("\n=== A1b: AI/ML in TITLE ===")
df = DB.execute(
    f"""
SELECT period2,
       100.0 * AVG(CASE WHEN regexp_matches(lower(title_best), '{TITLE_RE}') THEN 1 ELSE 0 END) AS pct_aiml_title,
       COUNT(*) AS n
FROM swe GROUP BY 1 ORDER BY 1
"""
).fetchdf()
print(df.to_string(index=False))

print("\n=== A1b: AI/ML strong body signal (>=2 distinct ML terms) ===")
TERMS = [
    r"\bmachine[- ]learning\b",
    r"\bdeep[- ]learning\b",
    r"\bml\b",
    r"\bllms?\b",
    r"\bneural[- ]networks?\b",
    r"\bpytorch\b",
    r"\btensorflow\b",
    r"\bscikit[- ]learn\b|\bsklearn\b",
    r"\btransformers?\b",
    r"\bnlp\b",
    r"\bfine[- ]tuning\b",
    r"\bgenerative ai\b|\bgen[- ]?ai\b",
    r"\bhuggingface\b|\bhugging face\b",
    r"\brag\b|\bretrieval[- ]augmented\b",
]
sum_expr = " + ".join(
    [f"(CASE WHEN regexp_matches(lower(text_best), '{p}') THEN 1 ELSE 0 END)" for p in TERMS]
)
q = f"""
SELECT period2,
       100.0 * AVG(CASE WHEN ({sum_expr}) >= 2 THEN 1 ELSE 0 END) AS pct_strong_aiml,
       100.0 * AVG(CASE WHEN ({sum_expr}) >= 3 THEN 1 ELSE 0 END) AS pct_very_strong_aiml,
       COUNT(*) AS n
FROM swe GROUP BY 1 ORDER BY 1
"""
print(DB.execute(q).fetchdf().to_string(index=False))
