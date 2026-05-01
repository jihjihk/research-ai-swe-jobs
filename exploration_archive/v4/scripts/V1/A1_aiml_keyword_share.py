"""V1 Part A task 1 — AI/ML domain keyword share by period.

Defines an 'AI/ML posting' as any posting mentioning at least one of a small
reference set of AI/ML terms in either description_core_llm (when available)
or description. Computes share by period and reports the period delta.
Independent of T09/T12 topic modeling.
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
       COALESCE(NULLIF(description_core_llm, ''), description_core, description) AS text_best
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

# A ~conservative AI/ML domain regex (word boundaries for short acronyms,
# case-insensitive). Reports individual contributions and the union.
TERMS = {
    "machine_learning": r"\bmachine[- ]learning\b",
    "deep_learning": r"\bdeep[- ]learning\b",
    "ml_acr": r"\bml\b",
    "ai_acr": r"\bai\b",
    "llm": r"\bllms?\b",
    "neural_net": r"\bneural[- ]networks?\b",
    "pytorch": r"\bpytorch\b",
    "tensorflow": r"\btensorflow\b",
    "scikit": r"\bscikit[- ]learn\b|\bsklearn\b",
    "transformer": r"\btransformers?\b",
    "nlp": r"\bnlp\b",
    "model_training": r"\bmodel (training|fine[- ]tuning)\b|\bfine[- ]tuning\b",
    "generative_ai": r"\bgenerative ai\b|\bgen[- ]?ai\b",
}

# Build a big OR that matches any term. DuckDB supports regexp_matches; we'll
# pre-lower text once.
UNION_RE = "|".join(TERMS.values())

print("\n=== Q1: AI/ML domain share by period (union of reference terms) ===")
q = f"""
SELECT period2,
       COUNT(*) AS n_total,
       SUM(CASE WHEN regexp_matches(lower(text_best), '{UNION_RE}') THEN 1 ELSE 0 END) AS n_aiml,
       100.0 * SUM(CASE WHEN regexp_matches(lower(text_best), '{UNION_RE}') THEN 1 ELSE 0 END) / COUNT(*) AS pct_aiml
FROM swe
GROUP BY 1 ORDER BY 1
"""
df = DB.execute(q).fetchdf()
print(df.to_string(index=False))
delta = float(df.loc[df["period2"] == "2026", "pct_aiml"].iloc[0]) - float(
    df.loc[df["period2"] == "2024", "pct_aiml"].iloc[0]
)
print(f"\nUnion AI/ML share delta 2024->2026: {delta:+.2f} pp")

print("\n=== Q1b: Per-term contribution ===")
rows = []
for name, pat in TERMS.items():
    q = f"""
    SELECT period2,
           100.0 * AVG(CASE WHEN regexp_matches(lower(text_best), '{pat}') THEN 1 ELSE 0 END) AS pct
    FROM swe GROUP BY 1 ORDER BY 1
    """
    sub = DB.execute(q).fetchdf()
    p24 = float(sub.loc[sub["period2"] == "2024", "pct"].iloc[0])
    p26 = float(sub.loc[sub["period2"] == "2026", "pct"].iloc[0])
    rows.append((name, p24, p26, p26 - p24))
rows.sort(key=lambda r: -r[3])
print(f"{'term':20s} {'pct_2024':>10s} {'pct_2026':>10s} {'delta_pp':>10s}")
for name, p24, p26, d in rows:
    print(f"{name:20s} {p24:>10.2f} {p26:>10.2f} {d:>+10.2f}")
