"""V1 Part A task 5 — AI mention growth."""
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

TERMS = {
    "ai_acr": r"\bai\b",
    "artificial_intelligence": r"\bartificial intelligence\b",
    "machine_learning": r"\bmachine[- ]learning\b",
    "ml_acr": r"\bml\b",
    "llm": r"\bllms?\b",
    "gpt": r"\bgpt(-?[0-9])?\b",
    "claude": r"\bclaude\b",
    "copilot": r"\bcopilot\b",
    "rag": r"\brag\b",
    "agent": r"\bagents?\b",
    "agentic": r"\bagentic\b",
}
UNION = "|".join(TERMS.values())

print("\n=== Union 'any AI term' share by period ===")
print(DB.execute(
    f"""
SELECT period2,
       COUNT(*) AS n,
       100.0 * AVG(CASE WHEN regexp_matches(lower(text_best), '{UNION}') THEN 1 ELSE 0 END) AS pct_any_ai
FROM swe GROUP BY 1 ORDER BY 1
    """
).fetchdf().to_string(index=False))

print("\n=== Per-term share ===")
rows = []
for name, pat in TERMS.items():
    df = DB.execute(
        f"""
SELECT period2,
       100.0 * AVG(CASE WHEN regexp_matches(lower(text_best), '{pat}') THEN 1 ELSE 0 END) AS pct
FROM swe GROUP BY 1 ORDER BY 1
        """
    ).fetchdf()
    p24 = float(df.loc[df['period2'] == '2024', 'pct'].iloc[0])
    p26 = float(df.loc[df['period2'] == '2026', 'pct'].iloc[0])
    rows.append((name, p24, p26, p26 - p24))
rows.sort(key=lambda r: -r[3])
print(f"{'term':24s} {'pct_2024':>10s} {'pct_2026':>10s} {'delta_pp':>10s}")
for name, p24, p26, d in rows:
    print(f"{name:24s} {p24:>10.2f} {p26:>10.2f} {d:>+10.2f}")
