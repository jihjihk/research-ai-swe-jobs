"""V1 Part B — sample keyword matches for precision assessment.

Samples matches for three indicators:
  6. naive 'hire' indicator (\\b(hire|hiring|recruit\\w*)\\b)
  7. strict mentoring indicator ('mentor engineers', 'mentor junior', 'coach engineers')
  8. 'agent/agentic' emerging term (2026 only)

Produces sample texts (title + 400-char snippet around match) for manual
review. Writes results to exploration/tables/V1/B_samples.txt.
"""
import duckdb
import re
import random

random.seed(42)

DB = duckdb.connect()
DB.execute(
    "CREATE VIEW u AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')"
)
DB.execute(
    """
CREATE VIEW swe AS
SELECT uid, title,
       CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
            WHEN source='scraped' THEN '2026' END AS period2,
       COALESCE(NULLIF(description_core_llm, ''), description_core, description) AS text_best
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)


DB.execute("SELECT setseed(0.42)")

def sample_matches(pattern_sql, label, n_per_period=25, periods=("2024", "2026")):
    out = []
    for p in periods:
        q = f"""
        SELECT uid, title, text_best FROM (
          SELECT uid, title, text_best FROM swe
          WHERE period2 = '{p}'
            AND regexp_matches(lower(text_best), '{pattern_sql}')
        )
        ORDER BY random()
        LIMIT {n_per_period}
        """
        rows = DB.execute(q).fetchall()
        for uid, title, text in rows:
            # Find one match and extract a 400-char window
            m = re.search(pattern_sql, text.lower())
            if m:
                s = max(0, m.start() - 150)
                e = min(len(text), m.end() + 250)
                snippet = text[s:e].replace("\n", " ")
            else:
                snippet = text[:400].replace("\n", " ")
            out.append((label, p, uid, title, snippet))
    return out


OUT_PATH = "/home/jihgaboot/gabor/job-research/exploration/tables/V1/B_samples.txt"
with open(OUT_PATH, "w") as f:
    f.write("=" * 100 + "\n")
    f.write("B6: naive 'hire' indicator — \\b(hire|hiring|recruit\\w*)\\b\n")
    f.write("=" * 100 + "\n")
    for label, p, uid, title, snippet in sample_matches(
        r"\b(hire|hiring|recruit\w*)\b", "B6_hire"
    ):
        f.write(f"\n--- {label} | {p} | {uid} ---\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"SNIPPET: {snippet}\n")

    f.write("\n\n" + "=" * 100 + "\n")
    f.write("B7: strict mentoring — 'mentor engineers' / 'mentor junior' / 'coach engineers'\n")
    f.write("=" * 100 + "\n")
    for label, p, uid, title, snippet in sample_matches(
        r"mentor (engineers?|juniors?)|coach engineers?", "B7_mentor"
    ):
        f.write(f"\n--- {label} | {p} | {uid} ---\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"SNIPPET: {snippet}\n")

    f.write("\n\n" + "=" * 100 + "\n")
    f.write("B8: 'agent' / 'agentic' (2026 only) — is it AI agent or other?\n")
    f.write("=" * 100 + "\n")
    for label, p, uid, title, snippet in sample_matches(
        r"\bagentic\b|\bagents?\b", "B8_agent", n_per_period=25, periods=("2026",)
    ):
        f.write(f"\n--- {label} | {p} | {uid} ---\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"SNIPPET: {snippet}\n")

print(f"Wrote samples to {OUT_PATH}")
