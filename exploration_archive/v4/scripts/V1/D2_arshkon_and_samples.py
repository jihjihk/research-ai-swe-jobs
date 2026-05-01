"""V1 Part D continued — arshkon presence and entry posting samples."""
import duckdb
import pandas as pd
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 160)

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
       CASE
         WHEN llm_classification_coverage = 'labeled'         THEN seniority_llm
         WHEN llm_classification_coverage = 'rule_sufficient' THEN seniority_final
         ELSE NULL
       END AS seniority_best_available
FROM u
WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
"""
)

# Top 20 by combined
top_combined = [
    "TikTok", "Affirm", "Canonical", "ByteDance", "Cisco", "Epic",
    "Jobs via Dice", "SMX", "WayUp", "Google", "Uber", "Leidos",
    "General Motors", "SkillStorm", "Amazon", "SynergisticIT",
    "Lockheed Martin", "Emonics LLC", "HP", "Applied Materials"
]

print("=== Arshkon 2024 presence for top 20 combined contributors ===")
names = "', '".join([c.replace("'", "''") for c in top_combined])
q = f"""
SELECT company_name_canonical,
       COUNT(*) AS n_2024_arshkon,
       COUNT(*) FILTER (WHERE seniority_native = 'entry') AS n_entry_2024_arshkon,
       100.0 * COUNT(*) FILTER (WHERE seniority_native = 'entry') / NULLIF(COUNT(*), 0) AS pct_entry_2024
FROM swe
WHERE source = 'kaggle_arshkon'
  AND company_name_canonical IN ('{names}')
GROUP BY 1 ORDER BY n_2024_arshkon DESC
"""
print(DB.execute(q).fetchdf().to_string(index=False))

print("\n=== YOE top 20 arshkon presence ===")
top_yoe = [
    "Google", "Jobs via Dice", "Walmart", "Qualcomm", "SpaceX",
    "Booz Allen Hamilton", "Wells Fargo", "Amazon", "Microsoft",
    "Meta", "Leidos", "Deloitte", "Cisco", "PricewaterhouseCoopers",
    "KPMG US", "LinkedIn", "Uber", "Visa", "Northrop Grumman", "Esri"
]
names_y = "', '".join([c.replace("'", "''") for c in top_yoe])
q2 = f"""
SELECT company_name_canonical,
       COUNT(*) AS n_2024_arshkon,
       COUNT(*) FILTER (WHERE seniority_native='entry') AS n_entry,
       100.0 * COUNT(*) FILTER (WHERE seniority_native='entry') / NULLIF(COUNT(*), 0) AS pct_entry_2024
FROM swe
WHERE source = 'kaggle_arshkon'
  AND company_name_canonical IN ('{names_y}')
GROUP BY 1 ORDER BY n_2024_arshkon DESC
"""
print(DB.execute(q2).fetchdf().to_string(index=False))

# Samples from top 5 combined-column contributors
print("\n=== Samples: 3 entry postings (combined-col) from top 5 ===")
for c in ["TikTok", "Affirm", "Canonical", "ByteDance", "Cisco", "WayUp", "SynergisticIT", "SkillStorm", "Emonics LLC"]:
    c_esc = c.replace("'", "''")
    q3 = f"""
    SELECT uid, title, substr(COALESCE(NULLIF(description_core_llm,''), description_core, description), 1, 250) AS snippet,
           seniority_best_available, yoe_extracted
    FROM swe
    WHERE period2='2026' AND company_name_canonical='{c_esc}' AND seniority_best_available='entry'
    ORDER BY uid LIMIT 3
    """
    df = DB.execute(q3).fetchdf()
    print(f"\n--- {c} ---")
    if df.empty:
        print("(no combined-col entry rows)")
    else:
        print(df.to_string(index=False))

# Quick check: how many postings from "Jobs via Dice" are aggregator?
print("\n=== Is 'Jobs via Dice' an aggregator? ===")
print(DB.execute(
    """
SELECT period2, is_aggregator, COUNT(*) FROM swe
WHERE company_name_canonical='Jobs via Dice' GROUP BY 1,2 ORDER BY 1,2
    """
).fetchdf().to_string(index=False))
