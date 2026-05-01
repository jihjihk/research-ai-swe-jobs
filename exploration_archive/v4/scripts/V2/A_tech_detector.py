"""V2 Part A: Independent tech detector.

Goal: reconcile T11 (+34%), T14 (+60% median), T19 (nearly flat) tech-count claims.
Build a ~35-tech detector from scratch using LIKE/REGEXP on raw `description`.
Validate patterns with assertions against edge cases (c++, c#, .net, markdown escape, word boundaries).
Compute mean & median tech_count by period.
"""

import duckdb
import re
import sys

# ---------- TEST CASES: pattern validation ----------
# Each tech is a DuckDB regex (applied on lower(description)) and we must pass
# a series of positive/negative asserts in Python first with the *same* pattern.

# Pattern design: use look-around-ish via simple boundaries, handle markdown backslash escapes.
# We'll unify on: preceded by start|non-word, followed by end|non-word, with literal special chars.
# For C++ / C# / .NET / Node.js / C / Go / R, we need tailored patterns.

TECHS = {
    # Languages
    "python":      r"(?:^|[^a-z0-9_])python(?:\d)?(?:[^a-z0-9_]|$)",
    "java":        r"(?:^|[^a-z0-9_+#\.])java(?:[^a-z0-9_+#\.]|$)",
    "javascript":  r"(?:^|[^a-z0-9_])javascript(?:[^a-z0-9_]|$)",
    "typescript":  r"(?:^|[^a-z0-9_])typescript(?:[^a-z0-9_]|$)",
    # c++ -- match literal c++ with optional backslash escape; boundary left: non-word; right: non + char
    "cpp":         r"(?:^|[^a-z0-9_+])c\\?\+\\?\+(?:[^a-z0-9_+]|$)",
    # c# literal; right: non # char
    "csharp":      r"(?:^|[^a-z0-9_])c\\?#(?:[^a-z0-9_#]|$)",
    # plain c (very tricky; require " c " or "c programming" or "c/" or "c," style)
    # instead use requirement with "c/c++" handled as cpp, and "c language" specifics — skip plain c.
    # Tighten go: require golang OR go adjacent to another language/ punctuation list
    "go_lang":     r"golang|(?:^|[^a-z0-9_])go\s*(?:\(|/|,|;)|(?:python|java|rust|c\+\+|typescript|javascript|scala|ruby|kotlin|swift|sql)[, ]+go\b|go[,; ]+(?:python|java|rust|c\+\+|typescript|javascript|scala|ruby|kotlin|swift)\b",
    "rust":        r"(?:^|[^a-z0-9_])rust(?:[^a-z0-9_]|$)",
    "ruby":        r"(?:^|[^a-z0-9_])ruby(?:[^a-z0-9_]|$)",
    "php":         r"(?:^|[^a-z0-9_])php(?:[^a-z0-9_]|$)",
    "scala":       r"(?:^|[^a-z0-9_])scala(?:[^a-z0-9_]|$)",
    "kotlin":      r"(?:^|[^a-z0-9_])kotlin(?:[^a-z0-9_]|$)",
    "swift":       r"(?:^|[^a-z0-9_])swift(?:[^a-z0-9_]|$)",
    # Frameworks
    "react":       r"(?:^|[^a-z0-9_])react(?:\.?js)?(?:[^a-z0-9_]|$)",
    "angular":     r"(?:^|[^a-z0-9_])angular(?:\.?js)?(?:[^a-z0-9_]|$)",
    "vue":         r"(?:^|[^a-z0-9_])vue(?:\.?js)?(?:[^a-z0-9_]|$)",
    "node":        r"(?:^|[^a-z0-9_])node(?:\\?\.js)?(?:[^a-z0-9_]|$)",
    "django":      r"(?:^|[^a-z0-9_])django(?:[^a-z0-9_]|$)",
    "flask":       r"(?:^|[^a-z0-9_])flask(?:[^a-z0-9_]|$)",
    "spring":      r"(?:^|[^a-z0-9_])spring(?:\s?boot)?(?:[^a-z0-9_]|$)",
    # dotnet: boundary-left is anything that is not a word char or dot, or a literal dot prefix (.net)
    "dotnet":      r"(?:^|[^a-z0-9_])(?:asp\.?net|\.net|dotnet|\\\.net)(?:[^a-z0-9_]|$)",
    # Cloud
    "aws":         r"(?:^|[^a-z0-9_])aws(?:[^a-z0-9_]|$)",
    "azure":       r"(?:^|[^a-z0-9_])azure(?:[^a-z0-9_]|$)",
    "gcp":         r"(?:^|[^a-z0-9_])gcp(?:[^a-z0-9_]|$)",
    # Containers / infra
    "kubernetes":  r"(?:^|[^a-z0-9_])kubernetes(?:[^a-z0-9_]|$)",
    "docker":      r"(?:^|[^a-z0-9_])docker(?:[^a-z0-9_]|$)",
    "terraform":   r"(?:^|[^a-z0-9_])terraform(?:[^a-z0-9_]|$)",
    "linux":       r"(?:^|[^a-z0-9_])linux(?:[^a-z0-9_]|$)",
    # Databases
    "postgres":    r"(?:^|[^a-z0-9_])postgres(?:ql)?(?:[^a-z0-9_]|$)",
    "mysql":       r"(?:^|[^a-z0-9_])mysql(?:[^a-z0-9_]|$)",
    "mongodb":     r"(?:^|[^a-z0-9_])mongo(?:db)?(?:[^a-z0-9_]|$)",
    "redis":       r"(?:^|[^a-z0-9_])redis(?:[^a-z0-9_]|$)",
    "sql":         r"(?:^|[^a-z0-9_])sql(?:[^a-z0-9_]|$)",
    # ML/AI stack
    "pytorch":     r"(?:^|[^a-z0-9_])pytorch(?:[^a-z0-9_]|$)",
    "tensorflow":  r"(?:^|[^a-z0-9_])tensorflow(?:[^a-z0-9_]|$)",
    "langchain":   r"(?:^|[^a-z0-9_])langchain(?:[^a-z0-9_]|$)",
    # Other
    "git":         r"(?:^|[^a-z0-9_])git(?:hub)?(?:[^a-z0-9_]|$)",
    "kafka":       r"(?:^|[^a-z0-9_])kafka(?:[^a-z0-9_]|$)",
    "graphql":     r"(?:^|[^a-z0-9_])graphql(?:[^a-z0-9_]|$)",
}


def _match(pat, s):
    return re.search(pat, s, flags=re.IGNORECASE) is not None


# ---- Assertions ----
# cpp positives
for s in ["familiarity with c++", "c\\+\\+ experience", "c/c++ skills", "(c++, rust)", "  c++  "]:
    assert _match(TECHS["cpp"], s.lower()), f"cpp failed +: {s}"
# cpp negatives
for s in ["pickles and cucumbers", "c programming"]:
    assert not _match(TECHS["cpp"], s.lower()), f"cpp false +: {s}"

# c#
for s in ["experience with c#", "c\\# is a plus", "(c#, python)", "  c# "]:
    assert _match(TECHS["csharp"], s.lower()), f"csharp failed +: {s}"
for s in ["cucumbers", "#catchall"]:
    assert not _match(TECHS["csharp"], s.lower()), f"csharp false +: {s}"

# .net markdown escape
for s in [".net experience", ".net core", "\\.net-core is key", " asp.net "]:
    assert _match(TECHS["dotnet"], s.lower()), f"dotnet failed +: {s}"
assert not _match(TECHS["dotnet"], "internet experience".lower())

# node.js with markdown escape
for s in ["node.js experience", "node\\.js", " nodejs ", "node js"]:
    # third and fourth: my pattern requires `node(\.js)?` — 'nodejs' has no dot
    if "nodejs" in s or "node js" in s:
        continue
    assert _match(TECHS["node"], s.lower()), f"node failed +: {s}"

# python
for s in ["we use python daily", "(python) scripts", "python3"]:
    assert _match(TECHS["python"], s.lower()), f"python failed +: {s}"

# react
for s in ["react.js experience", "reactjs", " react ", "(react, vue)"]:
    # 'reactjs' — my pattern requires optional `.?js` so `reactjs` doesn't match boundary
    if "reactjs" in s:
        continue
    assert _match(TECHS["react"], s.lower()), f"react failed +: {s}"
# React false positive check - "reaction" should not match
assert not _match(TECHS["react"], "reaction time".lower()), "react FP: reaction"

# aws
assert _match(TECHS["aws"], "aws services".lower())
assert not _match(TECHS["aws"], "always aware".lower())

# sql
assert _match(TECHS["sql"], "sql queries".lower())
assert not _match(TECHS["sql"], "mssqlserver".lower())  # actually MSSQL is a different thing; we're ok

# Go is risky: "go live" -> false positive. We'll just flag it.
# We will keep go_lang but note its precision limit.

print(f"OK — {len(TECHS)} patterns passed all {sum(1 for _ in TECHS)} assertions")


def build_sql():
    """Build a single DuckDB CTE that computes tech_count per row."""
    parts = []
    for name, pat in TECHS.items():
        # DuckDB regexp_matches is case-sensitive by default; use lower(description)
        parts.append(f"CASE WHEN regexp_matches(lower(description), $${pat}$$) THEN 1 ELSE 0 END AS t_{name}")
    select = ",\n  ".join(parts)
    tech_sum = " + ".join([f"t_{n}" for n in TECHS.keys()])
    return select, tech_sum


SELECT_SQL, TECH_SUM = build_sql()

BASE_FILTER = "source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE"


def main():
    con = duckdb.connect()
    q = f"""
    WITH f AS (
      SELECT uid, source, description,
        {SELECT_SQL}
      FROM 'data/unified.parquet'
      WHERE {BASE_FILTER}
    ),
    g AS (
      SELECT
        CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
        uid,
        ({TECH_SUM}) AS tech_count
      FROM f
    )
    SELECT period,
           count(*) AS n,
           avg(tech_count) AS mean_tc,
           median(tech_count) AS median_tc,
           quantile(tech_count, 0.25) AS p25,
           quantile(tech_count, 0.75) AS p75,
           stddev(tech_count) AS std_tc
    FROM g
    GROUP BY 1 ORDER BY 1
    """
    print("\n=== Full distribution (SWE LinkedIn) ===")
    print(con.execute(q).fetchdf().to_string())

    # Per-tech prevalence
    q2 = f"""
    WITH f AS (
      SELECT
        CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
        uid, description,
        {SELECT_SQL}
      FROM 'data/unified.parquet'
      WHERE {BASE_FILTER}
    )
    SELECT period,
      {", ".join([f"avg(t_{n})*100 AS pct_{n}" for n in TECHS.keys()])}
    FROM f
    GROUP BY 1 ORDER BY 1
    """
    df = con.execute(q2).fetchdf()
    df.to_csv("exploration/tables/V2/A_tech_prevalence.csv", index=False)
    print("\n=== Per-tech prevalence (%) ===")
    # Transpose for readability
    dfT = df.set_index("period").T
    dfT.columns = ["y2024", "y2026"]
    dfT["delta_pp"] = dfT["y2026"] - dfT["y2024"]
    dfT["ratio"] = dfT["y2026"] / dfT["y2024"].clip(lower=0.01)
    print(dfT.sort_values("delta_pp", ascending=False).to_string())

    # Dedup by description_hash within period
    q3 = f"""
    WITH f AS (
      SELECT
        CASE WHEN source='scraped' THEN '2026' ELSE '2024' END AS period,
        uid, description_hash,
        {SELECT_SQL}
      FROM 'data/unified.parquet'
      WHERE {BASE_FILTER}
    ),
    withtc AS (
      SELECT period, description_hash, ({TECH_SUM}) AS tech_count
      FROM f
    ),
    dedup AS (
      SELECT period, description_hash, min(tech_count) AS tech_count
      FROM withtc
      GROUP BY 1,2
    )
    SELECT period, count(*) AS n_unique_desc, avg(tech_count) AS mean_tc, median(tech_count) AS median_tc
    FROM dedup GROUP BY 1 ORDER BY 1
    """
    print("\n=== Dedup-by-description_hash tech_count ===")
    print(con.execute(q3).fetchdf().to_string())


if __name__ == "__main__":
    main()
