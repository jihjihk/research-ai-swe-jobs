"""V1.3 — Alternative explanations for Gate 2 headlines.

1. AI tool/framework explosion — drop top-10 AI-mentioning companies, does
   the rise survive?
2. Tech network modularity rise — recompute with cap=20 (instead of 50).
3. T13 responsibilities classifier FP sample (handled separately in notebook).
4. T09 employer-template artifact check — asaniczka rows in topics 10/13/17/19?
5. T08 arshkon-only entry-share flip — compute entry-of-all (not of-known).
"""

from __future__ import annotations

import duckdb
import pandas as pd

UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
META = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
TECH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet"
ARCH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_archetype_labels.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"

con = duckdb.connect()


# -----------------------------------------------------------------------------
# Alt 1: drop top 10 AI-mentioning companies, does AI rise survive?
# -----------------------------------------------------------------------------
def alt_1_drop_top_ai_companies() -> None:
    print("=" * 72)
    print("ALT 1: AI rise after dropping top 10 AI-mentioning companies")
    print("=" * 72)

    # Build ai_any per row joined to company; operate on full SWE frame
    ai_cols = [
        "claude_tool", "copilot", "langchain", "agents_framework", "embedding",
        "llm", "rag", "openai_api", "claude_api", "prompt_engineering",
        "fine_tuning", "mcp", "transformer_arch", "machine_learning",
        "deep_learning", "pytorch", "tensorflow", "cursor_tool", "chatgpt",
        "gemini_tool", "codex_tool", "huggingface", "nlp", "langgraph",
    ]
    tech_sel = ",".join([f"CAST(t.{c} AS INTEGER) AS {c}" for c in ai_cols])
    q = f"""
    WITH base AS (
      SELECT m.uid, m.source, u.company_name_canonical, {tech_sel}
      FROM '{META}' m
      INNER JOIN '{TECH}' t USING (uid)
      INNER JOIN '{UNI}' u USING (uid)
    )
    SELECT *, CASE WHEN ({ '+'.join(ai_cols) }) > 0 THEN 1 ELSE 0 END AS ai_any
    FROM base
    """
    df = con.execute(q).fetchdf()
    df["period_bucket"] = df["source"].apply(lambda s: "2026" if s == "scraped" else "2024")

    # Top 10 AI-mentioning companies in 2026
    scr = df[df["period_bucket"] == "2026"]
    top10 = (
        scr[scr["ai_any"] == 1]
        .groupby("company_name_canonical")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    print(f"Top 10 AI-mentioning 2026 companies: {top10}")

    # Primary (all)
    p = df.groupby("period_bucket")["ai_any"].mean()
    # Drop top 10
    df2 = df[~df["company_name_canonical"].isin(top10)]
    p2 = df2.groupby("period_bucket")["ai_any"].mean()

    print()
    print(f"Primary aggregate AI-any:  2024 {p['2024']*100:.2f}% -> 2026 {p['2026']*100:.2f}%  delta +{(p['2026']-p['2024'])*100:.2f} pp")
    print(f"Drop-top-10 AI companies:  2024 {p2['2024']*100:.2f}% -> 2026 {p2['2026']*100:.2f}%  delta +{(p2['2026']-p2['2024'])*100:.2f} pp")
    print(f"Multiplicative primary:    {p['2026']/p['2024']:.2f}x")
    print(f"Multiplicative drop-top10: {p2['2026']/p2['2024']:.2f}x")


# -----------------------------------------------------------------------------
# Alt 5: Arshkon-only entry-of-all (not of-known)
# -----------------------------------------------------------------------------
def alt_5_entry_of_all() -> None:
    print()
    print("=" * 72)
    print("ALT 5: Arshkon-only entry-share denominator sensitivity")
    print("=" * 72)

    q = f"""
    SELECT source, seniority_final, COUNT(*) n
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND source IN ('kaggle_arshkon', 'scraped')
    GROUP BY source, seniority_final
    ORDER BY source, seniority_final
    """
    df = con.execute(q).fetchdf()
    print(df.to_string())

    def shares(d: pd.DataFrame, label: str) -> dict:
        total = d["n"].sum()
        entry = d[d["seniority_final"] == "entry"]["n"].sum()
        unknown = d[d["seniority_final"] == "unknown"]["n"].sum()
        known = total - unknown
        return {
            "slice": label,
            "n_total": int(total),
            "n_unknown": int(unknown),
            "pct_unknown": unknown / total,
            "entry_of_all": entry / total,
            "entry_of_known": entry / known if known else float("nan"),
        }

    rows = [
        shares(df[df["source"] == "kaggle_arshkon"], "arshkon_only_2024"),
        shares(df[df["source"] == "scraped"], "scraped_2026"),
    ]
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    print()
    print("Gate 2 flip claim: 7.72% -> 6.70% entry-of-known (-1.0 pp)")
    print("Testing whether direction holds under entry-of-all denominator:")
    dd = out.iloc[1]["entry_of_all"] - out.iloc[0]["entry_of_all"]
    dk = out.iloc[1]["entry_of_known"] - out.iloc[0]["entry_of_known"]
    print(f"  entry-of-all:   {out.iloc[0]['entry_of_all']*100:.2f}% -> {out.iloc[1]['entry_of_all']*100:.2f}%  delta {dd*100:+.2f} pp")
    print(f"  entry-of-known: {out.iloc[0]['entry_of_known']*100:.2f}% -> {out.iloc[1]['entry_of_known']*100:.2f}%  delta {dk*100:+.2f} pp")
    out.to_csv(f"{OUT}/V1_alt5_entry_denominator.csv", index=False)


# -----------------------------------------------------------------------------
# Alt 4: asaniczka rows in employer-template topics?
# -----------------------------------------------------------------------------
def alt_4_asaniczka_in_templates() -> None:
    print()
    print("=" * 72)
    print("ALT 4: asaniczka rows in employer-template artifact topics (10,13,17,19)")
    print("=" * 72)

    q = f"""
    SELECT a.archetype, a.archetype_name, m.source,
           COUNT(*) n
    FROM '{ARCH}' a
    INNER JOIN '{META}' m USING (uid)
    WHERE a.archetype >= 0
    GROUP BY a.archetype, a.archetype_name, m.source
    ORDER BY a.archetype, m.source
    """
    df = con.execute(q).fetchdf()
    template_ids = [10, 13, 17, 19]
    templates = df[df["archetype"].isin(template_ids)]
    print("Per-archetype template membership:")
    print(templates.to_string())

    # Aggregate: asaniczka share of template-topic rows
    asan_templates = templates[templates["source"] == "kaggle_asaniczka"]["n"].sum()
    total_templates = templates["n"].sum()
    asan_all = df[df["source"] == "kaggle_asaniczka"]["n"].sum()
    total_all = df["n"].sum()
    print()
    print(f"asaniczka in template topics: {asan_templates:,} / {total_templates:,} ({asan_templates/total_templates*100:.1f}%)")
    print(f"asaniczka in all topics:      {asan_all:,} / {total_all:,} ({asan_all/total_all*100:.1f}%)")
    print(f"Enrichment vs. base: {(asan_templates/total_templates)/(asan_all/total_all):.2f}x")


if __name__ == "__main__":
    alt_1_drop_top_ai_companies()
    alt_5_entry_of_all()
    alt_4_asaniczka_in_templates()
