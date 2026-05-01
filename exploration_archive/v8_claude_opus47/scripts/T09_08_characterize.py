"""T09 Step 8: Characterize BERTopic archetypes.

For each cluster (using BERTopic reduced labels):
- Top 20 c-TF-IDF terms
- Seniority distribution (% of each seniority_final level in this cluster)
- J2 entry share of known-seniority PER archetype PER period
  (CRITICAL for H1 testing)
- Period distribution (% of each period in this cluster)
- Average description_cleaned_length, yoe_extracted, tech_count
- AI-mention binary share per archetype per period
- Archetype proportion over time (share-of-SWE framing)

Writes:
  exploration/tables/T09/archetype_characterization.csv
  exploration/tables/T09/archetype_period_share.csv
  exploration/tables/T09/archetype_entry_j2_by_period.csv
  exploration/tables/T09/archetype_ai_share_by_period.csv
  exploration/artifacts/T09/archetype_names.csv (content labels)
"""
import os
import re
import numpy as np
import pandas as pd
import duckdb

OUTDIR = "exploration/artifacts/T09"
TABLES = "exploration/tables/T09"

# Content-based name lookup -- NOT RQ-driven.
# Keyed by topic ID in the mts=30 seed=SEED BERTopic run.
# These names come from inspecting the top-20 terms per topic.
ARCHETYPE_NAMES = {
    -1: "unassigned",
    0: "generic_software_engineer",
    1: "data_engineering",
    2: "ai_ml_engineering",
    3: "cloud_devops",
    4: "systems_engineering",
    5: "frontend_react",
    6: "java_spring_backend",
    7: "mobile_android_ios",
    8: "network_security_linux",
    9: "customer_project_engineer",
    10: "technology_leadership",
    11: "aws_cloud_services",
    12: "dotnet_web_sql",
    13: "qa_test_automation",
    14: "python_backend_developer",
    15: "test_engineering",
    16: "azure_dotnet_web",
    17: "recommendation_ecommerce",
    18: "angular_frontend",
    19: "backend_platform",
    20: "internship_new_grad",
    21: "golang_cloud",
}


def ai_mention(text):
    """Broad AI-mention regex from calibration table."""
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r"\b(ai|a\.i\.|llm|llms|gpt|chatgpt|gemini|claude|copilot|cursor|"
        r"generative ai|gen[- ]ai|machine learning|deep learning|rag|"
        r"langchain|langgraph|agentic|prompt engineering|openai|anthropic|"
        r"foundation model)\b",
        text.lower()))


def main():
    df = pd.read_parquet(f"{OUTDIR}/sample_with_assignments.parquet")
    print(f"Loaded {len(df)} rows with assignments")

    # Merge tech_count from shared tech matrix
    con = duckdb.connect()
    tech = con.execute("""
        SELECT uid FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'
    """).fetchdf()
    tech_cols = con.execute("DESCRIBE SELECT * FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'").fetchdf()
    tech_col_names = [c for c in tech_cols["column_name"].tolist() if c != "uid"]
    tech_mat = con.execute(f"""
        SELECT uid, {' + '.join(['CAST(' + c + ' AS INT)' for c in tech_col_names])} AS tech_count
        FROM 'exploration/artifacts/shared/swe_tech_matrix.parquet'
    """).fetchdf()
    df = df.merge(tech_mat, on="uid", how="left")
    print(f"After tech_count merge: {len(df)}, tech_count NaN = {df.tech_count.isna().sum()}")

    # AI mention (binary)
    df["ai_mention"] = df["description_cleaned"].apply(ai_mention)

    # Name each archetype
    df["archetype_name"] = df["topic_reduced"].map(ARCHETYPE_NAMES).fillna("unknown")

    # Period coarse buckets -- for characterization we combine 2026-03 and 2026-04
    df["period_bucket"] = df["period"].map({
        "2024-04": "2024-04",
        "2024-01": "2024-01",
        "2026-03": "2026",
        "2026-04": "2026",
    })

    # --- Per-archetype characterization ---
    rows = []
    for arch_id in sorted(df["topic_reduced"].unique()):
        sub = df[df["topic_reduced"] == arch_id]
        name = ARCHETYPE_NAMES.get(arch_id, "unknown")
        sen_counts = sub["seniority_final"].value_counts().to_dict()
        total = len(sub)
        known = sub[sub.seniority_final != "unknown"]
        n_known = len(known)
        j2_share = 0.0
        if n_known > 0:
            j2_share = (known["seniority_final"].isin(["entry", "associate"]).sum()
                         / n_known)
        s1_share = 0.0
        if n_known > 0:
            s1_share = (known["seniority_final"].isin(["mid-senior", "director"]).sum()
                         / n_known)
        rows.append({
            "archetype_id": arch_id,
            "archetype_name": name,
            "n": total,
            "share_of_sample_pct": round(100.0 * total / len(df), 2),
            "share_known_entry_associate_j2": round(j2_share, 3),
            "share_known_mid_director_s1": round(s1_share, 3),
            "share_mid_senior": round(sub.seniority_final.eq("mid-senior").mean(), 3),
            "share_entry": round(sub.seniority_final.eq("entry").mean(), 3),
            "share_associate": round(sub.seniority_final.eq("associate").mean(), 3),
            "share_director": round(sub.seniority_final.eq("director").mean(), 3),
            "share_unknown": round(sub.seniority_final.eq("unknown").mean(), 3),
            "share_aggregator": round(sub.is_aggregator.mean(), 3),
            "share_2024_01": round(sub.period.eq("2024-01").mean(), 3),
            "share_2024_04": round(sub.period.eq("2024-04").mean(), 3),
            "share_2026_03": round(sub.period.eq("2026-03").mean(), 3),
            "share_2026_04": round(sub.period.eq("2026-04").mean(), 3),
            "mean_desc_length": round(sub.description_cleaned_length.mean(), 0),
            "mean_yoe": round(sub.yoe_extracted.mean(), 2) if not sub.yoe_extracted.isna().all() else None,
            "mean_tech_count": round(sub.tech_count.mean(), 2),
            "ai_mention_share": round(sub.ai_mention.mean(), 3),
        })
    char_df = pd.DataFrame(rows)
    char_df.to_csv(f"{TABLES}/archetype_characterization.csv", index=False)
    print(char_df.to_string(index=False))

    # --- Archetype proportion over periods (share-of-SWE framing) ---
    xtab = pd.crosstab(df.archetype_name, df.period_bucket, normalize="columns") * 100
    xtab = xtab.round(2)
    xtab["delta_2024avg_to_2026"] = xtab["2026"] - 0.5 * (xtab.get("2024-01", 0) + xtab.get("2024-04", 0))
    xtab.to_csv(f"{TABLES}/archetype_period_share.csv")
    print("\n-- Archetype share by period (share of SWE within each period) --")
    print(xtab.sort_values("delta_2024avg_to_2026", ascending=False).to_string())

    # --- Entry share (J2) per archetype per period (CRITICAL for H1) ---
    entry_rows = []
    for name in sorted(df.archetype_name.unique()):
        for period in ["2024-01", "2024-04", "2026"]:
            sub = df[(df.archetype_name == name) & (df.period_bucket == period)]
            known = sub[sub.seniority_final != "unknown"]
            n_known = len(known)
            n_total = len(sub)
            j2_share = (known.seniority_final.isin(["entry", "associate"]).sum() / n_known) if n_known > 0 else float("nan")
            entry_rows.append({
                "archetype": name,
                "period": period,
                "n_total": n_total,
                "n_known_seniority": n_known,
                "j2_entry_associate_share": round(j2_share, 3) if not pd.isna(j2_share) else None,
            })
    entry_df = pd.DataFrame(entry_rows)
    # Pivot for readability
    epivot = entry_df.pivot(index="archetype", columns="period", values="j2_entry_associate_share")
    epivot["n_known_2024_01"] = entry_df[entry_df.period=="2024-01"].set_index("archetype")["n_known_seniority"]
    epivot["n_known_2024_04"] = entry_df[entry_df.period=="2024-04"].set_index("archetype")["n_known_seniority"]
    epivot["n_known_2026"] = entry_df[entry_df.period=="2026"].set_index("archetype")["n_known_seniority"]
    epivot.to_csv(f"{TABLES}/archetype_entry_j2_by_period.csv")
    print("\n-- J2 entry share (of known seniority) by archetype × period --")
    print(epivot.round(3).to_string())

    # --- AI mention by archetype × period ---
    ai_rows = []
    for name in sorted(df.archetype_name.unique()):
        for period in ["2024-01", "2024-04", "2026"]:
            sub = df[(df.archetype_name == name) & (df.period_bucket == period)]
            share = sub.ai_mention.mean() if len(sub) > 0 else float("nan")
            ai_rows.append({"archetype": name, "period": period,
                            "n": len(sub),
                            "ai_mention_share": round(share, 3) if not pd.isna(share) else None})
    ai_df = pd.DataFrame(ai_rows)
    apivot = ai_df.pivot(index="archetype", columns="period", values="ai_mention_share")
    apivot.to_csv(f"{TABLES}/archetype_ai_share_by_period.csv")
    print("\n-- AI mention share by archetype × period --")
    print(apivot.round(3).to_string())

    # Save archetype names mapping
    arch_names = pd.DataFrame([{"archetype_id": k, "archetype_name": v}
                               for k, v in ARCHETYPE_NAMES.items()])
    arch_names.to_csv(f"{OUTDIR}/archetype_names.csv", index=False)


if __name__ == "__main__":
    main()
