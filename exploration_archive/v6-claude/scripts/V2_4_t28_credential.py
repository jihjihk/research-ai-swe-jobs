"""V2.4 — Independent T28 credential-stack-gap re-derivation.

Defines a credential stack as number of distinct categories (out of 6) mentioned:
  1. YOE (years-experience pattern)
  2. Degree (BS/MS/PhD)
  3. Certification
  4. Security clearance
  5. Specific tech_count ≥ 3
  6. Leadership/mentor pattern

Then computes per-archetype (entry - mid-senior) gap in 2024 and 2026, and Δgap.
Reports the 7/10 sign-flip finding.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exploration" / "tables" / "V2"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    con = duckdb.connect()
    # Join unified + tech matrix + archetype labels
    sql = """
    SELECT u.uid,
           CASE WHEN u.period LIKE '2024%' THEN '2024' ELSE '2026' END AS year,
           u.seniority_final,
           u.description_core_llm AS text,
           (t.llm OR t.rag OR t.agents_framework OR t.copilot OR t.claude_api OR t.claude_tool
            OR t.cursor_tool OR t.gemini_tool OR t.codex_tool OR t.chatgpt OR t.openai_api
            OR t.prompt_engineering OR t.fine_tuning OR t.mcp OR t.embedding OR t.transformer_arch
            OR t.machine_learning OR t.deep_learning OR t.pytorch OR t.tensorflow OR t.langchain
            OR t.langgraph OR t.nlp OR t.huggingface)::INT AS any_ai_broad,
           (t.python::INT + t.java::INT + t.aws::INT + t.azure::INT + t.gcp::INT + t.javascript::INT + t.typescript::INT + t.react::INT
            + t.docker::INT + t.kubernetes::INT + t.sql_lang::INT + t.rest_api::INT + t.kafka::INT + t.spark::INT
            + t.redis::INT + t.postgres::INT + t.cicd::INT + t.terraform::INT + t.linux::INT
            + t.nodejs::INT + t.agile::INT) AS tech_count_approx,
           a.archetype,
           a.archetype_name,
           u.yoe_extracted
    FROM read_parquet('data/unified.parquet') u
    LEFT JOIN read_parquet('exploration/artifacts/shared/swe_tech_matrix.parquet') t
        USING (uid)
    LEFT JOIN read_parquet('exploration/artifacts/shared/swe_archetype_labels.parquet') a
        USING (uid)
    WHERE u.is_swe = TRUE AND u.source_platform = 'linkedin'
      AND u.is_english = TRUE AND u.date_flag = 'ok'
      AND u.seniority_final IN ('entry', 'mid-senior')
      AND u.llm_extraction_coverage = 'labeled'
    """
    df = con.execute(sql).fetchdf()
    print(f"Loaded {len(df)} rows")

    # Compute credential-stack depth (6 categories):
    # 1. YOE present (or years-experience mention)
    # 2. Degree mention (BS/MS/PhD/bachelor/master/doctorate)
    # 3. Certification (cert*, AWS cert, PMP, etc.)
    # 4. Security clearance
    # 5. tech_count_approx >= 3
    # 6. Leadership / mentorship
    text = df["text"].fillna("").str.lower()

    yoe_rx = re.compile(r"\b(\d+)\+?\s*(?:years?|yrs?)\b")
    degree_rx = re.compile(r"\b(b\.?s\.?|m\.?s\.?|phd|ph\.d\.|bachelor|master|doctorate)\b")
    cert_rx = re.compile(r"\b(certif(?:ied|ication|y)|aws\s+cert|azure\s+cert|google\s+cert|pmp|cissp|ckad|cka|scrum\s+master)\b")
    clear_rx = re.compile(r"\b(clearance|secret|ts\/sci|top\s+secret|poly|polygraph)\b")
    lead_rx = re.compile(r"\b(mentor(?:s|ed|ing|ship)?|lead(?:s|ing|er(?:ship)?)?|manage|supervis(?:e|ing)|technical\s+direction|direct\s+reports)\b")

    df["cred_yoe"] = text.apply(lambda t: bool(yoe_rx.search(t))).astype(int)
    df["cred_degree"] = text.apply(lambda t: bool(degree_rx.search(t))).astype(int)
    df["cred_cert"] = text.apply(lambda t: bool(cert_rx.search(t))).astype(int)
    df["cred_clear"] = text.apply(lambda t: bool(clear_rx.search(t))).astype(int)
    df["cred_tech"] = (df["tech_count_approx"].fillna(0) >= 3).astype(int)
    df["cred_lead"] = text.apply(lambda t: bool(lead_rx.search(t))).astype(int)
    df["cred_stack"] = df[["cred_yoe", "cred_degree", "cred_cert", "cred_clear", "cred_tech", "cred_lead"]].sum(axis=1)

    # Restrict to labeled archetypes
    df = df.dropna(subset=["archetype"])
    df = df[df["archetype"] != -2]

    # Count rows per (archetype, year, seniority)
    counts = (
        df.groupby(["archetype_name", "archetype", "year", "seniority_final"]).size().unstack(["year", "seniority_final"], fill_value=0)
    )
    print("\nArchetype row counts:")
    print(counts.head(20))

    # For each archetype, compute (entry - mid-senior) cred_stack gap in 2024 and 2026
    rows = []
    for (arch_name, arch_id), sub in df.groupby(["archetype_name", "archetype"]):
        gaps = {}
        ok = True
        for year in ["2024", "2026"]:
            ys = sub[sub["year"] == year]
            e = ys[ys["seniority_final"] == "entry"]["cred_stack"]
            m = ys[ys["seniority_final"] == "mid-senior"]["cred_stack"]
            if len(e) < 5 or len(m) < 100:
                ok = False
                break
            gaps[f"gap_{year}"] = e.mean() - m.mean()
            gaps[f"n_entry_{year}"] = len(e)
            gaps[f"n_ms_{year}"] = len(m)
        if ok:
            gaps["archetype"] = arch_name
            gaps["archetype_id"] = int(arch_id)
            gaps["delta_gap"] = gaps["gap_2026"] - gaps["gap_2024"]
            rows.append(gaps)

    gap_df = pd.DataFrame(rows).sort_values("delta_gap")
    print("\nPer-archetype credential stack gap (entry − mid-senior) per period:")
    print(gap_df[["archetype", "archetype_id", "gap_2024", "gap_2026", "delta_gap", "n_entry_2024", "n_ms_2024", "n_entry_2026", "n_ms_2026"]])

    gap_df.to_csv(OUT / "V2_4_credential_gap.csv", index=False)

    # Sign flip count
    flipped = ((gap_df["gap_2024"] > 0) & (gap_df["gap_2026"] < 0)).sum()
    converged = (gap_df["delta_gap"] < 0).sum()
    print(f"\nTotal archetypes with valid data: {len(gap_df)}")
    print(f"Sign-flipped (2024 positive → 2026 negative): {flipped}")
    print(f"Δgap negative (converged): {converged}")

    # Sensitivity: stricter n ≥ 200 for mid-senior
    rows200 = []
    for (arch_name, arch_id), sub in df.groupby(["archetype_name", "archetype"]):
        gaps = {}
        ok = True
        for year in ["2024", "2026"]:
            ys = sub[sub["year"] == year]
            e = ys[ys["seniority_final"] == "entry"]["cred_stack"]
            m = ys[ys["seniority_final"] == "mid-senior"]["cred_stack"]
            if len(e) < 5 or len(m) < 200:
                ok = False
                break
            gaps[f"gap_{year}"] = e.mean() - m.mean()
        if ok:
            gaps["archetype"] = arch_name
            gaps["delta_gap"] = gaps["gap_2026"] - gaps["gap_2024"]
            rows200.append(gaps)
    g200 = pd.DataFrame(rows200).sort_values("delta_gap")
    print(f"\nStricter threshold (n_ms ≥ 200): {len(g200)} archetypes")
    flipped200 = ((g200["gap_2024"] > 0) & (g200["gap_2026"] < 0)).sum()
    converged200 = (g200["delta_gap"] < 0).sum()
    print(f"  Sign-flipped: {flipped200}")
    print(f"  Converged: {converged200}")
    if len(g200):
        print(g200[["archetype", "gap_2024", "gap_2026", "delta_gap"]])


if __name__ == "__main__":
    main()
