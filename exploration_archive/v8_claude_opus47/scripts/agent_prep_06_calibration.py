"""Agent Prep step 6: keystone calibration table.

For each metric: compute arshkon_value, asaniczka_value, scraped_value (all on
default-filtered SWE LinkedIn), within_2024 and cross-period effects, and an
SNR-based verdict.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET_UNIFIED = ROOT / "data/unified.parquet"
PARQUET_CLEAN = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
PARQUET_TECH = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
OUT_PATH = ROOT / "exploration/artifacts/shared/calibration_table.csv"

# Broad AI mention regex (spec)
AI_BROAD = re.compile(
    r"\b(ai|artificial intelligence|ml|machine learning|llm|large language model|"
    r"generative ai|genai|copilot|cursor|claude|chatgpt|openai|anthropic|gemini|"
    r"gpt|prompt engineering|rag|retrieval augmented|agent|langchain)\b",
    re.IGNORECASE,
)

AI_TOOL_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt|gemini|codex|mcp|llamaindex|"
    r"langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|"
    r"huggingface|hugging face)\b",
    re.IGNORECASE,
)

MGMT_BROAD = re.compile(r"\b(manage|mentor|coach|hire|lead|team|stakeholder|coordinate)\b", re.IGNORECASE)
MGMT_STRICT = re.compile(r"\b(manage|mentor|coach|hire|direct reports|performance review|headcount|people management)\b", re.IGNORECASE)
ORG_SCOPE = re.compile(r"\b(ownership|end[- ]to[- ]end|cross[- ]functional|autonomous|initiative|stakeholder|architect|system design|distributed system)\b", re.IGNORECASE)
SOFT_SKILL = re.compile(r"\b(collaboration|communication|problem[- ]solving|teamwork|leadership)\b", re.IGNORECASE)


def cohen_d(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """Cohen's d with pooled SD (assumes independent samples)."""
    x_vals = x_vals[~np.isnan(x_vals)]
    y_vals = y_vals[~np.isnan(y_vals)]
    nx, ny = len(x_vals), len(y_vals)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = x_vals.mean(), y_vals.mean()
    vx, vy = x_vals.var(ddof=1), y_vals.var(ddof=1)
    pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return float("nan")
    return (mx - my) / pooled_sd


def pooled_mean(x_val: float, x_n: int, y_val: float, y_n: int) -> float:
    if x_n + y_n == 0:
        return float("nan")
    if math.isnan(x_val) or math.isnan(y_val):
        return float("nan")
    return (x_val * x_n + y_val * y_n) / (x_n + y_n)


def snr_verdict(snr_arsh: float, snr_pooled: float) -> str:
    vals = [v for v in (snr_arsh, snr_pooled) if not math.isnan(v)]
    if not vals:
        return "undefined"
    mx = max(vals)
    if mx >= 2.0:
        return "above_noise"
    if mx < 1.0:
        return "below_noise"
    return "marginal"


def main() -> None:
    con = duckdb.connect()

    print("Loading raw data for length / AI / management metrics...")
    df = con.execute(
        f"""
        SELECT
            uid, source, description, description_length,
            seniority_final, seniority_3level,
            is_aggregator, yoe_extracted
        FROM read_parquet('{PARQUET_UNIFIED}')
        WHERE source_platform='linkedin'
          AND is_english=true
          AND date_flag='ok'
          AND is_swe=true
        """
    ).df()
    print(f"Unified SWE rows: {len(df):,}")

    print("Joining cleaned text + tech matrix...")
    # Pull cleaned_len and text_source + tech_count
    clean = con.execute(
        f"""
        SELECT uid, length(description_cleaned) AS clean_len, text_source
        FROM read_parquet('{PARQUET_CLEAN}')
        """
    ).df()
    tech = con.execute(
        f"""
        SELECT * FROM read_parquet('{PARQUET_TECH}')
        """
    ).df()
    tech_cols = [c for c in tech.columns if c != "uid"]
    tech["tech_count"] = tech[tech_cols].sum(axis=1)
    tech_counts = tech[["uid", "tech_count"]]

    df = df.merge(clean, on="uid", how="left").merge(tech_counts, on="uid", how="left")
    df["tech_count"] = df["tech_count"].fillna(0).astype(int)
    print(f"After joins: {len(df):,}")

    # Pre-compute AI mention flags on raw description (spec says raw)
    print("Computing AI/mgmt/org/soft regex flags on raw description...")
    desc = df["description"].fillna("").values

    def flag_col(regex):
        return np.array([bool(regex.search(t)) for t in desc])

    df["ai_mention_binary"] = flag_col(AI_BROAD)
    df["ai_tool_binary"] = flag_col(AI_TOOL_STRICT)
    df["mgmt_broad_binary"] = flag_col(MGMT_BROAD)
    df["mgmt_strict_binary"] = flag_col(MGMT_STRICT)
    df["org_scope_binary"] = flag_col(ORG_SCOPE)
    df["soft_skill_binary"] = flag_col(SOFT_SKILL)

    # AI mention counts for density — use raw description
    print("Computing AI mention counts for density...")
    df["ai_count"] = np.array([len(AI_BROAD.findall(t)) for t in desc])

    # Partition by source
    df_arsh = df[df["source"] == "kaggle_arshkon"]
    df_asan = df[df["source"] == "kaggle_asaniczka"]
    df_scr = df[df["source"] == "scraped"]
    n_arsh, n_asan, n_scr = len(df_arsh), len(df_asan), len(df_scr)
    print(f"Per source: arshkon={n_arsh:,}, asaniczka={n_asan:,}, scraped={n_scr:,}")

    # -------------------------------------------------------------------
    # Helper: compute per-source + cross metric values for a given metric
    # -------------------------------------------------------------------
    rows = []

    def add_metric(name: str, arsh_val: float, asan_val: float, scr_val: float,
                   effect_mode: str = "diff", n_arsh: int = n_arsh, n_asan: int = n_asan,
                   arsh_sd: float | None = None, asan_sd: float | None = None):
        """effect_mode: 'diff' = simple arithmetic; 'cohens_d' requires arrays so handled separately."""
        pooled = pooled_mean(arsh_val, n_arsh, asan_val, n_asan)
        within = arsh_val - asan_val if not (math.isnan(arsh_val) or math.isnan(asan_val)) else float("nan")
        cross_a = scr_val - arsh_val if not (math.isnan(scr_val) or math.isnan(arsh_val)) else float("nan")
        cross_p = scr_val - pooled if not (math.isnan(scr_val) or math.isnan(pooled)) else float("nan")
        snr_a = abs(cross_a) / abs(within) if (within not in (0, None) and not math.isnan(within) and within != 0) else float("nan")
        snr_p = abs(cross_p) / abs(within) if (within not in (0, None) and not math.isnan(within) and within != 0) else float("nan")
        verdict = snr_verdict(snr_a, snr_p)
        rows.append({
            "metric": name,
            "arshkon_value": arsh_val,
            "asaniczka_value": asan_val,
            "scraped_value": scr_val,
            "n_arshkon": n_arsh,
            "n_asaniczka": n_asan,
            "n_scraped": n_scr,
            "within_2024_effect": within,
            "pooled_2024_value": pooled,
            "cross_period_effect_arshkon": cross_a,
            "cross_period_effect_pooled": cross_p,
            "snr_arshkon": snr_a,
            "snr_pooled": snr_p,
            "calibration_verdict": verdict,
        })

    # description_length mean / median
    add_metric("description_length_mean",
               df_arsh["description_length"].mean(),
               df_asan["description_length"].mean(),
               df_scr["description_length"].mean())
    add_metric("description_length_median",
               df_arsh["description_length"].median(),
               df_asan["description_length"].median(),
               df_scr["description_length"].median())

    # description_cleaned_length mean / median
    add_metric("description_cleaned_length_mean",
               df_arsh["clean_len"].mean(),
               df_asan["clean_len"].mean(),
               df_scr["clean_len"].mean())
    add_metric("description_cleaned_length_median",
               df_arsh["clean_len"].median(),
               df_asan["clean_len"].median(),
               df_scr["clean_len"].median())

    # yoe_extracted median (of known YOE)
    add_metric("yoe_extracted_median",
               df_arsh.loc[df_arsh["yoe_extracted"].notna(), "yoe_extracted"].median(),
               df_asan.loc[df_asan["yoe_extracted"].notna(), "yoe_extracted"].median(),
               df_scr.loc[df_scr["yoe_extracted"].notna(), "yoe_extracted"].median())

    # tech_count mean / median
    add_metric("tech_count_mean",
               df_arsh["tech_count"].mean(),
               df_asan["tech_count"].mean(),
               df_scr["tech_count"].mean())
    add_metric("tech_count_median",
               df_arsh["tech_count"].median(),
               df_asan["tech_count"].median(),
               df_scr["tech_count"].median())

    # tech_count_density (per 1K chars of cleaned description)
    def tech_density(g):
        clean_chars = g["clean_len"].sum()
        if clean_chars == 0:
            return float("nan")
        return 1000.0 * g["tech_count"].sum() / clean_chars

    add_metric("tech_count_density_per_1k",
               tech_density(df_arsh), tech_density(df_asan), tech_density(df_scr))

    # ai_mention_binary share (rows matching broad AI regex on raw description)
    add_metric("ai_mention_binary_share",
               df_arsh["ai_mention_binary"].mean(),
               df_asan["ai_mention_binary"].mean(),
               df_scr["ai_mention_binary"].mean())

    # ai_mention_density — AI mentions per 1K chars of cleaned description
    def ai_density(g):
        clean_chars = g["clean_len"].sum()
        if clean_chars == 0:
            return float("nan")
        return 1000.0 * g["ai_count"].sum() / clean_chars

    add_metric("ai_mention_density_per_1k",
               ai_density(df_arsh), ai_density(df_asan), ai_density(df_scr))

    # ai_tool_binary
    add_metric("ai_tool_binary_share",
               df_arsh["ai_tool_binary"].mean(),
               df_asan["ai_tool_binary"].mean(),
               df_scr["ai_tool_binary"].mean())

    # management broad / strict
    add_metric("management_broad_binary_share",
               df_arsh["mgmt_broad_binary"].mean(),
               df_asan["mgmt_broad_binary"].mean(),
               df_scr["mgmt_broad_binary"].mean())
    add_metric("management_strict_binary_share",
               df_arsh["mgmt_strict_binary"].mean(),
               df_asan["mgmt_strict_binary"].mean(),
               df_scr["mgmt_strict_binary"].mean())

    # org_scope_binary
    add_metric("org_scope_binary_share",
               df_arsh["org_scope_binary"].mean(),
               df_asan["org_scope_binary"].mean(),
               df_scr["org_scope_binary"].mean())

    # soft_skill_binary
    add_metric("soft_skill_binary_share",
               df_arsh["soft_skill_binary"].mean(),
               df_asan["soft_skill_binary"].mean(),
               df_scr["soft_skill_binary"].mean())

    # aggregator_share
    add_metric("aggregator_share",
               df_arsh["is_aggregator"].mean(),
               df_asan["is_aggregator"].mean(),
               df_scr["is_aggregator"].mean())

    # yoe_known_share
    add_metric("yoe_known_share",
               df_arsh["yoe_extracted"].notna().mean(),
               df_asan["yoe_extracted"].notna().mean(),
               df_scr["yoe_extracted"].notna().mean())

    # J1 / J2 / J3 / S1
    add_metric("j1_entry_share",
               (df_arsh["seniority_final"] == "entry").mean(),
               (df_asan["seniority_final"] == "entry").mean(),
               (df_scr["seniority_final"] == "entry").mean())
    add_metric("j2_entry_or_associate_share",
               df_arsh["seniority_final"].isin(["entry", "associate"]).mean(),
               df_asan["seniority_final"].isin(["entry", "associate"]).mean(),
               df_scr["seniority_final"].isin(["entry", "associate"]).mean())

    def yoe_share_leq_2(g):
        yoe = g.loc[g["yoe_extracted"].notna(), "yoe_extracted"]
        if len(yoe) == 0:
            return float("nan")
        return (yoe <= 2).mean()

    add_metric("j3_yoe_leq_2_share",
               yoe_share_leq_2(df_arsh), yoe_share_leq_2(df_asan), yoe_share_leq_2(df_scr))
    add_metric("s1_senior_share",
               df_arsh["seniority_final"].isin(["mid-senior", "director"]).mean(),
               df_asan["seniority_final"].isin(["mid-senior", "director"]).mean(),
               df_scr["seniority_final"].isin(["mid-senior", "director"]).mean())

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out)} metrics)")
    print("\n=== Calibration table ===")
    print(out[["metric", "arshkon_value", "asaniczka_value", "scraped_value",
               "within_2024_effect", "cross_period_effect_pooled",
               "snr_pooled", "calibration_verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
