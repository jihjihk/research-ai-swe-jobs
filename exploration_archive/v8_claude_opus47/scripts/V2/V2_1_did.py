"""V2.1 — Independent T18 DiD re-derivation (DuckDB-native for speed).

Re-derives (SWE change) − (control change) for:
  ai_strict_binary, ai_broad_binary, tech_count mean,
  description_cleaned_length median, org_scope_binary, mgmt_strict_binary
with bootstrap 95% CI.

Patterns are independently re-declared (V1-refined) and run via DuckDB's
regexp_matches. Uses description_core_llm when available, falls back to
description.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = REPO / "data/unified.parquet"
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DuckDB regex literal strings (must escape for DuckDB string)
AI_STRICT = r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b"
AI_BROAD = r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face|ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b"
MGMT_STRICT = r"\b(mentor(s|ed|ing|ship)?|coach(es|ed|ing)?|(hire\s+and\s+(develop|manage|grow))|(hire(\s+a)?\s+team)|(hiring\s+(and|manager|engineers|team|plan))|headcount|performance[- ]review)\b"
ORG_SCOPE = r"\b(stakeholder(s)?|cross[- ]functional|leadership|strategic|vision|roadmap|mentor(ing|ship)?|influence|drive\s+(the|our|business|product|strategy)|partner(ship|ing)?\s+with|executive|c[- ]?suite|align(ment)?|org(anization)?)\b"

# Tech patterns for tech_count
TECHS = [
    r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bgolang\b", r"\brust\b", r"\bc\+\+\b", r"\bc#\b", r"\bruby\b",
    r"\bkotlin\b", r"\bswift\b", r"\bscala\b", r"\bphp\b", r"\bsql\b",
    r"\breact\b", r"\bangular\b", r"\bvue\.?js\b", r"\bnext\.?js\b",
    r"\bhtml\b", r"\bcss\b", r"\bnode\.?js\b", r"\bdjango\b", r"\bflask\b",
    r"\bspring\b", r"\b\.net\b", r"\baws\b", r"\bazure\b", r"\bgcp\b",
    r"\bkubernetes\b", r"\bdocker\b", r"\bterraform\b", r"\bpostgres\w*\b",
    r"\bmysql\b", r"\bmongo\w*\b", r"\bredis\b", r"\bkafka\b", r"\bspark\b",
    r"\bsnowflake\b", r"\btensorflow\b", r"\bpytorch\b", r"\bpandas\b",
    r"\bjupyter\b", r"\blangchain\b", r"\brag\b", r"\bcopilot\b", r"\bcursor\b",
    r"\bopenai\b", r"\bclaude\b", r"\bagile\b", r"\bscrum\b", r"\bmicroservices?\b",
    r"\bjenkins\b", r"\bgithub\s+actions\b", r"\blinux\b", r"\bbash\b", r"\bnumpy\b",
]
# Build a single CASE-sum expression for tech_count
def tech_count_expr(col: str) -> str:
    parts = []
    for pat in TECHS:
        parts.append(
            f"(CASE WHEN regexp_matches(lower({col}), '{pat}') THEN 1 ELSE 0 END)"
        )
    return " + ".join(parts)


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    text_col = "COALESCE(description_core_llm, description, '')"
    text_lower = f"lower({text_col})"
    tech_cnt = tech_count_expr(text_col)

    q = f"""
    SELECT
      uid,
      CASE WHEN period LIKE '2024%' THEN '2024'
           WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket,
      CASE WHEN is_swe THEN 'swe'
           WHEN is_swe_adjacent THEN 'adjacent'
           WHEN is_control THEN 'control' END AS grp,
      CAST(regexp_matches({text_lower}, '{AI_STRICT}') AS INTEGER) AS ai_strict,
      CAST(regexp_matches({text_lower}, '{AI_BROAD}') AS INTEGER) AS ai_broad,
      CAST(regexp_matches({text_lower}, '{MGMT_STRICT}') AS INTEGER) AS mgmt_strict,
      CAST(regexp_matches({text_lower}, '{ORG_SCOPE}') AS INTEGER) AS org_scope,
      ({tech_cnt}) AS tech_count,
      LENGTH({text_col}) AS desc_len,
      is_aggregator
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin'
      AND is_english=true
      AND date_flag='ok'
      AND (is_swe OR is_swe_adjacent OR is_control)
    """
    print("[V2.1] running DuckDB regex pass...")
    df = con.execute(q).df()
    df = df[df["period_bucket"].isin(["2024", "2026"])]
    df = df[df["grp"].isin(["swe", "adjacent", "control"])]
    print(f"[V2.1] computed features on {len(df)} rows")
    df.to_parquet(OUT_DIR / "V2_1_posting_features.parquet", index=False)

    # Group × period means (binary -> share; tech_count -> mean; desc_len -> median)
    metrics_mean = ["ai_strict", "ai_broad", "mgmt_strict", "org_scope", "tech_count"]
    rows = []
    for grp in ["swe", "adjacent", "control"]:
        for per in ["2024", "2026"]:
            sub = df[(df["grp"] == grp) & (df["period_bucket"] == per)]
            row = {"grp": grp, "period": per, "n": len(sub)}
            for m in metrics_mean:
                row[f"{m}_mean"] = sub[m].mean()
            row["desc_len_median"] = sub["desc_len"].median()
            rows.append(row)
    gp = pd.DataFrame(rows)
    gp.to_csv(OUT_DIR / "V2_1_group_period.csv", index=False)
    print("\n[V2.1] Group × Period:")
    print(gp.to_string(index=False))

    # DiD: (swe 2026 - swe 2024) - (control 2026 - control 2024)
    def d(grp, per, col):
        sub = df[(df["grp"] == grp) & (df["period_bucket"] == per)][col]
        if col == "desc_len":
            return sub.median()
        return sub.mean()

    rows = []
    for m in metrics_mean + ["desc_len"]:
        swe_d = d("swe", "2026", m) - d("swe", "2024", m)
        ctrl_d = d("control", "2026", m) - d("control", "2024", m)
        adj_d = d("adjacent", "2026", m) - d("adjacent", "2024", m)
        did_ctrl = swe_d - ctrl_d
        did_adj = swe_d - adj_d
        pct_ctrl = did_ctrl / swe_d if abs(swe_d) > 1e-9 else float("nan")
        pct_adj = did_adj / swe_d if abs(swe_d) > 1e-9 else float("nan")
        rows.append({
            "metric": m,
            "swe_delta": swe_d,
            "control_delta": ctrl_d,
            "adj_delta": adj_d,
            "did_vs_control": did_ctrl,
            "did_vs_adjacent": did_adj,
            "pct_swe_only_vs_control": pct_ctrl,
            "pct_swe_only_vs_adjacent": pct_adj,
        })
    did_df = pd.DataFrame(rows)
    did_df.to_csv(OUT_DIR / "V2_1_did_headline.csv", index=False)
    print("\n[V2.1] DiD Headline:")
    print(did_df.to_string(index=False))

    # Bootstrap CI for did_vs_control
    print("\n[V2.1] bootstrap (300 reps)...")
    np.random.seed(42)
    ci_rows = []
    for m in metrics_mean + ["desc_len"]:
        # Subgroup arrays
        arrs = {}
        for grp in ["swe", "control"]:
            for per in ["2024", "2026"]:
                arrs[(grp, per)] = df[(df["grp"] == grp) & (df["period_bucket"] == per)][m].values
        boot = []
        for _ in range(300):
            vals = {}
            for (grp, per), a in arrs.items():
                ix = np.random.randint(0, len(a), size=len(a))
                if m == "desc_len":
                    vals[(grp, per)] = np.median(a[ix])
                else:
                    vals[(grp, per)] = a[ix].mean()
            b_did = (vals[("swe", "2026")] - vals[("swe", "2024")]) - (
                vals[("control", "2026")] - vals[("control", "2024")]
            )
            boot.append(b_did)
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
        ci_rows.append({"metric": m, "did_lo95": lo, "did_hi95": hi})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(OUT_DIR / "V2_1_did_ci.csv", index=False)
    print(ci_df.to_string(index=False))

    # Stricter control: non-aggregator only
    print("\n[V2.1] alt stricter control: non-aggregator only:")
    sub_sa = df[df["is_aggregator"] != True]  # drop aggregators
    for m in metrics_mean + ["desc_len"]:
        def d2(g, p):
            s = sub_sa[(sub_sa["grp"] == g) & (sub_sa["period_bucket"] == p)][m]
            return s.median() if m == "desc_len" else s.mean()
        swe_d = d2("swe", "2026") - d2("swe", "2024")
        ctrl_d = d2("control", "2026") - d2("control", "2024")
        did = swe_d - ctrl_d
        base_swe_d = d("swe", "2026", m) - d("swe", "2024", m)
        base_ctrl_d = d("control", "2026", m) - d("control", "2024", m)
        base_did = base_swe_d - base_ctrl_d
        pct_diff = abs(did - base_did) / abs(base_did) if abs(base_did) > 1e-9 else float("nan")
        print(f"  {m}: full_did={base_did:.5f}, no_agg_did={did:.5f}, pct_change_from_full={pct_diff*100:.1f}%")


if __name__ == "__main__":
    main()
