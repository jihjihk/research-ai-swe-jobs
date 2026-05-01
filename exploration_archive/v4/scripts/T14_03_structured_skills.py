"""T14 Steps 7-9: Structured skills baseline (asaniczka only) + structured vs
description-extracted validation + seniority-level skill differences.

Outputs:
  tables/T14/structured_skills_top100.csv
  tables/T14/structured_skills_frequency.csv (full)
  tables/T14/structured_vs_extracted.csv
  tables/T14/seniority_skills_chi2.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

ROOT = Path("/home/jihgaboot/gabor/job-research")
SHARED = ROOT / "exploration/artifacts/shared"
TABLES = ROOT / "exploration/tables/T14"
TABLES.mkdir(parents=True, exist_ok=True)


# Canonical mapping: (regex pattern, tech_col). Order matters — longer/more
# specific patterns come first.
import re as _re
STRUCT_RULES = [
    # AI/ML
    (r"\bhugging\s*face\b", "huggingface"),
    (r"\bhuggingface\b", "huggingface"),
    (r"\blangchain\b", "langchain"),
    (r"\bopenai\b", "openai_api"),
    (r"\bprompt\s*engineering\b", "prompt_engineering"),
    (r"\bmachine\s*learning\b", "machine_learning"),
    (r"\bdeep\s*learning\b", "deep_learning"),
    (r"\bnatural\s*language\s*processing\b", "nlp"),
    (r"\bnlp\b", "nlp"),
    (r"\bcomputer\s*vision\b", "computer_vision"),
    (r"\btensorflow\b", "tensorflow"),
    (r"\bpytorch\b", "pytorch"),
    (r"\bscikit[-\s]*learn\b", "scikit_learn"),
    (r"\bpandas\b", "pandas"),
    (r"\bnumpy\b", "numpy"),
    # Languages — TypeScript BEFORE JavaScript (and JavaScript BEFORE Java)
    (r"\btypescript\b", "typescript"),
    (r"\bjavascript\b", "javascript"),
    (r"\bjava\b", "java"),
    (r"\bpython\b", "python"),
    (r"\bgolang\b", "go_lang"),
    (r"\bgo\b", "go_lang"),
    (r"\brust\b", "rust"),
    (r"\bruby\b", "ruby"),
    (r"\bphp\b", "php"),
    (r"\bswift\b", "swift"),
    (r"\bkotlin\b", "kotlin"),
    (r"\bscala\b", "scala"),
    (r"c\+\+", "c_cpp"),
    (r"c#", "csharp"),
    (r"\.net(\s*core)?", "dotnet"),
    (r"\bdotnet\b", "dotnet"),
    (r"\bperl\b", "perl"),
    # Frontend
    (r"\breact\s*native\b", "react_native"),
    (r"\breact\b", "react"),
    (r"\bangular\b", "angular"),
    (r"\bvue(\.js)?\b", "vue"),
    (r"\bnext\.?js\b", "nextjs"),
    (r"\bsvelte\b", "svelte"),
    (r"\bhtml", "html_css"),
    (r"\bcss", "html_css"),
    (r"\bbootstrap\b", "bootstrap"),
    (r"\btailwind\b", "tailwind"),
    # Backend
    (r"\bnode(\.js)?\b", "nodejs"),
    (r"\bdjango\b", "django"),
    (r"\bflask\b", "flask"),
    (r"\bfastapi\b", "fastapi"),
    (r"\bspring\b", "spring"),
    (r"\brails\b", "rails"),
    (r"\bexpress\b", "express"),
    (r"\bgraphql\b", "graphql"),
    (r"\brest\s*api\b", "rest_api"),
    # Cloud/DevOps
    (r"\bamazon\s*web\s*services\b", "aws"),
    (r"\baws\b", "aws"),
    (r"\bazure\b", "azure"),
    (r"\bgoogle\s*cloud\b", "gcp"),
    (r"\bgcp\b", "gcp"),
    (r"\bkubernetes\b", "kubernetes"),
    (r"\bdocker\b", "docker"),
    (r"\bterraform\b", "terraform"),
    (r"\bansible\b", "ansible"),
    (r"\bhelm\b", "helm"),
    (r"\bjenkins\b", "jenkins"),
    (r"\bgithub\s*actions\b", "github_actions"),
    (r"\bgitlab\s*ci\b", "gitlab_ci"),
    (r"\bci/cd\b", "cicd"),
    (r"\bdevops\b", "devops_practice"),
    (r"\bmicroservices\b", "microservices"),
    (r"\bserverless\b", "serverless"),
    # Data
    (r"\bpostgres", "postgresql"),
    (r"\bmysql\b", "mysql"),
    (r"\bmongodb\b", "mongodb"),
    (r"\bredis\b", "redis"),
    (r"\bkafka\b", "kafka"),
    (r"\bspark\b", "spark"),
    (r"\bsnowflake\b", "snowflake"),
    (r"\bdatabricks\b", "databricks"),
    (r"\bdbt\b", "dbt"),
    (r"\belasticsearch\b", "elasticsearch"),
    (r"\bairflow\b", "airflow"),
    (r"\bbigquery\b", "bigquery"),
    (r"\boracle\b", "oracle_db"),
    (r"\bsql\s*server\b", "sqlserver"),
    (r"\bsql\b", "sql"),
    (r"\betl\b", "etl"),
    # Testing
    (r"\bunit\s*test", "unit_testing"),
    (r"\bintegration\s*test", "integration_testing"),
    (r"\bjunit\b", "junit"),
    (r"\bjest\b", "jest"),
    (r"\bpytest\b", "pytest"),
    (r"\bselenium\b", "selenium"),
    (r"\bcypress\b", "cypress"),
    (r"\bplaywright\b", "playwright"),
    (r"\btdd\b", "tdd"),
    # Methodologies
    (r"\bagile\b", "agile"),
    (r"\bscrum\b", "scrum"),
    (r"\bkanban\b", "kanban"),
    # Security
    (r"\bcyber\s*security\b", "security"),
    (r"\bsecurity\b", "security"),
    (r"\bencryption\b", "encryption"),
    (r"\boauth\b", "oauth"),
    # Mobile
    (r"\bandroid\b", "android"),
    (r"\bios\b", "ios"),
]
_COMPILED = [(_re.compile(p, _re.IGNORECASE), t) for p, t in STRUCT_RULES]


def structured_to_tech(s: str) -> str | None:
    for rx, t in _COMPILED:
        if rx.search(s):
            return t
    return None


def main():
    # Load structured skills
    ss = pq.read_table(SHARED / "asaniczka_structured_skills.parquet").to_pandas()
    print(f"  structured mentions: {len(ss)}  unique skills: {ss['skill'].nunique()}")

    # ---------- Step 7: frequency table + top 100 ----------
    freq = ss.groupby("skill").size().rename("n").reset_index()
    freq = freq.sort_values("n", ascending=False)
    total_rows = ss["uid"].nunique()
    freq["pct_rows"] = freq["n"] / total_rows
    freq.to_csv(TABLES / "structured_skills_frequency.csv", index=False)
    freq.head(100).to_csv(TABLES / "structured_skills_top100.csv", index=False)
    print(f"  wrote structured_skills_top100.csv; total unique asaniczka SWE rows w/ skills = {total_rows}")

    # ---------- Step 8: structured vs description-extracted validation ----------
    # Load asaniczka SWE uids and tech matrix
    con = duckdb.connect()
    meta = con.execute(f"""
        SELECT c.uid FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
        WHERE c.source='kaggle_asaniczka' AND NOT coalesce(c.is_aggregator, false)
    """).df()
    asan_uids = set(meta["uid"])
    tech = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
    tech = tech[tech["uid"].isin(asan_uids)].reset_index(drop=True)
    print(f"  asaniczka SWE rows (non-agg) in tech matrix: {len(tech)}")

    tech_cols = [c for c in tech.columns if c != "uid"]
    extracted_rate = {t: tech[t].mean() for t in tech_cols}

    # For structured: map each skill string to a tech col when possible; any-hit per uid
    ss_asan = ss[ss["uid"].isin(asan_uids)].copy()
    ss_asan["tech_col"] = ss_asan["skill"].astype(str).map(structured_to_tech)
    mapped = ss_asan.dropna(subset=["tech_col"])
    # any-mention matrix from structured side
    structured_any = mapped.groupby(["uid", "tech_col"]).size().unstack(fill_value=0).clip(upper=1)
    # Fill missing uids
    missing = list(asan_uids - set(structured_any.index))
    if missing:
        structured_any = structured_any.reindex(list(structured_any.index) + missing, fill_value=0)
    structured_any = structured_any.reindex(sorted(asan_uids), fill_value=0)
    # Rate per tech
    structured_rate = {t: structured_any[t].mean() if t in structured_any.columns else 0.0
                       for t in tech_cols}

    # Build comparison table with both rates where either > 0.01
    comp_rows = []
    for t in tech_cols:
        er = extracted_rate[t]
        sr = structured_rate[t]
        if er < 0.005 and sr < 0.005:
            continue
        comp_rows.append({
            "tech": t,
            "rate_extracted": er,
            "rate_structured": sr,
            "diff": sr - er,
            "abs_diff": abs(sr - er),
        })
    comp = pd.DataFrame(comp_rows).sort_values("abs_diff", ascending=False)
    # Rank correlation
    rho = comp["rate_extracted"].corr(comp["rate_structured"], method="spearman")
    rho_pearson = comp["rate_extracted"].corr(comp["rate_structured"], method="pearson")
    comp.to_csv(TABLES / "structured_vs_extracted.csv", index=False)
    print(f"  wrote structured_vs_extracted.csv  spearman rho={rho:.3f}  pearson r={rho_pearson:.3f}")

    # ---------- Step 9: seniority skill differences via chi2 ----------
    # Use combined best-available seniority (collapse to junior/mid/senior/unknown) on asaniczka
    sen = con.execute(f"""
        SELECT c.uid,
               CASE
                 WHEN u.llm_classification_coverage='labeled' THEN u.seniority_llm
                 WHEN u.llm_classification_coverage='rule_sufficient' THEN u.seniority_final
                 ELSE NULL
               END AS sb
        FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
        LEFT JOIN read_parquet('{ROOT}/data/unified.parquet') u ON c.uid=u.uid
        WHERE c.source='kaggle_asaniczka' AND NOT coalesce(c.is_aggregator,false)
    """).df()
    map3 = {"entry": "junior", "associate": "mid", "mid-senior": "senior",
            "director": "senior"}
    sen["s3"] = sen["sb"].map(map3)
    sen = sen.dropna(subset=["s3"])
    print(f"  asaniczka rows with best-available seniority: {len(sen)}")
    print(sen["s3"].value_counts().to_dict())

    # Join with structured skills
    ss_join = ss[ss["uid"].isin(sen["uid"])].copy()
    ss_join = ss_join.merge(sen[["uid", "s3"]], on="uid", how="left")
    # Aggregate skills appearing >=30 times overall
    skill_counts = ss_join["skill"].value_counts()
    common = skill_counts[skill_counts >= 30].index.tolist()
    print(f"  common skills (n>=30): {len(common)}")

    n_entry = int((sen["s3"] == "junior").sum())
    n_mid = int((sen["s3"] == "mid").sum())
    n_sen = int((sen["s3"] == "senior").sum())
    n_total = n_entry + n_mid + n_sen

    chi_rows = []
    # Entry vs mid-senior (pooled) comparison, per skill
    ms_uids = set(sen[sen["s3"].isin(["mid", "senior"])]["uid"])
    entry_uids = set(sen[sen["s3"] == "junior"]["uid"])
    n_entry = len(entry_uids)
    n_ms = len(ms_uids)
    if n_entry < 20 or n_ms < 20:
        print(f"  WARNING: n_entry={n_entry} n_ms={n_ms}; chi2 may be unreliable")

    # For each common skill, compute 2x2 table
    skill_uids = ss_join.groupby("skill")["uid"].apply(set)
    for skill in common:
        uids = skill_uids[skill]
        a = len(uids & entry_uids)  # entry w/ skill
        b = n_entry - a              # entry w/o skill
        c = len(uids & ms_uids)      # MS w/ skill
        d = n_ms - c
        if a + c < 5 or b + d < 5:
            continue
        table = np.array([[a, b], [c, d]])
        try:
            chi2, p, _, _ = chi2_contingency(table, correction=False)
        except ValueError:
            continue
        p_entry = a / n_entry if n_entry else 0
        p_ms = c / n_ms if n_ms else 0
        chi_rows.append({
            "skill": skill,
            "n_entry_total": n_entry,
            "n_ms_total": n_ms,
            "n_skill_entry": a,
            "n_skill_ms": c,
            "pct_entry": p_entry,
            "pct_ms": p_ms,
            "delta_entry_minus_ms": p_entry - p_ms,
            "chi2": chi2,
            "p": p,
        })
    chi_df = pd.DataFrame(chi_rows)
    if len(chi_df) > 0:
        # FDR
        _, q, _, _ = multipletests(chi_df["p"], method="fdr_bh")
        chi_df["q_fdr"] = q
        chi_df = chi_df.sort_values("delta_entry_minus_ms", ascending=False)
    chi_df.to_csv(TABLES / "seniority_skills_chi2.csv", index=False)
    sig = chi_df[chi_df.get("q_fdr", 1.0) < 0.05] if len(chi_df) else chi_df
    print(f"  wrote seniority_skills_chi2.csv  total={len(chi_df)}  sig(q<0.05)={len(sig)}")

    # ---------- Supplementary: extracted-tech chi2 using pooled 2024 (arshkon+asaniczka) ----------
    # More power for entry vs mid-senior because arshkon has native entry labels.
    meta2 = con.execute(f"""
        SELECT c.uid, c.seniority_3level, c.source, c.period
        FROM read_parquet('{SHARED}/swe_cleaned_text.parquet') c
        WHERE NOT coalesce(c.is_aggregator,false)
          AND c.period IN ('2024-01','2024-04')
          AND c.seniority_3level IN ('junior','mid','senior')
    """).df()
    tech2 = pq.read_table(SHARED / "swe_tech_matrix.parquet").to_pandas()
    merged2 = meta2.merge(tech2, on="uid", how="inner")
    is_entry = merged2["seniority_3level"] == "junior"
    is_ms = merged2["seniority_3level"].isin(["mid", "senior"])
    n_entry2 = int(is_entry.sum())
    n_ms2 = int(is_ms.sum())
    print(f"  pooled 2024: n_entry={n_entry2}  n_ms={n_ms2}")

    chi2_rows = []
    for t in tech_cols:
        a = int(merged2.loc[is_entry, t].sum())
        b = n_entry2 - a
        c = int(merged2.loc[is_ms, t].sum())
        d = n_ms2 - c
        if a + c < 10:
            continue
        table = np.array([[a, b], [c, d]])
        try:
            chi2_, p, _, _ = chi2_contingency(table, correction=False)
        except ValueError:
            continue
        p_e = a / n_entry2 if n_entry2 else 0
        p_m = c / n_ms2 if n_ms2 else 0
        chi2_rows.append({
            "tech": t,
            "n_entry_total": n_entry2, "n_ms_total": n_ms2,
            "n_tech_entry": a, "n_tech_ms": c,
            "pct_entry": p_e, "pct_ms": p_m,
            "delta_entry_minus_ms": p_e - p_m,
            "chi2": chi2_, "p": p,
        })
    chi2_df = pd.DataFrame(chi2_rows)
    if len(chi2_df) > 0:
        _, q, _, _ = multipletests(chi2_df["p"], method="fdr_bh")
        chi2_df["q_fdr"] = q
        chi2_df = chi2_df.sort_values("delta_entry_minus_ms")
    chi2_df.to_csv(TABLES / "seniority_extracted_tech_chi2_pooled2024.csv", index=False)
    sig2 = chi2_df[chi2_df.get("q_fdr", 1.0) < 0.05] if len(chi2_df) else chi2_df
    print(f"  wrote seniority_extracted_tech_chi2_pooled2024.csv  total={len(chi2_df)}  sig(q<0.05)={len(sig2)}")

    print("Done T14 step 03.")


if __name__ == "__main__":
    main()
