"""T18 — Cross-occupation boundary analysis.

Compares SWE vs SWE-adjacent vs control along:
  1. Parallel-trends metrics (seniority, AI-strict, length, scope, tech, requirements-section).
  2. SWE-specificity DiD: (SWE 2024→2026) - (control 2024→2026).
  3. Boundary shift SWE ↔ SWE-adjacent via TF-IDF cosine per period.
  4. Specific adjacent-role comparisons.
  5. AI-adoption gradient.
  6. Classifier sensitivity (simple regex) for requirements-section share.

Outputs (csv):
  - exploration/tables/T18/parallel_trends.csv  (load-bearing for Wave 3.5 T32)
  - exploration/tables/T18/did_table.csv
  - exploration/tables/T18/boundary_similarity.csv
  - exploration/tables/T18/adjacent_role_dynamics.csv
  - exploration/tables/T18/ai_gradient.csv
  - exploration/tables/T18/requirements_classifier_sensitivity.csv
  - exploration/tables/T18/migrating_terms.csv
  - exploration/tables/T18/swe_tier_sensitivity.csv
  - exploration/tables/T18/aggregator_sensitivity.csv

Design:
  Default filter: source_platform='linkedin' AND is_english AND date_flag='ok'.
  3 occupation groups ("occ"): SWE, SWE_ADJACENT, CONTROL.
  2 periods: 2024 (pooled arshkon + asaniczka) vs 2026 (scraped).
  Metrics follow Gate 2 pre-commits (V1-validated ai_strict, section classifier,
  length residualization not required here — DiD uses raw level differences).
"""

from __future__ import annotations

import re
import sys
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path("/home/jihgaboot/gabor/job-research").resolve()
DATA = REPO / "data" / "unified.parquet"
OUT_TAB = REPO / "exploration" / "tables" / "T18"
OUT_FIG = REPO / "exploration" / "figures" / "T18"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "exploration" / "scripts"))
from T13_section_classifier import classify_sections  # type: ignore  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)

# --- V1-validated AI-strict pattern (ai_strict), dropping fine-tuning from 2024 per Gate 2 pre-commit #9.
AI_STRICT_CORE = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|"
    r"pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
# Full ai_strict including fine-tuning. Use for 2026 where FT precision 0.95.
AI_STRICT_FULL = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tun(?:e|ed|ing)|rag|"
    r"vector databas(?:e|es)|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)

# V1-validated scope pattern (scope_v1_rebuilt, drops 'autonomous').
SCOPE_RE = re.compile(
    r"\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|stakeholder(?:s)?)\b",
    re.IGNORECASE,
)

# Simple regex sensitivity: alternative classifier for requirements section.
SIMPLE_REQS_RE = re.compile(
    r"(?:^|\n)\s*(?:\*+\s*)?(?:\*\*)?\s*"
    r"(?:requirements|qualifications|required skills|required qualifications|minimum qualifications|basic qualifications|what you('|'|’)ll need|must[\s-]have|skills?\s+(?:and|&)\s+qualifications?|what we('|'|’)re looking for)"
    r"\s*:?\s*(?:\*\*)?\s*(?:\n|$)",
    re.IGNORECASE,
)


def pick_text(row: pd.Series) -> str:
    """Prefer description_core_llm when labeled, else raw description."""
    if isinstance(row.get("description_core_llm"), str) and row.get("llm_extraction_coverage") == "labeled":
        return row["description_core_llm"]
    return row.get("description") or ""


def simple_reqs_share(text: str) -> float:
    """Share of characters from the first 'Requirements'-style header onward."""
    if not text:
        return 0.0
    m = SIMPLE_REQS_RE.search(text)
    if not m:
        return 0.0
    remaining = len(text) - m.start()
    return remaining / max(len(text), 1)


def compute_metrics(df: pd.DataFrame, use_full_ai: bool) -> dict:
    """Aggregate metrics on a dataframe slice.

    Returns dict of {metric: (value, n, extra)}.
    """
    n = len(df)
    if n == 0:
        return {}

    # AI-strict prevalence (use full incl. fine-tuning on 2026, drop FT on 2024)
    ai_re = AI_STRICT_FULL if use_full_ai else AI_STRICT_CORE
    # Use raw description for binary presence (boilerplate-insensitive).
    ai_hits = df["description"].fillna("").str.contains(ai_re).astype(int)
    ai_prev = float(ai_hits.mean())

    # Description length (characters) using description (raw).
    desc_len = df["description"].fillna("").str.len()
    length_mean = float(desc_len.mean())
    length_median = float(desc_len.median())

    # Scope-language prevalence (binary)
    scope_hits = df["description"].fillna("").str.contains(SCOPE_RE).astype(int)
    scope_prev = float(scope_hits.mean())

    # Tech mention count via local scan of cleaned text (LLM-text fallback: raw)
    # Simple tech regex: list of known stack tokens. For parallel comparisons
    # across groups, use a compact tech list.
    techs = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go ", "rust",
        "kubernetes", "docker", "aws", "azure", "gcp", "sql", "react", "angular",
        "spring", "django", "fastapi", "flask", "pytorch", "tensorflow",
        "jenkins", "terraform", "git", "rest api", "graphql", "kafka",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "spark",
        "ci/cd", "linux", "scala", "ruby", "swift", "kotlin", "php",
        ".net", "node.js", "express", "airflow", "snowflake", "dbt",
        "tableau", "power bi", "hadoop", "databricks",
    ]
    tech_re = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in techs) + r")(?![a-z0-9])", re.IGNORECASE)

    def count_techs(t):
        if not isinstance(t, str):
            return 0
        return len(set(m.group(0).lower() for m in tech_re.finditer(t)))

    tech_counts = df["description"].fillna("").apply(count_techs)
    tech_mean = float(tech_counts.mean())
    tech_median = float(tech_counts.median())

    # Seniority shares (seniority_final, primary)
    sen = df["seniority_final"].fillna("unknown")
    sen_dist = {f"sen_{k}_share": float((sen == k).mean()) for k in ("entry", "associate", "mid-senior", "director", "unknown")}

    # Requirements-section share via T13 classifier — use LLM text when available.
    # Compute on a random subsample for speed (cap at 3,000 per cell).
    reqs_df = df.copy()
    if len(reqs_df) > 3000:
        reqs_df = reqs_df.sample(n=3000, random_state=SEED)
    texts = reqs_df.apply(pick_text, axis=1)

    reqs_shares = []
    simple_reqs = []
    for t in texts:
        counts = classify_sections(t)
        total = counts.get("total", 0) or 1
        reqs_shares.append(counts.get("requirements", 0) / total)
        simple_reqs.append(simple_reqs_share(t))

    reqs_share_t13 = float(np.mean(reqs_shares)) if reqs_shares else 0.0
    reqs_share_simple = float(np.mean(simple_reqs)) if simple_reqs else 0.0

    return {
        "n": n,
        "ai_strict_prev": ai_prev,
        "length_mean": length_mean,
        "length_median": length_median,
        "scope_prev": scope_prev,
        "tech_count_mean": tech_mean,
        "tech_count_median": tech_median,
        "reqs_share_t13": reqs_share_t13,
        "reqs_share_simple": reqs_share_simple,
        **sen_dist,
    }


def load_base(extra_where: str = "") -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    q = f"""
    SELECT uid, source, period, scrape_date,
           title, description, description_core_llm, llm_extraction_coverage,
           seniority_final, seniority_3level,
           is_swe, is_swe_adjacent, is_control,
           is_aggregator, swe_classification_tier,
           yoe_min_years_llm
    FROM read_parquet('{DATA}')
    WHERE source_platform='linkedin' AND is_english AND date_flag='ok'
      AND (is_swe OR is_swe_adjacent OR is_control)
      {extra_where}
    """
    return con.execute(q).fetchdf()


def assign_occ(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp a single occupation group label per row. Precedence SWE > adjacent > control."""
    occ = pd.Series("other", index=df.index)
    occ.loc[df["is_control"].fillna(False)] = "CONTROL"
    occ.loc[df["is_swe_adjacent"].fillna(False)] = "SWE_ADJACENT"
    occ.loc[df["is_swe"].fillna(False)] = "SWE"
    df = df.copy()
    df["occ"] = occ
    return df[df["occ"] != "other"]


def assign_period(df: pd.DataFrame) -> pd.DataFrame:
    """Map source → period label: 2024 (arshkon+asaniczka), 2026 (scraped)."""
    p = pd.Series("other", index=df.index)
    p.loc[df["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])] = "2024"
    p.loc[df["source"] == "scraped"] = "2026"
    df = df.copy()
    df["period_label"] = p
    return df[df["period_label"] != "other"]


def step1_parallel_trends(base: pd.DataFrame) -> pd.DataFrame:
    """Compute parallel-trends metrics for each (occ, period)."""
    rows = []
    for occ in ("SWE", "SWE_ADJACENT", "CONTROL"):
        for period in ("2024", "2026"):
            sub = base[(base["occ"] == occ) & (base["period_label"] == period)]
            if len(sub) == 0:
                continue
            m = compute_metrics(sub, use_full_ai=(period == "2026"))
            rows.append({"occupation_group": occ, "period": period, **m})
    wide = pd.DataFrame(rows)

    # Long-format for Wave 3.5 T32 handoff: occupation_group, period, metric, value, n.
    metric_cols = [
        "ai_strict_prev", "length_mean", "length_median", "scope_prev",
        "tech_count_mean", "tech_count_median", "reqs_share_t13", "reqs_share_simple",
        "sen_entry_share", "sen_associate_share", "sen_mid-senior_share",
        "sen_director_share", "sen_unknown_share",
    ]
    long_rows = []
    for _, r in wide.iterrows():
        for m in metric_cols:
            long_rows.append({
                "occupation_group": r["occupation_group"],
                "period": r["period"],
                "metric": m,
                "value": r[m],
                "n": r["n"],
            })
    long = pd.DataFrame(long_rows)
    long.to_csv(OUT_TAB / "parallel_trends.csv", index=False)
    wide.to_csv(OUT_TAB / "parallel_trends_wide.csv", index=False)
    return wide


def step2_did(wide: pd.DataFrame) -> pd.DataFrame:
    """Compute DiD = (SWE Δ) − (CONTROL Δ) and (SWE Δ) − (ADJ Δ).

    For each metric, report both DiD contrasts, 95% CI for proportions,
    and a flag for magnitude > 30% direction-flip tripwire.
    """
    metric_cols = [
        "ai_strict_prev", "length_mean", "length_median", "scope_prev",
        "tech_count_mean", "tech_count_median", "reqs_share_t13", "reqs_share_simple",
        "sen_entry_share", "sen_mid-senior_share",
    ]

    def get(occ, period, metric):
        m = wide[(wide["occupation_group"] == occ) & (wide["period"] == period)]
        if m.empty:
            return np.nan, np.nan
        return float(m.iloc[0][metric]), int(m.iloc[0]["n"])

    rows = []
    for metric in metric_cols:
        swe_24, n_swe_24 = get("SWE", "2024", metric)
        swe_26, n_swe_26 = get("SWE", "2026", metric)
        adj_24, n_adj_24 = get("SWE_ADJACENT", "2024", metric)
        adj_26, n_adj_26 = get("SWE_ADJACENT", "2026", metric)
        ctl_24, n_ctl_24 = get("CONTROL", "2024", metric)
        ctl_26, n_ctl_26 = get("CONTROL", "2026", metric)

        swe_delta = swe_26 - swe_24
        adj_delta = adj_26 - adj_24
        ctl_delta = ctl_26 - ctl_24
        did_swe_ctl = swe_delta - ctl_delta
        did_swe_adj = swe_delta - adj_delta

        # Approximate CI for proportions using pooled variance of delta-of-deltas.
        ci_swe_ctl = None
        if metric.endswith("_prev") or metric.endswith("_share"):
            # Variance of p_hat: p(1-p)/n. Var of delta = var24 + var26. Var of DiD = varSWE + varCTL.
            def prop_var(p, n):
                return p * (1 - p) / n if n > 0 else 0.0
            var_swe = prop_var(swe_24, n_swe_24) + prop_var(swe_26, n_swe_26)
            var_ctl = prop_var(ctl_24, n_ctl_24) + prop_var(ctl_26, n_ctl_26)
            se = (var_swe + var_ctl) ** 0.5
            ci_swe_ctl = (did_swe_ctl - 1.96 * se, did_swe_ctl + 1.96 * se)

        # Materiality: does DiD flip direction of SWE Δ?
        direction_flip = (swe_delta > 0) != (did_swe_ctl > 0) if not np.isnan(did_swe_ctl) else None
        did_as_share_of_swe = (did_swe_ctl / swe_delta * 100) if abs(swe_delta) > 1e-9 else float("nan")

        rows.append({
            "metric": metric,
            "swe_2024": swe_24,
            "swe_2026": swe_26,
            "swe_delta": swe_delta,
            "adj_2024": adj_24,
            "adj_2026": adj_26,
            "adj_delta": adj_delta,
            "control_2024": ctl_24,
            "control_2026": ctl_26,
            "control_delta": ctl_delta,
            "did_swe_minus_control": did_swe_ctl,
            "did_swe_minus_adjacent": did_swe_adj,
            "ci95_swe_minus_control_lo": ci_swe_ctl[0] if ci_swe_ctl else None,
            "ci95_swe_minus_control_hi": ci_swe_ctl[1] if ci_swe_ctl else None,
            "swe_direction_flips_under_control_did": direction_flip,
            "did_as_pct_of_swe_delta": did_as_share_of_swe,
        })
    did = pd.DataFrame(rows)
    did.to_csv(OUT_TAB / "did_table.csv", index=False)
    return did


def step3_boundary_similarity(base: pd.DataFrame) -> pd.DataFrame:
    """Sample 200 SWE + 200 SWE-adjacent per period; TF-IDF cosine similarity of centroids.

    Also extract the top migrating terms (terms whose z-score similarity to SWE
    rose most in the SWE-adjacent corpus 2024→2026).
    """
    samples = {}
    for period in ("2024", "2026"):
        for occ in ("SWE", "SWE_ADJACENT"):
            pool = base[(base["occ"] == occ) & (base["period_label"] == period)
                        & base["description_core_llm"].notna()
                        & (base["llm_extraction_coverage"] == "labeled")]
            n = min(200, len(pool))
            if n == 0:
                samples[(occ, period)] = pool
                continue
            samples[(occ, period)] = pool.sample(n=n, random_state=SEED)

    # Fit TF-IDF on all 4 samples, compute centroid-level cosine between SWE and ADJ per period.
    all_texts, all_labels = [], []
    for (occ, period), df in samples.items():
        for t in df["description_core_llm"].fillna("").tolist():
            all_texts.append(t)
            all_labels.append((occ, period))

    if not all_texts:
        return pd.DataFrame()

    vec = TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1, 2), max_features=10000, stop_words="english")
    X = vec.fit_transform(all_texts)
    vocab = np.array(vec.get_feature_names_out())

    labels_arr = np.array(all_labels, dtype=object)
    centroids, means = {}, {}
    for (occ, period), _ in samples.items():
        mask = (labels_arr[:, 0] == occ) & (labels_arr[:, 1] == period)
        if mask.sum() == 0:
            continue
        mat = X[mask]
        centroid = np.asarray(mat.mean(axis=0)).ravel()
        centroids[(occ, period)] = centroid
        means[(occ, period)] = mat.mean(axis=0)

    # Pairwise cosine similarity
    rows = []
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    for period in ("2024", "2026"):
        if ("SWE", period) in centroids and ("SWE_ADJACENT", period) in centroids:
            rows.append({
                "period": period,
                "pair": "SWE_vs_SWE_ADJACENT",
                "cosine_similarity": cos(centroids[("SWE", period)], centroids[("SWE_ADJACENT", period)]),
                "n_swe": len(samples[("SWE", period)]),
                "n_adj": len(samples[("SWE_ADJACENT", period)]),
            })
    # Within-group cross-period
    for occ in ("SWE", "SWE_ADJACENT"):
        if (occ, "2024") in centroids and (occ, "2026") in centroids:
            rows.append({
                "period": "cross_period",
                "pair": f"{occ}_2024_vs_2026",
                "cosine_similarity": cos(centroids[(occ, "2024")], centroids[(occ, "2026")]),
                "n_swe": len(samples[(occ, "2024")]),
                "n_adj": len(samples[(occ, "2026")]),
            })

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(OUT_TAB / "boundary_similarity.csv", index=False)

    # Migrating terms: terms whose SWE-ADJ 2026 weight − SWE-ADJ 2024 weight is
    # largest AND whose SWE 2024 − SWE-ADJ 2024 gap was positive (SWE-characteristic
    # in 2024, then appeared in SWE-ADJ 2026).
    try:
        swe_24 = centroids.get(("SWE", "2024"))
        swe_26 = centroids.get(("SWE", "2026"))
        adj_24 = centroids.get(("SWE_ADJACENT", "2024"))
        adj_26 = centroids.get(("SWE_ADJACENT", "2026"))
        if all(v is not None for v in (swe_24, swe_26, adj_24, adj_26)):
            swe_char_2024 = swe_24 - adj_24
            adj_delta = adj_26 - adj_24
            migration = adj_delta * np.clip(swe_char_2024, 0, None)
            top_idx = np.argsort(migration)[::-1][:30]
            migrate_rows = []
            for i in top_idx:
                migrate_rows.append({
                    "term": vocab[i],
                    "swe_2024_tfidf": float(swe_24[i]),
                    "adj_2024_tfidf": float(adj_24[i]),
                    "adj_2026_tfidf": float(adj_26[i]),
                    "swe_chars_2024": float(swe_char_2024[i]),
                    "adj_delta": float(adj_delta[i]),
                    "migration_score": float(migration[i]),
                })
            migrate_df = pd.DataFrame(migrate_rows)
            migrate_df.to_csv(OUT_TAB / "migrating_terms.csv", index=False)
    except Exception as e:
        print(f"[migrate] skipped: {e}")

    return sim_df


def step4_adjacent_roles(base: pd.DataFrame) -> pd.DataFrame:
    """Specific adjacent roles: how do top SWE-adjacent title families change?"""
    # Define title families via case-insensitive regex on raw title.
    families = {
        "data_engineer": r"\bdata\s+engineer\b",
        "network_engineer": r"\bnetwork\s+engineer\b",
        "data_scientist": r"\bdata\s+scientist\b",
        "ml_engineer": r"\b(?:ml|machine learning|ai)\s+engineer\b",
        "devops_sre": r"\b(?:dev\s*ops|sre|site reliability)\b",
        "security_engineer": r"\bsecurity\s+engineer\b",
        "solutions_architect": r"\bsolutions?\s+architect\b",
        "systems_administrator": r"\bsystems?\s+administrator\b",
        "qa_quality_engineer": r"\b(?:qa|quality)\s+engineer\b",
        "data_analyst": r"\bdata\s+analyst\b",
    }

    titles = base["title"].fillna("").str.lower()
    rows = []
    for fam, pat in families.items():
        mask = titles.str.contains(pat, regex=True)
        sub = base[mask]
        for period in ("2024", "2026"):
            slice_ = sub[sub["period_label"] == period]
            if len(slice_) < 50:
                continue
            m = compute_metrics(slice_, use_full_ai=(period == "2026"))
            rows.append({"family": fam, "period": period, **m})

    fam_df = pd.DataFrame(rows)

    # Compute deltas per family
    deltas = []
    if not fam_df.empty:
        for fam in fam_df["family"].unique():
            d24 = fam_df[(fam_df["family"] == fam) & (fam_df["period"] == "2024")]
            d26 = fam_df[(fam_df["family"] == fam) & (fam_df["period"] == "2026")]
            if d24.empty or d26.empty:
                continue
            d24 = d24.iloc[0]
            d26 = d26.iloc[0]
            deltas.append({
                "family": fam,
                "n_2024": int(d24["n"]),
                "n_2026": int(d26["n"]),
                "ai_strict_2024": d24["ai_strict_prev"],
                "ai_strict_2026": d26["ai_strict_prev"],
                "ai_strict_delta": d26["ai_strict_prev"] - d24["ai_strict_prev"],
                "length_mean_2024": d24["length_mean"],
                "length_mean_2026": d26["length_mean"],
                "length_delta": d26["length_mean"] - d24["length_mean"],
                "tech_count_2024": d24["tech_count_mean"],
                "tech_count_2026": d26["tech_count_mean"],
                "tech_count_delta": d26["tech_count_mean"] - d24["tech_count_mean"],
            })
    delt_df = pd.DataFrame(deltas)
    delt_df.to_csv(OUT_TAB / "adjacent_role_dynamics.csv", index=False)
    fam_df.to_csv(OUT_TAB / "adjacent_role_raw.csv", index=False)

    # SWE indistinguishability check: compare each family's TF-IDF centroid in 2026
    # with SWE-2026 centroid, and with its own 2024 centroid.
    # Use sample size up to 300 per family per period; compare only families with n≥100.
    eligible = delt_df[delt_df["n_2026"] >= 100]["family"].tolist()
    texts, labels = [], []
    swe_by_period = {}

    for occ_period, df in [
        (("SWE", "2024"), base[(base["occ"] == "SWE") & (base["period_label"] == "2024")
                                & base["description_core_llm"].notna()
                                & (base["llm_extraction_coverage"] == "labeled")]),
        (("SWE", "2026"), base[(base["occ"] == "SWE") & (base["period_label"] == "2026")
                                & base["description_core_llm"].notna()
                                & (base["llm_extraction_coverage"] == "labeled")]),
    ]:
        n = min(500, len(df))
        if n > 0:
            samp = df.sample(n=n, random_state=SEED)
            swe_by_period[occ_period] = samp
            for t in samp["description_core_llm"].fillna(""):
                texts.append(t)
                labels.append(("SWE", occ_period[1]))

    fam_samples = {}
    for fam in eligible:
        pat = families[fam]
        mask = base["title"].fillna("").str.lower().str.contains(pat, regex=True) \
               & base["description_core_llm"].notna() \
               & (base["llm_extraction_coverage"] == "labeled")
        for period in ("2024", "2026"):
            pool = base[mask & (base["period_label"] == period)]
            n = min(300, len(pool))
            if n == 0:
                continue
            samp = pool.sample(n=n, random_state=SEED)
            fam_samples[(fam, period)] = samp
            for t in samp["description_core_llm"].fillna(""):
                texts.append(t)
                labels.append((fam, period))

    if texts:
        vec = TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1, 2),
                              max_features=10000, stop_words="english")
        X = vec.fit_transform(texts)
        labels_arr = np.array(labels, dtype=object)
        centroids = {}
        for key in set(map(tuple, labels)):
            mask = (labels_arr[:, 0] == key[0]) & (labels_arr[:, 1] == key[1])
            if mask.sum() == 0:
                continue
            c = np.asarray(X[mask].mean(axis=0)).ravel()
            centroids[key] = c

        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

        sim_rows = []
        for fam in eligible:
            swe_24 = centroids.get(("SWE", "2024"))
            swe_26 = centroids.get(("SWE", "2026"))
            f24 = centroids.get((fam, "2024"))
            f26 = centroids.get((fam, "2026"))
            if f24 is not None and swe_24 is not None:
                sim_24_to_swe = cos(f24, swe_24)
            else:
                sim_24_to_swe = None
            if f26 is not None and swe_26 is not None:
                sim_26_to_swe = cos(f26, swe_26)
            else:
                sim_26_to_swe = None
            if f24 is not None and f26 is not None:
                self_drift = cos(f24, f26)
            else:
                self_drift = None
            sim_rows.append({
                "family": fam,
                "cosine_to_swe_2024": sim_24_to_swe,
                "cosine_to_swe_2026": sim_26_to_swe,
                "delta_cosine_to_swe": (sim_26_to_swe - sim_24_to_swe) if (sim_24_to_swe is not None and sim_26_to_swe is not None) else None,
                "self_cosine_2024_vs_2026": self_drift,
            })
        pd.DataFrame(sim_rows).to_csv(OUT_TAB / "adjacent_similarity_to_swe.csv", index=False)

    return delt_df


def step5_ai_gradient(wide: pd.DataFrame) -> pd.DataFrame:
    """AI-strict prevalence gradient across occupations per period."""
    sub = wide[["occupation_group", "period", "ai_strict_prev", "n"]].copy()
    sub_pivot = sub.pivot(index="occupation_group", columns="period", values="ai_strict_prev")
    sub_pivot["delta_2024_to_2026"] = sub_pivot["2026"] - sub_pivot["2024"]
    sub_pivot = sub_pivot.reset_index()
    sub_pivot.to_csv(OUT_TAB / "ai_gradient.csv", index=False)
    return sub_pivot


def step6_requirements_classifier_sensitivity(wide: pd.DataFrame) -> pd.DataFrame:
    """For the requirements-section share DiD, compare T13 classifier to the simple regex."""
    def get(occ, period, metric):
        m = wide[(wide["occupation_group"] == occ) & (wide["period"] == period)]
        if m.empty:
            return np.nan
        return float(m.iloc[0][metric])

    rows = []
    for metric_pair in [("reqs_share_t13", "T13 classifier"), ("reqs_share_simple", "simple regex")]:
        metric, label = metric_pair
        swe_delta = get("SWE", "2026", metric) - get("SWE", "2024", metric)
        ctl_delta = get("CONTROL", "2026", metric) - get("CONTROL", "2024", metric)
        adj_delta = get("SWE_ADJACENT", "2026", metric) - get("SWE_ADJACENT", "2024", metric)
        did_swe_ctl = swe_delta - ctl_delta
        rows.append({
            "classifier": label,
            "metric": metric,
            "swe_delta": swe_delta,
            "adj_delta": adj_delta,
            "control_delta": ctl_delta,
            "did_swe_minus_control": did_swe_ctl,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TAB / "requirements_classifier_sensitivity.csv", index=False)

    # Direction flip flag at the DiD level
    t13 = df[df["metric"] == "reqs_share_t13"].iloc[0]
    simple = df[df["metric"] == "reqs_share_simple"].iloc[0]
    flip_swe = (t13["swe_delta"] > 0) != (simple["swe_delta"] > 0)
    flip_did = (t13["did_swe_minus_control"] > 0) != (simple["did_swe_minus_control"] > 0)
    with open(OUT_TAB / "requirements_classifier_flip_flags.json", "w") as fh:
        json.dump({"flip_swe_delta": bool(flip_swe), "flip_did": bool(flip_did)}, fh, indent=2)
    return df


def step7_sensitivities(base: pd.DataFrame) -> None:
    """Essential sensitivities:
       (a) aggregator exclusion — recompute wide with is_aggregator=False.
       (g) SWE tier — exclude title_lookup_llm rows from SWE.
    """
    # (a) Aggregator exclusion
    agg_base = base[~base["is_aggregator"].fillna(False)].copy()
    rows = []
    for occ in ("SWE", "SWE_ADJACENT", "CONTROL"):
        for period in ("2024", "2026"):
            sub = agg_base[(agg_base["occ"] == occ) & (agg_base["period_label"] == period)]
            if len(sub) == 0:
                continue
            m = compute_metrics(sub, use_full_ai=(period == "2026"))
            rows.append({"occupation_group": occ, "period": period, **m})
    agg_df = pd.DataFrame(rows)
    agg_df.to_csv(OUT_TAB / "aggregator_sensitivity.csv", index=False)

    # (g) SWE tier: drop title_lookup_llm from SWE.
    tier_base = base.copy()
    drop_mask = (tier_base["occ"] == "SWE") & (tier_base["swe_classification_tier"] == "title_lookup_llm")
    tier_base = tier_base[~drop_mask]
    rows = []
    for occ in ("SWE", "SWE_ADJACENT", "CONTROL"):
        for period in ("2024", "2026"):
            sub = tier_base[(tier_base["occ"] == occ) & (tier_base["period_label"] == period)]
            if len(sub) == 0:
                continue
            m = compute_metrics(sub, use_full_ai=(period == "2026"))
            rows.append({"occupation_group": occ, "period": period, **m})
    tier_df = pd.DataFrame(rows)
    tier_df.to_csv(OUT_TAB / "swe_tier_sensitivity.csv", index=False)


def step8_figures(wide: pd.DataFrame, did: pd.DataFrame) -> None:
    """Save parallel-trends + AI gradient plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fig 1: AI prevalence trajectory by group
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for occ, color in [("SWE", "C0"), ("SWE_ADJACENT", "C1"), ("CONTROL", "C2")]:
            sub = wide[wide["occupation_group"] == occ].sort_values("period")
            ax.plot(sub["period"].tolist(), sub["ai_strict_prev"].tolist(),
                    marker="o", label=occ, color=color)
        ax.set_ylabel("AI-strict prevalence (share of postings)")
        ax.set_xlabel("Period")
        ax.set_title("T18. AI-strict prevalence by occupation group, 2024 → 2026")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "fig1_ai_prevalence.png", dpi=120)
        plt.close(fig)

        # Fig 2: length median trajectory
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for occ, color in [("SWE", "C0"), ("SWE_ADJACENT", "C1"), ("CONTROL", "C2")]:
            sub = wide[wide["occupation_group"] == occ].sort_values("period")
            ax.plot(sub["period"].tolist(), sub["length_median"].tolist(),
                    marker="o", label=occ, color=color)
        ax.set_ylabel("Description length (chars, median)")
        ax.set_xlabel("Period")
        ax.set_title("T18. Description length by occupation group")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_FIG / "fig2_length.png", dpi=120)
        plt.close(fig)

        # Fig 3: DiD bar chart
        key_metrics = ["ai_strict_prev", "scope_prev", "length_median",
                       "tech_count_mean", "reqs_share_t13", "reqs_share_simple"]
        sub = did[did["metric"].isin(key_metrics)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        y = np.arange(len(sub))
        ax.barh(y, sub["did_swe_minus_control"], color="C0")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["metric"])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("DiD = (SWE Δ) − (CONTROL Δ)")
        ax.set_title("T18. Difference-in-differences: SWE vs CONTROL")
        fig.tight_layout()
        fig.savefig(OUT_FIG / "fig3_did.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[figures] skipped: {e}")


def main():
    print("Loading base data …")
    raw = load_base()
    raw = assign_occ(raw)
    raw = assign_period(raw)
    print(f"  n={len(raw):,} rows across SWE/adjacent/control × 2024/2026")

    print("Step 1: parallel trends …")
    wide = step1_parallel_trends(raw)
    print(wide.to_string())

    print("Step 2: DiD …")
    did = step2_did(wide)

    print("Step 3: boundary TF-IDF similarity …")
    sim = step3_boundary_similarity(raw)
    print(sim.to_string())

    print("Step 4: specific adjacent roles …")
    step4_adjacent_roles(raw)

    print("Step 5: AI gradient …")
    grad = step5_ai_gradient(wide)
    print(grad.to_string())

    print("Step 6: classifier sensitivity …")
    step6_requirements_classifier_sensitivity(wide)

    print("Step 7: sensitivities (aggregator, SWE tier) …")
    step7_sensitivities(raw)

    print("Step 8: figures …")
    step8_figures(wide, did)

    print("Done.")


if __name__ == "__main__":
    main()
