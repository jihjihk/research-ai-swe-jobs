"""
T20. Seniority boundary clarity (feature-based)

Measures how sharp the boundaries between seniority levels are and whether they
blurred between 2024 and 2026 using feature-based discrimination. Semantic
convergence was already ruled out in T15; T20 is the structured-feature
complement.

Outputs:
    exploration/tables/T20/*.csv
    exploration/figures/T20/*.png
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_TAB = ROOT / "exploration/tables/T20"
OUT_FIG = ROOT / "exploration/figures/T20"
OUT_TAB.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

T11_FEAT = ROOT / "exploration/tables/T11/T11_features.parquet"
CLEANED = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
ARCHETYPE = ROOT / "exploration/artifacts/shared/swe_archetype_labels.parquet"
UNIFIED = ROOT / "data/unified.parquet"


# ----- Validated AI / management patterns (T11/V1-verified) -----

# Strict mentoring patterns (T11 strict detector, ~100% precision per V1)
MENTOR_PATTERNS = [
    r"\bmentor(?:ing)?\s+(?:engineers?|junior|team|others|developers?)\b",
    r"\bcoach(?:ing)?\s+(?:engineers?|junior|developers?|team)\b",
    r"\bmentor and develop\b",
    r"\bmentorship\b",
]

# Strict people-management patterns
PEOPLE_MGMT_PATTERNS = [
    r"\bpeople manager\b",
    r"\bpeople management\b",
    r"\bdirect reports?\b",
    r"\bperformance reviews?\b",
    r"\bheadcount\b",
    r"\bmanag(?:e|ing)\s+(?:a\s+)?team of\b",
    r"\blead(?:ing)?\s+(?:a\s+)?team of\b",
    r"\binterview\s+engineers\b",
    r"\bhir(?:e|ing)\s+engineers\b",
    r"\bteam building\b",
]

# Validated AI mention patterns (using agentic etc., NOT bare 'agent')
AI_PATTERNS = [
    r"\bagentic\b",
    r"\bai\s+agent\b",
    r"\bmulti[- ]agent\b",
    r"\blarge language model\b",
    r"\bllms?\b",
    r"\bgenerative ai\b",
    r"\bgen[- ]?ai\b",
    r"\bfoundation models?\b",
    r"\bprompt engineer\b",
    r"\brag\b",
    r"\bretrieval[- ]augmented\b",
    r"\bcopilot\b",
    r"\bchatgpt\b",
    r"\bclaude\b",
    r"\blangchain\b",
    r"\blanggraph\b",
    r"\bvector (?:database|db|store)\b",
    r"\bfine[- ]?tun(?:e|ing)\b",
]

# Technology orchestration patterns
TECH_ORCH_PATTERNS = [
    r"\barchitecture review\b",
    r"\bcode review\b",
    r"\bsystem design\b",
    r"\btechnical direction\b",
    r"\bai orchestration\b",
    r"\bagentic\b",
    r"\bmulti[- ]agent\b",
    r"\bworkflow\b",
    r"\bautomation\b",
    r"\bevaluation\b",
    r"\bprompt engineer\b",
    r"\btool selection\b",
    r"\bquality gate\b",
    r"\bguardrails\b",
]

# Strategic scope
STRAT_PATTERNS = [
    r"\bstakeholders?\b",
    r"\bbusiness impact\b",
    r"\brevenue\b",
    r"\bproduct strategy\b",
    r"\broadmap\b",
    r"\bprioritization\b",
    r"\bresource allocation\b",
    r"\bcross[- ]functional\b",
]

MENTOR_RE = re.compile("|".join(MENTOR_PATTERNS), re.IGNORECASE)
PEOPLE_RE = re.compile("|".join(PEOPLE_MGMT_PATTERNS), re.IGNORECASE)
AI_RE = re.compile("|".join(AI_PATTERNS), re.IGNORECASE)
TECH_ORCH_RE = re.compile("|".join(TECH_ORCH_PATTERNS), re.IGNORECASE)
STRAT_RE = re.compile("|".join(STRAT_PATTERNS), re.IGNORECASE)


def _asserts():
    """Sanity asserts for regex patterns."""
    assert MENTOR_RE.search("You will mentor junior engineers and coach team members.")
    assert MENTOR_RE.search("provide mentorship to the team")
    assert not MENTOR_RE.search("we value honesty and integrity")
    assert PEOPLE_RE.search("you will manage a team of 5 engineers")
    assert PEOPLE_RE.search("5 direct reports")
    assert PEOPLE_RE.search("perform quarterly performance reviews")
    assert not PEOPLE_RE.search("software engineer on the backend team")
    assert AI_RE.search("build agentic workflows")
    assert AI_RE.search("design a multi-agent system")
    assert AI_RE.search("integrate LLMs into our platform")
    assert not AI_RE.search("insurance agent duties")  # bare 'agent' excluded
    assert TECH_ORCH_RE.search("lead code review and system design")
    assert STRAT_RE.search("work with stakeholders on the roadmap")
    print("[T20] regex asserts passed")


# ----- Data loading -----

def load_features() -> pd.DataFrame:
    """Load T11 features + join text + recompute strict language-profile densities."""
    con = duckdb.connect()
    t11 = con.execute(f"SELECT * FROM read_parquet('{T11_FEAT}')").fetchdf()

    cleaned = con.execute(
        f"""SELECT uid, description_cleaned, text_source, seniority_final_source
            FROM read_parquet('{CLEANED}')"""
    ).fetchdf()
    t11 = t11.merge(cleaned, on="uid", how="left")

    # Pull raw description and llm_classification_coverage + seniority_llm
    raw = con.execute(
        f"""SELECT uid, description, description_length, seniority_llm,
                   llm_classification_coverage, title, yoe_extracted as yoe_raw
            FROM read_parquet('{UNIFIED}')
            WHERE source_platform='linkedin' AND is_english=true
              AND date_flag='ok' AND is_swe=true"""
    ).fetchdf()
    t11 = t11.merge(raw[["uid", "description", "seniority_llm",
                         "llm_classification_coverage"]], on="uid", how="left")

    # Best-available text for pattern matching: cleaned if >=50 char else raw
    text = t11["description_cleaned"].fillna("")
    fallback = t11["description"].fillna("")
    use_fallback = text.str.len() < 50
    t11["text_for_regex"] = np.where(use_fallback, fallback, text)
    t11["text_len_effective"] = t11["text_for_regex"].str.len().clip(lower=1)

    # Recompute strict language-profile counts
    t11["mentor_strict_count"] = t11["text_for_regex"].apply(
        lambda s: len(MENTOR_RE.findall(s)) if s else 0
    )
    t11["people_mgmt_strict_count"] = t11["text_for_regex"].apply(
        lambda s: len(PEOPLE_RE.findall(s)) if s else 0
    )
    t11["ai_validated_count"] = t11["text_for_regex"].apply(
        lambda s: len(AI_RE.findall(s)) if s else 0
    )
    t11["tech_orch_count"] = t11["text_for_regex"].apply(
        lambda s: len(TECH_ORCH_RE.findall(s)) if s else 0
    )
    t11["strat_count"] = t11["text_for_regex"].apply(
        lambda s: len(STRAT_RE.findall(s)) if s else 0
    )

    per_1k = 1000.0 / t11["text_len_effective"]
    t11["mentor_density"] = t11["mentor_strict_count"] * per_1k
    t11["people_mgmt_density"] = t11["people_mgmt_strict_count"] * per_1k
    t11["ai_density"] = t11["ai_validated_count"] * per_1k
    t11["tech_orch_density"] = t11["tech_orch_count"] * per_1k
    t11["strat_density"] = t11["strat_count"] * per_1k

    t11["ai_mention_validated"] = (t11["ai_validated_count"] > 0).astype(int)

    # Build combined best-available column
    sba = np.where(
        t11["llm_classification_coverage"] == "labeled",
        t11["seniority_llm"],
        np.where(
            t11["llm_classification_coverage"] == "rule_sufficient",
            t11["seniority_final"],
            None,
        ),
    )
    t11["seniority_best_available_combined"] = sba

    # Year bucket — T11 feature `year` is string-typed
    t11["year_bucket"] = t11["year"].astype(str)

    return t11


# ----- Feature engineering for classifier -----

FEATURE_COLS = [
    "yoe_extracted_or_median",
    "tech_count",
    "ai_mention_validated",
    "scope_density",
    "mentor_density",
    "people_mgmt_density",
    "text_len",
    "edu_level",
]


def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    yoe = df["yoe_extracted"].copy()
    median = yoe.dropna().median()
    df["yoe_extracted_or_median"] = yoe.fillna(median)
    df["yoe_imputed_flag"] = yoe.isna().astype(int)
    # Ensure numeric types
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def run_boundary_classifier(df: pd.DataFrame, level_a: str, level_b: str,
                            seniority_col: str, period: str) -> dict:
    sub = df[df[seniority_col].isin([level_a, level_b])].copy()
    sub = sub[sub["year_bucket"] == period]
    n_a = int((sub[seniority_col] == level_a).sum())
    n_b = int((sub[seniority_col] == level_b).sum())
    if min(n_a, n_b) < 20:
        return {
            "period": period,
            "level_a": level_a,
            "level_b": level_b,
            "seniority_col": seniority_col,
            "n_a": n_a,
            "n_b": n_b,
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "top_features": None,
            "reason": "insufficient_n",
        }
    y = (sub[seniority_col] == level_b).astype(int).values
    X = sub[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    coefs = []
    for train_idx, test_idx in cv.split(Xs, y):
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        clf.fit(Xs[train_idx], y[train_idx])
        p = clf.predict_proba(Xs[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], p))
        coefs.append(clf.coef_[0])
    coef_mean = np.mean(coefs, axis=0)
    # top absolute coefficients
    order = np.argsort(-np.abs(coef_mean))
    top_features = [
        (FEATURE_COLS[i], float(coef_mean[i])) for i in order[:5]
    ]
    return {
        "period": period,
        "level_a": level_a,
        "level_b": level_b,
        "seniority_col": seniority_col,
        "n_a": n_a,
        "n_b": n_b,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "top_features": top_features,
        "reason": "ok",
    }


# ----- Centroid distances for feature-space analysis -----

def compute_centroids(df: pd.DataFrame, seniority_col: str) -> pd.DataFrame:
    rows = []
    for period in ["2024", "2026"]:
        for level in ["entry", "associate", "mid-senior", "director"]:
            sub = df[(df["year_bucket"] == period) & (df[seniority_col] == level)]
            if len(sub) == 0:
                continue
            row = {"period": period, "seniority": level, "n": len(sub)}
            for c in FEATURE_COLS:
                row[c] = float(sub[c].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def centroid_distances(centroids: pd.DataFrame) -> pd.DataFrame:
    """All pairwise Euclidean distances in standardized feature space."""
    feats = centroids[FEATURE_COLS].values
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    rows = []
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i >= j:
                continue
            d = float(np.linalg.norm(feats_s[i] - feats_s[j]))
            rows.append({
                "a_period": centroids.iloc[i]["period"],
                "a_seniority": centroids.iloc[i]["seniority"],
                "b_period": centroids.iloc[j]["period"],
                "b_seniority": centroids.iloc[j]["seniority"],
                "distance": d,
                "a_n": int(centroids.iloc[i]["n"]),
                "b_n": int(centroids.iloc[j]["n"]),
            })
    return pd.DataFrame(rows)


# ----- Main -----

def main():
    _asserts()
    print("[T20] loading features...")
    df = load_features()
    df = make_feature_matrix(df)
    print(f"[T20] n rows = {len(df)}")

    # Boundary analysis across operationalizations ---------------------------

    boundaries = [
        ("entry", "associate"),
        ("associate", "mid-senior"),
        ("mid-senior", "director"),
    ]

    results = []

    # OP1: combined best-available (only where populated)
    # OP2: seniority_final (fallback for rows outside LLM frame)
    # OP3: seniority_best_available_aug
    for col in [
        "seniority_best_available_combined",
        "seniority_best_available_aug",
        "seniority_final",
    ]:
        for period in ["2024", "2026"]:
            for a, b in boundaries:
                r = run_boundary_classifier(df, a, b, col, period)
                results.append(r)

    res_df = pd.DataFrame(results)

    # Expand top_features into string
    def _fmt(tf):
        if tf is None:
            return ""
        return "; ".join(f"{n}:{c:+.3f}" for n, c in tf)

    res_df["top_features_str"] = res_df["top_features"].apply(_fmt)
    res_out = res_df.drop(columns=["top_features"])
    res_out.to_csv(OUT_TAB / "boundary_aucs.csv", index=False)
    print(f"[T20] wrote boundary_aucs.csv ({len(res_out)} rows)")

    # AUC delta table --------------------------------------------------------
    delta_rows = []
    for col in res_df["seniority_col"].unique():
        for a, b in boundaries:
            r24 = res_df[(res_df["seniority_col"] == col) & (res_df["period"] == "2024")
                         & (res_df["level_a"] == a) & (res_df["level_b"] == b)]
            r26 = res_df[(res_df["seniority_col"] == col) & (res_df["period"] == "2026")
                         & (res_df["level_a"] == a) & (res_df["level_b"] == b)]
            if len(r24) == 0 or len(r26) == 0:
                continue
            row = {
                "seniority_col": col,
                "boundary": f"{a}↔{b}",
                "auc_2024": float(r24["auc_mean"].iloc[0]) if not np.isnan(r24["auc_mean"].iloc[0]) else None,
                "auc_2026": float(r26["auc_mean"].iloc[0]) if not np.isnan(r26["auc_mean"].iloc[0]) else None,
                "n_2024_a": int(r24["n_a"].iloc[0]),
                "n_2024_b": int(r24["n_b"].iloc[0]),
                "n_2026_a": int(r26["n_a"].iloc[0]),
                "n_2026_b": int(r26["n_b"].iloc[0]),
                "top_features_2024": r24["top_features_str"].iloc[0],
                "top_features_2026": r26["top_features_str"].iloc[0],
            }
            if row["auc_2024"] is not None and row["auc_2026"] is not None:
                row["delta_auc"] = row["auc_2026"] - row["auc_2024"]
            else:
                row["delta_auc"] = None
            delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT_TAB / "boundary_auc_deltas.csv", index=False)
    print(f"[T20] wrote boundary_auc_deltas.csv ({len(delta_df)} rows)")

    # YOE-based proxy boundary (entry defined as yoe<=2 vs >2) ----------------
    df_yoe = df.dropna(subset=["yoe_extracted"]).copy()
    df_yoe["yoe_bucket"] = np.where(df_yoe["yoe_extracted"] <= 2, "yoe_low",
                                    np.where(df_yoe["yoe_extracted"] <= 5,
                                             "yoe_mid", "yoe_high"))
    yoe_results = []
    for period in ["2024", "2026"]:
        for a, b in [("yoe_low", "yoe_mid"), ("yoe_mid", "yoe_high")]:
            # Remove yoe-derived feature (avoid circularity) by dropping the feature
            sub = df_yoe[df_yoe["yoe_bucket"].isin([a, b]) &
                         (df_yoe["year_bucket"] == period)].copy()
            n_a = int((sub["yoe_bucket"] == a).sum())
            n_b = int((sub["yoe_bucket"] == b).sum())
            if min(n_a, n_b) < 20:
                continue
            y = (sub["yoe_bucket"] == b).astype(int).values
            # drop yoe_extracted_or_median since it is definitionally the outcome
            feats = [f for f in FEATURE_COLS if f != "yoe_extracted_or_median"]
            X = sub[feats].values
            Xs = StandardScaler().fit_transform(X)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = []
            coefs = []
            for tr, te in cv.split(Xs, y):
                clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
                clf.fit(Xs[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Xs[te])[:, 1]))
                coefs.append(clf.coef_[0])
            coef = np.mean(coefs, axis=0)
            order = np.argsort(-np.abs(coef))
            top = [(feats[i], float(coef[i])) for i in order[:5]]
            yoe_results.append({
                "period": period, "boundary": f"{a}↔{b}",
                "n_a": n_a, "n_b": n_b,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "top_features": "; ".join(f"{n}:{c:+.3f}" for n, c in top),
            })
    yoe_df = pd.DataFrame(yoe_results)
    yoe_df.to_csv(OUT_TAB / "boundary_aucs_yoe_proxy.csv", index=False)
    print(f"[T20] wrote boundary_aucs_yoe_proxy.csv ({len(yoe_df)} rows)")

    # Centroid profile heatmap -----------------------------------------------
    centroids = compute_centroids(df, "seniority_final")
    centroids.to_csv(OUT_TAB / "centroids_seniority_final.csv", index=False)

    # Heatmap: rows = period×seniority, cols = features (standardized)
    labels = [f"{r['period']} {r['seniority']}" for _, r in centroids.iterrows()]
    mat = centroids[FEATURE_COLS].values
    # Standardize each column so cross-feature comparability
    mat_std = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(mat_std, xticklabels=FEATURE_COLS, yticklabels=labels,
                cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "z-score across cells"})
    ax.set_title("T20. Feature profiles by period × seniority (seniority_final)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "feature_profile_heatmap.png", dpi=120)
    plt.close()
    print("[T20] wrote feature_profile_heatmap.png")

    # Centroid distances (feature-space boundary sharpness) ------------------
    dist_df = centroid_distances(centroids)
    dist_df.to_csv(OUT_TAB / "centroid_distances.csv", index=False)

    # Adjacent-level distances: summarize
    adj_rows = []
    for period in ["2024", "2026"]:
        for a, b in boundaries:
            sub = dist_df[(dist_df["a_period"] == period) & (dist_df["b_period"] == period)
                          & (dist_df["a_seniority"] == a) & (dist_df["b_seniority"] == b)]
            if len(sub) == 0:
                # try flipped
                sub = dist_df[(dist_df["a_period"] == period) & (dist_df["b_period"] == period)
                              & (dist_df["a_seniority"] == b) & (dist_df["b_seniority"] == a)]
            if len(sub) == 0:
                continue
            adj_rows.append({
                "period": period, "boundary": f"{a}↔{b}",
                "distance": float(sub["distance"].iloc[0]),
            })
    adj_df = pd.DataFrame(adj_rows)
    adj_df.to_csv(OUT_TAB / "adjacent_centroid_distances.csv", index=False)

    # Missing middle: is associate drifting toward entry or mid-senior in 2026?
    mm_rows = []
    for period in ["2024", "2026"]:
        entry = centroids[(centroids.period == period) & (centroids.seniority == "entry")]
        assoc = centroids[(centroids.period == period) & (centroids.seniority == "associate")]
        mid = centroids[(centroids.period == period) & (centroids.seniority == "mid-senior")]
        if len(entry) == 0 or len(assoc) == 0 or len(mid) == 0:
            continue
        scaler = StandardScaler()
        stack = np.vstack([entry[FEATURE_COLS].values,
                           assoc[FEATURE_COLS].values,
                           mid[FEATURE_COLS].values])
        stack_s = scaler.fit_transform(centroids[FEATURE_COLS].values)
        # use same scaling as earlier heatmap
        # compute from within-period stack instead
        scaler = StandardScaler()
        stack_s = scaler.fit_transform(stack)
        d_assoc_entry = float(np.linalg.norm(stack_s[0] - stack_s[1]))
        d_assoc_mid = float(np.linalg.norm(stack_s[2] - stack_s[1]))
        mm_rows.append({
            "period": period,
            "n_entry": int(entry["n"].iloc[0]),
            "n_associate": int(assoc["n"].iloc[0]),
            "n_mid_senior": int(mid["n"].iloc[0]),
            "dist_assoc_to_entry": d_assoc_entry,
            "dist_assoc_to_mid": d_assoc_mid,
            "ratio": d_assoc_entry / (d_assoc_mid + 1e-9),
        })
    mm_df = pd.DataFrame(mm_rows)
    mm_df.to_csv(OUT_TAB / "missing_middle_associate.csv", index=False)
    print(f"[T20] wrote missing_middle_associate.csv ({len(mm_df)} rows)")

    # Domain-stratified boundary analysis ------------------------------------
    try:
        arch = duckdb.connect().execute(
            f"SELECT uid, archetype, archetype_name FROM read_parquet('{ARCHETYPE}')"
        ).fetchdf()
    except Exception as e:
        print(f"[T20] archetype load failed: {e}")
        arch = None

    if arch is not None:
        df_arch = df.merge(arch, on="uid", how="inner")
        print(f"[T20] domain-stratified n = {len(df_arch)}")

        # Group into broad domain buckets
        def domain_bucket(row):
            name = row["archetype_name"] or ""
            if "ai" in name.lower() or "learning" in name.lower() or "machine" in name.lower():
                return "AI/ML"
            if "frontend" in name.lower() or "react" in name.lower() or "web" in name.lower():
                return "Frontend/Web"
            if "embedded" in name.lower() or "firmware" in name.lower() or "radar" in name.lower():
                return "Embedded/Systems"
            if "data" in name.lower() or "analytics" in name.lower():
                return "Data"
            if "cloud" in name.lower() or "devops" in name.lower() or "sre" in name.lower() or "reliability" in name.lower():
                return "Cloud/Infra"
            if "net" in name.lower() or "java" in name.lower() or "spring" in name.lower() or "python / django" in name.lower():
                return "Backend/GeneralStack"
            if name.startswith("noise") or name.startswith("T00_software"):
                return "Generic/Noise"
            return "Other"

        df_arch["domain_bucket"] = df_arch.apply(domain_bucket, axis=1)
        domain_results = []
        for domain in df_arch["domain_bucket"].unique():
            if domain in ("Generic/Noise",):
                continue
            sub = df_arch[df_arch["domain_bucket"] == domain]
            for period in ["2024", "2026"]:
                # Use seniority_final for adequate sample sizes in archetype slice
                for a, b in boundaries:
                    r = run_boundary_classifier(sub, a, b, "seniority_final", period)
                    r["domain"] = domain
                    r["top_features_str"] = _fmt(r["top_features"])
                    r.pop("top_features")
                    domain_results.append(r)
        dom_df = pd.DataFrame(domain_results)
        dom_df.to_csv(OUT_TAB / "domain_stratified_aucs.csv", index=False)
        print(f"[T20] wrote domain_stratified_aucs.csv ({len(dom_df)} rows)")

    # Bar chart: AUC by boundary across operationalizations for 2024 vs 2026 -
    auc_plot = delta_df[delta_df["seniority_col"] == "seniority_final"].copy()
    auc_plot = auc_plot.dropna(subset=["auc_2024", "auc_2026"])
    if len(auc_plot):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(auc_plot))
        w = 0.35
        ax.bar(x - w/2, auc_plot["auc_2024"], w, label="2024", color="#4C72B0")
        ax.bar(x + w/2, auc_plot["auc_2026"], w, label="2026", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(auc_plot["boundary"])
        ax.set_ylabel("Cross-validated AUC")
        ax.set_ylim(0.5, 1.0)
        ax.set_title("T20. Feature-based boundary AUC by period (seniority_final)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_FIG / "boundary_auc_seniority_final.png", dpi=120)
        plt.close()
        print("[T20] wrote boundary_auc_seniority_final.png")

    # Sensitivity: aggregator exclusion + combined column as primary ---------
    df_noagg = df[df["is_aggregator"] == False]
    sens_rows = []
    for period in ["2024", "2026"]:
        for a, b in boundaries:
            r = run_boundary_classifier(df_noagg, a, b, "seniority_final", period)
            r["variant"] = "no_aggregators"
            r["top_features_str"] = _fmt(r["top_features"])
            r.pop("top_features")
            sens_rows.append(r)
    pd.DataFrame(sens_rows).to_csv(OUT_TAB / "sensitivity_no_aggregators.csv", index=False)
    print("[T20] wrote sensitivity_no_aggregators.csv")

    # Summary JSON -----------------------------------------------------------
    summary = {
        "n_rows": int(len(df)),
        "n_2024": int((df["year_bucket"] == "2024").sum()),
        "n_2026": int((df["year_bucket"] == "2026").sum()),
        "feature_cols": FEATURE_COLS,
        "tables": sorted(p.name for p in OUT_TAB.glob("*.csv")),
        "figures": sorted(p.name for p in OUT_FIG.glob("*.png")),
    }
    with open(OUT_TAB / "T20_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[T20] done")


if __name__ == "__main__":
    main()
