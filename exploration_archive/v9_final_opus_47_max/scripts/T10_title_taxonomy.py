"""T10. Title taxonomy evolution.

Maps how the SWE title landscape evolved between 2024 and 2026:
  1. Title vocabulary comparison (truly new / disappeared / shared).
  2. Title concentration (unique titles per 1K postings).
  3. Compound/hybrid AI titles.
  4. Title-to-content alignment (TF-IDF cosine on descriptions).
  5. Title inflation/deflation signals (seniority markers share).
  6. Emerging role categories.
  7. Persist disappearing titles list for Wave 3.5 T36.

Scope: SWE, LinkedIn, English, date_flag='ok'.
Primary period comparison: arshkon 2024-04 vs scraped (pooled 2026-03/04).
Sensitivities: aggregator-exclusion, company capping (20/50), seniority panel.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "unified.parquet"
OUT_TABLES = ROOT / "exploration" / "tables" / "T10"
OUT_FIGS = ROOT / "exploration" / "figures" / "T10"
OUT_SHARED = ROOT / "exploration" / "artifacts" / "shared"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

FILTER = (
    "is_swe=TRUE AND source_platform='linkedin' "
    "AND is_english=TRUE AND date_flag='ok'"
)


# ----------------------------- title normalization ----------------------------


TITLE_ABBREV = {
    r"\bsr\.\s": "senior ",
    r"\bsr\s": "senior ",
    r"\bsr\.$": "senior",
    r"\bsr$": "senior",
    r"\bjr\.\s": "junior ",
    r"\bjr\s": "junior ",
    r"\bjr\.$": "junior",
    r"\bjr$": "junior",
    r"\bmgr\b": "manager",
    r"\barchitect\.": "architect",
    r"\beng\.\s": "engineer ",
    r"\bsoftware dev\b": "software developer",
    r"\bswe\b": "software engineer",
    r"\bqa\b": "quality assurance",
    r"\bml\b": "machine learning",
    r"\ba\.i\.\b": "ai",
    r"\bai/ml\b": "ai machine learning",
    r"\bml/ai\b": "machine learning ai",
    r"\bml ops\b": "mlops",
    r"\bsre\b": "site reliability engineer",
    r"\bdevops\b": "devops",  # keep as is but ensure lowercase
    r"\bfull-stack\b": "full stack",
    r"\bfullstack\b": "full stack",
    r"\bfront-end\b": "frontend",
    r"\bback-end\b": "backend",
    r"\bfront end\b": "frontend",
    r"\bback end\b": "backend",
    r"\biii\b": "3",
    r"\bii\b": "2",
    r"\biv\b": "4",
    r"\blevel iii\b": "3",
    r"\blevel ii\b": "2",
}

SENIORITY_MARKERS = {
    "junior": r"\b(junior|entry[- ]level|new grad|graduate|intern|trainee|apprentice|associate)\b",
    "mid": r"\b(mid[- ]level|mid|intermediate)\b",
    "senior": r"\b(senior|sr|snr)\b",
    "lead": r"\b(lead|tech lead|team lead)\b",
    "principal": r"\bprincipal\b",
    "staff": r"\bstaff\b",
    "director": r"\b(director|vp|vice president|cto|head of)\b",
    "manager": r"\b(manager|engineering manager|eng manager|em)\b",
}

AI_TERMS = {
    "ai": r"\bai\b|\bartificial intelligence\b",
    "ml": r"\bmachine learning\b|\bml\b|\bdeep learning\b",
    "data": r"\bdata\b",
    "llm": r"\bllm\b|\blarge language model\b",
    "agent": r"\bagent(?:ic)?\b",
    "genai": r"\bgen[- ]?ai\b|\bgenerative ai\b",
}

# Assertion edge-case tests for the seniority-marker and AI-term regexes.
def _assert_regexes() -> None:
    # Seniority markers — positive
    assert re.search(SENIORITY_MARKERS["junior"], "junior software engineer")
    assert re.search(SENIORITY_MARKERS["junior"], "new grad engineer")
    assert re.search(SENIORITY_MARKERS["junior"], "associate engineer")
    assert re.search(SENIORITY_MARKERS["senior"], "senior software engineer")
    assert re.search(SENIORITY_MARKERS["senior"], "sr software engineer")
    assert re.search(SENIORITY_MARKERS["principal"], "principal engineer")
    assert re.search(SENIORITY_MARKERS["staff"], "staff engineer")
    assert re.search(SENIORITY_MARKERS["lead"], "lead software engineer")
    # Seniority markers — negative
    assert not re.search(SENIORITY_MARKERS["junior"], "software engineer")
    assert not re.search(SENIORITY_MARKERS["senior"], "software engineer")
    # NB: "principal" appearing in a title string is treated as a seniority marker;
    # the rare false positive like "principal component analyst" is tolerated (not a SWE title).
    assert re.search(SENIORITY_MARKERS["principal"], "principal software engineer")
    # AI terms — positive
    assert re.search(AI_TERMS["ai"], "ai engineer")
    assert re.search(AI_TERMS["ml"], "machine learning engineer")
    assert re.search(AI_TERMS["ml"], "ml engineer")
    assert re.search(AI_TERMS["data"], "data engineer")
    assert re.search(AI_TERMS["llm"], "llm engineer")
    assert re.search(AI_TERMS["agent"], "agent engineer")
    assert re.search(AI_TERMS["agent"], "agentic systems engineer")
    assert re.search(AI_TERMS["genai"], "gen ai engineer")
    assert re.search(AI_TERMS["genai"], "generative ai engineer")
    # AI terms — negative
    assert not re.search(AI_TERMS["ai"], "software engineer")
    assert not re.search(AI_TERMS["ml"], "systems engineer")
    assert not re.search(AI_TERMS["data"], "software engineer")


_assert_regexes()


def normalize_title(t: str | None) -> str:
    if t is None:
        return ""
    s = t.lower().strip()
    # strip a few punctuation marks but keep /, -, & so we can still see compounds
    s = re.sub(r"[\"\'`]", "", s)
    s = re.sub(r"\s+", " ", s)
    # apply abbreviations
    for pat, rep in TITLE_ABBREV.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"[,]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def seniority_marker(title: str) -> str:
    # single-label marker; priority: principal>staff>director>manager>senior>lead>mid>junior>none
    priority = [
        "principal",
        "staff",
        "director",
        "manager",
        "senior",
        "lead",
        "mid",
        "junior",
    ]
    for p in priority:
        if re.search(SENIORITY_MARKERS[p], title):
            return p
    return "none"


def ai_marker(title: str) -> tuple[bool, list[str]]:
    hits = []
    for k, pat in AI_TERMS.items():
        if re.search(pat, title):
            hits.append(k)
    return (len(hits) > 0, hits)


# ----------------------------- data loading -----------------------------------


def load_corpus() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT uid, source, period, title, description_core_llm, description,
               company_name_canonical, is_aggregator,
               seniority_final, seniority_final_source, seniority_3level,
               yoe_min_years_llm, llm_extraction_coverage
        FROM '{DATA}'
        WHERE {FILTER}
        """
    ).df()
    df["title_norm"] = df["title"].map(normalize_title)
    return df


# ----------------------------- company capping --------------------------------


def cap_by_company(df: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Cap the number of postings per canonical company."""
    rng = np.random.default_rng(seed)
    parts = []
    for co, g in df.groupby("company_name_canonical", dropna=False):
        if len(g) <= cap:
            parts.append(g)
        else:
            idx = rng.choice(len(g), size=cap, replace=False)
            parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True)


# ----------------------------- step 1: vocab ----------------------------------


def period_bucket(row) -> str:
    if row["source"] == "kaggle_arshkon":
        return "arshkon_2024"
    if row["source"] == "kaggle_asaniczka":
        return "asaniczka_2024"
    if row["source"] == "scraped":
        return "scraped_2026"
    return "other"


def step1_vocab(df: pd.DataFrame, label: str) -> dict:
    """Compute title vocabulary: new / disappeared / shared titles, primary comparison arshkon vs scraped (pooled 2026)."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    ars = df[df["bucket"] == "arshkon_2024"]
    asa = df[df["bucket"] == "asaniczka_2024"]
    scr = df[df["bucket"] == "scraped_2026"]
    pooled24 = pd.concat([ars, asa])

    def counts(d: pd.DataFrame) -> Counter:
        return Counter(d["title_norm"].tolist())

    c_ars = counts(ars)
    c_asa = counts(asa)
    c_scr = counts(scr)
    c_pool = counts(pooled24)

    # Primary comparison: arshkon ∩ scraped; disappearing = arshkon only; emerging = scraped only.
    # Use min-count thresholds so we don't chase singletons.
    new_titles = []  # titles in scraped ≥ 10 but arshkon < 3
    disappearing = []  # titles in arshkon ≥ 10 but scraped < 3 (Wave 3.5 T36 input)
    shared = []  # titles with ≥ 10 in BOTH
    for t in set(c_ars) | set(c_scr):
        a = c_ars.get(t, 0)
        s = c_scr.get(t, 0)
        if a >= 10 and s < 3:
            disappearing.append((t, a, s))
        if s >= 10 and a < 3:
            new_titles.append((t, a, s))
        if a >= 10 and s >= 10:
            shared.append((t, a, s))

    new_titles = sorted(new_titles, key=lambda x: -x[2])
    disappearing = sorted(disappearing, key=lambda x: -x[1])
    shared = sorted(shared, key=lambda x: -(x[1] + x[2]))

    # Also for pooled-2024 context (arshkon + asaniczka combined baseline)
    pooled_new = []  # in scraped ≥ 10 but pooled-2024 < 3
    pooled_dis = []  # in pooled-2024 ≥ 10 but scraped < 3
    for t in set(c_pool) | set(c_scr):
        p = c_pool.get(t, 0)
        s = c_scr.get(t, 0)
        if p >= 10 and s < 3:
            pooled_dis.append((t, p, s))
        if s >= 10 and p < 3:
            pooled_new.append((t, p, s))
    pooled_new = sorted(pooled_new, key=lambda x: -x[2])
    pooled_dis = sorted(pooled_dis, key=lambda x: -x[1])

    # save tables
    pd.DataFrame(new_titles, columns=["title_norm", "arshkon_n", "scraped_n"]).to_csv(
        OUT_TABLES / f"step1_new_titles_arshkon_vs_scraped_{label}.csv", index=False
    )
    pd.DataFrame(disappearing, columns=["title_norm", "arshkon_n", "scraped_n"]).to_csv(
        OUT_TABLES / f"step1_disappearing_titles_arshkon_vs_scraped_{label}.csv",
        index=False,
    )
    pd.DataFrame(shared, columns=["title_norm", "arshkon_n", "scraped_n"]).to_csv(
        OUT_TABLES / f"step1_shared_titles_arshkon_vs_scraped_{label}.csv",
        index=False,
    )
    pd.DataFrame(pooled_new, columns=["title_norm", "pooled_2024_n", "scraped_n"]).to_csv(
        OUT_TABLES / f"step1_new_titles_pooled2024_vs_scraped_{label}.csv", index=False
    )
    pd.DataFrame(pooled_dis, columns=["title_norm", "pooled_2024_n", "scraped_n"]).to_csv(
        OUT_TABLES / f"step1_disappearing_titles_pooled2024_vs_scraped_{label}.csv",
        index=False,
    )

    return {
        "n_distinct_arshkon": len(c_ars),
        "n_distinct_asaniczka": len(c_asa),
        "n_distinct_pooled2024": len(c_pool),
        "n_distinct_scraped": len(c_scr),
        "n_new_arshkon_scraped": len(new_titles),
        "n_disappearing_arshkon_scraped": len(disappearing),
        "n_shared_arshkon_scraped": len(shared),
        "n_new_pooled_scraped": len(pooled_new),
        "n_disappearing_pooled_scraped": len(pooled_dis),
        "n_rows_arshkon": len(ars),
        "n_rows_asaniczka": len(asa),
        "n_rows_pooled24": len(pooled24),
        "n_rows_scraped": len(scr),
    }


# ----------------------------- step 2: concentration --------------------------


def step2_concentration(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Unique titles per 1K postings, per source+period; + HHI concentration."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    rows = []
    # Also add scraped month split
    scr_by_month = {
        "scraped_2026-03": df[(df["source"] == "scraped") & (df["period"] == "2026-03")],
        "scraped_2026-04": df[(df["source"] == "scraped") & (df["period"] == "2026-04")],
    }
    groups = {
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "asaniczka_2024": df[df["bucket"] == "asaniczka_2024"],
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
        **scr_by_month,
    }
    for name, d in groups.items():
        if len(d) == 0:
            continue
        n = len(d)
        vc = d["title_norm"].value_counts()
        n_uniq = (vc > 0).sum()
        # HHI on title shares (0-1 scale, times 10000 gives classic HHI)
        shares = vc.values / n
        hhi = float((shares**2).sum())
        # top-10 and top-50 cumulative shares
        top10 = float(vc.head(10).sum() / n)
        top50 = float(vc.head(50).sum() / n)
        rows.append(
            dict(
                group=name,
                n_postings=n,
                n_unique_titles=int(n_uniq),
                unique_per_1k=round(1000 * n_uniq / n, 2),
                hhi=round(hhi, 6),
                top10_share=round(top10, 4),
                top50_share=round(top50, 4),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / f"step2_concentration_{label}.csv", index=False)
    return out


# ----------------------------- step 3: compound AI titles ---------------------


def step3_compound_ai(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    # compute AI marker per row
    ai_flags = df["title_norm"].apply(ai_marker)
    df["has_ai_term"] = [x[0] for x in ai_flags]
    df["ai_terms"] = [",".join(x[1]) for x in ai_flags]
    rows = []
    groups = {
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "asaniczka_2024": df[df["bucket"] == "asaniczka_2024"],
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026-03": df[(df["source"] == "scraped") & (df["period"] == "2026-03")],
        "scraped_2026-04": df[(df["source"] == "scraped") & (df["period"] == "2026-04")],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    for name, d in groups.items():
        if len(d) == 0:
            continue
        rec = dict(group=name, n=len(d))
        rec["share_any_ai_term"] = round(d["has_ai_term"].mean(), 4)
        # per-term share
        for term in AI_TERMS:
            rec[f"share_{term}"] = round(
                d["ai_terms"].str.contains(rf"\b{term}\b").mean(), 4
            )
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / f"step3_ai_compound_titles_{label}.csv", index=False)

    # Also: top-20 AI titles in 2026
    scr = df[df["bucket"] == "scraped_2026"]
    ai_titles_scr = (
        scr[scr["has_ai_term"]]["title_norm"].value_counts().head(30).reset_index()
    )
    ai_titles_scr.columns = ["title_norm", "n"]
    ai_titles_scr.to_csv(OUT_TABLES / f"step3_top_ai_titles_scraped_{label}.csv", index=False)

    # Share of 2026 AI titles that did not exist in arshkon
    c_ars = Counter(df[df["bucket"] == "arshkon_2024"]["title_norm"].tolist())
    ai_titles_new = [
        (t, int(n), int(c_ars.get(t, 0)))
        for t, n in Counter(
            scr[scr["has_ai_term"]]["title_norm"].tolist()
        ).most_common()
        if c_ars.get(t, 0) == 0
    ]
    pd.DataFrame(ai_titles_new, columns=["title_norm", "scraped_n", "arshkon_n"]).to_csv(
        OUT_TABLES / f"step3_ai_titles_absent_from_arshkon_{label}.csv", index=False
    )
    return out


# ----------------------------- step 4: title-to-content alignment -------------


def step4_content_alignment(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Top-10 titles shared between arshkon and scraped: TF-IDF cosine between
    pooled-2024 and scraped descriptions for each title."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    ars = df[df["bucket"] == "arshkon_2024"]
    asa = df[df["bucket"] == "asaniczka_2024"]
    scr = df[df["bucket"] == "scraped_2026"]

    # Pick top-10 by combined count in arshkon+scraped
    c_ars = Counter(ars["title_norm"])
    c_scr = Counter(scr["title_norm"])
    shared = [(t, c_ars[t] + c_scr[t]) for t in set(c_ars) & set(c_scr) if c_ars[t] >= 30 and c_scr[t] >= 30]
    shared = sorted(shared, key=lambda x: -x[1])
    top10 = [t for t, _ in shared[:10]]

    # Text = description_core_llm where labeled, else raw description
    def pick_text(row):
        if row["llm_extraction_coverage"] == "labeled" and pd.notna(
            row["description_core_llm"]
        ):
            return row["description_core_llm"]
        return row["description"] or ""

    df["text"] = df.apply(pick_text, axis=1)

    rows = []
    for title in top10:
        a_texts = df[(df["bucket"] == "arshkon_2024") & (df["title_norm"] == title)][
            "text"
        ].tolist()
        s_texts = df[(df["bucket"] == "scraped_2026") & (df["title_norm"] == title)][
            "text"
        ].tolist()
        as_texts = df[
            (df["bucket"] == "asaniczka_2024") & (df["title_norm"] == title)
        ]["text"].tolist()
        if len(a_texts) < 10 or len(s_texts) < 10:
            continue
        # Fit TF-IDF on ALL posting-level documents for this title across all three groups,
        # then take the group mean vector and compute cosine between group means.
        all_docs = a_texts + as_texts + s_texts
        group_ids = (
            ["arshkon"] * len(a_texts)
            + ["asaniczka"] * len(as_texts)
            + ["scraped"] * len(s_texts)
        )
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=50000,
            sublinear_tf=True,
            stop_words="english",
        )
        X = vec.fit_transform(all_docs)
        gids = np.array(group_ids)
        ars_mean = np.asarray(X[gids == "arshkon"].mean(axis=0)).ravel()
        scr_mean = np.asarray(X[gids == "scraped"].mean(axis=0)).ravel()
        pool_mean = np.asarray(
            X[(gids == "arshkon") | (gids == "asaniczka")].mean(axis=0)
        ).ravel()
        def _cos(u, v):
            denom = (np.linalg.norm(u) * np.linalg.norm(v)) or 1e-12
            return float(np.dot(u, v) / denom)
        ars_scr = _cos(ars_mean, scr_mean)
        pool_scr = _cos(pool_mean, scr_mean)
        rows.append(
            dict(
                title=title,
                n_arshkon=len(a_texts),
                n_asaniczka=len(as_texts),
                n_pooled2024=len(a_texts) + len(as_texts),
                n_scraped=len(s_texts),
                cos_arshkon_vs_scraped=round(ars_scr, 4),
                cos_pooled2024_vs_scraped=round(pool_scr, 4),
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / f"step4_title_content_alignment_{label}.csv", index=False)

    # Null calibration: within-2024 cosine (arshkon vs asaniczka) for same titles
    cal_rows = []
    for title in top10:
        a_texts = df[(df["bucket"] == "arshkon_2024") & (df["title_norm"] == title)][
            "text"
        ].tolist()
        as_texts = df[
            (df["bucket"] == "asaniczka_2024") & (df["title_norm"] == title)
        ]["text"].tolist()
        if len(a_texts) < 10 or len(as_texts) < 10:
            continue
        all_docs = a_texts + as_texts
        group_ids = np.array(["arshkon"] * len(a_texts) + ["asaniczka"] * len(as_texts))
        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=50000,
            sublinear_tf=True,
            stop_words="english",
        )
        X = vec.fit_transform(all_docs)
        ars_mean = np.asarray(X[group_ids == "arshkon"].mean(axis=0)).ravel()
        asa_mean = np.asarray(X[group_ids == "asaniczka"].mean(axis=0)).ravel()
        denom = (np.linalg.norm(ars_mean) * np.linalg.norm(asa_mean)) or 1e-12
        cos = float(np.dot(ars_mean, asa_mean) / denom)
        cal_rows.append(
            dict(
                title=title,
                n_arshkon=len(a_texts),
                n_asaniczka=len(as_texts),
                within_2024_cosine=round(cos, 4),
            )
        )
    pd.DataFrame(cal_rows).to_csv(
        OUT_TABLES / f"step4_within2024_cosine_calibration_{label}.csv", index=False
    )
    return out


# ----------------------------- step 5: inflation/deflation --------------------


def step5_inflation(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    df["seniority_marker"] = df["title_norm"].apply(seniority_marker)
    rows = []
    groups = {
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "asaniczka_2024": df[df["bucket"] == "asaniczka_2024"],
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026-03": df[(df["source"] == "scraped") & (df["period"] == "2026-03")],
        "scraped_2026-04": df[(df["source"] == "scraped") & (df["period"] == "2026-04")],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    markers = ["junior", "mid", "senior", "lead", "principal", "staff", "director", "manager", "none"]
    for name, d in groups.items():
        if len(d) == 0:
            continue
        rec = dict(group=name, n=len(d))
        for m in markers:
            rec[f"share_{m}"] = round((d["seniority_marker"] == m).mean(), 4)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / f"step5_title_inflation_{label}.csv", index=False)
    return out


# ----------------------------- step 6: emerging categories --------------------


EMERGING_PATTERNS = {
    "ai_engineering": r"\b(?:ai engineer|artificial intelligence engineer|generative ai|gen[- ]?ai|llm engineer|prompt engineer|agent engineer|agentic|foundation model)\b",
    "ml_infrastructure": r"\b(?:ml platform|ml ops|mlops|ml infrastructure|ml engineering|machine learning platform|model serving|model ops)\b",
    "data_engineering": r"\b(?:data engineer|data platform|data infrastructure|analytics engineer|etl engineer|data pipeline)\b",
    "platform_engineering": r"\b(?:platform engineer|internal platform|developer platform|developer experience|devex|devx)\b",
    "reliability_engineering": r"\b(?:site reliability engineer|reliability engineer|sre|production engineer)\b",
    "security_engineering": r"\b(?:security engineer|appsec|application security|devsecops|offensive security|red team|blue team)\b",
    "cloud_engineering": r"\b(?:cloud engineer|cloud architect|cloud infrastructure|aws engineer|azure engineer|gcp engineer)\b",
    "frontend_fullstack": r"\b(?:frontend|full stack|web engineer|ui engineer|react engineer|angular engineer|vue engineer)\b",
    "backend": r"\b(?:backend|back[- ]end engineer|api engineer|server engineer)\b",
    "mobile": r"\b(?:ios engineer|android engineer|mobile engineer|react native engineer|flutter engineer)\b",
    "embedded_firmware": r"\b(?:embedded engineer|firmware engineer|embedded software|rtos)\b",
    "qa_testing": r"\b(?:qa engineer|quality assurance|test engineer|sdet|automation engineer|quality engineer)\b",
    "data_science": r"\b(?:data scientist|applied scientist|research scientist|research engineer)\b",
    "blockchain_crypto": r"\b(?:blockchain|crypto engineer|smart contract|solidity|web3)\b",
    "robotics": r"\b(?:robotics engineer|autonomy engineer|perception engineer|slam engineer)\b",
}


def step6_categories(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    for k, pat in EMERGING_PATTERNS.items():
        df[f"cat_{k}"] = df["title_norm"].str.contains(pat, regex=True, na=False)
    rows = []
    groups = {
        "arshkon_2024": df[df["bucket"] == "arshkon_2024"],
        "asaniczka_2024": df[df["bucket"] == "asaniczka_2024"],
        "pooled_2024": df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])],
        "scraped_2026": df[df["bucket"] == "scraped_2026"],
    }
    for name, d in groups.items():
        if len(d) == 0:
            continue
        rec = dict(group=name, n=len(d))
        for k in EMERGING_PATTERNS:
            rec[f"share_{k}"] = round(d[f"cat_{k}"].mean(), 4)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / f"step6_emerging_categories_{label}.csv", index=False)

    # For the "emerging" label subset — category only present in scraped-2026 (e.g. ai_engineering)
    # What are the top-10 actual title strings in each category in 2026 scraped?
    scr = df[df["bucket"] == "scraped_2026"]
    expansions = []
    for k in EMERGING_PATTERNS:
        d = scr[scr[f"cat_{k}"]]
        vc = d["title_norm"].value_counts().head(10).reset_index()
        vc.columns = ["title_norm", "n"]
        vc["category"] = k
        expansions.append(vc)
    exp = pd.concat(expansions, ignore_index=True)
    exp.to_csv(OUT_TABLES / f"step6_emerging_category_examples_{label}.csv", index=False)
    return out


# ----------------------------- step 7: disappearing titles list ---------------


def step7_disappearing_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Wave 3.5 T36 input: titles with ≥10 arshkon postings and <3 scraped postings.
    Includes top employers 2024 column. Also writes extended-threshold variants
    (≥5 arshkon / <3 scraped; and pooled-2024 ≥10 / scraped <3) as sensitivity."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    ars = df[df["bucket"] == "arshkon_2024"]
    asa = df[df["bucket"] == "asaniczka_2024"]
    pool = df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]
    scr = df[df["bucket"] == "scraped_2026"]
    c_ars = Counter(ars["title_norm"])
    c_asa = Counter(asa["title_norm"])
    c_pool = Counter(pool["title_norm"])
    c_scr = Counter(scr["title_norm"])

    def build(c_src, min_src, src_df, fname):
        rows = []
        for t, a_n in c_src.items():
            s_n = c_scr.get(t, 0)
            if a_n >= min_src and s_n < 3:
                top_emp = (
                    src_df[src_df["title_norm"] == t]["company_name_canonical"]
                    .value_counts()
                    .head(5)
                    .index.tolist()
                )
                rows.append(
                    dict(
                        title_normalized=t,
                        arshkon_n=int(c_ars.get(t, 0)),
                        asaniczka_n=int(c_asa.get(t, 0)),
                        source_n=int(a_n),
                        scraped_n=int(s_n),
                        top_employers_2024=json.dumps(top_emp),
                    )
                )
        out = pd.DataFrame(rows).sort_values("source_n", ascending=False).reset_index(drop=True)
        out.to_csv(OUT_TABLES / fname, index=False)
        return out

    primary = build(c_ars, 10, ars, "disappearing_titles.csv")  # Wave 3.5 T36 primary
    build(c_ars, 5, ars, "disappearing_titles_arshkon_ge5.csv")  # wider sensitivity
    build(c_pool, 10, pool, "disappearing_titles_pooled2024_ge10.csv")  # pooled
    build(c_pool, 20, pool, "disappearing_titles_pooled2024_ge20.csv")  # pooled, stricter
    return primary


# ----------------------------- seniority-shift (T30 panel) --------------------


def seniority_shift_shared_titles(df: pd.DataFrame) -> pd.DataFrame:
    """For the 10 most common shared-both-periods titles, report seniority_final
    distribution 2024 vs 2026 (the T30 panel requires per-J/S-definition stratification,
    but here we report `seniority_final` categorical share for the pooled-2024 vs scraped
    comparison). Additionally report YOE-based J3/S4 shares."""
    df = df.copy()
    df["bucket"] = df.apply(period_bucket, axis=1)
    ars = df[df["bucket"] == "arshkon_2024"]
    scr = df[df["bucket"] == "scraped_2026"]
    pool = df[df["bucket"].isin(["arshkon_2024", "asaniczka_2024"])]
    c_ars = Counter(ars["title_norm"])
    c_scr = Counter(scr["title_norm"])
    shared = sorted(
        [t for t in set(c_ars) & set(c_scr) if c_ars[t] >= 30 and c_scr[t] >= 30],
        key=lambda x: -(c_ars[x] + c_scr[x]),
    )[:10]
    rows = []
    for t in shared:
        pd_rows = {}
        for label, d in [
            ("arshkon_2024", ars),
            ("pooled_2024", pool),
            ("scraped_2026", scr),
        ]:
            sub = d[d["title_norm"] == t]
            n = len(sub)
            if n == 0:
                continue
            # seniority_final 3level
            sf3 = sub["seniority_3level"].value_counts(normalize=True)
            # YOE J3 (≤2) and S4 (≥5), denominator = LLM-labeled
            lab = sub[sub["llm_extraction_coverage"] == "labeled"]
            n_lab = len(lab)
            j3 = (lab["yoe_min_years_llm"] <= 2).mean() if n_lab > 0 else None
            s4 = (lab["yoe_min_years_llm"] >= 5).mean() if n_lab > 0 else None
            pd_rows[label] = dict(
                n=n,
                n_lab=n_lab,
                share_entry=round(float(sf3.get("entry", 0.0)), 4),
                share_mid=round(float(sf3.get("mid", 0.0)), 4),
                share_senior=round(float(sf3.get("senior", 0.0)), 4),
                share_unknown=round(float(sf3.get("unknown", 0.0)), 4),
                j3_share=round(float(j3), 4) if j3 is not None else None,
                s4_share=round(float(s4), 4) if s4 is not None else None,
            )
        rows.append(dict(title=t, **{f"{k}__{kk}": v for k, r in pd_rows.items() for kk, v in r.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "step1_top10_shared_seniority_shift.csv", index=False)
    return out


# ----------------------------- main -------------------------------------------


def main():
    print("[T10] loading corpus ...")
    df = load_corpus()
    n_rows = len(df)
    print(f"[T10] {n_rows} SWE LinkedIn rows loaded")
    # Step A. Base (all rows, no cap, aggregators INCLUDED — default)
    print("[T10] step1 vocab — full")
    full_meta = step1_vocab(df, "full")
    print("[T10] step2 concentration — full")
    step2_concentration(df, "full")
    print("[T10] step3 AI compound — full")
    step3_compound_ai(df, "full")
    print("[T10] step4 content alignment — full")
    step4_content_alignment(df, "full")
    print("[T10] step5 inflation — full")
    step5_inflation(df, "full")
    print("[T10] step6 emerging categories — full")
    step6_categories(df, "full")
    print("[T10] step7 disappearing titles (Wave 3.5 T36 input)")
    dis = step7_disappearing_titles(df)
    print(f"[T10] disappearing titles: {len(dis)}")

    # Sensitivity: aggregator-excluded
    df_nonagg = df[~df["is_aggregator"].fillna(False)].copy()
    print(f"[T10] non-agg rows: {len(df_nonagg)}")
    print("[T10] step1 vocab — nonagg")
    step1_vocab(df_nonagg, "nonagg")
    print("[T10] step2 concentration — nonagg")
    step2_concentration(df_nonagg, "nonagg")
    print("[T10] step3 AI compound — nonagg")
    step3_compound_ai(df_nonagg, "nonagg")
    print("[T10] step5 inflation — nonagg")
    step5_inflation(df_nonagg, "nonagg")
    print("[T10] step6 emerging — nonagg")
    step6_categories(df_nonagg, "nonagg")

    # Sensitivity: company cap 20 and cap 50
    for cap in (20, 50):
        df_cap = cap_by_company(df, cap=cap, seed=42)
        print(f"[T10] cap={cap} rows: {len(df_cap)}")
        step1_vocab(df_cap, f"cap{cap}")
        step2_concentration(df_cap, f"cap{cap}")
        step3_compound_ai(df_cap, f"cap{cap}")
        step5_inflation(df_cap, f"cap{cap}")
        step6_categories(df_cap, f"cap{cap}")

    # Shared top10 seniority shift (uses unseparated data with T30 panel)
    seniority_shift_shared_titles(df)

    # Meta JSON
    with open(OUT_TABLES / "run_meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)
    print("[T10] done.")


if __name__ == "__main__":
    main()
