"""Build pilot_sample_v3.parquet — 30 SWE postings stratified by role family + skill + era.

Stratification:
  - 17 role-family slots (1 per family by title heuristic)
  - 5 multi-skill (3+ regex skill themes)
  - 4 zero-skill (no skill themes)
  - 4 era-balance fillers to ensure each (source, period) stratum is represented

Written to pilot_sample_v3.parquet with columns:
  uid, source, period, title, description_core_llm, description, company_name_effective,
  regex_skill_topics, n_regex_skill_topics, role_family_heuristic, stratum_v3
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
PARQUET = REPO / "data/unified_core.parquet"
VOCAB = REPO / "paper/vocab_lists/vocab_lists.json"
OUT_DIR = REPO / "paper/vocab_lists/llm_classification"
SAMPLE_PATH = OUT_DIR / "pilot_sample_v3.parquet"

STRATA_SOURCES = [
    ("kaggle_arshkon", "2024-04"),
    ("kaggle_asaniczka", "2024-01"),
    ("scraped", "2026-04"),
]

SEED = 20260507
WORD_LIKE = re.compile(r"^[A-Za-z0-9 \-']+$")

# 17 role families (matches role_families.md). Order = priority for title heuristic;
# more specific titles match first.
ROLE_FAMILIES_ORDERED = [
    "ai_llm_engineer",
    "ml_engineer",
    "data_engineer",
    "data_analytics",
    "research",
    "security",
    "qa_test",
    "devops_sre_platform",
    "legacy_specialist",
    "mobile",
    "embedded",
    "solutions_field",
    "infra_ops_admin",
    "people_manager",
    "frontend_web",
    "backend_api",
    "software_engineer_general",  # fallback
]

# Compile patterns for title heuristic. Apply in priority order; first match wins.
TITLE_HEURISTICS: list[tuple[str, re.Pattern]] = [
    ("ai_llm_engineer", re.compile(
        r"\b(ai engineer|llm engineer|llm|genai|generative ai|prompt engineer|"
        r"agent engineer|forward[- ]deployed engineer|fde|gen-ai|ai/ml engineer)\b",
        re.IGNORECASE)),
    ("ml_engineer", re.compile(
        r"\b(ml engineer|machine learning engineer|mlops|ml ops|"
        r"machine[- ]learning engineer|ai/ml)\b", re.IGNORECASE)),
    ("research", re.compile(
        r"\b(research scientist|applied scientist|research engineer)\b",
        re.IGNORECASE)),
    ("data_engineer", re.compile(
        r"\b(data engineer|data platform|etl engineer|data infrastructure|"
        r"warehouse engineer|big data engineer|analytics platform)\b",
        re.IGNORECASE)),
    ("data_analytics", re.compile(
        r"\b(data scientist|data analyst|analytics engineer|"
        r"business intelligence|bi developer|bi engineer|data analytics)\b",
        re.IGNORECASE)),
    ("legacy_specialist", re.compile(
        r"\b(mainframe|cobol|salesforce developer|servicenow developer|"
        r"sap developer|abap|oracle forms|peoplesoft|legacy systems?)\b",
        re.IGNORECASE)),
    ("security", re.compile(
        r"\b(security engineer|cybersecurity|cyber security|appsec|infosec|"
        r"application security|penetration tester|cyber )\b",
        re.IGNORECASE)),
    ("qa_test", re.compile(
        r"\b(qa engineer|qa automation|test automation|test engineer|"
        r"quality assurance|quality engineer|sdet|qa lead|qa/test)\b",
        re.IGNORECASE)),
    ("devops_sre_platform", re.compile(
        r"\b(devops|dev[- ]?ops|sre|site reliability|platform engineer|"
        r"infrastructure engineer|cloud engineer|kubernetes engineer)\b",
        re.IGNORECASE)),
    ("mobile", re.compile(
        r"\b(mobile engineer|mobile developer|ios developer|android developer|"
        r"ios engineer|android engineer|react native|flutter)\b",
        re.IGNORECASE)),
    ("embedded", re.compile(
        r"\b(embedded engineer|firmware|firmware engineer|iot engineer|"
        r"embedded software|embedded systems?|rtos|fpga)\b",
        re.IGNORECASE)),
    ("solutions_field", re.compile(
        r"\b(solutions engineer|customer engineer|field engineer|"
        r"implementation engineer|sales engineer|pre-sales|presales|"
        r"forward[- ]deployed)\b", re.IGNORECASE)),
    ("infra_ops_admin", re.compile(
        r"\b(database administrator|\bdba\b|sysadmin|systems administrator|"
        r"network engineer|cloud administrator|systems engineer)\b",
        re.IGNORECASE)),
    ("people_manager", re.compile(
        r"\b(engineering manager|software engineering manager|"
        r"development manager|head of engineering|director of engineering|"
        r"manager,? software)\b", re.IGNORECASE)),
    ("frontend_web", re.compile(
        r"\b(front[- ]?end engineer|front[- ]?end developer|web developer|"
        r"web engineer|ui developer|ux engineer|full[- ]stack engineer|"
        r"full[- ]stack developer|fullstack|react developer|vue developer|"
        r"angular developer|javascript developer)\b", re.IGNORECASE)),
    ("backend_api", re.compile(
        r"\b(back[- ]?end engineer|back[- ]?end developer|api engineer|"
        r"api developer|microservices engineer|server[- ]side|"
        r"backend developer|application developer)\b", re.IGNORECASE)),
]


def role_family_from_title(title: str) -> str:
    """First-match title heuristic; falls back to software_engineer_general."""
    if not title:
        return "software_engineer_general"
    for family, pat in TITLE_HEURISTICS:
        if pat.search(title):
            return family
    return "software_engineer_general"


# ---------- regex skill labelling (mirrors run_pilot.py with prefilter) ----------

def compile_pattern(kw: str) -> re.Pattern:
    kw_lc = kw.lower()
    if WORD_LIKE.match(kw_lc):
        return re.compile(rf"\b{re.escape(kw_lc)}\b", re.IGNORECASE)
    starts_word = kw_lc[0].isalnum()
    ends_word = kw_lc[-1].isalnum()
    pre = r"(?<![A-Za-z0-9])" if starts_word else r""
    post = r"(?![A-Za-z0-9])" if ends_word else r""
    return re.compile(pre + re.escape(kw_lc) + post, re.IGNORECASE)


def build_topic_patterns(vocab: dict):
    out: dict[str, list[tuple[str, re.Pattern]]] = {}
    for slug, t in vocab["topics"].items():
        kw_pats = []
        for c in t["core_concepts"]:
            for k in c["keywords"]:
                kw_pats.append((k.lower(), compile_pattern(k)))
        out[slug] = kw_pats
    return out


def label_topics(text: str, patterns: dict) -> list[str]:
    text_lc = text.lower() if text else ""
    if not text_lc:
        return []
    hit: set[str] = set()
    for slug, kw_pats in patterns.items():
        for kw_lc, pat in kw_pats:
            if kw_lc not in text_lc:
                continue
            if pat.search(text_lc):
                hit.add(slug)
                break
    return sorted(hit)


# ---------- sample selection ----------

def load_swe(con) -> pd.DataFrame:
    print(f"Loading SWE postings from {PARQUET.name} for {len(STRATA_SOURCES)} strata...", flush=True)
    frames = []
    for src, period in STRATA_SOURCES:
        q = f"""
        SELECT uid, source, period, title, description_core_llm, description, company_name_effective
        FROM read_parquet('{PARQUET}')
        WHERE source = ? AND period = ? AND is_swe = TRUE
              AND description_core_llm IS NOT NULL
              AND length(description_core_llm) >= 80
        """
        frames.append(con.execute(q, [src, period]).fetchdf())
    df = pd.concat(frames, ignore_index=True)
    print(f"  total in-frame: {len(df):,}", flush=True)
    return df


def label_all(df: pd.DataFrame, patterns: dict) -> pd.DataFrame:
    n = len(df)
    print(f"Labelling {n:,} postings ...", flush=True)
    skill_labels = []
    role_heur = []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        skill_labels.append(label_topics(row.description_core_llm or "", patterns))
        role_heur.append(role_family_from_title(row.title or ""))
        if (i + 1) % 5000 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            print(f"  {i+1:>6,}/{n:,}  ({rate:.0f}/s)", flush=True)
    df = df.copy()
    df["regex_skill_topics"] = skill_labels
    df["n_regex_skill_topics"] = df["regex_skill_topics"].map(len)
    df["role_family_heuristic"] = role_heur
    return df


def pick_sample(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    used: set[int] = set()
    picked: list[int] = []
    strata: list[str] = []

    def reserve(idx: int, label: str):
        used.add(idx)
        picked.append(idx)
        strata.append(label)

    print("\nStratum A — 1 per role family by title heuristic:", flush=True)
    for fam in ROLE_FAMILIES_ORDERED:
        cands = df.index[
            (df["role_family_heuristic"] == fam)
            & ~df.index.to_series().isin(used)
        ].tolist()
        if not cands:
            print(f"  {fam}: NONE FOUND", flush=True)
            continue
        # prefer postings with non-trivial description
        cands_full = [i for i in cands if len(df.loc[i, "description_core_llm"] or "") >= 200]
        pool = cands_full or cands
        idx = rng.choice(pool)
        reserve(idx, f"role:{fam}")
        print(f"  {fam:28s}: uid={df.loc[idx,'uid']} title={df.loc[idx,'title'][:60]!r}", flush=True)

    print("\nStratum B — 5 multi-skill (3+ regex skill themes):", flush=True)
    cands = df.index[
        (df["n_regex_skill_topics"] >= 3)
        & ~df.index.to_series().isin(used)
    ].tolist()
    rng.shuffle(cands)
    for idx in cands[:5]:
        reserve(idx, "skill:multi")
        print(f"  uid={df.loc[idx,'uid']} skills={df.loc[idx,'regex_skill_topics']} "
              f"title={df.loc[idx,'title'][:50]!r}", flush=True)

    print("\nStratum C — 4 zero-skill:", flush=True)
    cands = df.index[
        (df["n_regex_skill_topics"] == 0)
        & ~df.index.to_series().isin(used)
    ].tolist()
    rng.shuffle(cands)
    for idx in cands[:4]:
        reserve(idx, "skill:zero")
        print(f"  uid={df.loc[idx,'uid']} title={df.loc[idx,'title'][:50]!r}", flush=True)

    print("\nStratum D — 4 era-balance fillers:", flush=True)
    # ensure each (source, period) has >= 8 in final sample
    so_far = df.loc[picked]
    counts = so_far.groupby(["source", "period"]).size().to_dict()
    target = 8
    for (src, period) in STRATA_SOURCES:
        deficit = max(0, target - counts.get((src, period), 0))
        if deficit <= 0:
            continue
        cands = df.index[
            (df["source"] == src) & (df["period"] == period)
            & ~df.index.to_series().isin(used)
        ].tolist()
        rng.shuffle(cands)
        for idx in cands[:deficit]:
            reserve(idx, f"era:{src.split('_')[-1] if '_' in src else src}")
            print(f"  uid={df.loc[idx,'uid']} src={src} title={df.loc[idx,'title'][:50]!r}",
                  flush=True)

    sample = df.loc[picked].reset_index(drop=True)
    sample["stratum_v3"] = strata
    return sample


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = json.loads(VOCAB.read_text())
    patterns = build_topic_patterns(vocab)

    con = duckdb.connect()
    df = load_swe(con)
    df = label_all(df, patterns)

    print(f"\nRole-family heuristic distribution in corpus:")
    for fam, n in df["role_family_heuristic"].value_counts().items():
        print(f"  {fam:30s}  {n:>6,}")

    sample = pick_sample(df, rng)
    pq.write_table(pa.Table.from_pandas(sample), SAMPLE_PATH)
    print(f"\nWrote {SAMPLE_PATH} ({len(sample)} postings)")

    print(f"\nFinal sample composition:")
    print(f"  by stratum_v3:")
    for stratum, n in sample["stratum_v3"].value_counts().sort_index().items():
        print(f"    {stratum:30s}  {n}")
    print(f"  by source/period:")
    for (src, period), n in sample.groupby(["source", "period"]).size().items():
        print(f"    {src}/{period:8s}  {n}")
    print(f"  by role_family_heuristic:")
    for fam, n in sample["role_family_heuristic"].value_counts().items():
        print(f"    {fam:30s}  {n}")


if __name__ == "__main__":
    main()
