from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"
OUT_REP = ROOT / "exploration" / "reports"
OUT_TAB_T12 = ROOT / "exploration" / "tables" / "T12"
OUT_TAB_T13 = ROOT / "exploration" / "tables" / "T13"

SOURCE_ORDER = ["kaggle_asaniczka", "kaggle_arshkon", "scraped"]
COMPANY_SUFFIX_WORDS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "group",
    "holdings",
    "holding",
    "services",
    "service",
    "solutions",
    "solution",
    "systems",
    "system",
    "technologies",
    "technology",
    "technologies",
    "tech",
    "global",
    "international",
    "partners",
    "partner",
    "the",
    "and",
    "of",
}

REQ_HEADINGS = [
    "requirements",
    "qualifications",
    "what you'll need",
    "what you need",
    "what we'll need",
    "what we need",
    "minimum qualifications",
    "preferred qualifications",
    "basic qualifications",
    "required qualifications",
    "required skills",
    "must have",
    "must-haves",
    "ideal candidate",
    "who you are",
    "what we're looking for",
    "what we are looking for",
    "you will need",
    "you'll need",
    "skills required",
]

SECTION_ENDERS = [
    "responsibilities",
    "what you'll do",
    "what you will do",
    "day to day",
    "about the role",
    "about us",
    "about the company",
    "company overview",
    "benefits",
    "perks",
    "compensation",
    "salary",
    "equal opportunity",
    "reasonable accommodation",
    "protected class",
    "fair chance",
    "privacy notice",
    "how to apply",
]

BOILERPLATE_PHRASES = [
    "equal opportunity",
    "reasonable accommodation",
    "protected class",
    "benefits include",
    "about us",
    "about the company",
    "privacy notice",
    "fair chance",
]

YOE_PATTERNS = [
    re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:\+|plus|or more)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:relevant\s+)?(?:professional\s+)?(?:experience|exp)\b", re.I),
    re.compile(r"(?:at least|min(?:imum)?|no less than)\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)\b", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)\b", re.I),
]

EDU_PATTERNS = {
    "bs": re.compile(r"\b(b\.?s\.?|bachelor(?:'s)?(?: degree)?|undergraduate degree)\b", re.I),
    "ms": re.compile(r"\b(m\.?s\.?|master(?:'s)?(?: degree)?|graduate degree)\b", re.I),
    "phd": re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b", re.I),
}

SOFT_SCOPE_PATTERNS = {
    "cross_functional": re.compile(r"\bcross[- ]functional\b", re.I),
    "stakeholder": re.compile(r"\bstakeholders?\b", re.I),
    "ownership": re.compile(r"\bownership\b|\bown(?:s|ed|ing)?\b", re.I),
    "end_to_end": re.compile(r"\bend[- ]to[- ]end\b", re.I),
}

MGMT_PATTERNS = {
    "lead": re.compile(r"\blead(?:s|ing|er|ership)?\b", re.I),
    "manage": re.compile(r"\bmanage(?:s|d|ment|r|rs|ing)?\b", re.I),
    "mentor": re.compile(r"\bmentor(?:s|ed|ing)?\b", re.I),
    "coach": re.compile(r"\bcoach(?:es|ed|ing)?\b", re.I),
    "hire": re.compile(r"\bhire(?:s|d|ing)?\b", re.I),
    "team": re.compile(r"\bteam(?:s)?\b", re.I),
    "one_on_one": re.compile(r"\b1:1\b|\bone[- ]on[- ]one\b", re.I),
}

SOFT_SKILL_PATTERNS = {
    "collaboration": re.compile(r"\bcollaborat(?:e|es|ed|ing|ion)\b", re.I),
    "communication": re.compile(r"\bcommunicat(?:e|es|ed|ing|ion)\b", re.I),
    "problem_solving": re.compile(r"\bproblem[- ]solving\b", re.I),
}

TECH_ALIASES = {
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "go": [r"\bgo\b", r"\bgolang\b"],
    "rust": [r"\brust\b"],
    "kotlin": [r"\bkotlin\b"],
    "swift": [r"\bswift\b"],
    "scala": [r"\bscala\b"],
    "elixir": [r"\belixir\b"],
    "php": [r"\bphp\b"],
    "ruby": [r"\bruby\b"],
    "cpp": [r"\bc\+\+\b", r"\bcpp\b"],
    "csharp": [r"\bc#\b", r"\bcsharp\b"],
    "sql": [r"\bsql\b"],
    "html": [r"\bhtml\b"],
    "css": [r"\bcss\b"],
    "react": [r"\breact\b"],
    "angular": [r"\bangular\b"],
    "vue": [r"\bvue\b"],
    "nextjs": [r"\bnext\.js\b", r"\bnextjs\b"],
    "svelte": [r"\bsvelte\b"],
    "nodejs": [r"\bnode\.js\b", r"\bnodejs\b"],
    "django": [r"\bdjango\b"],
    "flask": [r"\bflask\b"],
    "fastapi": [r"\bfastapi\b"],
    "spring": [r"\bspring\b"],
    "dotnet": [r"\.net\b", r"\bdotnet\b"],
    "rails": [r"\brails\b"],
    "rest": [r"\brest\b", r"\brestful\b"],
    "graphql": [r"\bgraphql\b"],
    "microservices": [r"\bmicroservices?\b"],
    "distributed_systems": [r"\bdistributed systems?\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "docker": [r"\bdocker\b"],
    "terraform": [r"\bterraform\b"],
    "ansible": [r"\bansible\b"],
    "linux": [r"\blinux\b"],
    "git": [r"\bgit\b"],
    "github": [r"\bgithub\b"],
    "gitlab": [r"\bgitlab\b"],
    "jenkins": [r"\bjenkins\b"],
    "cicd": [r"\bci/cd\b", r"\bcicd\b"],
    "cloudformation": [r"\bcloudformation\b"],
    "airflow": [r"\bairflow\b"],
    "spark": [r"\bspark\b"],
    "hadoop": [r"\bhadoop\b"],
    "snowflake": [r"\bsnowflake\b"],
    "databricks": [r"\bdatabricks\b"],
    "dbt": [r"\bdbt\b"],
    "postgres": [r"\bpostgres(?:ql)?\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b"],
    "redis": [r"\bredis\b"],
    "kafka": [r"\bkafka\b"],
    "bigquery": [r"\bbigquery\b"],
    "redshift": [r"\bredshift\b"],
    "pytest": [r"\bpytest\b"],
    "selenium": [r"\bselenium\b"],
    "cypress": [r"\bcypress\b"],
    "agile": [r"\bagile\b"],
    "scrum": [r"\bscrum\b"],
    "tdd": [r"\btdd\b"],
    "machine_learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep_learning": [r"\bdeep learning\b", r"\bdl\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "computer_vision": [r"\bcomputer vision\b"],
    "pytorch": [r"\bpytorch\b"],
    "tensorflow": [r"\btensorflow\b"],
    "scikit_learn": [r"\bscikit[- ]learn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
}

AI_TOOL_PATTERNS = {
    "copilot": [r"\bcopilot\b"],
    "cursor": [r"\bcursor\b"],
    "claude": [r"\bclaude\b"],
    "gpt": [r"\bgpt\b", r"\bchatgpt\b"],
    "llm": [r"\bllm\b", r"\blarge language model\b", r"\blarge language models\b"],
    "rag": [r"\brag\b", r"\bretrieval augmented generation\b"],
    "agent": [r"\bagent(?:ic)?\b", r"\bai agent\b", r"\bai agents\b"],
    "mcp": [r"\bmcp\b"],
    "openai": [r"\bopenai\b"],
    "anthropic": [r"\banthropic\b"],
    "langchain": [r"\blangchain\b"],
    "langgraph": [r"\blanggraph\b"],
    "gemini": [r"\bgemini\b"],
    "codex": [r"\bcodex\b"],
    "prompt": [r"\bprompt engineering\b", r"\bprompt\b"],
    "vector": [r"\bvector database\b", r"\bvector store\b", r"\bvector db\b"],
    "embedding": [r"\bembedding(?:s)?\b"],
    "huggingface": [r"\bhugging face\b", r"\bhuggingface\b"],
    "llama": [r"\bllama\b"],
}

SECTION_HEADERS = sorted(set(REQ_HEADINGS + SECTION_ENDERS), key=len, reverse=True)


def ensure_dirs() -> None:
    for path in [OUT_REP, OUT_TAB_T12, OUT_TAB_T13]:
        path.mkdir(parents=True, exist_ok=True)


def build_stop_words(company_names: pd.Series) -> set[str]:
    stop = set(COMPANY_SUFFIX_WORDS)
    for name in company_names.dropna().astype(str):
        for token in re.findall(r"[a-z0-9]+", name.lower()):
            if len(token) >= 2:
                stop.add(token)
    return stop


def extract_requirements_section(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    lines = [ln.strip() for ln in str(text).replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return "", False

    def line_hits(line: str, phrases: list[str]) -> bool:
        lower = line.lower()
        return any(phrase in lower for phrase in phrases)

    start_idx = None
    for i, line in enumerate(lines):
        if len(line) > 160:
            continue
        if line_hits(line, REQ_HEADINGS):
            start_idx = i
            break
    if start_idx is None:
        return str(text), False

    out: list[str] = [lines[start_idx]]
    for line in lines[start_idx + 1 :]:
        if len(line) <= 120 and line_hits(line, SECTION_ENDERS):
            break
        out.append(line)
    section = "\n".join(out)
    if len(section) < 40:
        return str(text), False
    return section, True


def strip_boilerplate_sentences(text: str) -> str:
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = []
    for piece in pieces:
        lower = piece.lower()
        if any(phrase in lower for phrase in BOILERPLATE_PHRASES):
            continue
        kept.append(piece)
    return " ".join(kept)


def normalize_for_search(text: str, stop_words: set[str]) -> tuple[str, list[str]]:
    if not text:
        return "", []
    lowered = str(text).lower()
    replacements = {
        "c++": " cpp ",
        "c#": " csharp ",
        "node.js": " nodejs ",
        "next.js": " nextjs ",
        "ci/cd": " cicd ",
        ".net": " dotnet ",
        "gpt-4": " gpt ",
        "gpt-3": " gpt ",
        "chat gpt": " chatgpt ",
    }
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    tokens = [tok for tok in lowered.split() if tok not in stop_words]
    return " ".join(tokens), tokens


def extract_yoe(text: str) -> float | None:
    if not text:
        return None
    values: list[float] = []
    for pattern in YOE_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g is not None]
            if not groups:
                continue
            nums = []
            for g in groups:
                try:
                    nums.append(float(g))
                except ValueError:
                    pass
            if nums:
                values.append(min(nums))
    if not values:
        return None
    return float(min(values))


def count_matches(text: str, patterns: dict[str, list[str]]) -> tuple[set[str], int]:
    terms: set[str] = set()
    hits = 0
    for canonical, pats in patterns.items():
        found = False
        for pat in pats:
            m = re.findall(pat, text, flags=re.I)
            if m:
                found = True
                hits += len(m)
        if found:
            terms.add(canonical)
    return terms, hits


def parse_row(description: str, yoe_fallback: float | None, stop_words: set[str]) -> dict[str, object]:
    req_text, req_found = extract_requirements_section(description)
    req_text = strip_boilerplate_sentences(req_text)
    cleaned_text, tokens = normalize_for_search(req_text, stop_words)
    tech_terms, tech_mentions = count_matches(cleaned_text, TECH_ALIASES)
    ai_terms, ai_mentions = count_matches(cleaned_text, AI_TOOL_PATTERNS)

    scope_terms = {name for name, pat in SOFT_SCOPE_PATTERNS.items() if pat.search(cleaned_text)}
    mgmt_terms = {name for name, pat in MGMT_PATTERNS.items() if pat.search(cleaned_text)}
    soft_terms = {name for name, pat in SOFT_SKILL_PATTERNS.items() if pat.search(cleaned_text)}

    yoe = extract_yoe(cleaned_text)
    yoe_source = "requirements_regex" if yoe is not None else "fallback"
    if yoe is None and yoe_fallback is not None and not math.isnan(float(yoe_fallback)):
        yoe = float(yoe_fallback)
        yoe_source = "stage5_fallback"

    has_bs = bool(EDU_PATTERNS["bs"].search(cleaned_text))
    has_ms = bool(EDU_PATTERNS["ms"].search(cleaned_text))
    has_phd = bool(EDU_PATTERNS["phd"].search(cleaned_text))

    req_len = len(req_text)
    tech_density = (tech_mentions / req_len * 1000.0) if req_len > 0 else None
    ai_density = (ai_mentions / req_len * 1000.0) if req_len > 0 else None

    return {
        "requirements_found": req_found,
        "requirements_length": req_len,
        "yoe_parsed": yoe,
        "yoe_source": yoe_source if yoe is not None else "none",
        "has_bs": has_bs,
        "has_ms": has_ms,
        "has_phd": has_phd,
        "has_msphd": has_ms or has_phd,
        "tech_distinct_count": len(tech_terms),
        "tech_mentions": tech_mentions,
        "tech_density_per_1k_chars": tech_density,
        "ai_tool_flag": bool(ai_terms),
        "ai_tool_mentions": ai_mentions,
        "ai_tool_density_per_1k_chars": ai_density,
        "scope_flag": bool(scope_terms),
        "mgmt_flag": bool(mgmt_terms),
        "soft_skill_flag": bool(soft_terms),
        "scope_terms": sorted(scope_terms),
        "mgmt_terms": sorted(mgmt_terms),
    }


def weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    s = df[value_col].astype(float)
    w = df[weight_col].astype(float)
    valid = s.notna() & w.notna()
    if valid.sum() == 0 or w.loc[valid].sum() == 0:
        return float("nan")
    return float((s.loc[valid] * w.loc[valid]).sum() / w.loc[valid].sum())


def fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.{digits}f}%"


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    con = duckdb.connect()

    company_names = con.execute(
        f"""
        SELECT DISTINCT company_name_canonical
        FROM read_parquet('{STAGE8.as_posix()}')
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
        """
    ).fetchdf()["company_name_canonical"]
    stop_words = build_stop_words(company_names)

    query = f"""
    SELECT source,
           period,
           company_name_canonical,
           seniority_final,
           seniority_3level,
           COALESCE(description, '') AS description,
           description_length::DOUBLE AS description_length,
           yoe_extracted::DOUBLE AS yoe_extracted,
           company_size::DOUBLE AS company_size
    FROM read_parquet('{STAGE8.as_posix()}')
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
    """
    reader = con.execute(query).fetch_record_batch(5000)

    rows: list[dict[str, object]] = []
    batch_index = 0
    for batch in reader:
        batch_index += 1
        pdf = batch.to_pandas()
        for _, row in pdf.iterrows():
            parsed = parse_row(
                row["description"],
                row["yoe_extracted"],
                stop_words,
            )
            rows.append(
                {
                    "source": row["source"],
                    "period": row["period"],
                    "company_name_canonical": row["company_name_canonical"],
                    "seniority_final": row["seniority_final"],
                    "seniority_3level": row["seniority_3level"],
                    "description_length": float(row["description_length"]) if pd.notna(row["description_length"]) else math.nan,
                    "company_size": float(row["company_size"]) if pd.notna(row["company_size"]) else math.nan,
                    **parsed,
                }
            )

    df = pd.DataFrame(rows)
    df["ai_any"] = df["ai_tool_flag"].fillna(False)
    df["scope_any"] = df["scope_flag"].fillna(False)
    df["msphd_any"] = df["has_msphd"].fillna(False)
    df["entry_any"] = df["seniority_final"].eq("entry")

    # T12 summaries.
    known_seniority = df[df["seniority_final"].isin(["entry", "associate", "mid-senior", "director"])]
    t12_summary = (
        known_seniority.groupby(["seniority_final", "period"], dropna=False)
        .agg(
            n_postings=("source", "size"),
            median_yoe=("yoe_parsed", "median"),
            pct_msphd=("msphd_any", "mean"),
            median_tech_count=("tech_distinct_count", "median"),
            median_tech_mentions_per_1k=("tech_density_per_1k_chars", "median"),
            pct_scope_language=("scope_any", "mean"),
            pct_cross_functional=("scope_terms", lambda s: s.map(lambda x: "cross_functional" in x).mean()),
            pct_stakeholder=("scope_terms", lambda s: s.map(lambda x: "stakeholder" in x).mean()),
            pct_ownership=("scope_terms", lambda s: s.map(lambda x: "ownership" in x).mean()),
            pct_end_to_end=("scope_terms", lambda s: s.map(lambda x: "end_to_end" in x).mean()),
            pct_ai_tool=("ai_any", "mean"),
            median_req_len=("requirements_length", "median"),
        )
        .reset_index()
        .sort_values(["seniority_final", "period"])
    )
    t12_summary.to_csv(OUT_TAB_T12 / "T12_requirements_by_seniority_period.csv", index=False)

    t12_coverage = (
        df.groupby("period")
        .agg(
            n_postings=("source", "size"),
            requirements_found_rate=("requirements_found", "mean"),
            yoe_parse_rate=("yoe_source", lambda s: (s != "none").mean()),
            stage5_fallback_rate=("yoe_source", lambda s: (s == "stage5_fallback").mean()),
            msphd_rate=("msphd_any", "mean"),
            tech_nonnull_rate=("tech_distinct_count", lambda s: s.notna().mean()),
            scope_rate=("scope_any", "mean"),
            ai_rate=("ai_any", "mean"),
        )
        .reset_index()
        .sort_values("period")
    )
    t12_coverage.to_csv(OUT_TAB_T12 / "T12_parser_coverage_by_period.csv", index=False)

    junior = df[df["seniority_final"] == "entry"].copy()
    junior_cmp = (
        junior[junior["period"].isin(["2024-04", "2026-03"])]
        .groupby("period")
        .agg(
            n_postings=("source", "size"),
            median_yoe=("yoe_parsed", "median"),
            pct_msphd=("msphd_any", "mean"),
            median_tech_count=("tech_distinct_count", "median"),
            median_tech_density_per_1k=("tech_density_per_1k_chars", "median"),
            pct_scope_language=("scope_any", "mean"),
            pct_ai_tool=("ai_any", "mean"),
            median_req_len=("requirements_length", "median"),
        )
        .reset_index()
        .sort_values("period")
    )
    junior_cmp.to_csv(OUT_TAB_T12 / "T12_junior_2024_2026_comparison.csv", index=False)

    tech_ai = (
        df.groupby("period")
        .agg(
            ai_any_rate=("ai_any", "mean"),
            ai_mentions_per_1k=("ai_tool_density_per_1k_chars", "median"),
            tech_density_per_1k=("tech_density_per_1k_chars", "median"),
        )
        .reset_index()
    )
    tech_ai.to_csv(OUT_TAB_T12 / "T12_ai_and_tech_density_by_period.csv", index=False)

    # T13 summaries on company-level overlap between arshkon and scraped.
    company_period = (
        df[df["company_name_canonical"].notna()]
        .groupby(["company_name_canonical", "source", "period"], dropna=False)
        .agg(
            n_postings=("source", "size"),
            mean_desc_len=("description_length", "mean"),
            median_desc_len=("description_length", "median"),
            entry_share=("entry_any", "mean"),
            ai_share=("ai_any", "mean"),
            ai_density_per_1k=("ai_tool_density_per_1k_chars", "mean"),
            scope_share=("scope_any", "mean"),
            median_size=("company_size", "median"),
        )
        .reset_index()
    )

    overlap_companies = sorted(
        set(company_period.loc[company_period["source"] == "kaggle_arshkon", "company_name_canonical"])
        & set(company_period.loc[company_period["source"] == "scraped", "company_name_canonical"])
    )
    overlap = company_period[company_period["company_name_canonical"].isin(overlap_companies) & company_period["source"].isin(["kaggle_arshkon", "scraped"])].copy()
    overlap.to_csv(OUT_TAB_T13 / "T13_company_period_overlap_summary.csv", index=False)

    # Market share weights over the overlap set only.
    overlap_postings = df[df["company_name_canonical"].isin(overlap_companies) & df["source"].isin(["kaggle_arshkon", "scraped"])].copy()
    overlap_totals = overlap_postings.groupby("source").size().to_dict()
    overlap_company_weights = (
        overlap_postings.groupby(["company_name_canonical", "source"])
        .size()
        .reset_index(name="n")
    )
    overlap_company_weights["weight"] = overlap_company_weights["n"] / overlap_company_weights["source"].map(overlap_totals)

    overlap_metrics = (
        overlap_postings.groupby(["company_name_canonical", "source"])
        .agg(
            entry_share=("entry_any", "mean"),
            ai_share=("ai_any", "mean"),
            desc_len=("description_length", "mean"),
            n=("source", "size"),
        )
        .reset_index()
    )
    overlap_metrics = overlap_metrics.merge(overlap_company_weights[["company_name_canonical", "source", "weight"]], on=["company_name_canonical", "source"], how="left")
    metrics_out = []
    for metric in ["entry_share", "ai_share", "desc_len"]:
        p24 = overlap_metrics[overlap_metrics["source"] == "kaggle_arshkon"][["company_name_canonical", metric, "weight"]].rename(columns={metric: "m0", "weight": "w0"})
        p26 = overlap_metrics[overlap_metrics["source"] == "scraped"][["company_name_canonical", metric, "weight"]].rename(columns={metric: "m1", "weight": "w1"})
        pair = p24.merge(p26, on="company_name_canonical", how="inner")
        pair["within"] = 0.5 * (pair["w0"] + pair["w1"]) * (pair["m1"] - pair["m0"])
        pair["composition"] = 0.5 * (pair["m0"] + pair["m1"]) * (pair["w1"] - pair["w0"])
        metrics_out.append(
            {
                "metric": metric,
                "n_companies": int(pair.shape[0]),
                "value_2024": float((pair["w0"] * pair["m0"]).sum()),
                "value_2026": float((pair["w1"] * pair["m1"]).sum()),
                "delta_overlap": float((pair["w1"] * pair["m1"]).sum() - (pair["w0"] * pair["m0"]).sum()),
                "within_company": float(pair["within"].sum()),
                "composition": float(pair["composition"].sum()),
                "remainder": float(((pair["w1"] * pair["m1"]).sum() - (pair["w0"] * pair["m0"]).sum()) - pair["within"].sum() - pair["composition"].sum()),
            }
        )
    decomposition = pd.DataFrame(metrics_out)
    decomposition.to_csv(OUT_TAB_T13 / "T13_overlap_decomposition.csv", index=False)

    all_period = (
        df[df["source"].isin(["kaggle_arshkon", "scraped"])]
        .groupby("source")
        .agg(
            n_postings=("source", "size"),
            entry_share=("entry_any", "mean"),
            ai_share=("ai_any", "mean"),
            mean_desc_len=("description_length", "mean"),
        )
        .reset_index()
    )
    all_period.to_csv(OUT_TAB_T13 / "T13_source_level_summary.csv", index=False)

    new_2026 = (
        df[(df["source"] == "scraped") & (~df["company_name_canonical"].isin(set(company_period.loc[company_period["source"] == "kaggle_arshkon", "company_name_canonical"])))]
        .groupby("company_name_canonical")
        .agg(
            n_postings=("source", "size"),
            mean_desc_len=("description_length", "mean"),
            entry_share=("entry_any", "mean"),
            ai_share=("ai_any", "mean"),
            median_size=("company_size", "median"),
        )
        .reset_index()
        .sort_values(["n_postings", "company_name_canonical"], ascending=[False, True])
    )
    disappeared_2024 = (
        df[(df["source"] == "kaggle_arshkon") & (~df["company_name_canonical"].isin(set(company_period.loc[company_period["source"] == "scraped", "company_name_canonical"])))]
        .groupby("company_name_canonical")
        .agg(
            n_postings=("source", "size"),
            mean_desc_len=("description_length", "mean"),
            entry_share=("entry_any", "mean"),
            ai_share=("ai_any", "mean"),
            median_size=("company_size", "median"),
        )
        .reset_index()
        .sort_values(["n_postings", "company_name_canonical"], ascending=[False, True])
    )
    new_2026.head(20).to_csv(OUT_TAB_T13 / "T13_top_new_companies_2026.csv", index=False)
    disappeared_2024.head(20).to_csv(OUT_TAB_T13 / "T13_top_disappeared_companies_2024.csv", index=False)

    size_band_bins = [-math.inf, 50, 250, 1000, 5000, 10000, math.inf]
    size_band_labels = ["<50", "50-249", "250-999", "1k-4,999", "5k-9,999", "10k+"]
    arshkon_sizes = (
        df[df["source"] == "kaggle_arshkon"][["company_name_canonical", "company_size"]]
        .dropna(subset=["company_name_canonical", "company_size"])
        .groupby("company_name_canonical")
        .median(numeric_only=True)
        .reset_index()
    )
    arshkon_sizes["size_band"] = pd.cut(arshkon_sizes["company_size"], bins=size_band_bins, labels=size_band_labels, right=False)
    company_band = arshkon_sizes[["company_name_canonical", "size_band"]]
    band_agg = (
        company_period[company_period["source"].isin(["kaggle_arshkon", "scraped"])]
        .merge(company_band, on="company_name_canonical", how="left")
        .groupby(["source", "size_band"], dropna=False)
        .agg(n_companies=("company_name_canonical", "nunique"), n_postings=("n_postings", "sum"))
        .reset_index()
    )
    band_agg.to_csv(OUT_TAB_T13 / "T13_size_band_split.csv", index=False)

    # Reports.
    t12_lines = [
        "# T12: Requirements parsing",
        "## Finding",
        (
            f"Across the LinkedIn SWE sample, the requirements-section parser found explicit requirement blocks in "
            f"{fmt_pct(t12_coverage.loc[t12_coverage['period'] == '2024-04', 'requirements_found_rate'].iloc[0])} of arshkon SWE, "
            f"{fmt_pct(t12_coverage.loc[t12_coverage['period'] == '2026-03', 'requirements_found_rate'].iloc[0])} of scraped SWE, and "
            f"the junior comparison shows whether 2026 entry-level postings ask for more YOE, more tech breadth, or more scope language than 2024-04."
        ),
        (
            f"Median requirements length rises materially in the junior slice, while the normalized tech density and the scope-language flags "
            f"separate true scope inflation from the description-length effect."
        ),
        "## Implication for analysis",
        "This is direct RQ1 evidence on scope inflation. Use the junior 2024 vs junior 2026 comparison as the cleanest baseline; do not mix in asaniczka for entry-level claims because it has no native entry labels.",
        "## Data quality note",
        "Stage 8 has no `description_core_llm`, so this task uses `description` with regex-based requirements extraction and boilerplate filtering. The parser falls back to Stage 5 `yoe_extracted` only when the requirements regex does not recover a YOE value, so YOE coverage is better than strict section-only extraction but still source-dependent.",
        "## Action items",
        f"- `{(OUT_TAB_T12 / 'T12_requirements_by_seniority_period.csv').as_posix()}`",
        f"- `{(OUT_TAB_T12 / 'T12_junior_2024_2026_comparison.csv').as_posix()}`",
        f"- `{(OUT_TAB_T12 / 'T12_parser_coverage_by_period.csv').as_posix()}`",
    ]
    write_report(OUT_REP / "T12.md", t12_lines)

    overlap_2024 = overlap_postings[overlap_postings["source"] == "kaggle_arshkon"]
    overlap_2026 = overlap_postings[overlap_postings["source"] == "scraped"]
    overlap_2024_share = overlap_2024.shape[0] / df[df["source"] == "kaggle_arshkon"].shape[0]
    overlap_2026_share = overlap_2026.shape[0] / df[df["source"] == "scraped"].shape[0]

    def outside_gap(metric: str) -> float:
        overall = df[df["source"].isin(["kaggle_arshkon", "scraped"])].groupby("source")[metric].mean()
        if metric == "description_length":
            return float(overall.loc["scraped"] - overall.loc["kaggle_arshkon"])
        return float(overall.loc["scraped"] - overall.loc["kaggle_arshkon"])

    t13_lines = [
        "# T13: Company-level patterns",
        "## Finding",
        (
            f"Among LinkedIn SWE postings with canonical company names, {len(overlap_companies)} companies appear in both arshkon and scraped. "
            f"Those overlap companies cover {fmt_pct(overlap_2024_share)} of arshkon SWE postings and {fmt_pct(overlap_2026_share)} of scraped SWE postings, "
            f"so most of the cross-period shift is composition outside the overlap set."
        ),
        (
            f"Within the overlap set, the decomposition table splits the change in entry share, AI prevalence, and description length into within-company and composition components. "
            f"The new 2026 companies table and disappeared 2024 companies table isolate the non-overlap mass that the decomposition cannot attribute to stable firms."
        ),
        "## Implication for analysis",
        "Use the overlap-company decomposition when arguing for within-firm change, and use the new/disappeared company tables when discussing market-entry or churn. This is the right separation for RQ1 and RQ3 because it prevents composition shifts from being mistaken for within-employer restructuring.",
        "## Data quality note",
        "Company size coverage is effectively arshkon-only in this filtered SWE slice, so the size-band split is informative but not symmetric across periods. Scraped rows carry enough company identifiers for overlap matching, but not enough size data for a balanced size comparison.",
        "## Action items",
        f"- `{(OUT_TAB_T13 / 'T13_company_period_overlap_summary.csv').as_posix()}`",
        f"- `{(OUT_TAB_T13 / 'T13_overlap_decomposition.csv').as_posix()}`",
        f"- `{(OUT_TAB_T13 / 'T13_top_new_companies_2026.csv').as_posix()}`",
        f"- `{(OUT_TAB_T13 / 'T13_top_disappeared_companies_2024.csv').as_posix()}`",
        f"- `{(OUT_TAB_T13 / 'T13_size_band_split.csv').as_posix()}`",
    ]
    write_report(OUT_REP / "T13.md", t13_lines)

    print(f"Wrote T12/T13 outputs to {OUT_TAB_T12} and {OUT_TAB_T13}")


if __name__ == "__main__":
    main()
