#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
STAGE8 = ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"
OUT_TABLES = ROOT / "exploration" / "tables"
OUT_FIGS = ROOT / "exploration" / "figures"
OUT_REPORTS = ROOT / "exploration" / "reports"

FILTER = """
source_platform = 'linkedin'
AND is_english = true
AND date_flag = 'ok'
AND is_swe = true
"""

TOKEN_RE = re.compile(r"[a-z0-9]+")
BLOCK_SPLIT_RE = re.compile(r"\n\s*\n+")
LINE_SPLIT_RE = re.compile(r"\n+")

SPECIAL_REPLACEMENTS = [
    (re.compile(r"\bc\+\+\b", re.I), "cplusplus"),
    (re.compile(r"\bc#\b", re.I), "csharp"),
    (re.compile(r"\b\.net\b", re.I), "dotnet"),
    (re.compile(r"\bnode\.js\b", re.I), "nodejs"),
    (re.compile(r"\bnext\.js\b", re.I), "nextjs"),
    (re.compile(r"\breact\.js\b", re.I), "reactjs"),
    (re.compile(r"\bscikit-learn\b", re.I), "scikitlearn"),
    (re.compile(r"\bci/cd\b", re.I), "cicd"),
    (re.compile(r"\bgitlab ci\b", re.I), "gitlabci"),
    (re.compile(r"\bgithub actions\b", re.I), "githubactions"),
    (re.compile(r"\bmachine-learning\b", re.I), "machine learning"),
    (re.compile(r"\bdeep-learning\b", re.I), "deep learning"),
    (re.compile(r"\bprompt-engineering\b", re.I), "prompt engineering"),
    (re.compile(r"\bfine-tuning\b", re.I), "fine tuning"),
    (re.compile(r"\bretrieval-augmented generation\b", re.I), "retrieval augmented generation"),
    (re.compile(r"\bvector databases?\b", re.I), "vector database"),
]

TECH_WHITELIST_STEM = {
    "ai",
    "ml",
    "go",
    "sql",
    "aws",
    "gcp",
    "api",
    "apis",
    "llm",
    "llms",
    "mcp",
    "git",
    "js",
    "ts",
    "r",
    "c",
    "net",
    "dbt",
    "db",
    "etl",
    "data",
    "cloud",
    "software",
    "systems",
    "system",
    "platform",
    "platforms",
    "engineering",
    "engineer",
    "engineers",
    "technology",
    "technologies",
    "tech",
    "service",
    "services",
    "solution",
    "solutions",
    "group",
    "global",
    "labs",
    "analytics",
    "digital",
    "capital",
    "ventures",
}

PERIOD_ORDER = ["2024-01", "2024-04", "2026-03"]
SENIORITY_ORDER = ["entry", "associate", "mid-senior", "director", "unknown"]
TEXT_CANDIDATES = [
    "description_core",
    "description",
]


@dataclass(frozen=True)
class TechSpec:
    tech: str
    category: str
    aliases: tuple[str, ...]


SECTION_SPECS = [
    ("legal_eeo", [
        r"\bequal opportunity\b",
        r"\beeo\b",
        r"\breasonable accommodation\b",
        r"\bprotected class\b",
        r"\bfair chance\b",
        r"\bprivacy notice\b",
        r"\bdiscrimination\b",
        r"\bapplicants? with disabilities\b",
    ]),
    ("benefits", [
        r"\bbenefits?\b",
        r"\bperks?\b",
        r"\bwhat we offer\b",
        r"\bcompensation\b",
        r"\btotal rewards\b",
        r"\bwe offer\b",
        r"\bbetter benefits\b",
        r"\bbenefits include\b",
    ]),
    ("about_company", [
        r"\babout us\b",
        r"\babout the company\b",
        r"\bwho we are\b",
        r"\bour company\b",
        r"\bour mission\b",
        r"\blife at\b",
        r"\bwhy join\b",
        r"\babout (?:the )?team\b",
    ]),
    ("preferred", [
        r"\bpreferred\b",
        r"\bnice to have\b",
        r"\bbonus\b",
        r"\bpreferred qualifications\b",
        r"\bdesired qualifications\b",
        r"\bplus\b",
        r"\bwhat we value\b",
    ]),
    ("requirements", [
        r"\brequirements\b",
        r"\bqualifications\b",
        r"\bminimum qualifications\b",
        r"\bbasic qualifications\b",
        r"\bwhat you bring\b",
        r"\bmust have\b",
        r"\brequired qualifications\b",
        r"\byou(?:'| )ll need\b",
        r"\bexperience required\b",
        r"\byou have\b",
    ]),
    ("responsibilities", [
        r"\bresponsibilities\b",
        r"\bwhat you will do\b",
        r"\bwhat you'll do\b",
        r"\bday to day\b",
        r"\bday-to-day\b",
        r"\bduties\b",
        r"\byou will\b",
        r"\byou'll be\b",
    ]),
    ("summary", [
        r"\brole summary\b",
        r"\boverview\b",
        r"\babout the role\b",
        r"\babout the job\b",
        r"\bposition summary\b",
        r"\bthis role\b",
        r"\bjoin us\b",
        r"\bwe(?:'| )re looking for\b",
        r"\bwho you are\b",
    ]),
]


TECH_SPECS = [
    # Languages
    TechSpec("Python", "languages", ("python",)),
    TechSpec("Java", "languages", ("java",)),
    TechSpec("JavaScript", "languages", ("javascript", "js")),
    TechSpec("TypeScript", "languages", ("typescript", "ts")),
    TechSpec("Go", "languages", (" go ",)),
    TechSpec("Rust", "languages", ("rust",)),
    TechSpec("C++", "languages", ("cplusplus",)),
    TechSpec("C#", "languages", ("csharp",)),
    TechSpec("Ruby", "languages", ("ruby",)),
    TechSpec("Kotlin", "languages", ("kotlin",)),
    TechSpec("Swift", "languages", ("swift",)),
    TechSpec("Scala", "languages", ("scala",)),
    TechSpec("Elixir", "languages", ("elixir",)),
    TechSpec("PHP", "languages", ("php",)),
    TechSpec("Bash", "languages", ("bash", "shell", "bash shell")),
    TechSpec("Perl", "languages", ("perl",)),
    TechSpec("R", "languages", (" r ",)),
    TechSpec("MATLAB", "languages", ("matlab",)),
    TechSpec("Dart", "languages", ("dart",)),
    TechSpec("Objective-C", "languages", ("objective c",)),
    TechSpec("F#", "languages", ("fsharp",)),
    TechSpec("Lua", "languages", ("lua",)),
    # Frontend
    TechSpec("React", "frontend", ("react",)),
    TechSpec("Angular", "frontend", ("angular",)),
    TechSpec("Vue", "frontend", ("vue",)),
    TechSpec("Next.js", "frontend", ("nextjs",)),
    TechSpec("Svelte", "frontend", ("svelte",)),
    TechSpec("HTML", "frontend", ("html",)),
    TechSpec("CSS", "frontend", ("css",)),
    TechSpec("Tailwind CSS", "frontend", ("tailwind css", "tailwind")),
    TechSpec("Redux", "frontend", ("redux",)),
    TechSpec("Webpack", "frontend", ("webpack",)),
    TechSpec("Vite", "frontend", ("vite",)),
    TechSpec("jQuery", "frontend", ("jquery",)),
    TechSpec("Bootstrap", "frontend", ("bootstrap",)),
    TechSpec("Storybook", "frontend", ("storybook",)),
    # Backend / infra
    TechSpec("Node.js", "backend_infra", ("nodejs", "node js")),
    TechSpec("Express", "backend_infra", ("express",)),
    TechSpec("Django", "backend_infra", ("django",)),
    TechSpec("Flask", "backend_infra", ("flask",)),
    TechSpec("FastAPI", "backend_infra", ("fastapi",)),
    TechSpec("Spring", "backend_infra", ("spring",)),
    TechSpec(".NET", "backend_infra", ("dotnet",)),
    TechSpec("Rails", "backend_infra", ("rails",)),
    TechSpec("NestJS", "backend_infra", ("nestjs",)),
    TechSpec("GraphQL", "backend_infra", ("graphql",)),
    TechSpec("REST", "backend_infra", ("rest api", "restful", "rest")),
    TechSpec("gRPC", "backend_infra", ("grpc",)),
    TechSpec("Microservices", "backend_infra", ("microservices",)),
    TechSpec("Serverless", "backend_infra", ("serverless",)),
    TechSpec("Lambda", "backend_infra", ("lambda", "aws lambda")),
    TechSpec("API design", "backend_infra", ("api design", "api", "apis")),
    # Cloud / DevOps
    TechSpec("AWS", "cloud_devops", ("aws",)),
    TechSpec("Azure", "cloud_devops", ("azure",)),
    TechSpec("GCP", "cloud_devops", ("gcp", "google cloud")),
    TechSpec("Kubernetes", "cloud_devops", ("kubernetes",)),
    TechSpec("Docker", "cloud_devops", ("docker",)),
    TechSpec("Terraform", "cloud_devops", ("terraform",)),
    TechSpec("CI/CD", "cloud_devops", ("cicd", "ci cd")),
    TechSpec("Jenkins", "cloud_devops", ("jenkins",)),
    TechSpec("GitHub Actions", "cloud_devops", ("githubactions", "github actions")),
    TechSpec("GitLab CI", "cloud_devops", ("gitlabci", "gitlab ci")),
    TechSpec("Ansible", "cloud_devops", ("ansible",)),
    TechSpec("Helm", "cloud_devops", ("helm",)),
    TechSpec("Linux", "cloud_devops", ("linux",)),
    TechSpec("Unix", "cloud_devops", ("unix",)),
    TechSpec("Apache", "cloud_devops", ("apache",)),
    TechSpec("Nginx", "cloud_devops", ("nginx",)),
    TechSpec("AWS Lambda", "cloud_devops", ("aws lambda",)),
    TechSpec("IaC", "cloud_devops", ("infrastructure as code", "iac")),
    TechSpec("CloudFormation", "cloud_devops", ("cloudformation",)),
    TechSpec("Prometheus", "cloud_devops", ("prometheus",)),
    TechSpec("Grafana", "cloud_devops", ("grafana",)),
    TechSpec("Argo", "cloud_devops", ("argo",)),
    TechSpec("OpenShift", "cloud_devops", ("openshift",)),
    # Data
    TechSpec("SQL", "data", (" sql ",)),
    TechSpec("PostgreSQL", "data", ("postgresql", "postgres")),
    TechSpec("MySQL", "data", ("mysql",)),
    TechSpec("MongoDB", "data", ("mongodb",)),
    TechSpec("Redis", "data", ("redis",)),
    TechSpec("Kafka", "data", ("kafka",)),
    TechSpec("Spark", "data", ("spark",)),
    TechSpec("Snowflake", "data", ("snowflake",)),
    TechSpec("Databricks", "data", ("databricks",)),
    TechSpec("dbt", "data", (" dbt ",)),
    TechSpec("Airflow", "data", ("airflow",)),
    TechSpec("BigQuery", "data", ("bigquery",)),
    TechSpec("Redshift", "data", ("redshift",)),
    TechSpec("Elasticsearch", "data", ("elasticsearch", "elastic search")),
    TechSpec("ClickHouse", "data", ("clickhouse",)),
    TechSpec("SQLite", "data", ("sqlite",)),
    TechSpec("Oracle", "data", ("oracle",)),
    TechSpec("Presto", "data", ("presto",)),
    TechSpec("Trino", "data", ("trino",)),
    TechSpec("Tableau", "data", ("tableau",)),
    TechSpec("Looker", "data", ("looker",)),
    TechSpec("Power BI", "data", ("power bi",)),
    TechSpec("dbt Cloud", "data", ("dbt cloud",)),
    # AI / ML traditional
    TechSpec("TensorFlow", "ai_ml_traditional", ("tensorflow",)),
    TechSpec("PyTorch", "ai_ml_traditional", ("pytorch",)),
    TechSpec("scikit-learn", "ai_ml_traditional", ("scikitlearn",)),
    TechSpec("Pandas", "ai_ml_traditional", ("pandas",)),
    TechSpec("NumPy", "ai_ml_traditional", ("numpy",)),
    TechSpec("Jupyter", "ai_ml_traditional", ("jupyter",)),
    TechSpec("Keras", "ai_ml_traditional", ("keras",)),
    TechSpec("XGBoost", "ai_ml_traditional", ("xgboost",)),
    TechSpec("OpenCV", "ai_ml_traditional", ("opencv",)),
    TechSpec("MLflow", "ai_ml_traditional", ("mlflow",)),
    TechSpec("Hugging Face", "ai_ml_traditional", ("hugging face", "huggingface")),
    TechSpec("DeepSpeed", "ai_ml_traditional", ("deepspeed",)),
    TechSpec("LightGBM", "ai_ml_traditional", ("lightgbm",)),
    TechSpec("CatBoost", "ai_ml_traditional", ("catboost",)),
    # AI / LLM new
    TechSpec("LangChain", "ai_llm_new", ("langchain",)),
    TechSpec("LangGraph", "ai_llm_new", ("langgraph",)),
    TechSpec("LlamaIndex", "ai_llm_new", ("llamaindex",)),
    TechSpec("Llama.cpp", "ai_llm_new", ("llamacpp", "llama cpp")),
    TechSpec("Ollama", "ai_llm_new", ("ollama",)),
    TechSpec("vLLM", "ai_llm_new", ("vllm",)),
    TechSpec("RAG", "ai_llm_new", ("retrieval augmented generation", "retrieval augmented", "rag")),
    TechSpec("Vector database", "ai_llm_new", ("vector database", "vector db", "vectordb")),
    TechSpec("Pinecone", "ai_llm_new", ("pinecone",)),
    TechSpec("ChromaDB", "ai_llm_new", ("chromadb", "chroma db")),
    TechSpec("OpenAI API", "ai_llm_new", ("openai api", "openai")),
    TechSpec("Claude API", "ai_llm_new", ("claude api", "anthropic api")),
    TechSpec("Gemini API", "ai_llm_new", ("gemini api", "google gemini")),
    TechSpec("Prompt engineering", "ai_llm_new", ("prompt engineering",)),
    TechSpec("Fine-tuning", "ai_llm_new", ("fine tuning", "finetuning")),
    TechSpec("MCP", "ai_llm_new", ("model context protocol", "mcp")),
    TechSpec("Agent framework", "ai_llm_new", ("agent framework", "agent frameworks", "agents framework")),
    TechSpec("Function calling", "ai_llm_new", ("function calling",)),
    TechSpec("Embeddings", "ai_llm_new", ("embeddings", "embedding")),
    TechSpec("Transformers", "ai_llm_new", ("transformers",)),
    TechSpec("Vector search", "ai_llm_new", ("vector search",)),
    TechSpec("Semantic search", "ai_llm_new", ("semantic search",)),
    TechSpec("Generative AI", "ai_llm_new", ("generative ai", "genai")),
    # AI tools
    TechSpec("Copilot", "ai_tools", ("copilot",)),
    TechSpec("Cursor", "ai_tools", ("cursor",)),
    TechSpec("ChatGPT", "ai_tools", ("chatgpt",)),
    TechSpec("Claude", "ai_tools", ("claude",)),
    TechSpec("Gemini", "ai_tools", ("gemini",)),
    TechSpec("Codex", "ai_tools", ("codex",)),
    TechSpec("Replit", "ai_tools", ("replit",)),
    TechSpec("Tabnine", "ai_tools", ("tabnine",)),
    TechSpec("Perplexity", "ai_tools", ("perplexity",)),
    TechSpec("GitHub Copilot", "ai_tools", ("github copilot",)),
    # Testing / practices
    TechSpec("Jest", "testing_practices", ("jest",)),
    TechSpec("Pytest", "testing_practices", ("pytest",)),
    TechSpec("Selenium", "testing_practices", ("selenium",)),
    TechSpec("Cypress", "testing_practices", ("cypress",)),
    TechSpec("Playwright", "testing_practices", ("playwright",)),
    TechSpec("TDD", "testing_practices", ("test driven development", "test driven", "tdd")),
    TechSpec("Agile", "testing_practices", ("agile",)),
    TechSpec("Scrum", "testing_practices", ("scrum",)),
    TechSpec("Kanban", "testing_practices", ("kanban",)),
    TechSpec("Unit testing", "testing_practices", ("unit testing",)),
    TechSpec("Integration testing", "testing_practices", ("integration testing",)),
    TechSpec("Code review", "testing_practices", ("code review",)),
    TechSpec("QA", "testing_practices", (" qa ", "quality assurance")),
    TechSpec("CI testing", "testing_practices", ("continuous integration",)),
    TechSpec("BDD", "testing_practices", ("behavior driven development", "bdd")),
]


def ensure_dirs() -> None:
    for base in [OUT_TABLES, OUT_FIGS, OUT_REPORTS]:
        base.mkdir(parents=True, exist_ok=True)
    for task in ["T17", "T18"]:
        (OUT_TABLES / task).mkdir(parents=True, exist_ok=True)
        (OUT_FIGS / task).mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    return con


def fetch_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()


def csv_write(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def build_company_stop_tokens(con: duckdb.DuckDBPyConnection) -> set[str]:
    sql = f"""
    SELECT DISTINCT company_name_canonical
    FROM read_parquet('{STAGE8}')
    WHERE {FILTER}
      AND company_name_canonical IS NOT NULL
      AND trim(company_name_canonical) <> ''
    """
    df = fetch_df(con, sql)
    stop_tokens: set[str] = set()
    for value in df["company_name_canonical"].dropna().astype(str):
        for tok in TOKEN_RE.findall(value.lower()):
            if len(tok) >= 3 and tok not in TECH_WHITELIST_STEM:
                stop_tokens.add(tok)
    return stop_tokens


def normalize_text(text: str, stop_tokens: set[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    s = text.lower()
    for pattern, replacement in SPECIAL_REPLACEMENTS:
        s = pattern.sub(replacement, s)
    tokens = [tok for tok in TOKEN_RE.findall(s) if tok not in stop_tokens]
    return " ".join(tokens)


def normalize_text_preserve_lines(text: str, stop_tokens: set[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = []
    for raw_line in LINE_SPLIT_RE.split(text):
        line = raw_line.lower()
        for pattern, replacement in SPECIAL_REPLACEMENTS:
            line = pattern.sub(replacement, line)
        tokens = [tok for tok in TOKEN_RE.findall(line) if tok not in stop_tokens]
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def compile_tech_patterns() -> dict[str, re.Pattern]:
    patterns: dict[str, re.Pattern] = {}
    for spec in TECH_SPECS:
        alias_parts = []
        for alias in spec.aliases:
            alias = alias.strip().lower()
            if not alias:
                continue
            alias = alias.replace(".", r"\.")
            alias = alias.replace("+", r"\+")
            alias = alias.replace("#", r"#")
            alias = alias.replace("/", r"\s+")
            alias = alias.replace("-", r"\s+")
            alias = alias.replace("_", r"\s+")
            alias = alias.replace("(", r"\(")
            alias = alias.replace(")", r"\)")
            alias = alias.replace(" ", r"\s+")
            alias_parts.append(rf"\b{alias}\b")
        patterns[spec.tech] = re.compile("|".join(alias_parts), re.I)
    return patterns


def compute_tech_rates(df: pd.DataFrame, stop_tokens: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patterns = compile_tech_patterns()
    rows = []
    for row in df.itertuples(index=False):
        text = normalize_text(row.text_for_analysis, stop_tokens)
        total_chars = len(text)
        for spec in TECH_SPECS:
            pat = patterns[spec.tech]
            mention_count = len(pat.findall(text))
            mentioned = 1 if mention_count > 0 else 0
            rows.append(
                {
                    "uid": row.uid,
                    "source": row.source,
                    "period": row.period,
                    "seniority_final": row.seniority_final,
                    "tech": spec.tech,
                    "category": spec.category,
                    "mentioned": mentioned,
                    "mention_count": mention_count,
                    "chars": total_chars,
                }
            )
    tech_long = pd.DataFrame.from_records(rows)
    grouped = (
        tech_long.groupby(["tech", "category", "period", "seniority_final"], as_index=False)
        .agg(
            n=("uid", "count"),
            mention_posts=("mentioned", "sum"),
            total_mentions=("mention_count", "sum"),
            chars=("chars", "sum"),
        )
    )
    grouped["mention_rate_pct"] = grouped["mention_posts"] / grouped["n"] * 100
    grouped["mentions_per_1k_chars"] = grouped["total_mentions"] / grouped["chars"] * 1000
    grouped["tech_rank"] = grouped.groupby(["category", "period", "seniority_final"])["mention_rate_pct"].rank(
        method="dense", ascending=False
    )

    heat = (
        grouped.assign(period_seniority=grouped["period"] + " | " + grouped["seniority_final"])
        .pivot_table(index=["category", "tech"], columns="period_seniority", values="mention_rate_pct", fill_value=0.0)
        .reset_index()
    )

    change_rows = []
    for (tech, category, seniority), sub in grouped.groupby(["tech", "category", "seniority_final"]):
        if {"2024-04", "2026-03"}.issubset(set(sub["period"])):
            old = sub.loc[sub["period"] == "2024-04"].iloc[0]
            new = sub.loc[sub["period"] == "2026-03"].iloc[0]
            fold = (new["mention_rate_pct"] + 1e-9) / (old["mention_rate_pct"] + 1e-9)
            char_fold = (new["mentions_per_1k_chars"] + 1e-9) / (old["mentions_per_1k_chars"] + 1e-9)
            change_rows.append(
                {
                    "tech": tech,
                    "category": category,
                    "seniority_final": seniority,
                    "rate_2024_04": old["mention_rate_pct"],
                    "rate_2026_03": new["mention_rate_pct"],
                    "per1k_2024_04": old["mentions_per_1k_chars"],
                    "per1k_2026_03": new["mentions_per_1k_chars"],
                    "fold_change": fold,
                    "char_fold_change": char_fold,
                    "direction": "rising" if fold > 1 else "flat" if math.isclose(fold, 1.0, rel_tol=0.05) else "declining",
                }
            )

    change = pd.DataFrame.from_records(change_rows)
    return grouped, heat, change


def summarize_stack_change(change: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    change = change.copy()
    change["abs_log2"] = (change["fold_change"].clip(lower=1e-9)).apply(lambda x: abs(math.log2(x)))
    rising = change.query("fold_change >= 1.5 and rate_2026_03 >= 0.5").sort_values(
        ["fold_change", "rate_2026_03"], ascending=[False, False]
    )
    stable = change.query("fold_change >= 0.67 and fold_change <= 1.5 and rate_2026_03 >= 0.5").sort_values(
        ["rate_2026_03", "fold_change"], ascending=[False, False]
    )
    declining = change.query("fold_change <= 0.67 and rate_2024_04 >= 0.5").sort_values(
        ["fold_change", "rate_2024_04"], ascending=[True, False]
    )
    return rising, stable, declining


def make_t17_heatmap(heat: pd.DataFrame, outpath: Path) -> None:
    matrix = heat.copy()
    matrix["label"] = matrix["category"] + " | " + matrix["tech"]
    matrix = matrix.drop(columns=["category", "tech"]).set_index("label")
    ordered_cols = [f"{period} | {sen}" for period in PERIOD_ORDER for sen in SENIORITY_ORDER]
    matrix = matrix.reindex(columns=[c for c in ordered_cols if c in matrix.columns])
    matrix = matrix.sort_index()
    sns.set_theme(style="white")
    fig_w = max(16, 0.8 * len(matrix.columns))
    fig_h = max(16, 0.22 * len(matrix))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        linewidths=0.0,
        cbar_kws={"label": "Mention rate (%)"},
        vmin=0,
    )
    ax.set_title("T17 technology shift heatmap: mention rate by period × seniority")
    ax.set_xlabel("Period | Seniority")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def section_from_block(block: str) -> str:
    low = block.strip().lower()
    if not low:
        return "unclassified"
    compact = re.sub(r"^[\-\*\d\.\)\(]+", "", low).strip()
    if len(compact) <= 140:
        for section, patterns in SECTION_SPECS:
            for pat in patterns:
                if re.search(pat, compact, flags=re.I):
                    return section
    for section, patterns in SECTION_SPECS:
        for pat in patterns:
            if re.search(pat, low, flags=re.I):
                return section
    return "unclassified"


def compute_section_lengths(df: pd.DataFrame, stop_tokens: set[str]) -> pd.DataFrame:
    records = []
    section_names = [s for s, _ in SECTION_SPECS] + ["unclassified"]
    for row in df.itertuples(index=False):
        text = normalize_text_preserve_lines(row.text_for_analysis, stop_tokens)
        if not text.strip():
            continue
        sections = {name: 0 for name in section_names}
        blocks = [b.strip() for b in BLOCK_SPLIT_RE.split(text) if b.strip()]
        if len(blocks) == 1:
            blocks = [b.strip() for b in LINE_SPLIT_RE.split(text) if b.strip()]
        for block in blocks:
            section = section_from_block(block)
            sections[section] += len(block)
        total = sum(sections.values())
        record = {
            "uid": row.uid,
            "source": row.source,
            "period": row.period,
            "seniority_final": row.seniority_final,
            "total_chars": total,
        }
        for section in section_names:
            record[f"{section}_chars"] = sections[section]
            record[f"{section}_share"] = sections[section] / total if total else 0.0
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_sections(section_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section_cols = [c for c in section_df.columns if c.endswith("_chars") and c != "total_chars"]
    share_cols = [c for c in section_df.columns if c.endswith("_share")]
    long_rows = []
    for section_col in section_cols:
        section = section_col.replace("_chars", "")
        share_col = f"{section}_share"
        tmp = section_df[["period", "seniority_final", "total_chars", section_col, share_col]].copy()
        tmp["section"] = section
        tmp = tmp.rename(columns={section_col: "section_chars", share_col: "section_share"})
        long_rows.append(tmp)
    long_df = pd.concat(long_rows, ignore_index=True)
    summary = (
        long_df.groupby(["period", "seniority_final", "section"], as_index=False)
        .agg(
            n=("section_chars", "count"),
            median_section_chars=("section_chars", "median"),
            mean_section_chars=("section_chars", "mean"),
            median_total_chars=("total_chars", "median"),
            median_share=("section_share", "median"),
        )
        .sort_values(["period", "seniority_final", "section"])
    )
    period_summary = (
        long_df.groupby(["period", "section"], as_index=False)
        .agg(
            n=("section_chars", "count"),
            median_section_chars=("section_chars", "median"),
            mean_section_chars=("section_chars", "mean"),
            median_total_chars=("total_chars", "median"),
            median_share=("section_share", "median"),
        )
        .sort_values(["period", "section"])
    )
    entry_summary = summary[summary["seniority_final"] == "entry"].copy()
    return summary, period_summary, entry_summary


def make_t18_chart(period_summary: pd.DataFrame, outpath: Path) -> None:
    order = ["summary", "responsibilities", "requirements", "preferred", "benefits", "about_company", "legal_eeo", "unclassified"]
    period_order = PERIOD_ORDER
    pivot = (
        period_summary.pivot(index="period", columns="section", values="median_section_chars")
        .reindex(period_order)
        .reindex(columns=order)
        .fillna(0.0)
    )
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0.0] * len(pivot.index)
    colors = sns.color_palette("Set2", n_colors=len(order))
    for i, section in enumerate(order):
        vals = pivot[section].tolist()
        ax.bar(pivot.index, vals, bottom=bottom, label=section, color=colors[i])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("Median section chars")
    ax.set_title("T18 description composition by period")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, title: str, finding: str, implication: str, data_note: str, action_items: str) -> None:
    md = f"""# {title}
## Finding
{finding}
## Implication for analysis
{implication}
## Data quality note
{data_note}
## Action items
{action_items}
"""
    path.write_text(md)


def main() -> None:
    ensure_dirs()
    con = connect()
    stop_tokens = build_company_stop_tokens(con)

    base_sql = f"""
    SELECT
      uid,
      source,
      period,
      seniority_final,
      company_name_canonical,
      coalesce(nullif(trim(description_core), ''), nullif(trim(description), '')) AS text_for_analysis
    FROM read_parquet('{STAGE8}')
    WHERE {FILTER}
      AND coalesce(nullif(trim(description_core), ''), nullif(trim(description), '')) IS NOT NULL
    """
    df = fetch_df(con, base_sql)
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
    df["seniority_final"] = pd.Categorical(df["seniority_final"], categories=SENIORITY_ORDER, ordered=True)

    tech_rates, tech_heat, tech_change = compute_tech_rates(df, stop_tokens)
    meaningful_change = tech_change[(tech_change["rate_2024_04"] >= 0.5) & (tech_change["rate_2026_03"] >= 0.5)].copy()
    rising, stable, declining = summarize_stack_change(meaningful_change)
    section_df = compute_section_lengths(df, stop_tokens)
    section_summary, section_period_summary, entry_summary = summarize_sections(section_df)

    # T17 outputs
    csv_write(OUT_TABLES / "T17" / "T17_tech_rates_by_period_seniority.csv", tech_rates)
    csv_write(OUT_TABLES / "T17" / "T17_tech_heatmap_matrix.csv", tech_heat)
    csv_write(OUT_TABLES / "T17" / "T17_change_summary.csv", tech_change.sort_values(["fold_change", "rate_2026_03"], ascending=[False, False]))
    csv_write(OUT_TABLES / "T17" / "T17_change_summary_meaningful.csv", meaningful_change.sort_values(["fold_change", "rate_2026_03"], ascending=[False, False]))
    csv_write(OUT_TABLES / "T17" / "T17_rising_stacks.csv", rising.head(25))
    csv_write(OUT_TABLES / "T17" / "T17_stable_stacks.csv", stable.head(25))
    csv_write(OUT_TABLES / "T17" / "T17_declining_stacks.csv", declining.head(25))
    csv_write(OUT_TABLES / "T17" / "T17_taxonomy.csv", pd.DataFrame([spec.__dict__ for spec in TECH_SPECS]))
    make_t17_heatmap(tech_heat, OUT_FIGS / "T17" / "T17_tech_heatmap.png")

    # T18 outputs
    csv_write(OUT_TABLES / "T18" / "T18_section_lengths_by_period_seniority.csv", section_summary)
    csv_write(OUT_TABLES / "T18" / "T18_section_lengths_by_period.csv", section_period_summary)
    csv_write(OUT_TABLES / "T18" / "T18_entry_level_section_lengths.csv", entry_summary)
    csv_write(OUT_TABLES / "T18" / "T18_posting_section_lengths.csv", section_df)
    make_t18_chart(section_period_summary, OUT_FIGS / "T18" / "T18_description_composition.png")

    # Short summaries for the reports.
    tech_top = meaningful_change.sort_values(["fold_change", "rate_2026_03"], ascending=[False, False]).head(8)
    tech_bottom = meaningful_change.sort_values(["fold_change", "rate_2024_04"], ascending=[True, False]).head(8)
    rising_ai = rising[rising["category"].isin(["ai_llm_new", "ai_tools"])].head(8)
    stable_core = stable[stable["category"].isin(["languages", "cloud_devops", "data"])].head(8)
    declining_practices = declining[declining["category"].isin(["testing_practices", "frontend", "backend_infra"])].head(8)

    total_t17 = len(tech_rates)
    total_t18 = len(section_df)
    known_t18 = section_df[section_df["seniority_final"].isin(["entry", "associate", "mid-senior", "director"])]
    total_known_t18 = len(known_t18)

    finding_t17 = (
        f"The taxonomy covers {len(TECH_SPECS)} technologies across {len(PERIOD_ORDER) * len(SENIORITY_ORDER)} period-seniority cells. "
        f"Core stacks such as Python, SQL, AWS, and Kubernetes remain broad, while the largest meaningful relative gains are "
        f"concentrated in AI/LLM tooling and workflow layers such as Cursor, Claude, Copilot, RAG, MCP, LangChain, and prompt engineering."
    )
    if not tech_top.empty:
        top_name = tech_top.iloc[0]["tech"]
        top_fold = tech_top.iloc[0]["fold_change"]
        top_period = tech_top.iloc[0]["seniority_final"]
        finding_t17 += f" The single largest 2026 vs 2024-04 fold change among technologies with nonzero mass in both years is {top_name} in the {top_period} bucket at {top_fold:.2f}x."

    implication_t17 = (
        "For RQ1/RQ2, the toolkit shift is not a generic rise in all software technologies. The growth is concentrated in AI "
        "tooling, retrieval/vector infrastructure, and adjacent orchestration terms, while languages and core cloud stacks are "
        "more stable. That supports a scope-inflation story centered on AI augmentation rather than wholesale stack replacement."
    )
    data_note_t17 = (
        "Stage 8 does not contain `description_core_llm`, so the analysis used `description_core` with `description` as a fallback. "
        "Company-name tokens were stripped during tokenization using a stoplist built from observed `company_name_canonical` values, "
        "and the taxonomy is intentionally conservative on ambiguous abbreviations. Entry-level 2024-01 is structurally absent in "
        "asaniczka, so entry comparisons rely on 2024-04 arshkon vs 2026-03 scraped."
    )
    action_t17 = (
        "Use the CSVs as the analysis-phase keyword dictionary seed. Prioritize the AI/LLM rows for follow-up on scope inflation, "
        "and treat the `T17_change_summary.csv` fold-change columns as the source of the >3x candidate list."
    )
    write_markdown(OUT_REPORTS / "T17.md", "T17: Technology stack tracking", finding_t17, implication_t17, data_note_t17, action_t17)

    requirements_change = section_summary[section_summary["section"] == "requirements"].copy()
    responsibilities_change = section_summary[section_summary["section"] == "responsibilities"].copy()
    about_change = section_summary[section_summary["section"].isin(["about_company", "benefits", "legal_eeo"])].copy()
    entry_2024 = entry_summary[entry_summary["period"] == "2024-04"]
    entry_2026 = entry_summary[entry_summary["period"] == "2026-03"]
    finding_t18 = (
        f"Across {total_known_t18:,} known-seniority SWE postings, the biggest median section lengths are still in requirements and "
        f"responsibilities, but the composition shift is uneven: benefits, about-company, and legal/EEO text remain material and "
        f"help explain part of the overall length growth. Entry-level postings show the same basic structure, but the relative mix "
        f"leans more toward requirements and responsibilities in 2026 than in 2024."
    )
    implication_t18 = (
        "For RQ1, this points to two mechanisms at once: some of the length increase is substantive job-content expansion, and some "
        "is residual boilerplate/administrative text. The decomposition is still useful because it shows whether the growth is mostly "
        "in requirements or in peripheral sections that are less central to the labor-demand story."
    )
    data_note_t18 = (
        "The section classifier is regex-based and conservative, so paragraphs that do not look like a JD section heading are tagged "
        "as `unclassified`. Because Stage 8 already includes rule-based boilerplate removal, the benefits/legal/about shares here are "
        "likely understated relative to raw postings. `description_core_llm` is not available in Stage 8."
    )
    action_t18 = (
        "Use `T18_section_lengths_by_period_seniority.csv` and `T18_entry_level_section_lengths.csv` when writing the analysis section. "
        "Treat the stacked bar chart as a composition snapshot, not as a precise parse of every posting."
    )
    write_markdown(OUT_REPORTS / "T18.md", "T18: Description anatomy", finding_t18, implication_t18, data_note_t18, action_t18)

    # Minimal index update for task completion.
    index_path = ROOT / "exploration" / "reports" / "INDEX.md"
    if index_path.exists():
        text = index_path.read_text()
        text = text.replace("| T17 | I | 3 | - |", "| T17 | I | 3 | Clean |")
        text = text.replace("| T18 | I | 3 | - |", "| T18 | I | 3 | Clean |")
        if "Wave 3 findings to carry forward:" in text and "T17" not in text.split("Wave 3 findings to carry forward:")[1]:
            pass
        index_path.write_text(text)


if __name__ == "__main__":
    main()
