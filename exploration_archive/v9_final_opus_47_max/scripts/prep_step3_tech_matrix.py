"""Wave 1.5 Agent Prep - Step 3: Technology mention binary matrix.

Builds a binary matrix of ~100-120 tech mentions across SWE LinkedIn postings.
Uses raw `description` (boilerplate-insensitive) with backslash-escape cleanup.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = Path("exploration/artifacts/shared")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "swe_tech_matrix.parquet"

# Regex to strip backslash-escaped special chars from scraped markdown
ESCAPE_RE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")


def unescape(text: str) -> str:
    return ESCAPE_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Technology taxonomy
# Each entry: column_name -> regex pattern (case-insensitive). Use \b word
# boundaries where possible and negative lookarounds to avoid common false
# positives (e.g., Go matching "google").
# ---------------------------------------------------------------------------

TECH_PATTERNS: dict[str, str] = {
    # -- Languages ---------------------------------------------------------
    "python": r"\bpython\b",
    "java": r"\bjava\b(?!\s*script)",
    "javascript": r"\b(?:javascript|java\s*script|js)\b",
    "typescript": r"\b(?:typescript|type\s*script|\bts\b)\b",
    "go_lang": r"\bgo(?:lang)?\b",
    "rust": r"\brust\b",
    "c_plus_plus": r"\bc\+\+",
    "c_lang": r"\bc\b(?!\s*[a-z0-9+#\-])",
    "c_sharp": r"\bc\#",
    "ruby": r"\bruby\b",
    "kotlin": r"\bkotlin\b",
    "swift": r"\bswift\b(?!\s*(?:code|current|action))",
    "scala": r"\bscala\b",
    "php": r"\bphp\b",
    "r_lang": r"\b(?:r\s*(?:language|programming)|programming\s*in\s*r|using\s+r\b|r\s*/\s*python|r\s*,\s*python|python\s*,\s*r)\b",
    "sql": r"\bsql\b",
    "bash": r"\bbash\b",
    "perl": r"\bperl\b",
    "elixir": r"\belixir\b",
    "objective_c": r"\bobjective[- ]c\b",
    # -- Frontend frameworks ----------------------------------------------
    "react": r"\breact(?:\.?js)?\b",
    "angular": r"\bangular(?:\.?js)?\b",
    "vue": r"\bvue(?:\.?js)?\b",
    "nextjs": r"\bnext\.?js\b",
    "nuxt": r"\bnuxt(?:\.?js)?\b",
    "svelte": r"\bsvelte\b",
    "ember": r"\bember(?:\.?js)?\b",
    "jquery": r"\bjquery\b",
    # -- Backend frameworks -----------------------------------------------
    "nodejs": r"\bnode(?:\.?js)?\b",
    "express": r"\bexpress(?:\.?js)?\b",
    "django": r"\bdjango\b",
    "flask": r"\bflask\b",
    "fastapi": r"\bfastapi\b",
    "spring": r"\bspring(?:\s*(?:boot|framework|cloud))?\b",
    "dot_net": r"\.net\b|\bdotnet\b|\b\.net\s*(?:core|framework)\b|\basp\.net\b",
    "rails": r"\b(?:ruby\s+on\s+rails|rails)\b",
    "laravel": r"\blaravel\b",
    "graphql": r"\bgraphql\b",
    "rest_api": r"\brest(?:ful)?\s*(?:api|apis|services?)\b",
    # -- Cloud ------------------------------------------------------------
    "aws": r"\b(?:aws|amazon\s*web\s*services)\b",
    "azure": r"\b(?:microsoft\s+)?azure\b",
    "gcp": r"\b(?:gcp|google\s*cloud(?:\s*platform)?)\b",
    "cloudflare": r"\bcloud\s*flare\b|\bcloudflare\b",
    "heroku": r"\bheroku\b",
    "digital_ocean": r"\bdigital\s*ocean\b|\bdigitalocean\b",
    # -- Orchestration / DevOps -------------------------------------------
    "kubernetes": r"\b(?:kubernetes|k8s)\b",
    "docker": r"\bdocker\b",
    "terraform": r"\bterraform\b",
    "ansible": r"\bansible\b",
    "helm": r"\bhelm(?:\s*charts?)?\b",
    "argocd": r"\bargo\s*cd\b|\bargocd\b",
    "puppet": r"\bpuppet\b",
    "chef": r"\bchef\b(?!\s*(?:de|d'))",
    # -- CI/CD ------------------------------------------------------------
    "jenkins": r"\bjenkins\b",
    "github_actions": r"\bgithub\s*actions?\b",
    "circleci": r"\bcircle\s*ci\b|\bcircleci\b",
    "gitlab_ci": r"\bgitlab(?:\s*ci)?\b",
    "buildkite": r"\bbuild\s*kite\b|\bbuildkite\b",
    "travis_ci": r"\btravis\s*ci\b",
    # -- Databases --------------------------------------------------------
    "postgresql": r"\b(?:postgres(?:ql)?|pgsql)\b",
    "mysql": r"\bmysql\b",
    "mongodb": r"\bmongo\s*db\b|\bmongodb\b|\bmongo\b",
    "redis": r"\bredis\b",
    "cassandra": r"\bcassandra\b",
    "dynamodb": r"\bdynamo\s*db\b|\bdynamodb\b",
    "snowflake": r"\bsnowflake\b",
    "bigquery": r"\bbig\s*query\b|\bbigquery\b",
    "oracle_db": r"\boracle\s+(?:database|db|sql)\b",
    "sqlite": r"\bsqlite\b",
    "sql_server": r"\b(?:sql\s*server|mssql|ms\s*sql)\b",
    # -- Data pipelines ---------------------------------------------------
    "kafka": r"\bkafka\b",
    "spark": r"\b(?:apache\s+)?spark\b",
    "airflow": r"\b(?:apache\s+)?airflow\b",
    "dbt": r"\bdbt\b",
    "databricks": r"\bdatabricks\b",
    "elasticsearch": r"\belastic\s*search\b|\belasticsearch\b|\belk\s*stack\b",
    "flink": r"\b(?:apache\s+)?flink\b",
    "hadoop": r"\bhadoop\b",
    "beam": r"\b(?:apache\s+)?beam\b",
    # -- Traditional ML ---------------------------------------------------
    "tensorflow": r"\btensor\s*flow\b|\btensorflow\b",
    "pytorch": r"\bpy\s*torch\b|\bpytorch\b",
    "scikit_learn": r"\bscikit[- ]learn\b|\bsklearn\b",
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "jupyter": r"\bjupyter\b",
    "mlflow": r"\bml\s*flow\b|\bmlflow\b",
    "xgboost": r"\bxg\s*boost\b|\bxgboost\b",
    "keras": r"\bkeras\b",
    # -- LLM era ----------------------------------------------------------
    "langchain": r"\blang\s*chain\b|\blangchain\b",
    "llamaindex": r"\bllama\s*index\b|\bllamaindex\b",
    "rag": r"\brag\b(?!\s*(?:on|tag))|\bretrieval[- ]augmented[- ]generation\b",
    "vector_database": r"\bvector\s*(?:database|db|store)\b",
    "pinecone": r"\bpinecone\b",
    "weaviate": r"\bweaviate\b",
    "chroma": r"\bchroma(?:db)?\b",
    "hugging_face": r"\bhugging\s*face\b|\bhuggingface\b",
    "openai_api": r"\bopen\s*ai\s*(?:api|platform)\b|\bopenai\s*(?:api|platform|sdk)\b",
    "claude_api": r"\b(?:anthropic|claude)\s*(?:api|sdk)\b",
    "anthropic": r"\banthropic\b",
    "gemini": r"\bgemini(?:\s*(?:api|pro|1\.5))?\b",
    "prompt_engineering": r"\bprompt[- ]engineer(?:ing)?\b",
    "fine_tuning": r"\bfine[- ]tun(?:e|ing|ed)\b",
    "mcp": r"\bmcp\b|\bmodel\s+context\s+protocol\b",
    "llm": r"\b(?:llm|llms|large\s*language\s*models?)\b",
    "ai_agent": r"\b(?:ai|llm)\s*agents?\b|\bagentic\b",
    # -- AI tools ---------------------------------------------------------
    "copilot": r"\b(?:github\s+)?copilot\b",
    "cursor_tool": r"\bcursor\s*(?:ai|ide|editor)?\b",
    "chatgpt": r"\bchat\s*gpt\b|\bchatgpt\b",
    "claude_tool": r"\bclaude(?:\s+code|\s+desktop)\b",
    "codex": r"\bcodex\b",
    "tabnine": r"\btab\s*nine\b|\btabnine\b",
    "gpt_model": r"\bgpt[- ]?[0-9]\b|\bgpt[- ]?\d\.\d\b",
    # -- Testing ----------------------------------------------------------
    "jest": r"\bjest\b",
    "pytest": r"\bpy\s*test\b|\bpytest\b",
    "selenium": r"\bselenium\b",
    "cypress": r"\bcypress\b",
    "playwright": r"\bplaywright\b",
    "junit": r"\bj\s*unit\b|\bjunit\b",
    "mocha": r"\bmocha\b",
    # -- Observability ----------------------------------------------------
    "datadog": r"\bdata\s*dog\b|\bdatadog\b",
    "new_relic": r"\bnew\s*relic\b",
    "pagerduty": r"\bpager\s*duty\b|\bpagerduty\b",
    "grafana": r"\bgrafana\b",
    "prometheus": r"\bprometheus\b",
    "splunk": r"\bsplunk\b",
    "sentry": r"\bsentry\b",
    # -- Practices --------------------------------------------------------
    "agile": r"\bagile\b",
    "scrum": r"\bscrum\b",
    "tdd": r"\btdd\b|\btest[- ]driven\s+development\b",
    "bdd": r"\bbdd\b|\bbehavior[- ]driven\b",
    "ddd": r"\bddd\b|\bdomain[- ]driven\s+design\b",
    "microservices": r"\bmicro\s*services?\b|\bmicroservices?\b",
    "ci_cd": r"\bci\s*/\s*cd\b|\bcicd\b|\bcontinuous\s+(?:integration|delivery|deployment)\b",
    "event_driven": r"\bevent[- ]driven\b",
    "serverless": r"\bserverless\b|\baws\s*lambda\b|\bazure\s*functions?\b",
}


def _check_tests() -> None:
    """TDD-style inline assertions for regex behavior."""
    def has(name: str, s: str) -> bool:
        return re.search(TECH_PATTERNS[name], s, flags=re.IGNORECASE) is not None

    # Required positives
    assert has("c_plus_plus", "We use C++ and Java"), "C++ must match"
    assert has("c_sharp", "Strong C# skills"), "C# must match"
    assert has("dot_net", "Experience with .NET Core"), ".NET must match"
    assert has("nextjs", "Using Next.js and React"), "Next.js must match"
    assert has("nodejs", "Node.js experience"), "Node.js must match"
    # Required negatives
    assert not has("go_lang", "good company"), "Go must NOT match 'good'"
    assert not has("go_lang", "google cloud"), "Go must NOT match 'google'"
    assert not has("r_lang", "RESTful API"), "R must NOT match random R"
    assert not has("r_lang", "React framework"), "R must NOT match inside React"
    assert not has("java", "JavaScript ES6"), "Java must NOT match JavaScript"
    assert has("java", "Java 17 with Spring Boot"), "Java must match 'Java 17'"
    # Escape-fix verification
    raw_scraped = r"Strong \+\+ knowledge of C\+\+ and C\#"
    unesc = unescape(raw_scraped)
    assert "C++" in unesc, "unescape must produce 'C++'"
    assert has("c_plus_plus", unesc), "post-escape C++ must match"
    assert has("c_sharp", unesc), "post-escape C# must match"
    # RAG tests
    assert has("rag", "Using RAG for retrieval"), "RAG must match"
    assert not has("rag", "a rag on the floor"), "rag (no uppercase, but..)"
    # Actually our regex is case-insensitive; we look at \brag\b with neg lookahead
    # for "on" and "tag". "a rag on" WILL still match because the negative lookahead
    # is just for `rag on` attached. Let's accept: we've added (?!\s*(?:on|tag)).
    # Let's verify carefully:
    # "a rag on the floor" -> "\brag\b(?!\s*(?:on|tag))" -> after "rag", next is " on" -> excluded
    assert not has("rag", "a rag on the floor"), "rag (small-r) with 'on' must NOT match"
    print("[step3]  tech regex assertions passed")


def build_matrix() -> None:
    t0 = time.time()
    _check_tests()
    con = duckdb.connect()
    print("[step3] loading SWE LinkedIn raw descriptions")
    df = con.execute(
        """
        SELECT uid, description, source
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag='ok'
        """
    ).df()
    print(f"[step3] rows: {len(df)}")

    tech_names = list(TECH_PATTERNS.keys())
    # Pre-compile patterns with IGNORECASE
    compiled = {name: re.compile(pat, flags=re.IGNORECASE) for name, pat in TECH_PATTERNS.items()}

    # Output dict: tech -> list of bool
    n = len(df)
    out_cols: dict[str, list[bool]] = {name: [False] * n for name in tech_names}
    # Also capture pre-escape C++ rate on scraped for diagnostic
    pre_escape_cpp_scraped = 0
    pre_escape_csharp_scraped = 0
    pre_escape_dotnet_scraped = 0
    post_escape_cpp_scraped = 0
    post_escape_csharp_scraped = 0
    post_escape_dotnet_scraped = 0
    total_scraped = 0

    report_every = 5000
    desc_vals = df["description"].values
    src_vals = df["source"].values
    for i in range(n):
        raw = desc_vals[i]
        if not isinstance(raw, str) or not raw:
            continue
        # Compute diagnostic rates on scraped BEFORE applying escape fix
        is_scraped = src_vals[i] == "scraped"
        if is_scraped:
            total_scraped += 1
            if compiled["c_plus_plus"].search(raw):
                pre_escape_cpp_scraped += 1
            if compiled["c_sharp"].search(raw):
                pre_escape_csharp_scraped += 1
            if compiled["dot_net"].search(raw):
                pre_escape_dotnet_scraped += 1

        txt = unescape(raw)

        if is_scraped:
            if compiled["c_plus_plus"].search(txt):
                post_escape_cpp_scraped += 1
            if compiled["c_sharp"].search(txt):
                post_escape_csharp_scraped += 1
            if compiled["dot_net"].search(txt):
                post_escape_dotnet_scraped += 1

        for name, patt in compiled.items():
            if patt.search(txt):
                out_cols[name][i] = True

        if (i + 1) % report_every == 0:
            print(f"[step3]  processed {i+1}/{n}")

    # Build output table
    uids = df["uid"].tolist()
    cols = {"uid": uids}
    for name in tech_names:
        cols[name] = out_cols[name]
    table = pa.table(cols)
    pq.write_table(table, OUT_PATH, compression="zstd")
    print(f"[step3] wrote {len(uids)} rows x {len(tech_names)} techs -> {OUT_PATH}")

    # Save diagnostic numbers to a side file
    diag_path = OUT_DIR / "tech_escape_diagnostic.txt"
    diag_path.write_text(
        "metric\tpre_escape_rate_scraped\tpost_escape_rate_scraped\n"
        f"c_plus_plus\t{pre_escape_cpp_scraped/total_scraped:.4f}\t{post_escape_cpp_scraped/total_scraped:.4f}\n"
        f"c_sharp\t{pre_escape_csharp_scraped/total_scraped:.4f}\t{post_escape_csharp_scraped/total_scraped:.4f}\n"
        f"dot_net\t{pre_escape_dotnet_scraped/total_scraped:.4f}\t{post_escape_dotnet_scraped/total_scraped:.4f}\n"
        f"n_scraped\t{total_scraped}\n"
    )
    elapsed = time.time() - t0
    print(f"[step3] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    build_matrix()
