"""Wave 1.5 Agent Prep — Step 3: technology mention binary matrix.

Binary matrix, one row per SWE LinkedIn posting (default filter),
columns = uid + one boolean per technology. ~100-120 regex patterns.

Scanned on `description_cleaned` (which already lowercases and preserves
+/#/./- characters inside tokens like c++, c#, node.js, ci/cd).

Output: exploration/artifacts/shared/swe_tech_matrix.parquet
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CLEANED = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
)
OUT = Path(
    "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet"
)

# Patterns apply to the *cleaned* text (already lowercased, tokens include +/#/./-).
# Use word boundaries that work with custom punctuation by wrapping with
# spaces or start/end anchors. We therefore test against padded strings.

# Each entry: (column_name, regex_pattern)
TECH_PATTERNS: list[tuple[str, str]] = [
    # Languages
    ("python", r"(?:^|\s)python(?:$|\s)"),
    ("java", r"(?:^|\s)java(?:$|\s)"),  # avoid javascript by using word-like check
    ("javascript", r"(?:^|\s)(?:javascript|js)(?:$|\s)"),
    ("typescript", r"(?:^|\s)(?:typescript|ts)(?:$|\s)"),
    ("go_lang", r"(?:^|\s)(?:golang|go-lang)(?:$|\s)"),  # 'go' alone is too noisy
    ("rust", r"(?:^|\s)rust(?:$|\s)"),
    ("c_plus_plus", r"(?:^|\s)c\+\+(?:$|\s)"),
    ("c_sharp", r"(?:^|\s)c#(?:$|\s)"),
    ("c_lang", r"(?:^|\s)c(?:$|\s)"),  # will be noisy but kept per spec
    ("ruby", r"(?:^|\s)ruby(?:$|\s)"),
    ("kotlin", r"(?:^|\s)kotlin(?:$|\s)"),
    ("swift", r"(?:^|\s)swift(?:$|\s)"),
    ("scala", r"(?:^|\s)scala(?:$|\s)"),
    ("php", r"(?:^|\s)php(?:$|\s)"),
    ("r_lang", r"(?:^|\s)r(?:$|\s)"),  # noisy
    ("perl", r"(?:^|\s)perl(?:$|\s)"),
    ("bash", r"(?:^|\s)bash(?:$|\s)"),
    ("shell", r"(?:^|\s)shell(?:$|\s)"),
    ("sql_lang", r"(?:^|\s)sql(?:$|\s)"),
    ("matlab", r"(?:^|\s)matlab(?:$|\s)"),
    ("objective_c", r"(?:^|\s)objective-c(?:$|\s)"),
    ("dart", r"(?:^|\s)dart(?:$|\s)"),
    ("haskell", r"(?:^|\s)haskell(?:$|\s)"),
    ("elixir", r"(?:^|\s)elixir(?:$|\s)"),
    ("solidity", r"(?:^|\s)solidity(?:$|\s)"),

    # Frontend
    ("react", r"(?:^|\s)(?:react|reactjs|react.js)(?:$|\s)"),
    ("angular", r"(?:^|\s)(?:angular|angularjs)(?:$|\s)"),
    ("vue", r"(?:^|\s)(?:vue|vuejs|vue.js)(?:$|\s)"),
    ("nextjs", r"(?:^|\s)(?:nextjs|next.js)(?:$|\s)"),
    ("svelte", r"(?:^|\s)svelte(?:$|\s)"),
    ("jquery", r"(?:^|\s)jquery(?:$|\s)"),
    ("html_css", r"(?:^|\s)(?:html|html5|css|css3)(?:$|\s)"),
    ("tailwind", r"(?:^|\s)tailwind(?:$|\s)"),
    ("webpack", r"(?:^|\s)webpack(?:$|\s)"),

    # Backend frameworks
    ("nodejs", r"(?:^|\s)(?:nodejs|node.js|node)(?:$|\s)"),
    ("django", r"(?:^|\s)django(?:$|\s)"),
    ("flask", r"(?:^|\s)flask(?:$|\s)"),
    ("spring", r"(?:^|\s)(?:spring|springboot|spring-boot)(?:$|\s)"),
    ("dotnet", r"(?:^|\s)(?:\.net|net|dotnet)(?:$|\s)"),
    ("rails", r"(?:^|\s)(?:rails|ruby-on-rails)(?:$|\s)"),
    ("fastapi", r"(?:^|\s)fastapi(?:$|\s)"),
    ("express", r"(?:^|\s)(?:express|expressjs)(?:$|\s)"),
    ("graphql", r"(?:^|\s)graphql(?:$|\s)"),
    ("rest_api", r"(?:^|\s)(?:rest|restful)(?:$|\s)"),
    ("grpc", r"(?:^|\s)grpc(?:$|\s)"),

    # Cloud / DevOps
    ("aws", r"(?:^|\s)(?:aws|amazon-web-services)(?:$|\s)"),
    ("azure", r"(?:^|\s)azure(?:$|\s)"),
    ("gcp", r"(?:^|\s)(?:gcp|google-cloud)(?:$|\s)"),
    ("kubernetes", r"(?:^|\s)(?:kubernetes|k8s)(?:$|\s)"),
    ("docker", r"(?:^|\s)docker(?:$|\s)"),
    ("terraform", r"(?:^|\s)terraform(?:$|\s)"),
    ("ansible", r"(?:^|\s)ansible(?:$|\s)"),
    ("cicd", r"(?:^|\s)(?:ci/cd|cicd|ci-cd)(?:$|\s)"),
    ("jenkins", r"(?:^|\s)jenkins(?:$|\s)"),
    ("github_actions", r"(?:^|\s)github-actions(?:$|\s)"),
    ("argocd", r"(?:^|\s)argocd(?:$|\s)"),
    ("linux", r"(?:^|\s)linux(?:$|\s)"),
    ("git", r"(?:^|\s)git(?:$|\s)"),
    ("helm", r"(?:^|\s)helm(?:$|\s)"),
    ("serverless", r"(?:^|\s)serverless(?:$|\s)"),
    ("lambda", r"(?:^|\s)lambda(?:$|\s)"),

    # Data / storage
    ("postgres", r"(?:^|\s)(?:postgres|postgresql)(?:$|\s)"),
    ("mysql", r"(?:^|\s)mysql(?:$|\s)"),
    ("mongodb", r"(?:^|\s)mongodb(?:$|\s)"),
    ("redis", r"(?:^|\s)redis(?:$|\s)"),
    ("kafka", r"(?:^|\s)kafka(?:$|\s)"),
    ("spark", r"(?:^|\s)spark(?:$|\s)"),
    ("hadoop", r"(?:^|\s)hadoop(?:$|\s)"),
    ("snowflake", r"(?:^|\s)snowflake(?:$|\s)"),
    ("databricks", r"(?:^|\s)databricks(?:$|\s)"),
    ("dbt", r"(?:^|\s)dbt(?:$|\s)"),
    ("elasticsearch", r"(?:^|\s)(?:elasticsearch|elastic-search)(?:$|\s)"),
    ("airflow", r"(?:^|\s)airflow(?:$|\s)"),
    ("bigquery", r"(?:^|\s)bigquery(?:$|\s)"),
    ("redshift", r"(?:^|\s)redshift(?:$|\s)"),
    ("cassandra", r"(?:^|\s)cassandra(?:$|\s)"),
    ("dynamodb", r"(?:^|\s)dynamodb(?:$|\s)"),

    # Traditional AI/ML
    ("tensorflow", r"(?:^|\s)tensorflow(?:$|\s)"),
    ("pytorch", r"(?:^|\s)pytorch(?:$|\s)"),
    ("scikit_learn", r"(?:^|\s)(?:scikit-learn|sklearn)(?:$|\s)"),
    ("pandas", r"(?:^|\s)pandas(?:$|\s)"),
    ("numpy", r"(?:^|\s)numpy(?:$|\s)"),
    ("jupyter", r"(?:^|\s)jupyter(?:$|\s)"),
    ("keras", r"(?:^|\s)keras(?:$|\s)"),
    ("xgboost", r"(?:^|\s)xgboost(?:$|\s)"),
    ("machine_learning", r"(?:^|\s)(?:machine-learning|ml)(?:$|\s)"),
    ("deep_learning", r"(?:^|\s)deep-learning(?:$|\s)"),
    ("nlp", r"(?:^|\s)nlp(?:$|\s)"),
    ("computer_vision", r"(?:^|\s)computer-vision(?:$|\s)"),

    # New AI / LLM
    ("llm", r"(?:^|\s)(?:llm|llms)(?:$|\s)"),
    ("langchain", r"(?:^|\s)langchain(?:$|\s)"),
    ("langgraph", r"(?:^|\s)langgraph(?:$|\s)"),
    ("rag", r"(?:^|\s)rag(?:$|\s)"),
    ("vector_db", r"(?:^|\s)(?:vector-db|vector-database)(?:$|\s)"),
    ("pinecone", r"(?:^|\s)pinecone(?:$|\s)"),
    ("chromadb", r"(?:^|\s)(?:chromadb|chroma)(?:$|\s)"),
    ("huggingface", r"(?:^|\s)(?:huggingface|hugging-face)(?:$|\s)"),
    ("openai_api", r"(?:^|\s)openai(?:$|\s)"),
    ("claude_api", r"(?:^|\s)anthropic(?:$|\s)"),
    ("prompt_engineering", r"(?:^|\s)prompt-engineering(?:$|\s)"),
    ("fine_tuning", r"(?:^|\s)(?:fine-tuning|finetuning)(?:$|\s)"),
    ("mcp", r"(?:^|\s)mcp(?:$|\s)"),
    ("agents_framework", r"(?:^|\s)(?:agent|agents|agentic)(?:$|\s)"),
    ("gpt", r"(?:^|\s)(?:gpt|gpt-4|gpt-3)(?:$|\s)"),
    ("transformer_arch", r"(?:^|\s)transformer(?:$|\s)"),
    ("embedding", r"(?:^|\s)embeddings?(?:$|\s)"),

    # AI tools (dev-facing)
    ("copilot", r"(?:^|\s)copilot(?:$|\s)"),
    ("cursor_tool", r"(?:^|\s)cursor(?:$|\s)"),
    ("chatgpt", r"(?:^|\s)chatgpt(?:$|\s)"),
    ("claude_tool", r"(?:^|\s)claude(?:$|\s)"),
    ("gemini_tool", r"(?:^|\s)gemini(?:$|\s)"),
    ("codex_tool", r"(?:^|\s)codex(?:$|\s)"),

    # Testing
    ("jest", r"(?:^|\s)jest(?:$|\s)"),
    ("pytest", r"(?:^|\s)pytest(?:$|\s)"),
    ("selenium", r"(?:^|\s)selenium(?:$|\s)"),
    ("cypress", r"(?:^|\s)cypress(?:$|\s)"),
    ("junit", r"(?:^|\s)junit(?:$|\s)"),
    ("playwright", r"(?:^|\s)playwright(?:$|\s)"),

    # Practices
    ("agile", r"(?:^|\s)agile(?:$|\s)"),
    ("scrum", r"(?:^|\s)scrum(?:$|\s)"),
    ("tdd", r"(?:^|\s)tdd(?:$|\s)"),
    ("kanban", r"(?:^|\s)kanban(?:$|\s)"),
    ("microservices", r"(?:^|\s)microservices?(?:$|\s)"),
]


def test_patterns() -> None:
    """Inline assertions for tricky patterns."""
    compiled = {name: re.compile(pat) for name, pat in TECH_PATTERNS}

    def pad(s: str) -> str:
        return " " + s + " "

    # C++
    assert compiled["c_plus_plus"].search(pad("c++ developer needed")) is not None
    assert compiled["c_plus_plus"].search(pad("python engineer")) is None
    # C#
    assert compiled["c_sharp"].search(pad("c# .net stack")) is not None
    assert compiled["c_sharp"].search(pad("python")) is None
    # .NET / net (our tokenizer turned .NET into 'net')
    assert compiled["dotnet"].search(pad("net developer")) is not None
    # CI/CD
    assert compiled["cicd"].search(pad("ci/cd pipeline")) is not None
    # Node.js (tokenizer keeps the dot)
    assert compiled["nodejs"].search(pad("node.js backend")) is not None
    assert compiled["nodejs"].search(pad("node backend")) is not None
    # Python
    assert compiled["python"].search(pad("python django flask")) is not None
    # Java vs JavaScript — ensure 'java' does not match inside 'javascript'
    assert compiled["java"].search(pad("java spring boot")) is not None
    assert compiled["java"].search(pad("javascript frontend")) is None
    # JS alias
    assert compiled["javascript"].search(pad("js html css")) is not None
    # Kubernetes
    assert compiled["kubernetes"].search(pad("k8s cluster")) is not None
    # REST
    assert compiled["rest_api"].search(pad("rest api")) is not None
    # LLM
    assert compiled["llm"].search(pad("llm openai")) is not None
    # Scala
    assert compiled["scala"].search(pad("scala spark")) is not None
    # Do not match scalable as scala
    assert compiled["scala"].search(pad("scalable systems")) is None
    # C alone would match, expected noisy
    assert compiled["c_lang"].search(pad("c c++")) is not None
    # go lang should require golang not bare 'go'
    assert compiled["go_lang"].search(pad("golang rust")) is not None
    assert compiled["go_lang"].search(pad("we go fast")) is None
    print("Tech pattern assertions PASSED.")


def main() -> None:
    t0 = time.time()
    test_patterns()

    compiled = [(name, re.compile(pat)) for name, pat in TECH_PATTERNS]

    tbl = pq.read_table(CLEANED, columns=["uid", "description_cleaned"])
    uids = tbl.column("uid").to_pylist()
    texts = tbl.column("description_cleaned").to_pylist()
    n = len(uids)
    print(f"Scanning {n:,} rows x {len(compiled)} technologies")

    # Build dict of column -> list[bool]
    cols = {name: [False] * n for name, _ in compiled}
    for i, txt in enumerate(texts):
        padded = " " + (txt or "") + " "
        for name, rx in compiled:
            if rx.search(padded):
                cols[name][i] = True
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1:,}/{n:,}")

    arrow_cols = {"uid": pa.array(uids, type=pa.string())}
    for name, _ in compiled:
        arrow_cols[name] = pa.array(cols[name], type=pa.bool_())
    out_tbl = pa.table(arrow_cols)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, OUT, compression="zstd")
    print(f"Wrote {OUT} with {out_tbl.num_columns - 1} tech cols, {n:,} rows")

    # Top-10 most prevalent
    prevalence = []
    for name, _ in compiled:
        prevalence.append((name, sum(cols[name])))
    prevalence.sort(key=lambda x: -x[1])
    print("\nTop 10 most-prevalent tech tokens (full corpus):")
    for name, ct in prevalence[:10]:
        print(f"  {name:24s} {ct:7,} ({100 * ct / n:5.1f}%)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
