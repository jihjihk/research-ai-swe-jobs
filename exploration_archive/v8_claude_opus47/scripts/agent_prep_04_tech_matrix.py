"""Agent Prep step 4: technology mention binary matrix + sanity check.

Build a wide-form one-row-per-uid boolean matrix using ~100-tech taxonomy
against `description_cleaned` from the cleaned-text artifact (which has
already had markdown backslash escapes removed).
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
IN_PATH = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
OUT_MATRIX = ROOT / "exploration/artifacts/shared/swe_tech_matrix.parquet"
OUT_SANITY = ROOT / "exploration/artifacts/shared/tech_matrix_sanity.csv"


# ---------------------------------------------------------------------------
# Taxonomy. Each entry: (column_name, list_of_regex_patterns). Regex run as
# case-insensitive on lowercased description_cleaned.
# ---------------------------------------------------------------------------
TAXONOMY: dict[str, list[str]] = {
    # Languages
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b(?!\s*script)"],
    "javascript": [r"\bjavascript\b", r"\bjs\b"],
    "typescript": [r"\btypescript\b", r"\bts\b"],
    "go": [r"\b(go|golang)\b"],
    "rust": [r"\brust\b"],
    "cpp": [r"\bc\+\+", r"\bcpp\b"],
    "csharp": [r"\bc#", r"\bcsharp\b"],
    "ruby": [r"\bruby\b"],
    "kotlin": [r"\bkotlin\b"],
    "swift": [r"\bswift\b"],
    "scala": [r"\bscala\b"],
    "php": [r"\bphp\b"],
    "r_language": [r"\br\b(?=\s*(?:programming|language|,|statistical|shiny))"],  # narrow
    "perl": [r"\bperl\b"],
    "bash": [r"\bbash\b"],
    "shell": [r"\bshell\s*scripting\b", r"\bshell\s*scripts?\b"],
    "sql": [r"\bsql\b"],

    # Frontend
    "react": [r"\breact\b", r"\breact\.js\b", r"\breactjs\b"],
    "angular": [r"\bangular\b"],
    "vue": [r"\bvue\b", r"\bvue\.js\b", r"\bvuejs\b"],
    "nextjs": [r"\bnext\.js\b", r"\bnextjs\b"],
    "svelte": [r"\bsvelte\b"],
    "jquery": [r"\bjquery\b"],
    "html": [r"\bhtml\b", r"\bhtml5\b"],
    "css": [r"\bcss\b", r"\bcss3\b"],
    "redux": [r"\bredux\b"],
    "tailwind": [r"\btailwind\b"],

    # Backend
    "nodejs": [r"\bnode\b", r"\bnode\.js\b", r"\bnodejs\b"],
    "django": [r"\bdjango\b"],
    "flask": [r"\bflask\b"],
    "spring": [r"\bspring\b", r"\bspring\s*boot\b"],
    "dotnet": [r"\.net\b", r"\bdotnet\b"],
    "rails": [r"\brails\b", r"\bruby\s*on\s*rails\b"],
    "fastapi": [r"\bfastapi\b"],
    "express": [r"\bexpress\.js\b", r"\bexpressjs\b", r"\bexpress\b"],
    "laravel": [r"\blaravel\b"],

    # Cloud / DevOps
    "aws": [r"\baws\b", r"\bamazon\s*web\s*services\b"],
    "azure": [r"\bazure\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle\s*cloud\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "docker": [r"\bdocker\b"],
    "terraform": [r"\bterraform\b"],
    "ansible": [r"\bansible\b"],
    "cicd": [r"\bci\s*/\s*cd\b", r"\bci\\/cd\b", r"\bcicd\b", r"\bci-cd\b"],
    "jenkins": [r"\bjenkins\b"],
    "github_actions": [r"\bgithub\s*actions?\b"],
    "argocd": [r"\bargo\s*cd\b", r"\bargocd\b"],
    "gitlab": [r"\bgitlab\b"],
    "helm": [r"\bhelm\b"],

    # Data
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "mysql": [r"\bmysql\b"],
    "mongodb": [r"\bmongodb\b", r"\bmongo\b"],
    "redis": [r"\bredis\b"],
    "kafka": [r"\bkafka\b"],
    "spark": [r"\bspark\b"],
    "snowflake": [r"\bsnowflake\b"],
    "databricks": [r"\bdatabricks\b"],
    "dbt": [r"\bdbt\b"],
    "elasticsearch": [r"\belasticsearch\b", r"\belastic\s*search\b"],
    "oracle": [r"\boracle\b"],
    "dynamodb": [r"\bdynamodb\b", r"\bdynamo\s*db\b"],
    "cassandra": [r"\bcassandra\b"],
    "bigquery": [r"\bbigquery\b", r"\bbig\s*query\b"],

    # ML/AI traditional
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "sklearn": [r"\bscikit\s*-?\s*learn\b", r"\bsklearn\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "jupyter": [r"\bjupyter\b"],
    "keras": [r"\bkeras\b"],
    "xgboost": [r"\bxgboost\b"],

    # ML/AI LLM-era
    "langchain": [r"\blangchain\b"],
    "langgraph": [r"\blanggraph\b"],
    "rag": [r"\brag\b", r"\bretrieval\s*augmented\b"],
    "vector_database": [r"\bvector\s*databases?\b", r"\bvector\s*dbs?\b", r"\bvector\s*stores?\b"],
    "pinecone": [r"\bpinecone\b"],
    "chromadb": [r"\bchromadb\b", r"\bchroma\s*db\b"],
    "huggingface": [r"\bhuggingface\b", r"\bhugging\s*face\b"],
    "openai": [r"\bopenai\b"],
    "claude": [r"\bclaude\b"],
    "gemini": [r"\bgemini\b"],
    "mcp": [r"\bmcp\b", r"\bmodel\s*context\s*protocol\b"],
    "llamaindex": [r"\bllamaindex\b", r"\bllama\s*index\b"],
    "anthropic": [r"\banthropic\b"],
    "ollama": [r"\bollama\b"],

    # AI tools
    "copilot": [r"\bcopilot\b", r"\bgithub\s*copilot\b"],
    "cursor": [r"\bcursor\b"],
    "chatgpt": [r"\bchatgpt\b", r"\bchat\s*gpt\b"],
    "codex": [r"\bcodex\b"],
    "llm_token": [r"\bllm\b", r"\bllms\b"],
    "prompt_engineering": [r"\bprompt\s*engineering\b"],
    "fine_tuning": [r"\bfine[\s-]*tuning\b", r"\bfinetuning\b"],
    "agent_framework": [r"\bagent\s*frameworks?\b", r"\bagentic\s*frameworks?\b"],

    # Testing
    "jest": [r"\bjest\b"],
    "pytest": [r"\bpytest\b"],
    "selenium": [r"\bselenium\b"],
    "cypress": [r"\bcypress\b"],
    "junit": [r"\bjunit\b"],
    "mocha": [r"\bmocha\b"],
    "playwright": [r"\bplaywright\b"],

    # Practices
    "agile": [r"\bagile\b"],
    "scrum": [r"\bscrum\b"],
    "tdd": [r"\btdd\b", r"\btest[\s-]*driven\s*development\b"],
    "devops": [r"\bdevops\b"],
    "sre": [r"\bsre\b", r"\bsite\s*reliability\b"],
    "microservices": [r"\bmicroservices?\b", r"\bmicro[\s-]*services?\b"],
}


def _inline_asserts() -> None:
    """TDD asserts per spec — verify critical regex before running at scale."""
    assert re.search(r"\bc\+\+", "knowledge of c++", re.I)
    assert re.search(r"\bc\+\+", "c++ experience required", re.I)
    # We intentionally don't match "c plus plus" — not special-handled.
    assert not re.search(r"\bc\+\+", "c plus plus", re.I)
    assert re.search(r"\bc#", "c# and .net developer", re.I)
    assert re.search(r"\.net\b", "using .net framework", re.I)
    # Java vs javascript separation
    assert re.search(r"\bjava\b(?!\s*script)", "java experience", re.I)
    assert not re.search(r"\bjava\b(?!\s*script)", "javascript experience", re.I)
    # Go vs other
    assert re.search(r"\b(go|golang)\b", "experience with go", re.I)
    assert re.search(r"\b(go|golang)\b", "golang experience", re.I)
    # R-language — avoid matching "version r" (we keep narrow context)
    assert re.search(r"\br\b(?=\s*(?:programming|language|,|statistical|shiny))",
                     "r programming", re.I)
    assert not re.search(r"\br\b(?=\s*(?:programming|language|,|statistical|shiny))",
                         "version r", re.I)
    print("Inline TDD asserts passed.")


def main() -> None:
    _inline_asserts()

    # Compile
    compiled: dict[str, list[re.Pattern]] = {
        col: [re.compile(p, re.IGNORECASE) for p in patterns]
        for col, patterns in TAXONOMY.items()
    }
    tech_cols = list(TAXONOMY.keys())
    print(f"Taxonomy: {len(tech_cols)} techs")

    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT uid, source, description_cleaned
        FROM read_parquet('{IN_PATH}')
        """
    ).df()
    print(f"Loaded {len(df):,} rows from cleaned-text artifact")

    # Build matrix in chunks
    N = len(df)
    bool_mat = np.zeros((N, len(tech_cols)), dtype=bool)

    CHUNK = 5000
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        texts = df["description_cleaned"].values[start:end]
        for ti, (col, patterns) in enumerate(zip(tech_cols, compiled.values())):
            for p in patterns:
                # OR-accumulate across patterns for the same tech
                for i, t in enumerate(texts):
                    if not bool_mat[start + i, ti] and p.search(t or ""):
                        bool_mat[start + i, ti] = True
        if (start // CHUNK) % 5 == 0:
            print(f"  scanned {end:,}/{N:,}")

    # Build output DataFrame
    out = pd.DataFrame(bool_mat, columns=tech_cols)
    out.insert(0, "uid", df["uid"].values)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, OUT_MATRIX, compression="snappy")
    size_mb = OUT_MATRIX.stat().st_size / (1024 * 1024)
    print(f"Wrote tech matrix to {OUT_MATRIX} ({size_mb:.2f} MB, {len(tech_cols)} techs)")

    # Sanity check — compute per-tech mention rates per source
    df["source"] = df["source"].astype(str)
    per_tech_rows = []
    src_arsh = df["source"] == "kaggle_arshkon"
    src_asan = df["source"] == "kaggle_asaniczka"
    src_scr = df["source"] == "scraped"

    n_arsh = int(src_arsh.sum())
    n_asan = int(src_asan.sum())
    n_scr = int(src_scr.sum())

    for ti, col in enumerate(tech_cols):
        arsh_rate = bool_mat[src_arsh.values, ti].mean() if n_arsh else float("nan")
        asan_rate = bool_mat[src_asan.values, ti].mean() if n_asan else float("nan")
        scr_rate = bool_mat[src_scr.values, ti].mean() if n_scr else float("nan")

        if arsh_rate > 0 and scr_rate > 0:
            ratio = arsh_rate / scr_rate
        else:
            ratio = float("nan")

        flag = ""
        note = ""
        if arsh_rate < 0.001 and scr_rate < 0.001 and asan_rate < 0.001:
            flag = "low_prevalence"
            note = "all three sources < 0.1%; regex may be too narrow or tech rarely mentioned"
        elif np.isnan(ratio):
            flag = "missing_rate"
            note = "zero rate in either arshkon or scraped"
        elif ratio < 0.33 or ratio > 3.0:
            flag = "ratio_outlier"
            # Investigate: check if raw text still has backslash escapes for this tech
            if col in {"cpp", "csharp", "dotnet"}:
                note = "tech is backslash-escape-sensitive; see residual check below"
            else:
                note = "arshkon/scraped ratio outside [0.33, 3] — review for real market shift vs tokenization"

        per_tech_rows.append({
            "tech": col,
            "arshkon_rate": arsh_rate,
            "asaniczka_rate": asan_rate,
            "scraped_rate": scr_rate,
            "arsh_to_scraped_ratio": ratio,
            "flag": flag,
            "investigation_note": note,
        })

    sanity_df = pd.DataFrame(per_tech_rows)
    sanity_df.to_csv(OUT_SANITY, index=False)
    flagged = sanity_df[sanity_df["flag"] != ""]
    print(f"\nSanity: {len(flagged)} / {len(sanity_df)} techs flagged")
    if len(flagged):
        print(flagged.to_string(index=False))

    # Residual check for escape-sensitive techs. Look for `\+\+` or `\#` or `\.net`
    # in the ORIGINAL raw description to confirm unescape worked.
    print("\nResidual backslash check on raw description (should be near zero after fix):")
    r = con.execute("""
        SELECT source,
               SUM(CASE WHEN regexp_matches(description, '\\\\\\+\\\\\\+') THEN 1 ELSE 0 END) AS raw_cpp_bs,
               SUM(CASE WHEN regexp_matches(description, 'c\\\\\\#') THEN 1 ELSE 0 END) AS raw_csharp_bs,
               SUM(CASE WHEN regexp_matches(description, '\\\\\\.net') THEN 1 ELSE 0 END) AS raw_dotnet_bs
        FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')
        WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
        GROUP BY source
        ORDER BY source
    """).df()
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
