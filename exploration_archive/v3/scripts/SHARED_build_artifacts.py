#!/usr/bin/env python3
"""
Wave 1.5 — Build shared analytical artifacts for Wave 2+ agents.

Outputs (all to exploration/artifacts/shared/):
  1. swe_cleaned_text.parquet — cleaned description text with metadata
  2. company_stoplist.txt — company-name tokens for text cleaning
  3. swe_embeddings.npy + swe_embedding_index.parquet — sentence-transformer embeddings
  4. swe_tech_matrix.parquet — technology mention binary matrix
  5. asaniczka_structured_skills.parquet — parsed skills from asaniczka

Run: ./.venv/bin/python exploration/scripts/SHARED_build_artifacts.py
"""

import time
import re
import os
import sys
from collections import Counter, OrderedDict

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/unified.parquet"
OUT_DIR = "exploration/artifacts/shared"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE_EMBED = 256

# Technology terms that must NOT be stripped even if they appear as company names.
# These are critical for downstream tech-mention detection.
TECH_SAFELIST = {
    # Languages & runtimes
    'python', 'java', 'javascript', 'typescript', 'golang', 'rust', 'ruby',
    'kotlin', 'swift', 'scala', 'php', 'perl', 'lua', 'dart', 'elixir',
    'haskell', 'clojure', 'matlab', 'groovy', 'bash',
    # Frameworks & tools
    'react', 'angular', 'vue', 'svelte', 'django', 'flask', 'fastapi',
    'spring', 'rails', 'express', 'node', 'electron', 'flutter', 'redux',
    'bootstrap', 'tailwind', 'webpack', 'vite', 'graphql', 'grpc',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform', 'jenkins',
    'ansible', 'helm', 'prometheus', 'grafana', 'datadog', 'splunk',
    'nginx', 'linux', 'git', 'github', 'gitlab', 'argocd',
    # Data
    'postgres', 'postgresql', 'mysql', 'mongodb', 'mongo', 'redis', 'kafka',
    'spark', 'pyspark', 'snowflake', 'databricks', 'dbt', 'elasticsearch',
    'elastic', 'dynamodb', 'cassandra', 'bigquery', 'airflow', 'rabbitmq',
    'hadoop', 'oracle', 'sql',
    # AI/ML
    'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'jupyter',
    'langchain', 'langgraph', 'pinecone', 'chromadb', 'chroma', 'openai',
    'anthropic', 'copilot', 'cursor', 'chatgpt', 'claude', 'gemini', 'codex',
    # Testing
    'jest', 'pytest', 'selenium', 'cypress', 'junit', 'mocha', 'playwright',
    # Practices & other
    'agile', 'scrum', 'kanban', 'devops', 'microservices', 'microservice',
    # Companies that are also tech terms — keep the tech meaning
    'apple', 'google', 'meta', 'amazon',
}


def flush_print(msg):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 0: Build company stoplist
# ---------------------------------------------------------------------------
def build_company_stoplist(con):
    """Extract company name tokens to use as a stoplist."""
    flush_print("\n=== Step 0: Building company name stoplist ===")
    t0 = time.time()

    rows = con.sql("""
        SELECT DISTINCT company_name_canonical
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND company_name_canonical IS NOT NULL
    """, params=[DATA_PATH]).fetchall()

    # Tokenize all company names
    token_counts = Counter()
    for (name,) in rows:
        tokens = re.split(r'[\s,.\-/&()\[\]]+', name.lower())
        for t in tokens:
            t = t.strip("'\"")
            if len(t) >= 3:  # minimum 3 chars to avoid noise
                token_counts[t] += 1

    # Exclude generic English words and very common business terms
    generic_words = {
        'the', 'and', 'for', 'inc', 'llc', 'ltd', 'corp', 'company',
        'group', 'services', 'solutions', 'systems', 'technology', 'technologies',
        'consulting', 'digital', 'global', 'international', 'national',
        'north', 'south', 'east', 'west', 'new', 'united', 'american',
        'usa', 'one', 'first', 'next', 'best', 'top', 'pro',
        'health', 'healthcare', 'medical', 'financial', 'insurance',
        'energy', 'power', 'network', 'networks', 'data', 'cloud',
        'software', 'engineering', 'design', 'creative', 'media',
        'management', 'partners', 'associates', 'staffing', 'talent',
        'jobs', 'career', 'careers', 'hiring', 'team',
        'not', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
        'will', 'can', 'may', 'should', 'would', 'could', 'shall',
        'this', 'that', 'these', 'those', 'with', 'from', 'about',
        'into', 'over', 'under', 'between', 'through', 'after', 'before',
        'above', 'below', 'each', 'every', 'both', 'all', 'any', 'few',
        'more', 'most', 'some', 'such', 'only', 'also', 'very',
        'just', 'than', 'then', 'now', 'here', 'there', 'when',
        'where', 'how', 'what', 'which', 'who', 'whom', 'why',
    }

    # Also exclude purely numeric tokens, very short tokens, and tech safelist
    stoplist_tokens = sorted([
        t for t, c in token_counts.items()
        if t not in generic_words
        and t not in TECH_SAFELIST
        and len(t) >= 3
        and not t.isdigit()
        and not re.match(r'^[\d%$#@]+$', t)
    ])

    outpath = os.path.join(OUT_DIR, "company_stoplist.txt")
    with open(outpath, 'w') as f:
        for t in stoplist_tokens:
            f.write(t + '\n')

    elapsed = time.time() - t0
    flush_print(f"  Company stoplist: {len(stoplist_tokens)} tokens from {len(rows)} companies ({elapsed:.1f}s)")
    return set(stoplist_tokens)


# ---------------------------------------------------------------------------
# Step 1: Cleaned text parquet
# ---------------------------------------------------------------------------
def build_cleaned_text(con, company_tokens_unused):
    """Build swe_cleaned_text.parquet with best-available description text.

    Text cleaning strategy:
    - Strip complete company name strings (multi-word, case-insensitive) from each
      row's own company_name_canonical, rather than stripping individual tokens globally.
      This avoids destroying tech terms like "machine learning" that happen to appear
      in some company names.
    - Strip English stopwords at the individual token level.
    """
    flush_print("\n=== Step 1: Building cleaned text artifact ===")
    t0 = time.time()

    from nltk.corpus import stopwords
    english_stops = set(stopwords.words('english'))

    # Pull data using DuckDB — do text selection in SQL for efficiency
    flush_print("  Querying SWE rows with text selection...")
    df = con.sql("""
        SELECT
            uid,
            CASE
                WHEN llm_extraction_coverage = 'labeled'
                     AND description_core_llm IS NOT NULL
                     AND length(trim(description_core_llm)) > 0
                THEN description_core_llm
                WHEN description_core IS NOT NULL
                     AND length(trim(description_core)) > 0
                THEN description_core
                ELSE description
            END AS text_raw,
            CASE
                WHEN llm_extraction_coverage = 'labeled'
                     AND description_core_llm IS NOT NULL
                     AND length(trim(description_core_llm)) > 0
                THEN 'llm'
                WHEN description_core IS NOT NULL
                     AND length(trim(description_core)) > 0
                THEN 'rule'
                ELSE 'raw'
            END AS text_source,
            source,
            period,
            seniority_final,
            seniority_3level,
            is_aggregator,
            company_name_canonical,
            metro_area,
            yoe_extracted,
            swe_classification_tier,
            seniority_final_source
        FROM read_parquet(?)
        WHERE source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
    """, params=[DATA_PATH]).fetchdf()

    n = len(df)
    flush_print(f"  Fetched {n} rows")

    # Build a cache of compiled company-name patterns
    # For each unique company_name_canonical, compile a regex that matches the
    # full string (case-insensitive, word boundary). This strips the actual
    # company name without destroying individual tokens.
    flush_print("  Building company name patterns...")
    unique_companies = df['company_name_canonical'].dropna().unique()
    company_patterns = {}
    for name in unique_companies:
        if not name or len(name.strip()) < 2:
            continue
        # Escape regex special chars, match as whole phrase with word boundaries
        escaped = re.escape(name.strip())
        try:
            company_patterns[name] = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
        except re.error:
            pass  # skip malformed patterns
    flush_print(f"  Compiled {len(company_patterns)} company name patterns")

    # Precompile stopword pattern
    token_pattern = re.compile(r'\b\w+\b')
    whitespace_collapse = re.compile(r'\s+')

    flush_print("  Cleaning text (company name + stopword removal)...")

    texts_raw = df['text_raw'].tolist()
    companies = df['company_name_canonical'].tolist()
    cleaned = []
    for i, (text, company) in enumerate(zip(texts_raw, companies)):
        if not text or not isinstance(text, str):
            cleaned.append('')
            continue

        # Step 1: Strip this row's company name as a complete phrase
        if company and company in company_patterns:
            text = company_patterns[company].sub(' ', text)

        # Step 2: Strip English stopwords token-by-token
        result_parts = []
        last_end = 0
        for m in token_pattern.finditer(text):
            result_parts.append(text[last_end:m.start()])
            token = m.group()
            if token.lower() not in english_stops:
                result_parts.append(token)
            last_end = m.end()
        result_parts.append(text[last_end:])
        text = ''.join(result_parts)

        # Collapse whitespace
        text = whitespace_collapse.sub(' ', text).strip()
        cleaned.append(text)

        if i % 10000 == 0 and i > 0:
            flush_print(f"    Cleaned {i}/{n} rows")

    df['description_cleaned'] = cleaned

    # Drop temporary column
    df.drop(columns=['text_raw'], inplace=True)

    outpath = os.path.join(OUT_DIR, "swe_cleaned_text.parquet")
    df.to_parquet(outpath, index=False)

    # Stats
    source_dist = df['text_source'].value_counts().to_dict()
    elapsed = time.time() - t0
    flush_print(f"  Saved {n} rows to {outpath} ({elapsed:.1f}s)")
    flush_print(f"  Text source distribution: {source_dist}")

    return outpath, n, source_dist, elapsed


# ---------------------------------------------------------------------------
# Step 2: Sentence-transformer embeddings
# ---------------------------------------------------------------------------
def build_embeddings():
    """Compute sentence-transformer embeddings on cleaned text."""
    flush_print("\n=== Step 2: Computing sentence-transformer embeddings ===")
    t0 = time.time()

    # Read the cleaned text artifact
    table = pq.read_table(
        os.path.join(OUT_DIR, "swe_cleaned_text.parquet"),
        columns=['uid', 'description_cleaned']
    )
    uids = table.column('uid').to_pylist()
    texts = table.column('description_cleaned').to_pylist()
    n = len(texts)
    flush_print(f"  Processing {n} texts")

    # Truncate to ~512 tokens (approx 3000 chars)
    texts_truncated = []
    for t in texts:
        if t is None:
            t = ''
        texts_truncated.append(t[:3000])

    # Load model
    flush_print("  Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode in batches
    flush_print(f"  Encoding in batches of {BATCH_SIZE_EMBED}...")
    all_embeddings = []
    total_batches = (n + BATCH_SIZE_EMBED - 1) // BATCH_SIZE_EMBED
    for i in range(0, n, BATCH_SIZE_EMBED):
        batch = texts_truncated[i:i + BATCH_SIZE_EMBED]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)
        batch_num = i // BATCH_SIZE_EMBED + 1
        if batch_num % 20 == 0 or batch_num == total_batches:
            flush_print(f"    Batch {batch_num}/{total_batches} ({min(i + BATCH_SIZE_EMBED, n)}/{n} rows)")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    flush_print(f"  Embedding shape: {embeddings.shape}")

    # Save
    emb_path = os.path.join(OUT_DIR, "swe_embeddings.npy")
    np.save(emb_path, embeddings)
    flush_print(f"  Saved embeddings to {emb_path}")

    idx_table = pa.table({
        'row_index': pa.array(range(n), type=pa.int32()),
        'uid': pa.array(uids, type=pa.string())
    })
    idx_path = os.path.join(OUT_DIR, "swe_embedding_index.parquet")
    pq.write_table(idx_table, idx_path)
    flush_print(f"  Saved embedding index to {idx_path}")

    elapsed = time.time() - t0
    flush_print(f"  Embedding computation complete: {n} rows, {embeddings.shape[1]}d ({elapsed:.1f}s)")
    return n, embeddings.shape[1], elapsed


# ---------------------------------------------------------------------------
# Step 3: Technology mention binary matrix
# ---------------------------------------------------------------------------
def build_tech_matrix():
    """Scan cleaned text for ~120 technology mentions using regex."""
    flush_print("\n=== Step 3: Building technology mention matrix ===")
    t0 = time.time()

    tech_taxonomy = OrderedDict([
        # === Languages ===
        ('lang_python',       r'\bpython\b'),
        ('lang_java',         r'\bjava\b(?!\s*script)'),
        ('lang_javascript',   r'\bjavascript\b|\bjs\b'),
        ('lang_typescript',   r'\btypescript\b'),
        ('lang_go',           r'\bgolang\b|\bgo\s+(?:lang|programming|language)\b|\b(?:written\s+in|experience\s+(?:with|in)|proficiency\s+in|knowledge\s+of)\s+go\b'),
        ('lang_rust',         r'\brust\b(?:\s+(?:lang|programming|language))?'),
        ('lang_c_cpp',        r'\bc\+\+\b|\bcpp\b|\bC/C\+\+\b|\bC\+\+/C\b'),
        ('lang_csharp',       r'\bc#\b|\bcsharp\b|\bc\s+sharp\b'),
        ('lang_ruby',         r'\bruby\b'),
        ('lang_kotlin',       r'\bkotlin\b'),
        ('lang_swift',        r'\bswift\b'),
        ('lang_scala',        r'\bscala\b'),
        ('lang_php',          r'\bphp\b'),
        ('lang_r',            r'(?:\bR\b\s*(?:programming|language|studio)|(?:experience\s+(?:with|in)|proficiency\s+in)\s+R\b)'),
        ('lang_sql',          r'\bsql\b'),
        ('lang_bash_shell',   r'\bbash\b|\bshell\s+script'),
        ('lang_perl',         r'\bperl\b'),
        ('lang_lua',          r'\blua\b'),
        ('lang_dart',         r'\bdart\b'),
        ('lang_elixir',       r'\belixir\b'),
        ('lang_haskell',      r'\bhaskell\b'),
        ('lang_clojure',      r'\bclojure\b'),
        ('lang_matlab',       r'\bmatlab\b'),

        # === Frontend ===
        ('fe_react',          r'\breact\b(?:\.?js)?|\breactjs\b'),
        ('fe_angular',        r'\bangular\b(?:\.?js)?|\bangularjs\b'),
        ('fe_vue',            r'\bvue\b(?:\.?js)?|\bvuejs\b'),
        ('fe_nextjs',         r'\bnext\.?js\b'),
        ('fe_svelte',         r'\bsvelte\b'),
        ('fe_html_css',       r'\bhtml\b|\bcss\b'),
        ('fe_sass_scss',      r'\bsass\b|\bscss\b'),
        ('fe_tailwind',       r'\btailwind\b'),
        ('fe_bootstrap',      r'\bbootstrap\b'),
        ('fe_webpack',        r'\bwebpack\b'),
        ('fe_vite',           r'\bvite\b'),
        ('fe_redux',          r'\bredux\b'),
        ('fe_graphql',        r'\bgraphql\b'),
        ('fe_flutter',        r'\bflutter\b'),
        ('fe_react_native',   r'\breact\s+native\b'),
        ('fe_electron',       r'\belectron\b'),

        # === Backend ===
        ('be_nodejs',         r'\bnode\.?js\b|\bnodejs\b'),
        ('be_django',         r'\bdjango\b'),
        ('be_flask',          r'\bflask\b'),
        ('be_fastapi',        r'\bfastapi\b'),
        ('be_spring',         r'\bspring\b(?:\s+boot)?'),
        ('be_dotnet',         r'\.net\b|\bdotnet\b|\basp\.net\b'),
        ('be_rails',          r'\brails\b|\bruby\s+on\s+rails\b'),
        ('be_express',        r'\bexpress\.?js\b|\bexpressjs\b'),
        ('be_grpc',           r'\bgrpc\b'),
        ('be_rest_api',       r'\brest\s*(?:ful)?\s*api\b'),
        ('be_microservices',  r'\bmicroservice[s]?\b'),

        # === Cloud / DevOps ===
        ('cloud_aws',         r'\baws\b|\bamazon\s+web\s+services\b'),
        ('cloud_azure',       r'\bazure\b'),
        ('cloud_gcp',         r'\bgcp\b|\bgoogle\s+cloud\b'),
        ('devops_kubernetes', r'\bkubernetes\b|\bk8s\b'),
        ('devops_docker',     r'\bdocker\b'),
        ('devops_terraform',  r'\bterraform\b'),
        ('devops_cicd',       r'\bci\s*/\s*cd\b|\bcicd\b|\bcontinuous\s+(?:integration|delivery|deployment)\b'),
        ('devops_jenkins',    r'\bjenkins\b'),
        ('devops_github_actions', r'\bgithub\s+actions\b'),
        ('devops_argocd',     r'\bargocd\b|\bargo\s+cd\b'),
        ('devops_ansible',    r'\bansible\b'),
        ('devops_helm',       r'\bhelm\b'),
        ('devops_prometheus', r'\bprometheus\b'),
        ('devops_grafana',    r'\bgrafana\b'),
        ('devops_datadog',    r'\bdatadog\b'),
        ('devops_splunk',     r'\bsplunk\b'),
        ('devops_nginx',      r'\bnginx\b'),
        ('devops_linux',      r'\blinux\b'),
        ('devops_git',        r'\bgit\b(?!hub)|\bgithub\b|\bgitlab\b'),
        ('devops_cloudformation', r'\bcloudformation\b'),

        # === Data ===
        ('data_postgresql',   r'\bpostgres(?:ql)?\b'),
        ('data_mysql',        r'\bmysql\b'),
        ('data_mongodb',      r'\bmongodb\b|\bmongo\b'),
        ('data_redis',        r'\bredis\b'),
        ('data_kafka',        r'\bkafka\b'),
        ('data_spark',        r'\bspark\b|\bpyspark\b'),
        ('data_snowflake',    r'\bsnowflake\b'),
        ('data_databricks',   r'\bdatabricks\b'),
        ('data_dbt',          r'\bdbt\b'),
        ('data_elasticsearch',r'\belasticsearch\b|\belastic\s+search\b'),
        ('data_dynamodb',     r'\bdynamodb\b'),
        ('data_cassandra',    r'\bcassandra\b'),
        ('data_bigquery',     r'\bbigquery\b|\bbig\s+query\b'),
        ('data_airflow',      r'\bairflow\b'),
        ('data_rabbitmq',     r'\brabbitmq\b'),
        ('data_etl',          r'\betl\b'),
        ('data_hadoop',       r'\bhadoop\b'),
        ('data_nosql',        r'\bnosql\b'),
        ('data_oracle_db',    r'\boracle\b'),
        ('data_sql_server',   r'\bsql\s+server\b|\bmssql\b'),

        # === AI/ML traditional ===
        ('ml_tensorflow',     r'\btensorflow\b'),
        ('ml_pytorch',        r'\bpytorch\b'),
        ('ml_scikit_learn',   r'\bscikit[\s\-]?learn\b|\bsklearn\b'),
        ('ml_pandas',         r'\bpandas\b'),
        ('ml_numpy',          r'\bnumpy\b'),
        ('ml_jupyter',        r'\bjupyter\b'),
        ('ml_keras',          r'\bkeras\b'),
        ('ml_mlops',          r'\bmlops\b'),
        ('ml_computer_vision', r'\bcomputer\s+vision\b|\bopencv\b'),
        ('ml_nlp',            r'\bnatural\s+language\s+processing\b|\bnlp\b'),
        ('ml_deep_learning',  r'\bdeep\s+learning\b'),
        ('ml_machine_learning', r'\bmachine\s+learning\b'),

        # === AI/LLM new ===
        ('ai_langchain',      r'\blangchain\b'),
        ('ai_langgraph',      r'\blanggraph\b'),
        ('ai_rag',            r'\bretrieval[\s\-]augmented\s+generation\b|\brag\b'),
        ('ai_vector_db',      r'\bvector\s+(?:database|db|store)\b'),
        ('ai_pinecone',       r'\bpinecone\b'),
        ('ai_chromadb',       r'\bchromadb\b|\bchroma\b'),
        ('ai_huggingface',    r'\bhugging\s*face\b'),
        ('ai_openai_api',     r'\bopenai\b'),
        ('ai_claude_api',     r'\bclaude\s+api\b|\banthropic\s+api\b|\banthropic\b'),
        ('ai_prompt_eng',     r'\bprompt\s+engineering\b'),
        ('ai_fine_tuning',    r'\bfine[\s\-]?tun(?:e|ing)\b'),
        ('ai_mcp',            r'\bmodel\s+context\s+protocol\b'),
        ('ai_agent_frameworks', r'\bagent\s+framework[s]?\b|\bai\s+agent[s]?\b|\bagentic\b'),
        ('ai_llm',            r'\bllm[s]?\b|\blarge\s+language\s+model[s]?\b'),
        ('ai_generative_ai',  r'\bgenerative\s+ai\b|\bgen\s*ai\b'),
        ('ai_transformers',   r'\btransformer[s]?\s+(?:model|architecture)\b'),
        ('ai_embeddings',     r'\bembedding[s]?\b'),

        # === AI tools ===
        ('tool_copilot',      r'\bcopilot\b|\bgithub\s+copilot\b'),
        ('tool_cursor',       r'\bcursor\s+(?:ai|ide|editor)\b'),
        ('tool_chatgpt',      r'\bchatgpt\b|\bchat\s+gpt\b'),
        ('tool_claude',       r'\bclaude\b'),
        ('tool_gemini',       r'\bgemini\b'),
        ('tool_codex',        r'\bcodex\b'),

        # === Testing ===
        ('test_jest',         r'\bjest\b'),
        ('test_pytest',       r'\bpytest\b'),
        ('test_selenium',     r'\bselenium\b'),
        ('test_cypress',      r'\bcypress\b'),
        ('test_junit',        r'\bjunit\b'),
        ('test_mocha',        r'\bmocha\b'),
        ('test_playwright',   r'\bplaywright\b'),
        ('test_unit_testing', r'\bunit\s+test(?:s|ing)?\b'),
        ('test_integration',  r'\bintegration\s+test(?:s|ing)?\b'),

        # === Practices ===
        ('practice_agile',    r'\bagile\b'),
        ('practice_scrum',    r'\bscrum\b'),
        ('practice_tdd',      r'\btdd\b|\btest[\s\-]driven\s+development\b'),
        ('practice_kanban',   r'\bkanban\b'),
        ('practice_devops',   r'\bdevops\b'),
        ('practice_sre',      r'\bsre\b|\bsite\s+reliability\b'),

        # === Mobile ===
        ('mobile_ios',        r'\bios\b'),
        ('mobile_android',    r'\bandroid\b'),

        # === Security ===
        ('security_oauth',    r'\boauth\b'),
        ('security_sso',      r'\bsso\b|\bsingle\s+sign[\s\-]on\b'),
        ('security_devsecops', r'\bdevsecops\b'),
        ('security_owasp',    r'\bowasp\b'),
    ])

    # Compile all patterns
    compiled = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in tech_taxonomy.items()]
    tech_names = [name for name, _ in compiled]

    # Read cleaned text
    table = pq.read_table(
        os.path.join(OUT_DIR, "swe_cleaned_text.parquet"),
        columns=['uid', 'description_cleaned']
    )
    uids = table.column('uid').to_pylist()
    texts = table.column('description_cleaned').to_pylist()
    n = len(texts)
    flush_print(f"  Scanning {n} texts for {len(compiled)} technology patterns")

    # Build matrix — use numpy for memory efficiency
    matrix = np.zeros((n, len(compiled)), dtype=np.bool_)

    for i, text in enumerate(texts):
        if not text:
            continue
        for j, (name, pat) in enumerate(compiled):
            if pat.search(text):
                matrix[i, j] = True
        if i % 10000 == 0 and i > 0:
            flush_print(f"    Processed {i}/{n} rows")

    flush_print(f"    Processed {n}/{n} rows")

    # Build pyarrow table
    arrays = [pa.array(uids, type=pa.string())]
    names = ['uid']
    for j, name in enumerate(tech_names):
        arrays.append(pa.array(matrix[:, j].tolist(), type=pa.bool_()))
        names.append(name)

    out_table = pa.table(dict(zip(names, arrays)))
    outpath = os.path.join(OUT_DIR, "swe_tech_matrix.parquet")
    pq.write_table(out_table, outpath)

    # Summary stats
    mention_counts = {}
    for j, name in enumerate(tech_names):
        mention_counts[name] = int(matrix[:, j].sum())
    top_20 = sorted(mention_counts.items(), key=lambda x: -x[1])[:20]

    elapsed = time.time() - t0
    flush_print(f"  Saved tech matrix: {n} rows x {len(tech_names)} technologies ({elapsed:.1f}s)")
    flush_print(f"  Top 20 technologies:")
    for name, count in top_20:
        flush_print(f"    {name}: {count} ({100*count/n:.1f}%)")

    return n, len(tech_names), elapsed


# ---------------------------------------------------------------------------
# Step 4: Asaniczka structured skills
# ---------------------------------------------------------------------------
def build_asaniczka_skills(con):
    """Parse comma-separated skills from asaniczka SWE rows."""
    flush_print("\n=== Step 4: Building asaniczka structured skills ===")
    t0 = time.time()

    rows = con.sql("""
        SELECT uid, COALESCE(asaniczka_skills, skills_raw) as skills_text
        FROM read_parquet(?)
        WHERE source = 'kaggle_asaniczka'
          AND source_platform = 'linkedin'
          AND is_english = true
          AND date_flag = 'ok'
          AND is_swe = true
          AND (asaniczka_skills IS NOT NULL OR skills_raw IS NOT NULL)
    """, params=[DATA_PATH]).fetchall()

    uids = []
    skills_list = []
    for uid, skills_text in rows:
        if skills_text:
            parsed = [s.strip() for s in skills_text.split(',') if s.strip()]
            for skill in parsed:
                uids.append(uid)
                skills_list.append(skill)

    table = pa.table({
        'uid': pa.array(uids, type=pa.string()),
        'skill': pa.array(skills_list, type=pa.string())
    })
    outpath = os.path.join(OUT_DIR, "asaniczka_structured_skills.parquet")
    pq.write_table(table, outpath)

    n_rows = len(rows)
    n_skills = len(skills_list)
    n_unique = len(set(skills_list))
    elapsed = time.time() - t0
    flush_print(f"  Parsed {n_skills} skill mentions ({n_unique} unique) from {n_rows} rows ({elapsed:.1f}s)")
    return n_rows, n_skills, n_unique, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    overall_t0 = time.time()
    flush_print("=" * 70)
    flush_print("Wave 1.5: Building shared analytical artifacts")
    flush_print("=" * 70)

    con = duckdb.connect()

    # Step 0: Company stoplist
    company_tokens = build_company_stoplist(con)

    # Step 1: Cleaned text
    ct_path, ct_rows, ct_dist, ct_time = build_cleaned_text(con, company_tokens)

    # Step 2: Embeddings
    emb_rows, emb_dims, emb_time = build_embeddings()

    # Step 3: Tech matrix
    tm_rows, tm_techs, tm_time = build_tech_matrix()

    # Step 4: Asaniczka skills
    ask_rows, ask_skills, ask_unique, ask_time = build_asaniczka_skills(con)

    # Summary
    overall_time = time.time() - overall_t0
    flush_print("\n" + "=" * 70)
    flush_print("BUILD COMPLETE")
    flush_print("=" * 70)
    flush_print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f}m)")
    flush_print(f"\nArtifacts in {OUT_DIR}/:")
    flush_print(f"  swe_cleaned_text.parquet     - {ct_rows} rows, text_source: {ct_dist}")
    flush_print(f"  company_stoplist.txt         - {len(company_tokens)} tokens")
    flush_print(f"  swe_embeddings.npy           - {emb_rows} x {emb_dims} float32")
    flush_print(f"  swe_embedding_index.parquet  - {emb_rows} rows")
    flush_print(f"  swe_tech_matrix.parquet      - {tm_rows} rows x {tm_techs} tech columns")
    flush_print(f"  asaniczka_structured_skills.parquet - {ask_rows} source rows, {ask_skills} skill mentions, {ask_unique} unique")

    # Write README
    readme = f"""# Shared Analytical Artifacts -- Wave 1.5

Built: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total build time: {overall_time:.1f}s ({overall_time/60:.1f}m)

## Contents

### swe_cleaned_text.parquet
- **Rows:** {ct_rows}
- **Columns:** uid, description_cleaned, text_source, source, period, seniority_final, seniority_3level, is_aggregator, company_name_canonical, metro_area, yoe_extracted, swe_classification_tier, seniority_final_source
- **Text source distribution:**
  - llm: {ct_dist.get('llm', 0)} rows ({100*ct_dist.get('llm', 0)/ct_rows:.1f}%)
  - rule: {ct_dist.get('rule', 0)} rows ({100*ct_dist.get('rule', 0)/ct_rows:.1f}%)
  - raw: {ct_dist.get('raw', 0)} rows ({100*ct_dist.get('raw', 0)/ct_rows:.1f}%)
- **Build time:** {ct_time:.1f}s
- **Description:** Best-available description text for each SWE LinkedIn row. Priority: description_core_llm (where llm_extraction_coverage='labeled') > description_core > description. Company name tokens and English stopwords stripped from text.

### company_stoplist.txt
- **Tokens:** {len(company_tokens)}
- **Description:** Unique tokens (3+ chars) from company_name_canonical values across SWE rows. Excludes generic English words and numeric-only tokens. Used during text cleaning.

### swe_embeddings.npy
- **Shape:** {emb_rows} x {emb_dims} (float32)
- **Model:** all-MiniLM-L6-v2 (sentence-transformers)
- **Build time:** {emb_time:.1f}s
- **Description:** Sentence-transformer embeddings of description_cleaned (first ~3000 chars, ~512 tokens). Row order matches swe_embedding_index.parquet.

### swe_embedding_index.parquet
- **Rows:** {emb_rows}
- **Columns:** row_index (int32), uid (string)
- **Description:** Maps embedding matrix row index to uid. Join with other artifacts on uid.

### swe_tech_matrix.parquet
- **Rows:** {tm_rows}
- **Columns:** uid + {tm_techs} boolean technology columns
- **Build time:** {tm_time:.1f}s
- **Description:** Binary matrix of technology mentions detected via regex in description_cleaned. Covers languages, frameworks, cloud/devops, data, AI/ML, AI tools, testing, practices, mobile, and security.

### asaniczka_structured_skills.parquet
- **Source rows:** {ask_rows}
- **Total skill mentions:** {ask_skills}
- **Unique skills:** {ask_unique}
- **Build time:** {ask_time:.1f}s
- **Columns:** uid (string), skill (string)
- **Description:** Parsed comma-separated skills from asaniczka SWE rows. Long format (one row per uid-skill pair).

## Filters Applied

All artifacts use the default SQL filters:
```sql
WHERE source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
```

## Known Issues / Partial Coverage

- **Text quality varies by source:** LLM-cleaned text covers ~45% of rows (Kaggle with llm_extraction_coverage='labeled'). Remaining rows use rule-based boilerplate removal (~44% accuracy) or raw text. Filter on text_source='llm' for highest quality text subset.
- **Scraped data has NO LLM-cleaned text** -- all ~24K scraped rows use rule-based or raw fallback.
- **Company stoplist** excludes generic English words, numbers, and short tokens, but may still contain tokens that coincidentally match non-company words in descriptions.
- **Technology regex patterns:** 'go' is restricted to clear programming contexts (golang, "experience with go", etc.) to avoid false positives. Some patterns like 'rust', 'swift', 'flask' may have minor false-positive rates in non-tech contexts.
- **ai_rag** pattern may match the standalone word "rag" in non-AI contexts; rates should be low in SWE job postings.
"""

    with open(os.path.join(OUT_DIR, "README.md"), 'w') as f:
        f.write(readme)
    flush_print(f"\nREADME written to {OUT_DIR}/README.md")


if __name__ == '__main__':
    main()
