#!/usr/bin/env python3
"""
Build shared analytical artifacts for Wave 2+ exploration agents.

Produces:
  1. swe_cleaned_text.parquet — cleaned description text with metadata
  2. swe_embeddings.npy + swe_embedding_index.parquet — sentence-transformer embeddings
  3. swe_tech_matrix.parquet — technology mention binary matrix
  4. company_stoplist.txt — company name token stoplist
  5. asaniczka_structured_skills.parquet — parsed skills for asaniczka SWE rows
  6. calibration_table.csv — within-2024 vs cross-period effect size calibration

Usage:
    .venv/bin/python exploration/artifacts/shared/build_shared_artifacts.py
"""

import time
import re
import gc
import sys
import os
from pathlib import Path
from collections import Counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent.parent.parent
DATA = PROJECT / "data" / "unified.parquet"
OUT = PROJECT / "exploration" / "artifacts" / "shared"
OUT.mkdir(parents=True, exist_ok=True)

WALL_START = time.time()


def elapsed():
    return f"{time.time() - WALL_START:.1f}s"


# ── SQL base filter ──────────────────────────────────────────────────────────
BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

# ============================================================================
# Step 0: Build company name stoplist (needed before cleaned text)
# ============================================================================
print(f"[{elapsed()}] Step 0: Building company name stoplist ...")

con = duckdb.connect()
company_names = con.execute(f"""
    SELECT DISTINCT company_name_canonical
    FROM read_parquet('{DATA}')
    WHERE {BASE_FILTER}
      AND company_name_canonical IS NOT NULL
      AND LENGTH(company_name_canonical) > 0
""").fetchall()

# Tokenize company names: split on whitespace and common punctuation
company_tokens = set()
for (name,) in company_names:
    tokens = re.split(r'[\s,.\-/&()\[\]]+', name.lower())
    for t in tokens:
        t = t.strip("'\"")
        if len(t) >= 2:  # skip single chars
            company_tokens.add(t)

# Remove very common English words that happen to appear in company names
# but are useful in descriptions (keep only company-specific tokens)
generic_words = {
    'the', 'and', 'for', 'inc', 'llc', 'ltd', 'corp', 'co', 'company',
    'group', 'services', 'solutions', 'technologies', 'technology', 'tech',
    'systems', 'software', 'global', 'international', 'digital', 'data',
    'health', 'financial', 'consulting', 'labs', 'studio', 'studios',
    'network', 'networks', 'media', 'partners', 'capital', 'industries',
    'enterprises', 'holdings', 'management', 'of', 'in', 'on', 'at', 'to',
    'is', 'it', 'by', 'us', 'or', 'an', 'as', 'if', 'do', 'no', 'we',
}
# Keep all tokens for company-name stripping (including generic ones)
# The stoplist is used specifically to remove company name fragments from descriptions

stoplist_path = OUT / "company_stoplist.txt"
sorted_tokens = sorted(company_tokens)
with open(stoplist_path, 'w') as f:
    for token in sorted_tokens:
        f.write(token + '\n')

print(f"[{elapsed()}] Company stoplist: {len(sorted_tokens)} tokens from {len(company_names)} companies")
con.close()

# ============================================================================
# Step 1: Build cleaned text column
# ============================================================================
print(f"\n[{elapsed()}] Step 1: Building cleaned text column ...")

con = duckdb.connect()

# Load the base data we need
rows = con.execute(f"""
    SELECT
        uid,
        description_core_llm,
        llm_extraction_coverage,
        description,
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
    FROM read_parquet('{DATA}')
    WHERE {BASE_FILTER}
    ORDER BY uid
""").fetchall()

con.close()
print(f"[{elapsed()}] Loaded {len(rows)} SWE rows")

# Build English stopwords set (simple hardcoded set to avoid NLTK download issues)
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
}

# Build company name pattern for stripping (only multi-word company names)
# We'll strip exact company names, not individual tokens (too aggressive)
company_name_patterns = {}
for (name,) in company_names:
    # Create case-insensitive pattern for each company name
    escaped = re.escape(name.lower())
    company_name_patterns[name.lower()] = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)

# For efficiency, just do simple case-insensitive replacement of long company names
# Short names (1-2 chars) are too risky
long_company_names = sorted(
    [name.lower() for (name,) in company_names if len(name) >= 4],
    key=len, reverse=True  # longest first to avoid partial matches
)


def clean_text(text, company_name=None):
    """Clean a job description text."""
    if not text or not text.strip():
        return ""

    result = text

    # Strip the specific company name if provided
    if company_name and len(company_name) >= 4:
        result = re.sub(re.escape(company_name), ' ', result, flags=re.IGNORECASE)

    # Remove URLs
    result = re.sub(r'https?://\S+', ' ', result)
    # Remove email addresses
    result = re.sub(r'\S+@\S+\.\S+', ' ', result)

    # Collapse whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    # Remove stopwords (word-level)
    words = result.split()
    words = [w for w in words if w.lower() not in ENGLISH_STOPWORDS]
    result = ' '.join(words)

    return result


# Process each row
uids = []
descriptions_cleaned = []
text_sources = []
sources = []
periods = []
seniority_finals = []
seniority_3levels = []
is_aggregators = []
company_names_canonical = []
metro_areas = []
yoe_extracteds = []
swe_tiers = []
seniority_final_sources = []

text_source_counts = Counter()

for row in rows:
    (uid, desc_llm, llm_cov, desc_raw,
     source, period, sen_final, sen_3, is_agg,
     comp_canon, metro, yoe, swe_tier, sen_src) = row

    # Boilerplate-sensitive consumers must filter to text_source = 'llm'.
    if llm_cov == 'labeled' and desc_llm and len(desc_llm.strip()) > 0:
        text = desc_llm
        tsrc = 'llm'
    else:
        text = desc_raw if desc_raw else ''
        tsrc = 'raw'

    cleaned = clean_text(text, comp_canon)

    text_source_counts[tsrc] += 1
    uids.append(uid)
    descriptions_cleaned.append(cleaned)
    text_sources.append(tsrc)
    sources.append(source)
    periods.append(period)
    seniority_finals.append(sen_final)
    seniority_3levels.append(sen_3)
    is_aggregators.append(bool(is_agg) if is_agg is not None else False)
    company_names_canonical.append(comp_canon)
    metro_areas.append(metro)
    yoe_extracteds.append(float(yoe) if yoe is not None else None)
    swe_tiers.append(swe_tier)
    seniority_final_sources.append(sen_src)

# Save as parquet
table = pa.table({
    'uid': pa.array(uids, type=pa.string()),
    'description_cleaned': pa.array(descriptions_cleaned, type=pa.string()),
    'text_source': pa.array(text_sources, type=pa.string()),
    'source': pa.array(sources, type=pa.string()),
    'period': pa.array(periods, type=pa.string()),
    'seniority_final': pa.array(seniority_finals, type=pa.string()),
    'seniority_3level': pa.array(seniority_3levels, type=pa.string()),
    'is_aggregator': pa.array(is_aggregators, type=pa.bool_()),
    'company_name_canonical': pa.array(company_names_canonical, type=pa.string()),
    'metro_area': pa.array(metro_areas, type=pa.string()),
    'yoe_extracted': pa.array(yoe_extracteds, type=pa.float64()),
    'swe_classification_tier': pa.array(swe_tiers, type=pa.string()),
    'seniority_final_source': pa.array(seniority_final_sources, type=pa.string()),
})

cleaned_path = OUT / "swe_cleaned_text.parquet"
pq.write_table(table, cleaned_path)

print(f"[{elapsed()}] Cleaned text saved: {len(uids)} rows")
print(f"  Text source distribution: {dict(text_source_counts)}")

# Free memory
del rows, company_name_patterns, long_company_names
gc.collect()

# ============================================================================
# Step 2: Sentence-transformer embeddings
# ============================================================================
print(f"\n[{elapsed()}] Step 2: Computing sentence-transformer embeddings ...")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"[{elapsed()}] Model loaded")

# Process in batches of 256
BATCH_SIZE = 256
n_rows = len(descriptions_cleaned)
embedding_dim = 384  # all-MiniLM-L6-v2 output dimension
embeddings = np.zeros((n_rows, embedding_dim), dtype=np.float32)

# Truncate to ~512 tokens (rough: 512 * 5 chars per token = 2560 chars)
# The model's tokenizer handles actual truncation, but pre-truncating saves memory
MAX_CHARS = 2560

n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
for i in range(0, n_rows, BATCH_SIZE):
    batch_idx = i // BATCH_SIZE
    end = min(i + BATCH_SIZE, n_rows)
    batch_texts = [
        (t[:MAX_CHARS] if t else "") for t in descriptions_cleaned[i:end]
    ]
    batch_embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True
    )
    embeddings[i:end] = batch_embeddings

    if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
        print(f"[{elapsed()}]   Batch {batch_idx + 1}/{n_batches} done")

    del batch_texts, batch_embeddings
    gc.collect()

# Save embeddings
emb_path = OUT / "swe_embeddings.npy"
np.save(emb_path, embeddings)

# Save index
idx_table = pa.table({
    'uid': pa.array(uids, type=pa.string()),
})
idx_path = OUT / "swe_embedding_index.parquet"
pq.write_table(idx_table, idx_path)

print(f"[{elapsed()}] Embeddings saved: {embeddings.shape} ({embeddings.nbytes / 1e6:.1f} MB)")

# Free embedding memory
del model, embeddings
gc.collect()

# ============================================================================
# Step 3: Technology mention binary matrix
# ============================================================================
print(f"\n[{elapsed()}] Step 3: Building technology mention matrix ...")

# Define ~110 technology patterns with regex variations
TECH_PATTERNS = {
    # ── Programming Languages ──
    'python': r'\bpython\b',
    'java': r'\bjava\b(?!\s*script)',
    'javascript': r'\bjavascript\b|\bjs\b',
    'typescript': r'\btypescript\b|\bts\b(?:\s+/\s*js)?',
    'go_lang': r'\bgolang\b|\bgo\s+lang\b|\bgo\b(?:\s+programming)?',
    'rust': r'\brust\b(?:\s+lang)?',
    'c_cpp': r'\bc\+\+\b|\bcpp\b|\bc\s+programming\b|\bc\s+language\b',
    'c_lang': r'\bc\b(?:\s*/\s*c\+\+|\s+programming|\s+language)',
    'csharp': r'\bc#\b|\bcsharp\b|c\s+sharp\b|\.net\s+c#',
    'ruby': r'\bruby\b',
    'kotlin': r'\bkotlin\b',
    'swift': r'\bswift\b(?:\s+programming|\s+language)?',
    'scala': r'\bscala\b',
    'php': r'\bphp\b',
    'r_lang': r'\br\b(?:\s+programming|\s+language|\s+studio)',
    'perl': r'\bperl\b',
    'lua': r'\blua\b',
    'elixir': r'\belixir\b',
    'haskell': r'\bhaskell\b',
    'clojure': r'\bclojure\b',
    'erlang': r'\berlang\b',
    'dart': r'\bdart\b',
    'julia': r'\bjulia\b(?:\s+lang)?',
    'sql': r'\bsql\b',
    'bash_shell': r'\bbash\b|\bshell\s+script',
    'groovy': r'\bgroovy\b',
    'objective_c': r'\bobjective[\-\s]c\b',
    'assembly': r'\bassembly\b(?:\s+lang)?',

    # ── Frontend Frameworks ──
    'react': r'\breact\b|\breactjs\b|\breact\.js\b',
    'angular': r'\bangular\b|\bangularjs\b',
    'vue': r'\bvue\b|\bvuejs\b|\bvue\.js\b',
    'nextjs': r'\bnext\.js\b|\bnextjs\b|\bnext\s+js\b',
    'svelte': r'\bsvelte\b',
    'nuxtjs': r'\bnuxt\b',
    'remix': r'\bremix\b',
    'ember': r'\bember\b',
    'jquery': r'\bjquery\b',
    'tailwind': r'\btailwind\b',
    'bootstrap': r'\bbootstrap\b',
    'html_css': r'\bhtml\b|\bcss\b',
    'sass_less': r'\bsass\b|\bless\b|\bscss\b',
    'webpack': r'\bwebpack\b|\bvite\b|\besbuild\b',

    # ── Backend Frameworks ──
    'nodejs': r'\bnode\.?js\b|\bnode\b(?:\s+js)?',
    'django': r'\bdjango\b',
    'flask': r'\bflask\b',
    'fastapi': r'\bfastapi\b|\bfast\s*api\b',
    'spring': r'\bspring\b(?:\s+boot|\s+framework|\s+mvc)?',
    'dotnet': r'\.net\b|\bdotnet\b|\basp\.net\b',
    'rails': r'\brails\b|\bruby\s+on\s+rails\b',
    'express': r'\bexpress\b(?:\.js|\s+js)?',
    'gin': r'\bgin\b(?:\s+framework)?',
    'laravel': r'\blaravel\b',
    'graphql': r'\bgraphql\b',
    'grpc': r'\bgrpc\b',
    'rest_api': r'\brest\b(?:\s*ful|\s+api)',

    # ── Cloud & Infrastructure ──
    'aws': r'\baws\b|\bamazon\s+web\s+services\b',
    'azure': r'\bazure\b|\bmicrosoft\s+azure\b',
    'gcp': r'\bgcp\b|\bgoogle\s+cloud\b',
    'kubernetes': r'\bkubernetes\b|\bk8s\b',
    'docker': r'\bdocker\b',
    'terraform': r'\bterraform\b',
    'ansible': r'\bansible\b',
    'helm': r'\bhelm\b',
    'cloudformation': r'\bcloudformation\b',
    'pulumi': r'\bpulumi\b',
    'serverless': r'\bserverless\b|\blambda\b',
    'microservices': r'\bmicroservices?\b',

    # ── CI/CD & DevOps ──
    'cicd': r'\bci\s*/\s*cd\b|\bcicd\b|\bcontinuous\s+(?:integration|delivery|deployment)\b',
    'jenkins': r'\bjenkins\b',
    'github_actions': r'\bgithub\s+actions\b',
    'gitlab_ci': r'\bgitlab\b(?:\s+ci)?',
    'circleci': r'\bcircleci\b|\bcircle\s+ci\b',
    'argocd': r'\bargocd\b|\bargo\s+cd\b',
    'datadog': r'\bdatadog\b',
    'prometheus': r'\bprometheus\b',
    'grafana': r'\bgrafana\b',
    'splunk': r'\bsplunk\b',
    'newrelic': r'\bnew\s*relic\b',

    # ── Databases & Data ──
    'postgresql': r'\bpostgre(?:sql|s)?\b',
    'mysql': r'\bmysql\b',
    'mongodb': r'\bmongo(?:db)?\b',
    'redis': r'\bredis\b',
    'dynamodb': r'\bdynamo\s*db\b',
    'cassandra': r'\bcassandra\b',
    'elasticsearch': r'\belasticsearch\b|\belastic\s+search\b|\belastic\b',
    'kafka': r'\bkafka\b',
    'rabbitmq': r'\brabbitmq\b|\brabbit\s*mq\b',
    'spark': r'\bspark\b|\bpyspark\b',
    'snowflake': r'\bsnowflake\b',
    'databricks': r'\bdatabricks\b',
    'dbt': r'\bdbt\b',
    'airflow': r'\bairflow\b',
    'bigquery': r'\bbigquery\b|\bbig\s+query\b',
    'redshift': r'\bredshift\b',
    'oracle_db': r'\boracle\b(?:\s+db)?',
    'sqlserver': r'\bsql\s+server\b|\bmssql\b',
    'neo4j': r'\bneo4j\b',
    'etl': r'\betl\b',

    # ── AI/ML Frameworks ──
    'tensorflow': r'\btensorflow\b|\btf\b(?:\s+\d)',
    'pytorch': r'\bpytorch\b|\btorch\b',
    'scikit_learn': r'\bscikit[\-\s]learn\b|\bsklearn\b',
    'pandas': r'\bpandas\b',
    'numpy': r'\bnumpy\b',
    'keras': r'\bkeras\b',
    'huggingface': r'\bhugging\s*face\b|\bhf\b(?:\s+model)',
    'mlflow': r'\bmlflow\b',
    'sagemaker': r'\bsagemaker\b',
    'mlops': r'\bmlops\b',
    'computer_vision': r'\bcomputer\s+vision\b|\bcv\b(?:\s+model)',
    'nlp': r'\bnlp\b|\bnatural\s+language\s+processing\b',
    'deep_learning': r'\bdeep\s+learning\b',
    'machine_learning': r'\bmachine\s+learning\b|\bml\b',

    # ── AI/LLM Specific ──
    'langchain': r'\blangchain\b',
    'rag': r'\brag\b|\bretrieval[\-\s]augmented\b',
    'vector_databases': r'\bvector\s+(?:db|database|store)\b|\bpinecone\b|\bchromadb\b|\bweaviate\b|\bmilvus\b|\bqdrant\b',
    'pinecone': r'\bpinecone\b',
    'openai_api': r'\bopenai\b(?:\s+api)?',
    'claude_api': r'\bclaude\b(?:\s+api)?|\banthropic\b',
    'prompt_engineering': r'\bprompt\s+engineering\b|\bprompt\s+design\b',
    'fine_tuning': r'\bfine[\-\s]tun(?:e|ing|ed)\b',
    'mcp': r'\bmcp\b|\bmodel\s+context\s+protocol\b',
    'llm': r'\bllm\b|\blarge\s+language\s+model\b',
    'generative_ai': r'\bgenerative\s+ai\b|\bgen\s*ai\b',
    'ai_agents': r'\bai\s+agent\b|\bagent(?:ic|s)\b',
    'transformers': r'\btransformer(?:s)?\b(?:\s+model)?',

    # ── AI Tools ──
    'copilot': r'\bcopilot\b|\bco[\-\s]pilot\b',
    'cursor_ide': r'\bcursor\b(?:\s+ide)?',
    'chatgpt': r'\bchatgpt\b|\bchat\s+gpt\b',
    'claude_tool': r'\bclaude\b',
    'gemini': r'\bgemini\b',
    'codex': r'\bcodex\b',

    # ── Testing ──
    'jest': r'\bjest\b',
    'pytest': r'\bpytest\b',
    'selenium': r'\bselenium\b',
    'cypress': r'\bcypress\b',
    'junit': r'\bjunit\b',
    'mocha': r'\bmocha\b',
    'playwright': r'\bplaywright\b',
    'unit_testing': r'\bunit\s+test',
    'integration_testing': r'\bintegration\s+test',
    'tdd': r'\btdd\b|\btest[\-\s]driven\b',

    # ── Methodologies ──
    'agile': r'\bagile\b',
    'scrum': r'\bscrum\b',
    'kanban': r'\bkanban\b',
    'devops_practice': r'\bdevops\b',
    'sre': r'\bsre\b|\bsite\s+reliability\b',

    # ── Mobile ──
    'ios': r'\bios\b',
    'android': r'\bandroid\b',
    'react_native': r'\breact\s+native\b',
    'flutter': r'\bflutter\b',

    # ── Security ──
    'security': r'\bsecurity\b|\bcybersecurity\b',
    'oauth': r'\boauth\b',
    'encryption': r'\bencryption\b|\bcryptograph',
}

# Compile all patterns
compiled_patterns = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in TECH_PATTERNS.items()
}

print(f"[{elapsed()}] Scanning {len(compiled_patterns)} technology patterns across {len(descriptions_cleaned)} rows ...")

# Build tech matrix
tech_names = sorted(compiled_patterns.keys())
tech_columns = {name: [] for name in tech_names}

for idx, text in enumerate(descriptions_cleaned):
    text_lower = text.lower() if text else ""
    for name in tech_names:
        tech_columns[name].append(bool(compiled_patterns[name].search(text_lower)) if text_lower else False)

    if (idx + 1) % 10000 == 0:
        print(f"[{elapsed()}]   {idx + 1}/{len(descriptions_cleaned)} rows scanned")

# Build parquet table
tech_arrays = {'uid': pa.array(uids, type=pa.string())}
for name in tech_names:
    tech_arrays[name] = pa.array(tech_columns[name], type=pa.bool_())

tech_table = pa.table(tech_arrays)
tech_path = OUT / "swe_tech_matrix.parquet"
pq.write_table(tech_table, tech_path)

print(f"[{elapsed()}] Tech matrix saved: {len(uids)} rows x {len(tech_names)} technologies")

# Show top 20 most mentioned techs
tech_counts = {name: sum(tech_columns[name]) for name in tech_names}
top_20 = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"  Top 20 technologies:")
for name, count in top_20:
    print(f"    {name}: {count} ({100*count/len(uids):.1f}%)")

del tech_columns, tech_arrays
gc.collect()

# ============================================================================
# Step 5: Asaniczka structured skills
# ============================================================================
print(f"\n[{elapsed()}] Step 5: Parsing asaniczka structured skills ...")

con = duckdb.connect()
skills_rows = con.execute(f"""
    SELECT uid, skills_raw
    FROM read_parquet('{DATA}')
    WHERE {BASE_FILTER}
      AND source = 'kaggle_asaniczka'
      AND skills_raw IS NOT NULL
      AND LENGTH(skills_raw) > 0
""").fetchall()
con.close()

skill_uids = []
skill_values = []
for uid, skills_raw in skills_rows:
    skills = [s.strip() for s in skills_raw.split(',') if s.strip()]
    for skill in skills:
        skill_uids.append(uid)
        skill_values.append(skill)

skills_table = pa.table({
    'uid': pa.array(skill_uids, type=pa.string()),
    'skill': pa.array(skill_values, type=pa.string()),
})
skills_path = OUT / "asaniczka_structured_skills.parquet"
pq.write_table(skills_table, skills_path)

n_unique_uids = len(set(skill_uids))
n_unique_skills = len(set(skill_values))
print(f"[{elapsed()}] Asaniczka skills saved: {len(skill_uids)} skill mentions from {n_unique_uids} rows, {n_unique_skills} unique skills")

del skill_uids, skill_values, skills_rows
gc.collect()

# ============================================================================
# Step 6: Calibration table
# ============================================================================
print(f"\n[{elapsed()}] Step 6: Building calibration table ...")

import scipy.stats as stats

con = duckdb.connect()

# Load metrics data per source
metrics_query = f"""
    SELECT
        uid,
        source,
        period,
        seniority_final,
        seniority_3level,
        description_length,
        yoe_extracted,
        is_aggregator,
        is_remote_inferred,
        description,
        description_core_llm,
        llm_extraction_coverage
    FROM read_parquet('{DATA}')
    WHERE {BASE_FILTER}
"""

arshkon = con.execute(metrics_query + " AND source = 'kaggle_arshkon'").fetchdf()
asaniczka = con.execute(metrics_query + " AND source = 'kaggle_asaniczka'").fetchdf()
scraped = con.execute(metrics_query + " AND source = 'scraped'").fetchdf()
con.close()

print(f"[{elapsed()}] Loaded metrics: arshkon={len(arshkon)}, asaniczka={len(asaniczka)}, scraped={len(scraped)}")

# Load tech matrix for tech counts
tech_df = pq.read_table(tech_path).to_pandas()
# Merge tech counts
tech_cols = [c for c in tech_df.columns if c != 'uid']
tech_df['tech_count'] = tech_df[tech_cols].sum(axis=1)

# AI-related columns
ai_cols = [c for c in tech_cols if c in [
    'llm', 'generative_ai', 'ai_agents', 'openai_api', 'claude_api', 'claude_tool',
    'chatgpt', 'copilot', 'cursor_ide', 'gemini', 'codex', 'prompt_engineering',
    'fine_tuning', 'langchain', 'rag', 'vector_databases', 'mcp', 'transformers',
    'huggingface'
]]
tech_df['ai_keyword_count'] = tech_df[ai_cols].sum(axis=1)
tech_df['has_ai_keyword'] = tech_df['ai_keyword_count'] > 0

# ML columns
ml_cols = [c for c in tech_cols if c in [
    'tensorflow', 'pytorch', 'scikit_learn', 'keras', 'mlflow', 'sagemaker',
    'mlops', 'deep_learning', 'machine_learning', 'computer_vision', 'nlp'
]]
tech_df['has_ml_keyword'] = tech_df[ml_cols].sum(axis=1) > 0

tech_uid_map = tech_df.set_index('uid')[['tech_count', 'ai_keyword_count', 'has_ai_keyword', 'has_ml_keyword']].to_dict('index')


def add_tech_cols(df):
    df['tech_count'] = df['uid'].map(lambda u: tech_uid_map.get(u, {}).get('tech_count', 0))
    df['ai_keyword_count'] = df['uid'].map(lambda u: tech_uid_map.get(u, {}).get('ai_keyword_count', 0))
    df['has_ai_keyword'] = df['uid'].map(lambda u: tech_uid_map.get(u, {}).get('has_ai_keyword', False))
    df['has_ml_keyword'] = df['uid'].map(lambda u: tech_uid_map.get(u, {}).get('has_ml_keyword', False))
    return df


arshkon = add_tech_cols(arshkon)
asaniczka = add_tech_cols(asaniczka)
scraped = add_tech_cols(scraped)

# Define text-based metrics
management_re = re.compile(r'\b(?:manage|lead|mentor|supervise|oversee|coach|direct\s+reports?|team\s+lead)\b', re.IGNORECASE)
scope_re = re.compile(r'\b(?:architect|design\s+system|end[\-\s]to[\-\s]end|full[\-\s]stack|cross[\-\s]functional|strategic|roadmap|stakeholder)\b', re.IGNORECASE)
soft_skill_re = re.compile(r'\b(?:communicate|collaboration|teamwork|interpersonal|problem[\-\s]solv|critical\s+think|leadership|adaptab|creative|initiative)\b', re.IGNORECASE)


def text_flags(df):
    texts = df['description'].fillna('')
    df['has_management'] = texts.apply(lambda t: bool(management_re.search(t)))
    df['has_scope'] = texts.apply(lambda t: bool(scope_re.search(t)))
    df['has_soft_skill'] = texts.apply(lambda t: bool(soft_skill_re.search(t)))
    return df


arshkon = text_flags(arshkon)
asaniczka = text_flags(asaniczka)
scraped = text_flags(scraped)


def cohens_d(group1, group2):
    """Compute Cohen's d for two groups (continuous)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m2 - m1) / pooled_std


def prop_diff(s1, s2):
    """Proportion difference and effect size (h) for binary."""
    p1 = s1.mean() if len(s1) > 0 else float('nan')
    p2 = s2.mean() if len(s2) > 0 else float('nan')
    diff = p2 - p1
    # Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))) if not (np.isnan(p1) or np.isnan(p2)) else float('nan')
    return p1, p2, diff, h


# Build calibration rows
calibration_rows = []


def add_continuous_metric(name, arshkon_vals, asaniczka_vals, scraped_vals, stat='mean'):
    a_vals = arshkon_vals.dropna()
    b_vals = asaniczka_vals.dropna()
    s_vals = scraped_vals.dropna()

    if stat == 'mean':
        a_val = a_vals.mean() if len(a_vals) > 0 else float('nan')
        b_val = b_vals.mean() if len(b_vals) > 0 else float('nan')
        s_val = s_vals.mean() if len(s_vals) > 0 else float('nan')
    elif stat == 'median':
        a_val = a_vals.median() if len(a_vals) > 0 else float('nan')
        b_val = b_vals.median() if len(b_vals) > 0 else float('nan')
        s_val = s_vals.median() if len(s_vals) > 0 else float('nan')

    within_d = cohens_d(a_vals, b_vals)
    cross_d = cohens_d(a_vals, s_vals)
    ratio = cross_d / within_d if within_d != 0 and not np.isnan(within_d) and not np.isnan(cross_d) else float('nan')

    calibration_rows.append({
        'metric': name,
        'type': 'continuous',
        'stat': stat,
        'arshkon_value': round(a_val, 2) if not np.isnan(a_val) else None,
        'asaniczka_value': round(b_val, 2) if not np.isnan(b_val) else None,
        'scraped_value': round(s_val, 2) if not np.isnan(s_val) else None,
        'within_2024_effect_d': round(within_d, 4) if not np.isnan(within_d) else None,
        'cross_period_effect_d': round(cross_d, 4) if not np.isnan(cross_d) else None,
        'calibration_ratio': round(ratio, 2) if not np.isnan(ratio) else None,
        'arshkon_n': len(a_vals),
        'asaniczka_n': len(b_vals),
        'scraped_n': len(s_vals),
    })


def add_binary_metric(name, arshkon_vals, asaniczka_vals, scraped_vals):
    a_p, b_p, within_diff, within_h = prop_diff(arshkon_vals.astype(float), asaniczka_vals.astype(float))
    a_p2, s_p, cross_diff, cross_h = prop_diff(arshkon_vals.astype(float), scraped_vals.astype(float))

    ratio = cross_h / within_h if within_h != 0 and not np.isnan(within_h) and not np.isnan(cross_h) else float('nan')

    calibration_rows.append({
        'metric': name,
        'type': 'binary',
        'stat': 'proportion',
        'arshkon_value': round(a_p, 4) if not np.isnan(a_p) else None,
        'asaniczka_value': round(b_p, 4) if not np.isnan(b_p) else None,
        'scraped_value': round(s_p, 4) if not np.isnan(s_p) else None,
        'within_2024_effect_d': round(within_h, 4) if not np.isnan(within_h) else None,
        'cross_period_effect_d': round(cross_h, 4) if not np.isnan(cross_h) else None,
        'calibration_ratio': round(ratio, 2) if not np.isnan(ratio) else None,
        'arshkon_n': len(arshkon_vals),
        'asaniczka_n': len(asaniczka_vals),
        'scraped_n': len(scraped_vals),
    })


# Continuous metrics
add_continuous_metric('description_length_mean', arshkon['description_length'], asaniczka['description_length'], scraped['description_length'], 'mean')
add_continuous_metric('description_length_median', arshkon['description_length'], asaniczka['description_length'], scraped['description_length'], 'median')
add_continuous_metric('yoe_extracted_mean', arshkon['yoe_extracted'], asaniczka['yoe_extracted'], scraped['yoe_extracted'], 'mean')
add_continuous_metric('yoe_extracted_median', arshkon['yoe_extracted'], asaniczka['yoe_extracted'], scraped['yoe_extracted'], 'median')
add_continuous_metric('tech_count_mean', arshkon['tech_count'], asaniczka['tech_count'], scraped['tech_count'], 'mean')
add_continuous_metric('tech_count_median', arshkon['tech_count'], asaniczka['tech_count'], scraped['tech_count'], 'median')
add_continuous_metric('ai_keyword_count_mean', arshkon['ai_keyword_count'], asaniczka['ai_keyword_count'], scraped['ai_keyword_count'], 'mean')

# Binary metrics
add_binary_metric('has_ai_keyword', arshkon['has_ai_keyword'], asaniczka['has_ai_keyword'], scraped['has_ai_keyword'])
add_binary_metric('has_ml_keyword', arshkon['has_ml_keyword'], asaniczka['has_ml_keyword'], scraped['has_ml_keyword'])
add_binary_metric('has_management_indicator', arshkon['has_management'], asaniczka['has_management'], scraped['has_management'])
add_binary_metric('has_scope_term', arshkon['has_scope'], asaniczka['has_scope'], scraped['has_scope'])
add_binary_metric('has_soft_skill', arshkon['has_soft_skill'], asaniczka['has_soft_skill'], scraped['has_soft_skill'])
add_binary_metric('is_aggregator', arshkon['is_aggregator'], asaniczka['is_aggregator'], scraped['is_aggregator'])

# Seniority-based metrics (seniority_final)
add_binary_metric('seniority_entry_rate', (arshkon['seniority_final'] == 'entry'), (asaniczka['seniority_final'] == 'entry'), (scraped['seniority_final'] == 'entry'))
add_binary_metric('seniority_associate_rate', (arshkon['seniority_final'] == 'associate'), (asaniczka['seniority_final'] == 'associate'), (scraped['seniority_final'] == 'associate'))
add_binary_metric('seniority_midsrnr_rate', (arshkon['seniority_final'] == 'mid-senior'), (asaniczka['seniority_final'] == 'mid-senior'), (scraped['seniority_final'] == 'mid-senior'))
add_binary_metric('seniority_director_rate', (arshkon['seniority_final'] == 'director'), (asaniczka['seniority_final'] == 'director'), (scraped['seniority_final'] == 'director'))
add_binary_metric('seniority_unknown_rate', (arshkon['seniority_final'] == 'unknown'), (asaniczka['seniority_final'] == 'unknown'), (scraped['seniority_final'] == 'unknown'))

# YOE-derived binary
add_binary_metric('has_yoe', arshkon['yoe_extracted'].notna(), asaniczka['yoe_extracted'].notna(), scraped['yoe_extracted'].notna())

# Remote
add_binary_metric('is_remote', arshkon['is_remote_inferred'].fillna(False), asaniczka['is_remote_inferred'].fillna(False), scraped['is_remote_inferred'].fillna(False))

# Description quality
add_binary_metric('has_llm_text', (arshkon['llm_extraction_coverage'] == 'labeled'), (asaniczka['llm_extraction_coverage'] == 'labeled'), (scraped['llm_extraction_coverage'] == 'labeled'))

# Tech-specific binary metrics
for tech in ['python', 'java', 'javascript', 'typescript', 'react', 'aws', 'kubernetes', 'docker', 'sql', 'llm', 'copilot']:
    a_tech = arshkon['uid'].isin(tech_df[tech_df[tech]]['uid'])
    b_tech = asaniczka['uid'].isin(tech_df[tech_df[tech]]['uid'])
    s_tech = scraped['uid'].isin(tech_df[tech_df[tech]]['uid'])
    add_binary_metric(f'tech_{tech}', a_tech, b_tech, s_tech)

import pandas as pd
cal_df = pd.DataFrame(calibration_rows)
cal_path = OUT / "calibration_table.csv"
cal_df.to_csv(cal_path, index=False)

print(f"[{elapsed()}] Calibration table saved: {len(cal_df)} metrics")
print(f"  Calibration ratio summary (cross-period/within-2024 effect size):")
valid_ratios = cal_df['calibration_ratio'].dropna()
print(f"    Mean: {valid_ratios.mean():.2f}, Median: {valid_ratios.median():.2f}")
print(f"    Range: [{valid_ratios.min():.2f}, {valid_ratios.max():.2f}]")

del arshkon, asaniczka, scraped, tech_df
gc.collect()

# ============================================================================
# Summary
# ============================================================================
wall_time = time.time() - WALL_START
print(f"\n{'='*60}")
print(f"Shared preprocessing complete in {wall_time:.0f}s ({wall_time/60:.1f}m)")
print(f"{'='*60}")
print(f"Artifacts saved to: {OUT}")
print(f"  1. swe_cleaned_text.parquet — {len(uids)} rows")
print(f"     Text source: {dict(text_source_counts)}")
print(f"  2. swe_embeddings.npy — {len(uids)} x 384 float32")
print(f"     swe_embedding_index.parquet — uid index")
print(f"  3. swe_tech_matrix.parquet — {len(uids)} x {len(tech_names)} booleans")
print(f"  4. company_stoplist.txt — {len(sorted_tokens)} tokens")
print(f"  5. asaniczka_structured_skills.parquet — {n_unique_uids} rows, {n_unique_skills} unique skills")
print(f"  6. calibration_table.csv — {len(calibration_rows)} metrics")
