#!/usr/bin/env python3
"""
Build remaining shared artifacts (tech matrix, skills, calibration).
Assumes swe_cleaned_text.parquet, company_stoplist.txt, and embeddings already exist.
"""
import sys
import time
import re
import gc
from pathlib import Path
from collections import Counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parent.parent.parent.parent
DATA = PROJECT / "data" / "unified.parquet"
OUT = PROJECT / "exploration" / "artifacts" / "shared"

WALL_START = time.time()

def elapsed():
    return f"{time.time() - WALL_START:.1f}s"

BASE_FILTER = """
    source_platform = 'linkedin'
    AND is_english = true
    AND date_flag = 'ok'
    AND is_swe = true
"""

# ============================================================================
# Load cleaned text UIDs and descriptions
# ============================================================================
print(f"[{elapsed()}] Loading cleaned text ...", flush=True)
ct = pq.read_table(OUT / "swe_cleaned_text.parquet", columns=['uid', 'description_cleaned'])
uids = ct.column('uid').to_pylist()
descriptions_cleaned = ct.column('description_cleaned').to_pylist()
del ct
gc.collect()
print(f"[{elapsed()}] Loaded {len(uids)} rows", flush=True)

# ============================================================================
# Step 3: Technology mention binary matrix
# ============================================================================
print(f"\n[{elapsed()}] Step 3: Building technology mention matrix ...", flush=True)

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

compiled_patterns = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in TECH_PATTERNS.items()
}

tech_names = sorted(compiled_patterns.keys())
print(f"[{elapsed()}] Scanning {len(compiled_patterns)} technology patterns ...", flush=True)

# Build tech matrix efficiently
tech_columns = {name: np.zeros(len(uids), dtype=bool) for name in tech_names}

for idx, text in enumerate(descriptions_cleaned):
    if not text:
        continue
    text_lower = text.lower()
    for name in tech_names:
        if compiled_patterns[name].search(text_lower):
            tech_columns[name][idx] = True

    if (idx + 1) % 10000 == 0:
        print(f"[{elapsed()}]   {idx + 1}/{len(descriptions_cleaned)} rows scanned", flush=True)

print(f"[{elapsed()}]   {len(descriptions_cleaned)}/{len(descriptions_cleaned)} rows scanned", flush=True)

# Build parquet table
tech_arrays = {'uid': pa.array(uids, type=pa.string())}
for name in tech_names:
    tech_arrays[name] = pa.array(tech_columns[name].tolist(), type=pa.bool_())

tech_table = pa.table(tech_arrays)
tech_path = OUT / "swe_tech_matrix.parquet"
pq.write_table(tech_table, tech_path)

print(f"[{elapsed()}] Tech matrix saved: {len(uids)} rows x {len(tech_names)} technologies", flush=True)

# Show top 20
tech_counts = {name: int(tech_columns[name].sum()) for name in tech_names}
top_20 = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"  Top 20 technologies:", flush=True)
for name, count in top_20:
    print(f"    {name}: {count} ({100*count/len(uids):.1f}%)", flush=True)

del tech_columns, tech_arrays
gc.collect()

# ============================================================================
# Step 5: Asaniczka structured skills
# ============================================================================
print(f"\n[{elapsed()}] Step 5: Parsing asaniczka structured skills ...", flush=True)

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
for uid_val, skills_raw in skills_rows:
    skills = [s.strip() for s in skills_raw.split(',') if s.strip()]
    for skill in skills:
        skill_uids.append(uid_val)
        skill_values.append(skill)

skills_table = pa.table({
    'uid': pa.array(skill_uids, type=pa.string()),
    'skill': pa.array(skill_values, type=pa.string()),
})
skills_path = OUT / "asaniczka_structured_skills.parquet"
pq.write_table(skills_table, skills_path)

n_unique_uids = len(set(skill_uids))
n_unique_skills = len(set(skill_values))
print(f"[{elapsed()}] Asaniczka skills saved: {len(skill_uids)} skill mentions from {n_unique_uids} rows, {n_unique_skills} unique skills", flush=True)

del skill_uids, skill_values, skills_rows
gc.collect()

# ============================================================================
# Step 6: Calibration table
# ============================================================================
print(f"\n[{elapsed()}] Step 6: Building calibration table ...", flush=True)

import pandas as pd
import scipy.stats as stats

con = duckdb.connect()

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

print(f"[{elapsed()}] Loaded metrics: arshkon={len(arshkon)}, asaniczka={len(asaniczka)}, scraped={len(scraped)}", flush=True)

# Load tech matrix for tech counts
tech_df = pq.read_table(tech_path).to_pandas()
tech_cols = [c for c in tech_df.columns if c != 'uid']
tech_df['tech_count'] = tech_df[tech_cols].sum(axis=1)

ai_cols = [c for c in tech_cols if c in [
    'llm', 'generative_ai', 'ai_agents', 'openai_api', 'claude_api', 'claude_tool',
    'chatgpt', 'copilot', 'cursor_ide', 'gemini', 'codex', 'prompt_engineering',
    'fine_tuning', 'langchain', 'rag', 'vector_databases', 'mcp', 'transformers',
    'huggingface'
]]
tech_df['ai_keyword_count'] = tech_df[ai_cols].sum(axis=1)
tech_df['has_ai_keyword'] = tech_df['ai_keyword_count'] > 0

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

# Text-based metrics
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
    p1 = s1.mean() if len(s1) > 0 else float('nan')
    p2 = s2.mean() if len(s2) > 0 else float('nan')
    diff = p2 - p1
    h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))) if not (np.isnan(p1) or np.isnan(p2)) else float('nan')
    return p1, p2, diff, h


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
        'arshkon_value': round(float(a_val), 2) if not np.isnan(a_val) else None,
        'asaniczka_value': round(float(b_val), 2) if not np.isnan(b_val) else None,
        'scraped_value': round(float(s_val), 2) if not np.isnan(s_val) else None,
        'within_2024_effect_d': round(float(within_d), 4) if not np.isnan(within_d) else None,
        'cross_period_effect_d': round(float(cross_d), 4) if not np.isnan(cross_d) else None,
        'calibration_ratio': round(float(ratio), 2) if not np.isnan(ratio) else None,
        'arshkon_n': int(len(a_vals)),
        'asaniczka_n': int(len(b_vals)),
        'scraped_n': int(len(s_vals)),
    })


def add_binary_metric(name, arshkon_vals, asaniczka_vals, scraped_vals):
    a_f = arshkon_vals.astype(float)
    b_f = asaniczka_vals.astype(float)
    s_f = scraped_vals.astype(float)

    a_p, b_p, within_diff, within_h = prop_diff(a_f, b_f)
    a_p2, s_p, cross_diff, cross_h = prop_diff(a_f, s_f)

    ratio = cross_h / within_h if within_h != 0 and not np.isnan(within_h) and not np.isnan(cross_h) else float('nan')

    calibration_rows.append({
        'metric': name,
        'type': 'binary',
        'stat': 'proportion',
        'arshkon_value': round(float(a_p), 4) if not np.isnan(a_p) else None,
        'asaniczka_value': round(float(b_p), 4) if not np.isnan(b_p) else None,
        'scraped_value': round(float(s_p), 4) if not np.isnan(s_p) else None,
        'within_2024_effect_d': round(float(within_h), 4) if not np.isnan(within_h) else None,
        'cross_period_effect_d': round(float(cross_h), 4) if not np.isnan(cross_h) else None,
        'calibration_ratio': round(float(ratio), 2) if not np.isnan(ratio) else None,
        'arshkon_n': int(len(a_f)),
        'asaniczka_n': int(len(b_f)),
        'scraped_n': int(len(s_f)),
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

# Seniority-based metrics
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

cal_df = pd.DataFrame(calibration_rows)
cal_path = OUT / "calibration_table.csv"
cal_df.to_csv(cal_path, index=False)

print(f"[{elapsed()}] Calibration table saved: {len(cal_df)} metrics", flush=True)
valid_ratios = cal_df['calibration_ratio'].dropna()
print(f"  Calibration ratio summary (cross-period/within-2024 effect size):", flush=True)
print(f"    Mean: {valid_ratios.mean():.2f}, Median: {valid_ratios.median():.2f}", flush=True)
print(f"    Range: [{valid_ratios.min():.2f}, {valid_ratios.max():.2f}]", flush=True)

# Print notable metrics
print(f"\n  Notable calibration entries:", flush=True)
for _, row in cal_df.iterrows():
    r = row['calibration_ratio']
    if r is not None and not np.isnan(r) and abs(r) > 2:
        print(f"    {row['metric']}: ratio={r:.2f} (within={row['within_2024_effect_d']}, cross={row['cross_period_effect_d']})", flush=True)

del arshkon, asaniczka, scraped, tech_df
gc.collect()

print(f"\n[{elapsed()}] All remaining artifacts complete.", flush=True)
