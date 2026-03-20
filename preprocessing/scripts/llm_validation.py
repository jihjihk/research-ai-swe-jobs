#!/usr/bin/env python3
"""
LLM-based validation (Tier 2) for the SWE labor market preprocessing pipeline.
Uses claude CLI to validate classification, seniority, boilerplate, and ghost job labels.

Memory constraint: 31GB RAM — uses pyarrow iter_batches for large reads.
Optimized: single-pass sampling to avoid repeated full scans.
"""

import json
import subprocess
import time
import random
import sys
import os
from datetime import datetime
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import pyarrow as pa

CLAUDE_CLI = "/home/jihgaboot/.local/bin/claude"
STAGE5_PATH = "/home/jihgaboot/gabor/job-research/preprocessing/intermediate/stage5_classification.parquet"
STAGE8_PATH = "/home/jihgaboot/gabor/job-research/preprocessing/intermediate/stage8_final.parquet"
OUTPUT_RESPONSES = "/home/jihgaboot/gabor/job-research/preprocessing/intermediate/llm_validation_responses.json"
OUTPUT_REPORT = "/home/jihgaboot/gabor/job-research/preprocessing/logs/llm_validation_report.md"

CALL_DELAY = 1.5  # seconds between claude calls

random.seed(42)  # reproducibility

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def call_claude(prompt, retries=2):
    """Call claude CLI and parse JSON response."""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                [CLAUDE_CLI, "-p", prompt, "--output-format", "json"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"  claude returned code {result.returncode}: {result.stderr[:200]}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(3)
                    continue
                return None

            outer = json.loads(result.stdout)
            response_text = outer.get("result", "")

            # Try to extract JSON from the response
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            # Find the JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                return parsed
            else:
                return {"raw_response": response_text}

        except subprocess.TimeoutExpired:
            print(f"  claude call timed out (attempt {attempt+1})", file=sys.stderr)
            if attempt < retries:
                time.sleep(3)
                continue
            return None
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(3)
                continue
            return {"raw_response": result.stdout[:500] if result else ""}
        except Exception as e:
            print(f"  Unexpected error: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(3)
                continue
            return None

    return None


def safe_str(val, max_len=None):
    """Safely convert a value to string, truncating if needed."""
    if val is None:
        return ""
    s = str(val)
    if max_len:
        s = s[:max_len]
    return s


def sample_all_stage5():
    """Single-pass through stage5 to collect all needed samples using reservoir sampling."""
    print("Scanning stage5 parquet (single pass)...")

    cols = ['job_id', 'title', 'company_name', 'description', 'description_core',
            'description_length', 'core_length', 'is_swe', 'is_swe_adjacent',
            'seniority_final', 'boilerplate_flag']

    # Buckets for reservoir sampling
    buckets = {
        'swe_true': [],       # Task 1: is_swe=True (need 50)
        'swe_adj': [],        # Task 1: is_swe_adjacent=True, not is_swe (need 25)
        'swe_false_border': [],  # Task 1: borderline non-SWE (need 25)
        'sen_entry': [],      # Task 2: SWE + entry level (need 25)
        'sen_mid': [],        # Task 2: SWE + mid-senior level (need 25)
        'sen_unknown': [],    # Task 2: SWE + unknown (need 25)
        'sen_associate': [],  # Task 2: SWE + associate (need 25)
        'boilerplate': [],    # Task 3: SWE + boilerplate removed (need 50)
    }
    targets = {
        'swe_true': 50, 'swe_adj': 25, 'swe_false_border': 25,
        'sen_entry': 25, 'sen_mid': 25, 'sen_unknown': 25, 'sen_associate': 25,
        'boilerplate': 50,
    }
    counts = {k: 0 for k in buckets}

    pf = pq.ParquetFile(STAGE5_PATH)
    total_rows = 0

    for batch in pf.iter_batches(batch_size=50000, columns=cols):
        tbl = pa.Table.from_batches([batch])
        for i in range(tbl.num_rows):
            total_rows += 1
            row = {col: tbl.column(col)[i].as_py() for col in cols}

            is_swe = row.get('is_swe', False)
            is_adj = row.get('is_swe_adjacent', False)
            title_lower = safe_str(row.get('title', '')).lower()
            sen = row.get('seniority_final', '')

            # Task 1 buckets
            if is_swe:
                counts['swe_true'] += 1
                idx = counts['swe_true']
                if len(buckets['swe_true']) < targets['swe_true']:
                    buckets['swe_true'].append(row)
                else:
                    j = random.randint(0, idx - 1)
                    if j < targets['swe_true']:
                        buckets['swe_true'][j] = row

            if is_adj and not is_swe:
                counts['swe_adj'] += 1
                idx = counts['swe_adj']
                if len(buckets['swe_adj']) < targets['swe_adj']:
                    buckets['swe_adj'].append(row)
                else:
                    j = random.randint(0, idx - 1)
                    if j < targets['swe_adj']:
                        buckets['swe_adj'][j] = row

            if not is_swe and not is_adj and ('engineer' in title_lower or 'developer' in title_lower):
                counts['swe_false_border'] += 1
                idx = counts['swe_false_border']
                if len(buckets['swe_false_border']) < targets['swe_false_border']:
                    buckets['swe_false_border'].append(row)
                else:
                    j = random.randint(0, idx - 1)
                    if j < targets['swe_false_border']:
                        buckets['swe_false_border'][j] = row

            # Task 2 buckets (SWE only)
            if is_swe:
                bucket_key = None
                if sen == 'entry level':
                    bucket_key = 'sen_entry'
                elif sen == 'mid-senior level':
                    bucket_key = 'sen_mid'
                elif sen == 'unknown':
                    bucket_key = 'sen_unknown'
                elif sen == 'associate':
                    bucket_key = 'sen_associate'

                if bucket_key:
                    counts[bucket_key] += 1
                    idx = counts[bucket_key]
                    if len(buckets[bucket_key]) < targets[bucket_key]:
                        buckets[bucket_key].append(row)
                    else:
                        j = random.randint(0, idx - 1)
                        if j < targets[bucket_key]:
                            buckets[bucket_key][j] = row

            # Task 3: boilerplate removed
            if is_swe:
                desc = safe_str(row.get('description', ''))
                core = safe_str(row.get('description_core', ''))
                dl = row.get('description_length') or len(desc)
                cl = row.get('core_length') or len(core)
                if desc != core and desc and core and dl > 0 and cl < dl * 0.8:
                    counts['boilerplate'] += 1
                    idx = counts['boilerplate']
                    if len(buckets['boilerplate']) < targets['boilerplate']:
                        buckets['boilerplate'].append(row)
                    else:
                        j = random.randint(0, idx - 1)
                        if j < targets['boilerplate']:
                            buckets['boilerplate'][j] = row

    print(f"  Scanned {total_rows} rows")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} samples (from {counts[k]} candidates)")

    return buckets


def sample_all_stage8():
    """Single-pass through stage8 to collect ghost job samples."""
    print("Scanning stage8 parquet (single pass)...")

    cols = ['job_id', 'title', 'description', 'is_swe', 'seniority_3level', 'ghost_job_risk']
    buckets = {
        'ghost_high': [],
        'ghost_low': [],
    }
    targets = {'ghost_high': 25, 'ghost_low': 25}
    counts = {k: 0 for k in buckets}

    pf = pq.ParquetFile(STAGE8_PATH)
    total_rows = 0

    for batch in pf.iter_batches(batch_size=50000, columns=cols):
        tbl = pa.Table.from_batches([batch])
        for i in range(tbl.num_rows):
            total_rows += 1
            row = {col: tbl.column(col)[i].as_py() for col in cols}

            if not row.get('is_swe') or row.get('seniority_3level') != 'junior':
                continue

            ghost = row.get('ghost_job_risk', '')
            if ghost == 'high':
                counts['ghost_high'] += 1
                idx = counts['ghost_high']
                if len(buckets['ghost_high']) < targets['ghost_high']:
                    buckets['ghost_high'].append(row)
                else:
                    j = random.randint(0, idx - 1)
                    if j < targets['ghost_high']:
                        buckets['ghost_high'][j] = row
            elif ghost == 'low':
                counts['ghost_low'] += 1
                idx = counts['ghost_low']
                if len(buckets['ghost_low']) < targets['ghost_low']:
                    buckets['ghost_low'].append(row)
                else:
                    j = random.randint(0, idx - 1)
                    if j < targets['ghost_low']:
                        buckets['ghost_low'][j] = row

    print(f"  Scanned {total_rows} rows")
    for k, v in buckets.items():
        print(f"  {k}: {len(v)} samples (from {counts[k]} candidates)")

    return buckets


# ============================================================
# Task runners
# ============================================================

def run_task1(buckets):
    """Task 1: SWE Classification Validation."""
    print("\n=== Task 1: SWE Classification Validation ===")

    all_samples = []
    for r in buckets['swe_true']:
        r['_pipeline_label'] = 'SWE'
        all_samples.append(r)
    for r in buckets['swe_adj']:
        r['_pipeline_label'] = 'SWE_ADJACENT'
        all_samples.append(r)
    for r in buckets['swe_false_border']:
        r['_pipeline_label'] = 'NOT_SWE'
        all_samples.append(r)

    results = []
    for i, row in enumerate(all_samples):
        title = safe_str(row.get('title', ''), 80)
        print(f"  [{i+1}/{len(all_samples)}] {title}")
        desc = safe_str(row.get('description', ''), 400)
        company = safe_str(row.get('company_name', ''))

        prompt = f"""Title: {title}
Company: {company}
Description (first 400 chars): {desc}

Is this a software engineering role? Answer: SWE / SWE_ADJACENT / NOT_SWE
One sentence reason.
Respond as JSON: {{"classification": "...", "reason": "..."}}"""

        resp = call_claude(prompt)
        time.sleep(CALL_DELAY)

        results.append({
            'task': 'swe_classification',
            'job_id': row.get('job_id'),
            'title': row.get('title'),
            'company': row.get('company_name'),
            'pipeline_label': row['_pipeline_label'],
            'llm_response': resp
        })

    return results


def run_task2(buckets):
    """Task 2: Seniority Validation."""
    print("\n=== Task 2: Seniority Validation ===")

    all_samples = []
    for level_key, level_name in [('sen_entry', 'entry level'), ('sen_mid', 'mid-senior level'),
                                   ('sen_unknown', 'unknown'), ('sen_associate', 'associate')]:
        for r in buckets[level_key]:
            r['_pipeline_seniority'] = level_name
            all_samples.append(r)

    results = []
    for i, row in enumerate(all_samples):
        title = safe_str(row.get('title', ''), 80)
        print(f"  [{i+1}/{len(all_samples)}] {title}")
        desc = safe_str(row.get('description', ''), 500)

        prompt = f"""Title: {title}
Description (first 500 chars): {desc}

What seniority level is this role? Answer one of: entry level / associate / mid-senior level / director / unknown
What signals support this?
Respond as JSON: {{"seniority": "...", "signals": "..."}}"""

        resp = call_claude(prompt)
        time.sleep(CALL_DELAY)

        results.append({
            'task': 'seniority_validation',
            'job_id': row.get('job_id'),
            'title': row.get('title'),
            'pipeline_seniority': row['_pipeline_seniority'],
            'llm_response': resp
        })

    return results


def run_task3(buckets):
    """Task 3: Boilerplate Validation."""
    print("\n=== Task 3: Boilerplate Validation ===")

    results = []
    rows = buckets['boilerplate']
    for i, row in enumerate(rows):
        title = safe_str(row.get('title', ''), 80)
        print(f"  [{i+1}/{len(rows)}] {title}")
        desc_orig = safe_str(row.get('description', ''), 300)
        desc_core = safe_str(row.get('description_core', ''), 300)

        prompt = f"""ORIGINAL (first 300 chars): {desc_orig}
---
CORE (first 300 chars): {desc_core}

Was the boilerplate removal correct? Did it preserve job requirements? Did it remove actual job content?
Respond as JSON: {{"quality": "correct" | "over_removed" | "under_removed", "issue": "..."}}"""

        resp = call_claude(prompt)
        time.sleep(CALL_DELAY)

        results.append({
            'task': 'boilerplate_validation',
            'job_id': row.get('job_id'),
            'title': row.get('title'),
            'description_length': row.get('description_length'),
            'core_length': row.get('core_length'),
            'llm_response': resp
        })

    return results


def run_task4(buckets):
    """Task 4: Ghost Job Validation."""
    print("\n=== Task 4: Ghost Job Validation ===")

    all_samples = []
    for r in buckets['ghost_high']:
        r['_pipeline_ghost'] = 'high'
        all_samples.append(r)
    for r in buckets['ghost_low']:
        r['_pipeline_ghost'] = 'low'
        all_samples.append(r)

    results = []
    for i, row in enumerate(all_samples):
        title = safe_str(row.get('title', ''), 80)
        print(f"  [{i+1}/{len(all_samples)}] {title}")
        desc = safe_str(row.get('description', ''), 500)

        prompt = f"""Title: {title}
Seniority: entry level
Description (first 500 chars): {desc}

Are the requirements realistic for an entry-level role? Or do they suggest this is a "ghost job" (posting with inflated requirements)?
Respond as JSON: {{"assessment": "realistic" | "inflated" | "ghost_likely", "reason": "..."}}"""

        resp = call_claude(prompt)
        time.sleep(CALL_DELAY)

        results.append({
            'task': 'ghost_job_validation',
            'job_id': row.get('job_id'),
            'title': row.get('title'),
            'pipeline_ghost_risk': row['_pipeline_ghost'],
            'llm_response': resp
        })

    return results


# ============================================================
# Report Generation
# ============================================================
def generate_report(all_results):
    """Generate comprehensive validation report."""

    task1 = [r for r in all_results if r['task'] == 'swe_classification']
    task2 = [r for r in all_results if r['task'] == 'seniority_validation']
    task3 = [r for r in all_results if r['task'] == 'boilerplate_validation']
    task4 = [r for r in all_results if r['task'] == 'ghost_job_validation']

    lines = []
    lines.append("# LLM Validation Report (Tier 2)")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Validator: Claude (via `claude -p` CLI), single-shot classification")
    lines.append(f"Input: stage5_classification.parquet (tasks 1-3), stage8_final.parquet (task 4)")
    lines.append(f"Total LLM calls: {len(all_results)}")
    lines.append("")

    # ---- Task 1: SWE Classification ----
    lines.append("## 1. SWE Classification Validation")
    lines.append(f"\nSample: {len(task1)} postings (50 SWE, 25 SWE_ADJACENT, 25 NOT_SWE borderline)")
    lines.append("")

    def normalize_swe_label(resp):
        if not resp or 'classification' not in resp:
            return 'PARSE_ERROR'
        c = resp['classification'].upper().strip()
        if 'NOT' in c:
            return 'NOT_SWE'
        elif 'ADJACENT' in c:
            return 'SWE_ADJACENT'
        elif 'SWE' in c:
            return 'SWE'
        return 'PARSE_ERROR'

    agree_count = 0
    disagree_examples = []
    confusion = defaultdict(Counter)

    for r in task1:
        pipeline = r['pipeline_label']
        llm = normalize_swe_label(r.get('llm_response'))
        confusion[pipeline][llm] += 1
        if pipeline == llm:
            agree_count += 1
        else:
            if llm != 'PARSE_ERROR':
                disagree_examples.append(r)

    valid_count = sum(1 for r in task1 if normalize_swe_label(r.get('llm_response')) != 'PARSE_ERROR')
    agreement_rate = agree_count / valid_count * 100 if valid_count else 0

    lines.append(f"### Agreement rate: {agreement_rate:.1f}% ({agree_count}/{valid_count} valid responses)")
    lines.append("")

    # Confusion matrix
    labels = ['SWE', 'SWE_ADJACENT', 'NOT_SWE']
    lines.append("### Confusion matrix (rows = pipeline, cols = LLM)")
    lines.append("")
    header = "| Pipeline \\ LLM | " + " | ".join(labels) + " | PARSE_ERROR |"
    lines.append(header)
    lines.append("|" + "---|" * (len(labels) + 2))
    for pl in labels:
        row_vals = [str(confusion[pl][ll]) for ll in labels]
        row_vals.append(str(confusion[pl]['PARSE_ERROR']))
        lines.append(f"| **{pl}** | " + " | ".join(row_vals) + " |")
    lines.append("")

    # Per-stratum agreement
    for stratum in labels:
        stratum_items = [r for r in task1 if r['pipeline_label'] == stratum]
        stratum_agree = sum(1 for r in stratum_items if normalize_swe_label(r.get('llm_response')) == stratum)
        stratum_valid = sum(1 for r in stratum_items if normalize_swe_label(r.get('llm_response')) != 'PARSE_ERROR')
        rate = stratum_agree / stratum_valid * 100 if stratum_valid else 0
        lines.append(f"- **{stratum}** stratum agreement: {rate:.1f}% ({stratum_agree}/{stratum_valid})")
    lines.append("")

    # Disagreement examples
    if disagree_examples:
        lines.append("### Disagreement examples")
        lines.append("")
        for ex in disagree_examples[:10]:
            llm_label = normalize_swe_label(ex.get('llm_response'))
            reason = ex.get('llm_response', {}).get('reason', 'N/A')
            lines.append(f"- **\"{ex['title']}\"** at {ex.get('company', 'N/A')}")
            lines.append(f"  - Pipeline: {ex['pipeline_label']}, LLM: {llm_label}")
            lines.append(f"  - Reason: {reason}")
        lines.append("")

    # ---- Task 2: Seniority Validation ----
    lines.append("## 2. Seniority Validation")
    lines.append(f"\nSample: {len(task2)} SWE postings (25 per seniority level)")
    lines.append("")

    def normalize_seniority(resp):
        if not resp or 'seniority' not in resp:
            return 'PARSE_ERROR'
        s = resp['seniority'].lower().strip()
        if 'entry' in s or 'junior' in s:
            return 'entry level'
        elif 'associate' in s:
            return 'associate'
        elif 'director' in s or 'executive' in s or 'vp' in s:
            return 'director'
        elif 'mid' in s or 'senior' in s:
            return 'mid-senior level'
        elif 'unknown' in s or 'unclear' in s or 'not' in s:
            return 'unknown'
        return s

    sen_agree = 0
    sen_disagree = []
    sen_confusion = defaultdict(Counter)
    sen_levels = ['entry level', 'associate', 'mid-senior level', 'unknown']

    for r in task2:
        pipeline = r['pipeline_seniority']
        llm = normalize_seniority(r.get('llm_response'))
        sen_confusion[pipeline][llm] += 1
        if pipeline == llm:
            sen_agree += 1
        else:
            if llm != 'PARSE_ERROR':
                sen_disagree.append(r)

    sen_valid = sum(1 for r in task2 if normalize_seniority(r.get('llm_response')) != 'PARSE_ERROR')
    sen_rate = sen_agree / sen_valid * 100 if sen_valid else 0

    lines.append(f"### Agreement rate: {sen_rate:.1f}% ({sen_agree}/{sen_valid} valid responses)")
    lines.append("")

    # Confusion matrix - include all labels that appear in LLM responses
    all_llm_labels = set()
    for r in task2:
        lbl = normalize_seniority(r.get('llm_response'))
        if lbl != 'PARSE_ERROR':
            all_llm_labels.add(lbl)
    all_sen_labels = sorted(set(sen_levels) | all_llm_labels)

    lines.append("### Confusion matrix (rows = pipeline, cols = LLM)")
    lines.append("")
    header = "| Pipeline \\ LLM | " + " | ".join(all_sen_labels) + " | PARSE_ERROR |"
    lines.append(header)
    lines.append("|" + "---|" * (len(all_sen_labels) + 2))
    for pl in sen_levels:
        row_vals = [str(sen_confusion[pl][ll]) for ll in all_sen_labels]
        row_vals.append(str(sen_confusion[pl]['PARSE_ERROR']))
        lines.append(f"| **{pl}** | " + " | ".join(row_vals) + " |")
    lines.append("")

    # Per-stratum
    for stratum in sen_levels:
        stratum_items = [r for r in task2 if r['pipeline_seniority'] == stratum]
        stratum_agree = sum(1 for r in stratum_items if normalize_seniority(r.get('llm_response')) == stratum)
        stratum_valid = sum(1 for r in stratum_items if normalize_seniority(r.get('llm_response')) != 'PARSE_ERROR')
        rate = stratum_agree / stratum_valid * 100 if stratum_valid else 0
        lines.append(f"- **{stratum}** stratum agreement: {rate:.1f}% ({stratum_agree}/{stratum_valid})")
    lines.append("")

    # Disagreement examples
    if sen_disagree:
        lines.append("### Disagreement examples")
        lines.append("")
        for ex in sen_disagree[:10]:
            llm_sen = normalize_seniority(ex.get('llm_response'))
            signals = ex.get('llm_response', {}).get('signals', 'N/A')
            lines.append(f"- **\"{ex['title']}\"**")
            lines.append(f"  - Pipeline: {ex['pipeline_seniority']}, LLM: {llm_sen}")
            lines.append(f"  - Signals: {signals}")
        lines.append("")

    # ---- Task 3: Boilerplate Validation ----
    lines.append("## 3. Boilerplate Removal Validation")
    lines.append(f"\nSample: {len(task3)} SWE postings with boilerplate removed")
    lines.append("")

    bp_counts = Counter()
    bp_issues = []
    for r in task3:
        resp = r.get('llm_response', {})
        quality = resp.get('quality', 'PARSE_ERROR') if resp else 'PARSE_ERROR'
        quality = str(quality).lower().strip().replace('"', '').replace("'", "")
        if 'correct' in quality:
            quality = 'correct'
        elif 'over' in quality:
            quality = 'over_removed'
        elif 'under' in quality:
            quality = 'under_removed'
        bp_counts[quality] += 1
        if quality != 'correct':
            bp_issues.append(r)

    total_bp = sum(bp_counts.values())
    correct_pct = bp_counts.get('correct', 0) / total_bp * 100 if total_bp else 0

    lines.append(f"### Quality distribution")
    lines.append("")
    lines.append(f"| Quality | Count | Percentage |")
    lines.append(f"|---|---|---|")
    for q in ['correct', 'over_removed', 'under_removed']:
        cnt = bp_counts.get(q, 0)
        pct = cnt / total_bp * 100 if total_bp else 0
        lines.append(f"| {q} | {cnt} | {pct:.1f}% |")
    parse_err = sum(v for k, v in bp_counts.items() if k not in ['correct', 'over_removed', 'under_removed'])
    if parse_err:
        lines.append(f"| parse_error/other | {parse_err} | {parse_err/total_bp*100:.1f}% |")
    lines.append("")

    lines.append(f"**Correct removal rate: {correct_pct:.1f}%**")
    lines.append("")

    if bp_issues:
        lines.append("### Issues found")
        lines.append("")
        for ex in bp_issues[:8]:
            resp = ex.get('llm_response', {})
            quality = resp.get('quality', 'unknown') if resp else 'unknown'
            issue = resp.get('issue', 'N/A') if resp else 'N/A'
            dl = ex.get('description_length', '?')
            cl = ex.get('core_length', '?')
            lines.append(f"- **\"{ex['title']}\"** (orig: {dl} chars, core: {cl} chars)")
            lines.append(f"  - Quality: {quality}")
            lines.append(f"  - Issue: {issue}")
        lines.append("")

    # ---- Task 4: Ghost Job Validation ----
    lines.append("## 4. Ghost Job Risk Validation")
    lines.append(f"\nSample: {len(task4)} entry-level SWE postings (25 high-risk, 25 low-risk)")
    lines.append("")

    def normalize_ghost(resp):
        if not resp or 'assessment' not in resp:
            return 'PARSE_ERROR'
        a = resp['assessment'].lower().strip().replace('"', '').replace("'", "")
        if 'ghost' in a:
            return 'ghost_likely'
        elif 'inflat' in a:
            return 'inflated'
        elif 'realistic' in a:
            return 'realistic'
        return a

    ghost_confusion = defaultdict(Counter)
    ghost_disagree = []
    ghost_agree_count = 0

    for r in task4:
        pipeline = r['pipeline_ghost_risk']
        llm = normalize_ghost(r.get('llm_response'))
        ghost_confusion[pipeline][llm] += 1

        is_agree = False
        if pipeline == 'high' and llm in ('inflated', 'ghost_likely'):
            is_agree = True
        elif pipeline == 'low' and llm == 'realistic':
            is_agree = True

        r['_agree'] = is_agree
        r['_llm_assessment'] = llm
        if is_agree:
            ghost_agree_count += 1
        elif llm != 'PARSE_ERROR':
            ghost_disagree.append(r)

    ghost_valid = sum(1 for r in task4 if normalize_ghost(r.get('llm_response')) != 'PARSE_ERROR')
    ghost_rate = ghost_agree_count / ghost_valid * 100 if ghost_valid else 0

    lines.append(f"### Agreement rate: {ghost_rate:.1f}% ({ghost_agree_count}/{ghost_valid} valid responses)")
    lines.append("")

    ghost_labels = ['realistic', 'inflated', 'ghost_likely']
    lines.append("### Cross-tabulation (rows = pipeline risk, cols = LLM assessment)")
    lines.append("")
    header = "| Pipeline Risk \\ LLM | " + " | ".join(ghost_labels) + " | PARSE_ERROR |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ghost_labels) + 2))
    for risk in ['high', 'low']:
        row_vals = [str(ghost_confusion[risk][gl]) for gl in ghost_labels]
        row_vals.append(str(ghost_confusion[risk]['PARSE_ERROR']))
        lines.append(f"| **{risk}** | " + " | ".join(row_vals) + " |")
    lines.append("")

    for risk in ['high', 'low']:
        stratum_items = [r for r in task4 if r['pipeline_ghost_risk'] == risk]
        stratum_agree = sum(1 for r in stratum_items if r.get('_agree', False))
        stratum_valid = sum(1 for r in stratum_items if normalize_ghost(r.get('llm_response')) != 'PARSE_ERROR')
        rate = stratum_agree / stratum_valid * 100 if stratum_valid else 0
        lines.append(f"- **{risk} risk** stratum agreement: {rate:.1f}% ({stratum_agree}/{stratum_valid})")
    lines.append("")

    if ghost_disagree:
        lines.append("### Disagreement examples")
        lines.append("")
        for ex in ghost_disagree[:8]:
            reason = ex.get('llm_response', {}).get('reason', 'N/A') if ex.get('llm_response') else 'N/A'
            lines.append(f"- **\"{ex['title']}\"**")
            lines.append(f"  - Pipeline: {ex['pipeline_ghost_risk']} risk, LLM: {ex.get('_llm_assessment', '?')}")
            lines.append(f"  - Reason: {reason}")
        lines.append("")

    # ---- Summary ----
    lines.append("## 5. Overall Data Quality Assessment")
    lines.append("")
    lines.append("### Summary table")
    lines.append("")
    lines.append("| Validation Task | Sample Size | Agreement Rate | Assessment |")
    lines.append("|---|---|---|---|")

    def quality_label(rate):
        if rate >= 90:
            return "Excellent"
        elif rate >= 80:
            return "Good"
        elif rate >= 70:
            return "Acceptable"
        elif rate >= 60:
            return "Marginal"
        else:
            return "Needs review"

    lines.append(f"| SWE Classification | {len(task1)} | {agreement_rate:.1f}% | {quality_label(agreement_rate)} |")
    lines.append(f"| Seniority Labels | {len(task2)} | {sen_rate:.1f}% | {quality_label(sen_rate)} |")
    lines.append(f"| Boilerplate Removal | {len(task3)} | {correct_pct:.1f}% correct | {quality_label(correct_pct)} |")
    lines.append(f"| Ghost Job Risk | {len(task4)} | {ghost_rate:.1f}% | {quality_label(ghost_rate)} |")
    lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")

    if agreement_rate < 90:
        lines.append("- **SWE classification**: Review misclassified titles, especially at the SWE/SWE_ADJACENT boundary. Consider tightening keyword rules or adding a second-pass LLM check for borderline cases.")
    else:
        lines.append("- **SWE classification**: Pipeline labels are well-calibrated. No changes needed.")

    if sen_rate < 70:
        lines.append("- **Seniority imputation**: Significant disagreement between pipeline and LLM labels. Key issues: the 'unknown' bucket likely contains classifiable roles, and the associate/entry boundary needs refinement. Consider using description-based signals (years of experience mentions, requirement complexity) for a second-pass classification.")
    elif sen_rate < 80:
        lines.append("- **Seniority imputation**: Moderate agreement. Review the associate/mid-senior boundary and unknown classifications that the LLM resolved. Title-only imputation misses description-level signals.")
    elif sen_rate < 90:
        lines.append("- **Seniority imputation**: Good agreement. Minor boundary issues between adjacent levels.")
    else:
        lines.append("- **Seniority imputation**: Strong agreement with LLM labels.")

    if correct_pct < 70:
        lines.append("- **Boilerplate removal**: Significant quality issues. Audit the removal logic, especially for over-removal cases where job requirements are being stripped.")
    elif correct_pct < 85:
        lines.append("- **Boilerplate removal**: Moderate quality. Focus on reducing over-removal cases. Consider a more conservative removal threshold.")
    else:
        lines.append("- **Boilerplate removal**: Removal quality is good. Monitor edge cases.")

    if ghost_rate < 60:
        lines.append("- **Ghost job detection**: The ghost job risk heuristic shows weak agreement with LLM assessment. The heuristic may be capturing a different signal than requirement inflation. Consider recalibrating: use explicit requirement counts, years-of-experience extraction, and credential demands as features.")
    elif ghost_rate < 75:
        lines.append("- **Ghost job detection**: Moderate alignment. The heuristic captures some ghost job signals but misses nuance. Consider adding requirement-complexity features.")
    else:
        lines.append("- **Ghost job detection**: Ghost job risk labels show good alignment with LLM assessment.")

    lines.append("")
    lines.append("### Methodology notes")
    lines.append("")
    lines.append("- LLM validator: Claude (via `claude -p` CLI), single-shot classification")
    lines.append("- Each posting was evaluated independently with no context from other postings")
    lines.append("- Description text was truncated (300-500 chars) to fit prompt constraints; this may cause the LLM to miss signals in longer descriptions")
    lines.append("- Sampling: reservoir sampling with seed=42 for reproducibility")
    lines.append("- Agreement for ghost job risk is directional: pipeline 'high' should map to LLM 'inflated' or 'ghost_likely'; pipeline 'low' should map to LLM 'realistic'")
    lines.append(f"- Validation timestamp: {datetime.now().isoformat()}")
    lines.append("- This report is part of the 3-tier validation protocol (Tier 1: rule-based checks, Tier 2: LLM review, Tier 3: human audit)")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Starting LLM validation at {datetime.now()}")
    print(f"Stage5: {STAGE5_PATH}")
    print(f"Stage8: {STAGE8_PATH}")

    # Phase 1: Sample all data in single passes
    stage5_buckets = sample_all_stage5()
    stage8_buckets = sample_all_stage8()

    all_results = []

    # Phase 2: Run LLM validation tasks
    results1 = run_task1(stage5_buckets)
    all_results.extend(results1)
    print(f"\nTask 1 complete: {len(results1)} results")

    results2 = run_task2(stage5_buckets)
    all_results.extend(results2)
    print(f"\nTask 2 complete: {len(results2)} results")

    results3 = run_task3(stage5_buckets)
    all_results.extend(results3)
    print(f"\nTask 3 complete: {len(results3)} results")

    results4 = run_task4(stage8_buckets)
    all_results.extend(results4)
    print(f"\nTask 4 complete: {len(results4)} results")

    # Phase 3: Save results
    print(f"\nSaving raw responses to {OUTPUT_RESPONSES}")
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if not k.startswith('_') or k in ('_agree', '_llm_assessment')}
        clean_results.append(clean)

    with open(OUTPUT_RESPONSES, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)

    print(f"Generating report to {OUTPUT_REPORT}")
    report = generate_report(all_results)
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)

    print(f"\nValidation complete at {datetime.now()}")
    print(f"Total LLM calls: {len(all_results)}")
    print(f"Report: {OUTPUT_REPORT}")
    print(f"Raw responses: {OUTPUT_RESPONSES}")
