#!/usr/bin/env python3
"""
Stage 5: Classification (SWE / SWE-adjacent / control + seniority).

Seniority design (simplified 2026-04-10):
  - `seniority_final` is set from STRONG title-keyword rules only:
    `title_keyword` (junior, senior, lead, principal, staff, director, etc.)
    or `title_manager` (manager/lead with a family role hint). Stage 5 leaves
    rows that lack a strong rule as `seniority_final = 'unknown'`.
  - Stage 10 routes the unknown rows to the LLM and overwrites `seniority_final`
    with the LLM result for routed rows. There is no separate `seniority_llm`
    column — the LLM result lives in `seniority_final` with
    `seniority_final_source = 'llm'`.
  - Native LinkedIn labels (`seniority_native`) are passed through from Stage 1
    as a diagnostic; they no longer feed `seniority_final`.

SWE design:
  - Tier 1 regex, Tier 2 curated title lookup, Tier 2b embedding similarity
    (>= 0.85 = SWE, 0.70-0.85 = SWE_ADJACENT).

Memory-safe: builds lookup dicts from unique titles first, then streams
chunks through for the final write pass. 31GB RAM constraint.

Input:  intermediate/stage4_dedup.parquet  (~1.2M rows)
Output: intermediate/stage5_classification.parquet
"""

import gc
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file, promote_null_schema
from llm_shared import SENIORITY_3LEVEL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"
TIER2_LOOKUP_PATH = INTERMEDIATE_DIR / "tier2_title_lookup.parquet"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage5_classification.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 200_000
TIER2_REQUIRED_COLUMNS = {"title", "llm_swe_classification"}
TIER2_ALLOWED_CLASSES = {"SWE", "SWE_ADJACENT", "NOT_SWE"}

# ---------------------------------------------------------------------------
# 5a. SWE Classification -- Patterns
# ---------------------------------------------------------------------------


def normalize_swe_title_key(title) -> str:
    """
    Canonical SWE-classification key.

    Must stay aligned with Stage 1's `title_normalized` contract so the same
    title variant resolves identically when we build and apply the lookup.
    """
    if pd.isna(title):
        return ""

    value = str(title).lower().strip()
    if not value:
        return ""

    value = re.sub(r"\.net\b", "dotnet", value)
    value = re.sub(r"c#", "csharp", value)
    value = re.sub(r"\b(senior|sr\.?|junior|jr\.?|lead|staff|principal|distinguished)\b", "", value)
    value = re.sub(r"\b(i{1,3}|iv|v)\b", "", value)
    value = re.sub(r"\b[1-5]\b", "", value)
    value = re.sub(r"\s+with\s+(?:an?\s+)?(?:active\s+)?(?:department of defense\s+)?(?:top\s+secret\s+)?(?:secret\s+)?(?:sci\s+)?security\s+clearance\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*[-–—]\s*(remote|hybrid|onsite|on-site)\s*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s*[/,:;()\-]+\s*$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

# Stage 1 regex patterns (canonical, for reference / re-application)
SWE_INCLUDE = re.compile(
    r'(?i)\b('
    r'software\s*(engineer|developer|dev|development\s*engineer)|'
    r'swe\b|full[- ]?stack|front[- ]?end\s*(engineer|developer)|'
    r'back[- ]?end\s*(engineer|developer)|'
    r'web\s*(developer|engineer)|mobile\s*(developer|engineer)|'
    r'devops\s*(engineer)?|platform\s*engineer|'
    r'data\s*engineer|ml\s*engineer|machine\s*learning\s*engineer|'
    r'site\s*reliability\s*(engineer)?|'
    r'ai\s*engineer|llm\s*engineer|agent\s*engineer|'
    r'applied\s*(ai|ml)\s*engineer|prompt\s*engineer|'
    r'infrastructure\s*engineer|cloud\s*engineer|'
    r'application\s*(developer|engineer)|software\s*architect|'
    r'(?:mainframe|salesforce|service\s*now|servicenow)\s*(developer|engineer)|'
    r'founding\s*engineer|member\s*of\s*technical\s*staff|'
    r'product\s*engineer|systems?\s*engineer|'
    r'(python|java(?!script)|javascript|typescript|ruby|golang|go(?=\s*(developer|engineer|dev|programmer))|rust|kotlin|swift|scala|elixir|php|perl|clojure|haskell|dart|lua|csharp|c\+\+|dotnet)\s*(developer|engineer|dev|programmer)|'
    r'gui\s*(developer|engineer)|'
    r'react(\s*native)?\s*(developer|engineer)|'
    r'ios\s*(developer|engineer)|'
    r'android\s*(developer|engineer)'
    r')\b'
)

SWE_ADJACENT = re.compile(
    r'(?i)\b('
    r'(?:network|cloud\s+network)\s*engineer|'
    r'(?:security|cyber\s*security|cybersecurity|information\s*security|application\s*security|'
    r'network\s*security|cloud\s*security|infrastructure\s*security)\s*engineer|'
    r'(?:systems?|cloud|linux|windows)\s*administrator|'
    r'(?:database|sql\s*database|oracle\s*database)\s*administrator|'
    r'(?:vmware|sap\s*basis|windows)\s*engineer|'
    r'data\s*scientist|'
    r'(?:cloud|application|security|data|technical|infrastructure|salesforce|service\s*now|servicenow)\s*architect|'
    r'network\s*engineering'
    r')\b'
)

SWE_EXCLUDE = re.compile(
    r'(?i)\b('
    r'sales\s*engineer|support\s*engineer|field\s*(service|engineer)|'
    r'customer\s*(success|support)\s*engineer|'
    r'solutions?\s*(architect|engineer)|'
    r'systems?\s*administrator|'
    r'civil\s*engineer|mechanical\s*engineer|electrical\s*engineer|'
    r'chemical\s*engineer|industrial\s*engineer|'
    r'audio\s*engineer|recording\s*engineer|sound\s*engineer|'
    r'network\s*engineer|hardware\s*engineer|'
    r'(electrical|hardware|mechanical|rf|avionics|controls?)\s*systems?\s*engineer|'
    r'systems?\s*engineer\s*[-–—,]\s*(electrical|hardware|mechanical|rf|avionics)'
    r')\b'
)

CONTROL_PATTERN = re.compile(
    r'(?i)\b('
    r'civil\s*engineer|mechanical\s*engineer|'
    r'nurse|registered\s*nurse|nursing|'
    r'electrical\s*engineer|chemical\s*engineer|'
    r'accountant|accounting|'
    r'financial\s*analyst'
    r')\b'
)

# O*NET SOC 15-1252 reference titles for embedding similarity
SWE_REFS = [
    "Software Engineer", "Software Developer", "Software Development Engineer",
    "Full Stack Developer", "Frontend Engineer", "Backend Engineer", "Web Developer",
    "DevOps Engineer", "Data Engineer", "ML Engineer", "Machine Learning Engineer",
    "AI Engineer", "Site Reliability Engineer", "Platform Engineer", "Cloud Engineer",
    "Mobile Developer", "Infrastructure Engineer", "Software Architect",
    "Systems Engineer", "Application Developer", "QA Engineer", "Test Automation Engineer",
    "Embedded Software Engineer", "Firmware Engineer",
]

EMBEDDING_THRESHOLD_HIGH = 0.85   # cosine >= 0.85 => SWE (94% accuracy)
EMBEDDING_THRESHOLD_LOW = 0.70    # cosine 0.70-0.85 => SWE_ADJACENT

# Pre-filter: only titles containing tech-adjacent keywords get sent to the
# embedding model.
TECH_PREFILTER = re.compile(
    r'(?i)\b('
    r'engineer|developer|architect|programmer|coder|coding|'
    r'technical|tech|software|swe|devops|'
    r'full.?stack|front.?end|back.?end|'
    r'data|machine.?learn|ml\b|ai\b|llm|'
    r'cloud|infrastructure|platform|site.?reliab|sre|'
    r'embedded|firmware|mobile|web|app|application|'
    r'systems?|administrator|database|computing|computer|cyber|security|'
    r'automation|qa\b|quality\s*assur|test|'
    r'blockchain|crypto|defi|smart.?contract|'
    r'python|java|javascript|typescript|react|node|'
    r'golang|rust|ruby|scala|kotlin|swift|'
    r'elixir|haskell|clojure|dart|lua|perl|oracle|sap|vmware|'
    r'gui|react|ios|android|'
    r'it\b|information\s*tech'
    r')\b'
)


def is_primary_swe_title(title_key: str) -> bool:
    return bool(SWE_INCLUDE.search(title_key)) and not bool(SWE_EXCLUDE.search(title_key))


def is_adjacent_technical_title(title_key: str) -> bool:
    return bool(SWE_ADJACENT.search(title_key))


# ---------------------------------------------------------------------------
# 5a. SWE Classification -- Tier 2 artifact
# ---------------------------------------------------------------------------

def load_tier2_title_lookup() -> dict[str, str]:
    """
    Load and validate the curated title lookup artifact used before embeddings.

    Contract:
      - required columns: title, llm_swe_classification
      - allowed labels: SWE, SWE_ADJACENT, NOT_SWE
      - keys are normalized with the same canonical function used by Stage 1's
        `title_normalized`
    """
    if not TIER2_LOOKUP_PATH.exists():
        log.warning(f"  Title lookup artifact not found at {TIER2_LOOKUP_PATH}; falling back to regex + embedding")
        return {}

    stat = TIER2_LOOKUP_PATH.stat()
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    log.info(
        "  Title lookup artifact: %s (%s bytes, modified %s)",
        TIER2_LOOKUP_PATH.name,
        stat.st_size,
        mtime,
    )

    llm_df = pd.read_parquet(TIER2_LOOKUP_PATH)
    missing = TIER2_REQUIRED_COLUMNS - set(llm_df.columns)
    if missing:
        raise ValueError(
            f"Title lookup artifact missing required columns: {sorted(missing)}"
        )

    llm_df = llm_df[list(TIER2_REQUIRED_COLUMNS)].copy()
    llm_df["title_key"] = llm_df["title"].map(normalize_swe_title_key)
    llm_df["llm_swe_classification"] = llm_df["llm_swe_classification"].astype(str).str.strip()

    invalid_labels = sorted(set(llm_df["llm_swe_classification"]) - TIER2_ALLOWED_CLASSES)
    if invalid_labels:
        raise ValueError(
            f"Title lookup artifact has invalid labels: {invalid_labels}"
        )

    llm_df = llm_df[llm_df["title_key"] != ""].copy()
    if llm_df.empty:
        raise ValueError("Title lookup artifact produced no usable normalized title keys")

    conflict_counts = llm_df.groupby("title_key")["llm_swe_classification"].nunique()
    conflicting_keys = conflict_counts[conflict_counts > 1]
    if not conflicting_keys.empty:
        sample = conflicting_keys.index[:10].tolist()
        log.warning(
            "  Dropping %s ambiguous normalized title keys from the lookup artifact; sample keys: %s",
            len(conflicting_keys),
            sample,
        )
        llm_df = llm_df[~llm_df["title_key"].isin(conflicting_keys.index)].copy()

    rows_before = len(llm_df)
    llm_df = llm_df.drop_duplicates(subset=["title_key", "llm_swe_classification"])
    llm_df = llm_df.drop_duplicates(subset=["title_key"], keep="first")
    rows_deduped = rows_before - len(llm_df)
    if rows_deduped:
        log.info(f"  Deduped {rows_deduped:,} redundant title lookup rows after normalization")

    label_counts = Counter(llm_df["llm_swe_classification"])
    for label, count in label_counts.items():
        log.info(f"    {label}: {count}")

    return dict(zip(llm_df["title_key"], llm_df["llm_swe_classification"]))


# ---------------------------------------------------------------------------
# 5a. SWE Classification -- Build Lookup
# ---------------------------------------------------------------------------

def build_swe_lookup(input_path: Path) -> tuple[dict, dict]:
    """
    Build lookups from raw title:
      - swe_lookup -> (is_swe, is_swe_adjacent, confidence, tier)
      - control_lookup -> bool

    Tier 1: Regex (SWE_INCLUDE / SWE_EXCLUDE)
    Tier 2a: LLM-validated title lookup (500 titles)
    Tier 2b: Embedding similarity with split thresholds
    """
    log.info("--- 5a. Building SWE classification lookup ---")
    t0 = time.time()

    tier2_llm = load_tier2_title_lookup()
    log.info(f"  Loaded normalized title lookup entries: {len(tier2_llm)}")

    # Collect unique canonical SWE title keys plus raw titles for controls.
    pf = pq.ParquetFile(input_path)
    title_swe = {}
    title_adjacent = {}
    title_control = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title_normalized", "title"]):
        chunk = batch.to_pandas()
        for title_key in chunk["title_normalized"].map(normalize_swe_title_key).drop_duplicates():
            t = title_key
            if not t:
                continue
            if t not in title_swe:
                title_swe[t] = is_primary_swe_title(t)
                title_adjacent[t] = is_adjacent_technical_title(t)
        for raw_title in chunk["title"].dropna().astype(str).str.lower().str.strip().drop_duplicates():
            if raw_title not in title_control:
                title_control[raw_title] = bool(CONTROL_PATTERN.search(raw_title))
        del chunk, batch
        gc.collect()

    all_titles = list(title_swe.keys())
    log.info(f"  Unique titles: {len(all_titles):,}")

    # Tier 1: regex-classified titles
    regex_swe = {t for t, v in title_swe.items() if v}
    control_titles = {t for t, v in title_control.items() if v}
    log.info(f"  Tier 1 (regex) SWE: {len(regex_swe):,}")
    log.info(f"  Control group titles: {len(control_titles):,}")

    # Build initial lookup for regex-classified titles.
    # Format: title -> (is_swe, is_swe_adjacent, confidence, tier)
    lookup = {}
    for t in regex_swe:
        lookup[t] = (True, False, 1.0, "regex")
    for t in control_titles:
        if t not in lookup:
            lookup[t] = (False, False, 1.0, "regex")

    # Check LLM lookup for remaining titles
    unresolved = [t for t in all_titles if t not in lookup]
    log.info(f"  Unresolved titles (not regex, not control): {len(unresolved):,}")

    llm_resolved = 0
    llm_swe = 0
    llm_adjacent = 0
    llm_not_swe = 0
    still_unresolved = []

    for t in unresolved:
        if t in tier2_llm:
            classification = tier2_llm[t]
            if classification == "SWE":
                lookup[t] = (True, False, 0.90, "title_lookup_llm")
                llm_swe += 1
            elif classification == "SWE_ADJACENT":
                lookup[t] = (False, True, 0.85, "title_lookup_llm")
                llm_adjacent += 1
            else:  # NOT_SWE
                lookup[t] = (False, False, 0.90, "title_lookup_llm")
                llm_not_swe += 1
            llm_resolved += 1
        else:
            still_unresolved.append(t)

    log.info(f"  LLM lookup resolved: {llm_resolved:,} (SWE={llm_swe}, ADJ={llm_adjacent}, NOT={llm_not_swe})")
    log.info(f"  Still unresolved after LLM lookup: {len(still_unresolved):,}")

    del unresolved
    gc.collect()

    regex_adjacent_resolved = 0
    unresolved_after_adjacent = []
    for t in still_unresolved:
        if title_adjacent.get(t, False):
            lookup[t] = (False, True, 0.82, "regex")
            regex_adjacent_resolved += 1
        else:
            unresolved_after_adjacent.append(t)

    log.info(f"  Regex-adjacent resolved after LLM lookup: {regex_adjacent_resolved:,}")
    log.info(f"  Still unresolved after regex-adjacent pass: {len(unresolved_after_adjacent):,}")

    still_unresolved = unresolved_after_adjacent

    # Pre-filter for embedding candidates
    candidates = [t for t in still_unresolved if TECH_PREFILTER.search(str(t))]
    non_candidates = [t for t in still_unresolved if not TECH_PREFILTER.search(str(t))]
    log.info(f"  Embedding candidates (after tech pre-filter): {len(candidates):,}")
    log.info(f"  Non-candidates (clearly non-SWE, skipping): {len(non_candidates):,}")

    # Mark non-candidates immediately
    for t in non_candidates:
        lookup[t] = (False, False, 0.0, "unresolved")

    del title_swe, title_adjacent, all_titles, still_unresolved, non_candidates
    gc.collect()

    # -----------------------------------------------------------------------
    # Tier 2b: Embedding similarity with split thresholds
    # -----------------------------------------------------------------------
    if candidates:
        log.info("  Loading JobBERT-v2 model...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("TechWolf/JobBERT-v2")

        # Encode reference titles
        log.info(f"  Encoding {len(SWE_REFS)} reference titles...")
        ref_embeddings = model.encode(SWE_REFS, normalize_embeddings=True,
                                       show_progress_bar=False)

        # Encode candidate titles in batches
        log.info(f"  Encoding {len(candidates):,} candidate titles...")
        ENCODE_BATCH = 4096
        embed_swe_count = 0
        embed_adjacent_count = 0
        embed_not_swe_count = 0

        for i in range(0, len(candidates), ENCODE_BATCH):
            batch_titles = candidates[i:i + ENCODE_BATCH]
            batch_embeddings = model.encode(batch_titles, normalize_embeddings=True,
                                             show_progress_bar=False,
                                             batch_size=256)

            # Cosine similarity (normalized => dot product = cosine)
            sims = batch_embeddings @ ref_embeddings.T
            max_sims = sims.max(axis=1)

            for j, t in enumerate(batch_titles):
                sim = float(max_sims[j])
                if sim >= EMBEDDING_THRESHOLD_HIGH:
                    lookup[t] = (True, False, sim, "embedding_high")
                    embed_swe_count += 1
                elif sim >= EMBEDDING_THRESHOLD_LOW:
                    lookup[t] = (False, True, sim, "embedding_adjacent")
                    embed_adjacent_count += 1
                else:
                    lookup[t] = (False, False, sim, "unresolved")
                    embed_not_swe_count += 1

            if (i // ENCODE_BATCH) % 4 == 0:
                log.info(f"    Encoded {min(i + ENCODE_BATCH, len(candidates)):,}/{len(candidates):,} "
                         f"(SWE: {embed_swe_count:,}, ADJ: {embed_adjacent_count:,})")

            del batch_embeddings, sims, max_sims
            gc.collect()

        log.info(f"  Tier 2b (embedding) SWE (>=0.85): {embed_swe_count:,}")
        log.info(f"  Tier 2b (embedding) SWE_ADJACENT (0.70-0.85): {embed_adjacent_count:,}")
        log.info(f"  Tier 2b (embedding) NOT_SWE (<0.70): {embed_not_swe_count:,}")

        # Free model
        del model, ref_embeddings
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Summary
    unresolved_count = sum(1 for v in lookup.values() if v[3] == "unresolved")
    log.info(f"  Titles still unresolved after all tiers: {unresolved_count:,}")

    elapsed = time.time() - t0
    log.info(f"  SWE lookup built in {elapsed:.1f}s ({len(lookup):,} entries)")

    return lookup, title_control


# ---------------------------------------------------------------------------
# 5b. Seniority Classification -- Multi-signal, uniform
# ---------------------------------------------------------------------------

# Title keyword patterns -- explicit signals only
TITLE_INTERN = re.compile(r'\b(intern|internship|co-?op)\b', re.IGNORECASE)
TITLE_DIRECTOR = re.compile(
    r'\b(director|vp|vice\s*president|head\s+of|chief|cto|cio)\b', re.IGNORECASE
)
TITLE_SENIOR = re.compile(
    r'\b(senior|sr\.?|staff|principal|distinguished|fellow|architect)\b', re.IGNORECASE
)
TITLE_LEAD = re.compile(
    r'\b('
    r'tech\s*lead|engineering\s*lead|'
    r'lead\s+(?:software|data|platform|backend|back[- ]?end|frontend|front[- ]?end|'
    r'full[- ]?stack|devops|qa|test|ml|ai|cloud|infrastructure|systems?)'
    r')\b',
    re.IGNORECASE
)
TITLE_JUNIOR = re.compile(
    r'\b(junior|jr\.?|new\s*grad(?:uate)?|recent\s*grad(?:uate)?|fresh\s*grad(?:uate)?|entry[- ]?level|early\s*career)\b',
    re.IGNORECASE,
)
TITLE_MANAGERISH = re.compile(r'\b(manager|supervisor)\b', re.IGNORECASE)
TITLE_LEAD_GENERAL = re.compile(r'(^|[\s,/:()\-])lead\b|\blead(?=[\s,/:()\-]|$)', re.IGNORECASE)

ADJACENT_ROLE_HINT = re.compile(
    r'\b('
    r'qa|quality|test|network|data|application|infrastructure|architect|'
    r'analyst|security|ui|ux|web|cloud|automation|systems?|reliability|'
    r'analytics?|mlops'
    r')\b',
    re.IGNORECASE,
)
CONTROL_ROLE_HINT = re.compile(
    r'\b('
    r'nurse|rn\b|nursing|nurse\s*practitioner|accountant|accounting|'
    r'financial\s*analyst|civil|mechanical|electrical|chemical|engineer'
    r')\b',
    re.IGNORECASE,
)

# Description YOE patterns for contradiction checks only.
# This extractor works from raw description text.
YOE_WORD_TO_NUM = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
YOE_TOKEN = r'(?:0|[1-9]\d?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)'
YOE_VALUE = rf'(?<!\d)(?:{YOE_TOKEN})(?!\d)'
YOE_PLUS = r'(?:\\\+|\+|plus|or\s+more)'
YOE_QUALIFIER = (
    r'(?:recent|relevant|related|professional|technical|engineering|software|systems|'
    r'applied|direct|hands[- ]on|practical|working|work|development|developing|'
    r'industry|job[- ]related|overall|total|management|managerial|leadership|'
    r'operational|operations|architectural|architecture|clinical|manufacturing|'
    r'project|product|mobile|android|ios|embedded|cloud|security|ml|data|'
    r'packaging|electrical|sales|marketing|consulting)'
)
YOE_SCOPE_NOUN = (
    r'(?:experience|development|engineering|programming|design|architecture|'
    r'management|leadership|operations|manufacturing|analysis|analytics|support|'
    r'testing|implementation|administration|consulting|research|nursing|clinical|'
    r'sales|marketing|product|project|quality|compliance|security|cloud|data|'
    r'machine\s+learning|ml|ai|software|systems|mobile|android|ios|kotlin|java|'
    r'python|packaging|electrical|embedded|fabrication|supervis(?:ion|ory)|'
    r'customer\s+service|account\s+management)'
)
YOE_PATTERN_PRIORITY = {
    "experience_label": 4,
    "forward_exp": 3,
    "reverse_exp": 2,
    "years_domain": 2,
    "years_in_with": 1,
}
YOE_CANDIDATE_PATTERNS = [
    (
        "experience_label",
        re.compile(
            rf'\b(?:experience|experience\s+level|exp\.?)\s*(?:[:\-]|is\b)?\s*'
            rf'(?:greater\s+than\s+|minimum(?:\s+of)?\s+|at\s+least\s+)?'
            rf'(?P<low>{YOE_VALUE})(?:\s*{YOE_PLUS})?'
            rf'(?:\s*(?:-|–|to)\s*(?P<high>{YOE_VALUE})(?:\s*{YOE_PLUS})?)?'
            rf'\s*(?:years?|yrs?)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "forward_exp",
        re.compile(
            rf'(?P<low>{YOE_VALUE})(?:\s*{YOE_PLUS})?'
            rf'(?:\s*(?:-|–|to)\s*(?P<high>{YOE_VALUE})(?:\s*{YOE_PLUS})?)?'
            rf'\s*(?:years?|yrs?)(?:[\'’]s?)?(?:\s+of)?'
            rf'(?:\s+{YOE_QUALIFIER}){{0,5}}\s+(?:experience|exp\.?)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "reverse_exp",
        re.compile(
            rf'\b(?:experience|exp\.?)(?:\s+requirements?)?(?:\s*[:\-])?'
            rf'[^.;\n]{{0,80}}?(?P<low>{YOE_VALUE})(?:\s*{YOE_PLUS})?'
            rf'(?:\s*(?:-|–|to)\s*(?P<high>{YOE_VALUE})(?:\s*{YOE_PLUS})?)?'
            rf'\s*(?:years?|yrs?)\b',
            re.IGNORECASE,
        ),
    ),
    (
        "years_domain",
        re.compile(
            rf'(?P<low>{YOE_VALUE})(?:\s*{YOE_PLUS})?'
            rf'(?:\s*(?:-|–|to)\s*(?P<high>{YOE_VALUE})(?:\s*{YOE_PLUS})?)?'
            rf'\s*(?:years?|yrs?)(?:[\'’]s?)?(?:\s+of)?'
            rf'(?:\s+(?:recent|relevant|related|professional|technical|direct|hands[- ]on|'
            rf'practical|overall|total|advanced|applied|strong|demonstrated)){{0,3}}'
            rf'\s+(?:[a-z][\w/+&.-]*\s+){{0,4}}{YOE_SCOPE_NOUN}\b',
            re.IGNORECASE,
        ),
    ),
    (
        "years_in_with",
        re.compile(
            rf'(?P<low>{YOE_VALUE})(?:\s*{YOE_PLUS})?'
            rf'(?:\s*(?:-|–|to)\s*(?P<high>{YOE_VALUE})(?:\s*{YOE_PLUS})?)?'
            rf'\s*(?:years?|yrs?)(?:[\'’]s?)?'
            rf'\s+(?:as|in|with|writing|building|developing|supporting|designing|delivering|'
            rf'maintaining|troubleshooting|administering|operating|working(?:\s+in)?)\b[^.;\n]{{0,80}}',
            re.IGNORECASE,
        ),
    ),
]
YOE_SECTION_HEADER = re.compile(
    r'(?i)^(?:qualifications?|requirements?|required skills?|required qualifications?|'
    r'basic qualifications?|minimum qualifications?|preferred qualifications?|'
    r'preferred skills?|experience|experience level|job qualifications?|about us|'
    r'about our founder|about the company|company description|benefits|'
    r'compensation|salary|licenses?|licensure|certifications?)\s*:?\s*$'
)
YOE_REQUIREMENT_SECTION = re.compile(
    r'(?i)\b(?:requirements?|required skills?|required qualifications?|basic qualifications?|'
    r'minimum qualifications?|minimum requirements?|job qualifications?)\b'
)
YOE_PREFERRED_SECTION = re.compile(
    r'(?i)\b(?:preferred qualifications?|preferred skills?|nice to have|desired|bonus)\b'
)
YOE_HISTORY_SECTION = re.compile(
    r'(?i)\b(?:about us|about our founder|about the company|company description|who we are)\b'
)
YOE_CERT_SECTION = re.compile(
    r'(?i)\b(?:licenses?|licensure|certifications?|board certification)\b'
)
YOE_COMP_SECTION = re.compile(
    r'(?i)\b(?:compensation|salary|pay range|benefits)\b'
)
YOE_REQUIRED_CONTEXT = re.compile(
    r'(?i)\b(?:required|required qualifications?|requirements?|minimum(?:\s+of)?|at least|'
    r'must have|basic qualifications?|experience requirements?|typically requires|'
    r'candidate will have|ideal candidate|qualifications?)\b'
)
YOE_PREFERRED_CONTEXT = re.compile(
    r'(?i)\b(?:preferred|preferred qualifications?|nice to have|desired|bonus|plus|pluses)\b'
)
YOE_DEGREE_CONTEXT = re.compile(
    r"(?i)\b(?:bachelor(?:'s)?|master(?:'s)?|ph\.?d|advanced degree|bs\b|ms\b)\b"
)
YOE_OVERALL_CONTEXT = re.compile(
    r'(?i)\b(?:related|relevant|professional|technical|engineering|software|systems|'
    r'job-related|industry|overall|total|direct|applied|role|position|field|'
    r'leadership|management|managerial|operations|manufacturing|clinical|'
    r'architecture|marketing|sales|product|project)\b'
)
YOE_SUBSKILL_CONTEXT = re.compile(
    r'(?i)\b(?:with|in|on|testing|'
    r'aws|azure|gcp|java|python|c\+\+|linux|kubernetes|android|ios|sql|agile|cgmp)\b'
)
YOE_COMPANY_HISTORY = re.compile(
    r'(?i)\b(?:more than|over|nearly|almost)\s+\d+\+?\s+years?\s+of\s+experience\b'
)
YOE_HISTORY_CONTEXT = re.compile(
    r'(?i)\b(?:leadership team|our company|the company|our history|company history|'
    r'combined experience|we bring|founded|serving customers|for decades|community|'
    r'years of existence|years of service|original ownership|under original ownership|'
    r'family-owned|family owned|since \d{4}|in business|about our founder|'
    r'founder|editorial makeup artist)\b'
)
YOE_YEARS_AGO = re.compile(r'(?i)\byears?\s+ago\b')
YOE_AGE_CONTEXT = re.compile(r'(?i)\byears?\s+(?:old|older)\b')
YOE_COMPENSATION_CONTEXT = re.compile(
    r'(?i)(?:\$|salary|hourly|per\s+hour|per-hour|wage|compensation|pay\s+range|'
    r'salary\s+range|bonus|commission)\b'
)
YOE_TENURE_CONTEXT = re.compile(
    r'(?i)\b(?:within|after|following|upon)\s+\d+\s+years?\b|\b(?:in position|'
    r'in role|of service|of existence|in the .* community)\b'
)
YOE_CERTIFICATION_DEFERRAL_CONTEXT = re.compile(
    r'(?i)\b(?:certification requirement may be deferred|board certification requirement may be deferred|'
    r'may be deferred up to \d+\s+year|must obtain .* within \d+\s+year|'
    r'within \d+\s+year(?:s)? of hire|within \d+\s+year(?:s)? if approved|'
    r'probationary period|must obtain certification)\b'
)
YOE_SUBORDINATE_CONTEXT = re.compile(
    r'(?i)\b(?:with\s+at\s+least|including|of\s+which|focus\s+on|specializing\s+in|'
    r'in\s+a\s+supervisory\s+role|managerial\s+role|additional\s+experience)\b'
)
YOE_SUBSTITUTION_CONTEXT = re.compile(
    r'(?i)\b(?:equivalent\s+experience|equivalent\s+combination|or\s+equivalent|'
    r'in\s+lieu\s+of|substitut(?:e|ion)|in\s+substitution)\b'
)
YOE_INDUSTRY_SPECIFIC_CONTEXT = re.compile(
    r'(?i)\b(?:in\s+(?:the\s+)?[a-z/&+ -]{0,40}\s+industry|in\s+a\s+[a-z/&+ -]{0,40}\s+setting|'
    r'in\s+an?\s+[a-z/&+ -]{0,40}\s+environment|industry-specific|domain-specific)\b'
)
YOE_AUXILIARY_CONTEXT = re.compile(
    r'(?i)\b(?:lead|leadership|management|managerial|supervisory|team\s+lead)\s+experience\b'
)
YOE_PARALLEL_ALT_CONTEXT = re.compile(
    r'(?i)\b(?:or|alternatively|in lieu of|equivalent combination|equivalent experience)\b'
)

def family_role_hint(family: str):
    if family == "adjacent":
        return ADJACENT_ROLE_HINT
    if family == "control":
        return CONTROL_ROLE_HINT
    return None


def classify_seniority(title: str, family: str = "other") -> tuple[str, str]:
    """
    Strong-rule-only seniority classifier from title text.

    Returns (seniority_level, source).
      - source ∈ {"title_keyword", "title_manager", "unknown"}
      - level ∈ {"entry", "associate", "mid-senior", "director", "unknown"}

    Stage 5 only fires on high-confidence title patterns; rows that lack a
    strong rule fall through to "unknown" and are routed to the Stage 10 LLM.
    Description signals, weak title-level numbers, native labels, and
    title-prior heuristics were removed in the 2026-04-10 simplification.
    """
    title_str = str(title).lower().strip() if pd.notna(title) else ""
    if not title_str:
        return ("unknown", "unknown")

    if TITLE_INTERN.search(title_str):
        return ("entry", "title_keyword")

    if TITLE_DIRECTOR.search(title_str):
        return ("director", "title_keyword")

    if TITLE_SENIOR.search(title_str):
        return ("mid-senior", "title_keyword")

    role_hint = family_role_hint(family)
    if role_hint and role_hint.search(title_str):
        if TITLE_MANAGERISH.search(title_str) or TITLE_LEAD_GENERAL.search(title_str):
            return ("mid-senior", "title_manager")

    if TITLE_LEAD.search(title_str):
        return ("mid-senior", "title_keyword")

    if TITLE_JUNIOR.search(title_str):
        return ("entry", "title_keyword")

    return ("unknown", "unknown")


def normalize_yoe_text(description) -> str:
    """Normalize raw text for YOE extraction while preserving clause boundaries."""
    if pd.isna(description):
        return ""

    text = str(description)
    if not text.strip():
        return ""

    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\\+", "+")
    )
    text = re.sub(
        r'(?i)\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s*'
        r'\(\s*(\d{1,2})\s*\)',
        r'\1',
        text,
    )
    text = re.sub(r'(?i)\b(\d{1,2})\s+or\s+more\s+years\b', r'\1+ years', text)
    text = re.sub(
        r'(?i)\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+'
        r'or\s+more\s+years\b',
        lambda m: f"{parse_yoe_token(m.group(1))}+ years",
        text,
    )
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def parse_yoe_token(token) -> int | None:
    if token is None:
        return None
    raw = str(token).strip().lower()
    if not raw:
        return None
    raw = raw.replace("+", "").replace("\\", "").strip()
    if raw.isdigit():
        return int(raw)
    return YOE_WORD_TO_NUM.get(raw)


def is_yoe_header(line: str) -> bool:
    text = re.sub(r"\s+", " ", str(line or "").strip())
    if not text:
        return False
    if YOE_SECTION_HEADER.match(text):
        return True
    return text.endswith(":") and len(text.split()) <= 6


def classify_yoe_section(header: str, clause_text: str) -> tuple[str, int]:
    section_text = f"{header} {clause_text}".strip().lower()
    if YOE_HISTORY_SECTION.search(section_text):
        return "history", 0
    if YOE_COMP_SECTION.search(section_text):
        return "compensation", 5
    if YOE_CERT_SECTION.search(section_text):
        return "certification", 10
    if YOE_PREFERRED_SECTION.search(section_text):
        return "preferred", 20
    if YOE_REQUIREMENT_SECTION.search(section_text):
        return "required", 50
    if YOE_DEGREE_CONTEXT.search(section_text) or YOE_REQUIRED_CONTEXT.search(section_text):
        return "qualification", 40
    return "general", 30


def split_yoe_clauses(text: str) -> list[dict]:
    """Split a normalized description into section-aware clause records."""
    clauses: list[dict] = []
    if not text:
        return clauses

    current_header = ""
    offset = 0
    clause_index = 0
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            offset += len(raw_line) + 1
            continue

        if is_yoe_header(line):
            current_header = re.sub(r":\s*$", "", line).strip()
            offset += len(raw_line) + 1
            continue

        line_cursor = 0
        sentence_parts = []
        for chunk in re.split(r"[;•]", raw_line):
            sentence_parts.extend(re.split(r"(?<=[.!?])\s+", chunk))
        parts = [p.strip(" -*\t") for p in sentence_parts if p.strip(" -*\t")]
        for part in parts:
            part_start_local = raw_line.find(part, line_cursor)
            if part_start_local == -1:
                part_start_local = line_cursor
            line_cursor = part_start_local + len(part)
            clause_text = re.sub(r"\s+", " ", part.strip())
            if not clause_text:
                continue
            section_type, section_rank = classify_yoe_section(current_header, clause_text)
            clauses.append(
                {
                    "text": clause_text,
                    "header": current_header,
                    "section_type": section_type,
                    "section_rank": section_rank,
                    "section_index": clause_index,
                    "start": offset + part_start_local,
                }
            )
            clause_index += 1
        offset += len(raw_line) + 1

    if not clauses:
        section_type, section_rank = classify_yoe_section("", text)
        clauses.append(
            {
                "text": text,
                "header": "",
                "section_type": section_type,
                "section_rank": section_rank,
                "section_index": 0,
                "start": 0,
            }
        )
    return clauses


def extract_yoe_clause(text: str, start: int, end: int) -> str:
    """Return the nearest clause-like span surrounding a regex match."""
    boundaries = "\n.;:"
    left = max(text.rfind(ch, 0, start) for ch in boundaries)
    right_candidates = [text.find(ch, end) for ch in boundaries if text.find(ch, end) != -1]
    right = min(right_candidates) if right_candidates else len(text)
    snippet = text[left + 1:right].strip() if left != -1 else text[:right].strip()
    return re.sub(r"\s+", " ", snippet)


def classify_yoe_candidate(
    full_text: str,
    clause_info: dict,
    match: re.Match,
    pattern_name: str,
) -> dict | None:
    low = parse_yoe_token(match.group("low"))
    high = parse_yoe_token(match.groupdict().get("high"))
    if low is None:
        return None
    high = high if high is not None else low
    if low > high:
        low, high = high, low

    clause = clause_info["text"]
    clause_lower = clause.lower()
    absolute_start = clause_info["start"] + match.start()
    absolute_end = clause_info["start"] + match.end()
    degree_start = max(0, absolute_start - 160)
    degree_end = min(len(full_text), absolute_end + 120)
    degree_context = full_text[degree_start:degree_end]
    degree_lower = degree_context.lower()
    scope_start = max(0, absolute_start - 40)
    scope_end = min(len(full_text), absolute_end + 80)
    scope_context = full_text[scope_start:scope_end]
    scope_lower = scope_context.lower()
    text = re.sub(r"\s+", " ", match.group(0).strip())
    section_type = clause_info["section_type"]
    section_rank = clause_info["section_rank"]
    section_header = clause_info["header"]

    reject_reason = None
    if low < 1 or high > 20:
        reject_reason = "out_of_range"
    elif section_type == "history" or YOE_COMPANY_HISTORY.search(clause_lower) or YOE_HISTORY_CONTEXT.search(clause_lower):
        reject_reason = "company_history"
    elif YOE_YEARS_AGO.search(clause_lower):
        reject_reason = "years_ago"
    elif YOE_AGE_CONTEXT.search(scope_lower):
        reject_reason = "age_context"
    elif section_type == "compensation" or (YOE_COMPENSATION_CONTEXT.search(scope_lower) and "$" in scope_context):
        reject_reason = "compensation_context"
    elif YOE_TENURE_CONTEXT.search(scope_lower):
        reject_reason = "tenure_context"
    elif YOE_CERTIFICATION_DEFERRAL_CONTEXT.search(clause_lower):
        reject_reason = "certification_deferral"

    required = section_type in {"required", "qualification"} or bool(YOE_REQUIRED_CONTEXT.search(clause_lower))
    preferred = section_type == "preferred" or bool(YOE_PREFERRED_CONTEXT.search(clause_lower))
    degree_ladder = bool(YOE_DEGREE_CONTEXT.search(degree_lower))
    overall = pattern_name in {"experience_label", "forward_exp", "reverse_exp", "years_domain"} or bool(
        YOE_OVERALL_CONTEXT.search(scope_lower)
    ) or degree_ladder
    subordinate = bool(YOE_SUBORDINATE_CONTEXT.search(scope_lower))
    substitution = bool(YOE_SUBSTITUTION_CONTEXT.search(scope_lower))
    industry_specific = bool(YOE_INDUSTRY_SPECIFIC_CONTEXT.search(scope_lower))
    auxiliary = bool(YOE_AUXILIARY_CONTEXT.search(clause_lower))
    certification_deferral = bool(YOE_CERTIFICATION_DEFERRAL_CONTEXT.search(clause_lower))
    parallel_alt = degree_ladder and bool(YOE_PARALLEL_ALT_CONTEXT.search(clause_lower))
    history_like = section_type == "history" or bool(YOE_HISTORY_CONTEXT.search(clause_lower))
    compensation_like = section_type == "compensation" or bool(YOE_COMPENSATION_CONTEXT.search(scope_lower))
    tenure_like = bool(YOE_TENURE_CONTEXT.search(scope_lower))
    subskill = (
        pattern_name == "years_in_with"
        or industry_specific
        or auxiliary
        or (
            bool(YOE_SUBSKILL_CONTEXT.search(scope_lower))
            and pattern_name not in {"experience_label", "forward_exp", "years_domain"}
        )
    )
    if pattern_name == "years_domain":
        overall = True

    if subordinate and not degree_ladder and not required:
        subskill = True
    if substitution and not degree_ladder:
        subskill = True

    total_role = overall and not any(
        (
            subskill,
            subordinate,
            substitution,
            certification_deferral,
            history_like,
            compensation_like,
            tenure_like,
        )
    )

    if reject_reason is None and pattern_name == "reverse_exp" and not (
        required or overall or degree_ladder or section_type in {"required", "qualification"}
    ):
        reject_reason = "generic_reverse_order"

    return {
        "text": text,
        "start": absolute_start,
        "end": absolute_end,
        "pattern": pattern_name,
        "min_year": low,
        "max_year": high,
        "required": required,
        "preferred": preferred,
        "degree_ladder": degree_ladder,
        "overall": overall,
        "subskill": subskill,
        "subordinate": subordinate,
        "substitution": substitution,
        "industry_specific": industry_specific,
        "auxiliary": auxiliary,
        "certification_deferral": certification_deferral,
        "history_like": history_like,
        "compensation_like": compensation_like,
        "tenure_like": tenure_like,
        "total_role": total_role,
        "parallel_alt": parallel_alt,
        "section_type": section_type,
        "section_rank": section_rank,
        "section_header": section_header,
        "section_index": clause_info["section_index"],
        "reject_reason": reject_reason,
        "context": clause[:180],
    }


def collect_yoe_mentions(description: str) -> list[dict]:
    """Collect and de-duplicate candidate YOE mentions from raw description text."""
    text = normalize_yoe_text(description)
    if not text:
        return []

    candidates = []
    for clause_info in split_yoe_clauses(text):
        clause_text = clause_info["text"]
        for pattern_name, pattern in YOE_CANDIDATE_PATTERNS:
            for match in pattern.finditer(clause_text):
                candidate = classify_yoe_candidate(text, clause_info, match, pattern_name)
                if candidate is not None:
                    candidates.append(candidate)

    if not candidates:
        return []

    candidates.sort(
        key=lambda c: (
            c["section_index"],
            c["start"],
            -(c["end"] - c["start"]),
            -YOE_PATTERN_PRIORITY.get(c["pattern"], 0),
        )
    )

    deduped = []
    for candidate in candidates:
        overlaps_existing = False
        for kept in deduped:
            if (
                candidate["section_index"] == kept["section_index"]
                and candidate["start"] < kept["end"]
                and kept["start"] < candidate["end"]
            ):
                overlaps_existing = True
                break
        if not overlaps_existing:
            deduped.append(candidate)
    return deduped


def compact_yoe_mentions_json(mentions: list[dict]) -> str | None:
    if not mentions:
        return None
    payload = [
        {
            "txt": m["text"],
            "lo": m["min_year"],
            "hi": m["max_year"],
            "pat": m["pattern"],
            "req": m["required"],
            "pref": m["preferred"],
            "deg": m["degree_ladder"],
            "ovr": m["overall"],
            "sub": m["subskill"],
            "subord": m["subordinate"],
            "subst": m["substitution"],
            "ind": m["industry_specific"],
            "aux": m["auxiliary"],
            "cert": m["certification_deferral"],
            "hist": m["history_like"],
            "comp": m["compensation_like"],
            "ten": m["tenure_like"],
            "tot": m["total_role"],
            "par": m["parallel_alt"],
            "sec": m["section_type"],
            "rej": m["reject_reason"],
            "ctx": m["context"],
        }
        for m in mentions
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def choose_primary_yoe(mentions: list[dict]) -> tuple[dict | None, str]:
    """Resolve a posting-level YOE from candidate mentions."""
    valid = [m for m in mentions if m["reject_reason"] is None]
    if not valid:
        return None, "no_valid_mentions"

    required_total_role = [
        m
        for m in valid
        if m["required"]
        and not m["preferred"]
        and m["total_role"]
        and not m["substitution"]
        and not m["certification_deferral"]
    ]
    parallel_degree_mentions = [
        m
        for m in valid
        if m["degree_ladder"]
        and not m["preferred"]
        and m["parallel_alt"]
        and not m["certification_deferral"]
    ]
    overall_total_role = [
        m
        for m in valid
        if not m["preferred"]
        and m["total_role"]
        and not m["substitution"]
        and not m["certification_deferral"]
    ]
    preferred_total_role = [
        m
        for m in valid
        if m["preferred"] and m["total_role"] and not m["certification_deferral"]
    ]
    required_subskill = [
        m
        for m in valid
        if m["required"] and not m["preferred"] and not m["certification_deferral"]
    ]

    if len(parallel_degree_mentions) >= 2:
        primary = min(
            parallel_degree_mentions,
            key=lambda m: (m["min_year"], m["max_year"], m["start"]),
        )
        return primary, "degree_ladder_parallel_min_path"
    elif required_total_role:
        pool = required_total_role
        default_rule = "required_total_role"
    elif overall_total_role:
        pool = overall_total_role
        default_rule = "overall_total_role"
    elif preferred_total_role:
        pool = preferred_total_role
        default_rule = "preferred_total_role"
    elif required_subskill:
        pool = required_subskill
        default_rule = "required_subskill_fallback"
    else:
        degree_mentions = [m for m in valid if m["degree_ladder"] and not m["preferred"]]
        if len(degree_mentions) >= 2:
            primary = min(degree_mentions, key=lambda m: (m["min_year"], m["max_year"], m["start"]))
            return primary, "degree_ladder_min_path"
        if all(m["subskill"] for m in valid):
            return None, "subskill_only_abstain"
        pool = valid
        default_rule = "fallback_primary"

    primary = max(
        pool,
        key=lambda m: (
            m["section_rank"],
            int(m["required"] and not m["preferred"]),
            int(m["total_role"]),
            int(not m["substitution"]),
            -int(m["subordinate"]),
            -int(m["subskill"]),
            YOE_PATTERN_PRIORITY.get(m["pattern"], 0),
            m["min_year"],
            -m["start"],
        ),
    )

    if primary["max_year"] > primary["min_year"]:
        rule = "range_primary"
    elif primary["parallel_alt"] and primary["degree_ladder"]:
        rule = "degree_ladder_parallel_min_path"
    elif default_rule == "required_total_role":
        rule = "required_total_role"
    elif default_rule == "overall_total_role":
        rule = "overall_total_role"
    elif default_rule == "preferred_total_role":
        rule = "preferred_total_role"
    elif default_rule == "required_subskill_fallback":
        rule = "required_subskill_fallback"
    elif primary["preferred"]:
        rule = "preferred_fallback"
    elif primary["required"] and (primary["overall"] or primary["degree_ladder"]):
        rule = "required_primary"
    elif primary["overall"] or primary["degree_ladder"]:
        rule = "overall_primary"
    else:
        rule = default_rule
    return primary, rule


def extract_yoe_features(description: str) -> dict:
    """
    Extract posting-level YOE plus audit fields.

    `yoe_extracted` is now the resolved primary requirement.
    `yoe_min_extracted` and `yoe_max_extracted` preserve lower/upper bounds.
    """
    mentions = collect_yoe_mentions(description)
    mentions_json = compact_yoe_mentions_json(mentions)
    valid = [m for m in mentions if m["reject_reason"] is None]
    primary, rule = choose_primary_yoe(mentions)

    if primary is None:
        return {
            "yoe_extracted": np.nan,
            "yoe_min_extracted": np.nan,
            "yoe_max_extracted": np.nan,
            "yoe_match_count": len(valid),
            "yoe_resolution_rule": rule,
            "yoe_all_mentions_json": mentions_json,
        }

    return {
        "yoe_extracted": float(primary["min_year"]),
        "yoe_min_extracted": float(min(m["min_year"] for m in valid)),
        "yoe_max_extracted": float(max(m["max_year"] for m in valid)),
        "yoe_match_count": len(valid),
        "yoe_resolution_rule": rule,
        "yoe_all_mentions_json": mentions_json,
    }


def has_yoe_contradiction(seniority: str, yoe_extracted) -> bool:
    """Flag obviously inflated YOE requirements for lower-level roles."""
    if pd.isna(yoe_extracted):
        return False

    if seniority == "entry" and yoe_extracted >= 3:
        return True
    if seniority == "associate" and yoe_extracted >= 5:
        return True
    return False


def build_seniority_title_lookup(
    input_path: Path,
    swe_lookup: dict,
    control_lookup: dict,
) -> dict[str, tuple[str, str]]:
    """
    Build lookup: raw title (lowered) -> (seniority_level, source).

    Only stores entries where the strong-rule classifier returned a non-unknown
    result. Streaming Stage 5 reads from this lookup to avoid recomputing per
    row.

    CRITICAL: Uses the raw 'title' column, NOT 'title_normalized', because
    title_normalized strips seniority prefixes (Senior, Sr, Lead, Staff, etc.)
    which is exactly what we need to classify on.
    """
    log.info("--- 5b. Building seniority title lookup ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    unique_titles: dict[str, str] = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title", "title_normalized"]):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            raw_title = getattr(row, "title")
            if pd.isna(raw_title):
                continue
            raw_lower = str(raw_title).lower().strip()
            if not raw_lower:
                continue
            swe_key = normalize_swe_title_key(getattr(row, "title_normalized"))
            swe_tuple = swe_lookup.get(swe_key, (False, False, 0.0, "unresolved"))
            if control_lookup.get(raw_lower, False):
                family = "control"
            elif swe_tuple[1]:
                family = "adjacent"
            elif swe_tuple[0]:
                family = "swe"
            else:
                family = "other"
            unique_titles[raw_lower] = family
        del chunk, batch
        gc.collect()

    log.info(f"  Unique raw titles to classify: {len(unique_titles):,}")

    title_lookup: dict[str, tuple[str, str]] = {}
    for t, family in unique_titles.items():
        level, source = classify_seniority(t, family=family)
        if source != "unknown":
            title_lookup[t] = (level, source)

    resolved_counts = Counter(v[0] for v in title_lookup.values())
    log.info(f"  Strong-rule seniority resolved at title-only: {len(title_lookup):,}")
    for level, count in resolved_counts.most_common():
        log.info(f"    {level}: {count:,}")

    elapsed = time.time() - t0
    log.info(f"  Seniority title lookup built in {elapsed:.1f}s")

    return title_lookup


# ---------------------------------------------------------------------------
# 5c. 3-level bucketing
# ---------------------------------------------------------------------------


def map_3level(seniority: str) -> str:
    return SENIORITY_3LEVEL.get(str(seniority).lower().strip(), "unknown")


# ---------------------------------------------------------------------------
# Streaming write pass
# ---------------------------------------------------------------------------

def streaming_write(
    input_path: Path,
    output_path: Path,
    swe_lookup: dict,
    seniority_title_lookup: dict,
    control_lookup: dict,
):
    """
    Stream chunks, apply lookups, write incrementally.
    """
    log.info("--- Streaming write pass ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    output_schema = None  # Built from first chunk after new columns are added
    writer = None
    written = 0

    # Counters for reporting
    swe_tier_counts = Counter()
    seniority_final_source_counts = Counter()
    seniority_final_level_counts = Counter()
    swe_by_source_counts = Counter()
    adj_by_source_counts = Counter()
    total_by_source_counts = Counter()
    swe_only_tier_counts = Counter()
    adj_only_tier_counts = Counter()
    seniority_3level_counts = Counter()
    swe_final_level_counts = Counter()
    swe_seniority_3level_counts = Counter()

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        chunk = batch.to_pandas()
        n = len(chunk)

        # ---- 5a: SWE classification ----
        swe_results = chunk["title_normalized"].map(normalize_swe_title_key).map(
            lambda t: swe_lookup.get(t, (False, False, 0.0, "unresolved"))
        )
        chunk["is_swe"] = swe_results.apply(lambda x: x[0]).astype(bool)
        chunk["is_swe_adjacent"] = swe_results.apply(lambda x: x[1]).astype(bool)
        chunk["swe_confidence"] = swe_results.apply(lambda x: float(x[2])).astype(np.float64)
        chunk["swe_classification_tier"] = swe_results.apply(lambda x: x[3])

        # Count tiers
        tier_vc = chunk["swe_classification_tier"].value_counts()
        for tier, count in tier_vc.items():
            swe_tier_counts[tier] += int(count)

        del swe_results

        # ---- 5b: Seniority classification (strong-rule only) ----
        # CRITICAL: Use raw 'title' column (not title_normalized) because
        # title_normalized strips seniority prefixes (Senior, Sr, Lead, etc.)
        title_lower = chunk["title"].fillna("").str.lower().str.strip()
        chunk["is_control"] = title_lower.map(lambda t: control_lookup.get(t, False)).astype(bool)

        # Enforce mutual exclusion: SWE > SWE_adjacent > Control
        chunk.loc[chunk["is_swe"], "is_control"] = False
        chunk.loc[chunk["is_swe"], "is_swe_adjacent"] = False
        chunk.loc[chunk["is_swe_adjacent"], "is_control"] = False

        seniority_final_levels: list[str] = []
        seniority_final_sources: list[str] = []
        yoe_values: list[float] = []
        yoe_min_values: list[float] = []
        yoe_max_values: list[float] = []
        yoe_match_counts: list[int] = []
        yoe_resolution_rules: list[str] = []
        yoe_mentions_json: list[str | None] = []
        yoe_contradictions: list[bool] = []

        is_control_arr = chunk["is_control"].to_numpy()
        is_swe_adj_arr = chunk["is_swe_adjacent"].to_numpy()
        is_swe_arr = chunk["is_swe"].to_numpy()
        title_lower_arr = title_lower.to_numpy(dtype=object)
        description_arr = chunk["description"].to_numpy(dtype=object)

        for idx in range(n):
            title_val = title_lower_arr[idx]
            cached = seniority_title_lookup.get(title_val)
            if cached is not None:
                level, source = cached
            else:
                if is_control_arr[idx]:
                    family = "control"
                elif is_swe_adj_arr[idx]:
                    family = "adjacent"
                elif is_swe_arr[idx]:
                    family = "swe"
                else:
                    family = "other"
                level, source = classify_seniority(title_val, family=family)
            seniority_final_levels.append(level)
            seniority_final_sources.append(source)

            yoe_features = extract_yoe_features(description_arr[idx])
            yoe_values.append(yoe_features["yoe_extracted"])
            yoe_min_values.append(yoe_features["yoe_min_extracted"])
            yoe_max_values.append(yoe_features["yoe_max_extracted"])
            yoe_match_counts.append(yoe_features["yoe_match_count"])
            yoe_resolution_rules.append(yoe_features["yoe_resolution_rule"])
            yoe_mentions_json.append(yoe_features["yoe_all_mentions_json"])
            yoe_contradictions.append(has_yoe_contradiction(level, yoe_features["yoe_extracted"]))

        chunk["seniority_final"] = seniority_final_levels
        chunk["seniority_final_source"] = seniority_final_sources
        chunk["seniority_3level"] = chunk["seniority_final"].apply(map_3level)
        chunk["seniority_rule"] = chunk["seniority_final"]
        chunk["seniority_rule_source"] = chunk["seniority_final_source"]
        chunk["yoe_extracted"] = np.array(yoe_values, dtype=np.float64)
        chunk["yoe_min_extracted"] = np.array(yoe_min_values, dtype=np.float64)
        chunk["yoe_max_extracted"] = np.array(yoe_max_values, dtype=np.float64)
        chunk["yoe_match_count"] = np.array(yoe_match_counts, dtype=np.int16)
        chunk["yoe_resolution_rule"] = yoe_resolution_rules
        chunk["yoe_all_mentions_json"] = yoe_mentions_json
        chunk["yoe_seniority_contradiction"] = np.array(yoe_contradictions, dtype=bool)

        # Count seniority-final sources and levels
        for src, count in chunk["seniority_final_source"].value_counts().items():
            seniority_final_source_counts[src] += int(count)
        for level, count in chunk["seniority_final"].value_counts().items():
            seniority_final_level_counts[level] += int(count)
        for level, count in chunk.loc[chunk["is_swe"], "seniority_final"].value_counts().items():
            swe_final_level_counts[level] += int(count)
        for level, count in chunk["seniority_3level"].value_counts().items():
            seniority_3level_counts[level] += int(count)
        for level, count in chunk.loc[chunk["is_swe"], "seniority_3level"].value_counts().items():
            swe_seniority_3level_counts[level] += int(count)

        del seniority_final_levels, seniority_final_sources

        # Count by source/platform
        source_keys = chunk["source"].fillna("unknown") + "|" + chunk["source_platform"].fillna("unknown")
        for key, count in source_keys.value_counts().items():
            total_by_source_counts[key] += int(count)
        for key, count in source_keys[chunk["is_swe"]].value_counts().items():
            swe_by_source_counts[key] += int(count)
        for key, count in source_keys[chunk["is_swe_adjacent"]].value_counts().items():
            adj_by_source_counts[key] += int(count)

        for tier, count in chunk.loc[chunk["is_swe"], "swe_classification_tier"].value_counts().items():
            swe_only_tier_counts[tier] += int(count)
        for tier, count in chunk.loc[chunk["is_swe_adjacent"], "swe_classification_tier"].value_counts().items():
            adj_only_tier_counts[tier] += int(count)

        # ---- Force float64 on numeric columns for schema consistency ----
        chunk["swe_confidence"] = chunk["swe_confidence"].astype(np.float64)
        for col in chunk.select_dtypes(include=["object"]).columns:
            chunk[col] = chunk[col].astype("string")

        # ---- Write (cast to unified schema) ----
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if output_schema is None:
            output_schema = promote_null_schema(table.schema)
        table = table.cast(output_schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, output_schema)
        writer.write_table(table)

        written += n
        log.info(f"  Written {written:,}/{total_rows:,} rows")

        del chunk, table, batch
        gc.collect()

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    log.info(f"  Write pass complete: {written:,} rows in {elapsed:.1f}s")

    return {
        "written": written,
        "swe_tier_counts": swe_tier_counts,
        "seniority_final_source_counts": seniority_final_source_counts,
        "seniority_final_level_counts": seniority_final_level_counts,
        "swe_by_source_counts": swe_by_source_counts,
        "adj_by_source_counts": adj_by_source_counts,
        "total_by_source_counts": total_by_source_counts,
        "swe_only_tier_counts": swe_only_tier_counts,
        "adj_only_tier_counts": adj_only_tier_counts,
        "seniority_3level_counts": seniority_3level_counts,
        "swe_final_level_counts": swe_final_level_counts,
        "swe_seniority_3level_counts": swe_seniority_3level_counts,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(
    input_path: Path,
    output_path: Path,
    report_stats: dict,
):
    """Post-write validation and detailed reporting."""
    log.info("\n" + "=" * 60)
    log.info("STAGE 5 REPORT")
    log.info("=" * 60)

    written = report_stats["written"]
    swe_tier_counts = report_stats["swe_tier_counts"]
    seniority_final_source_counts = report_stats["seniority_final_source_counts"]
    seniority_final_level_counts = report_stats["seniority_final_level_counts"]
    swe_by_source_counts = report_stats["swe_by_source_counts"]
    adj_by_source_counts = report_stats["adj_by_source_counts"]
    total_by_source_counts = report_stats["total_by_source_counts"]
    swe_only_tier_counts = report_stats["swe_only_tier_counts"]
    adj_only_tier_counts = report_stats["adj_only_tier_counts"]
    seniority_3level_counts = report_stats["seniority_3level_counts"]
    swe_final_level_counts = report_stats["swe_final_level_counts"]
    swe_seniority_3level_counts = report_stats["swe_seniority_3level_counts"]

    # --- SWE classification report ---
    log.info("\n--- 5a. SWE Classification ---")
    total = sum(swe_tier_counts.values())

    log.info(f"  {'Tier':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    for tier in ["regex", "title_lookup_llm", "embedding_high", "embedding_adjacent", "unresolved"]:
        c = swe_tier_counts.get(tier, 0)
        pct = c / total * 100 if total > 0 else 0
        log.info(f"  {tier:<25} {c:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<25} {total:>10,}")

    # SWE by data source
    log.info("\n  SWE counts by data source:")
    log.info(f"  {'Source':<30} {'SWE':>8} {'Total':>10} {'Pct':>8}")
    log.info(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    total_swe = 0
    total_rows = 0
    for src in sorted(total_by_source_counts):
        swe = swe_by_source_counts.get(src, 0)
        total = total_by_source_counts.get(src, 0)
        total_swe += swe
        total_rows += total
        pct = swe / total * 100 if total else 0
        log.info(f"  {src:<30} {swe:>8,} {total:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<30} {total_swe:>8,} {total_rows:>10,} {total_swe/total_rows*100:>7.1f}%")

    # SWE-adjacent by data source
    log.info("\n  SWE-adjacent counts by data source:")
    log.info(f"  {'Source':<30} {'Adjacent':>8} {'Total':>10} {'Pct':>8}")
    log.info(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    total_adj = 0
    for src in sorted(total_by_source_counts):
        adj = adj_by_source_counts.get(src, 0)
        total = total_by_source_counts.get(src, 0)
        total_adj += adj
        pct = adj / total * 100 if total else 0
        log.info(f"  {src:<30} {adj:>8,} {total:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<30} {total_adj:>8,} {total_rows:>10,} {total_adj/total_rows*100:>7.1f}%")

    # SWE by classification tier
    log.info("\n  SWE classification tier breakdown (SWE=True only):")
    for tier, count in swe_only_tier_counts.most_common():
        log.info(f"    {tier:<25} {count:>10,}")

    # SWE-adjacent by classification tier
    log.info("\n  SWE-adjacent tier breakdown (is_swe_adjacent=True only):")
    for tier, count in adj_only_tier_counts.most_common():
        log.info(f"    {tier:<25} {count:>10,}")

    # --- Seniority report ---
    log.info("\n--- 5b. Seniority (strong-rule pass; LLM fills the rest in Stage 10) ---")
    log.info(f"\n  seniority_final_source distribution:")
    log.info(f"  {'Source':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    total_final = sum(seniority_final_source_counts.values())
    for src in ["title_keyword", "title_manager", "unknown"]:
        c = seniority_final_source_counts.get(src, 0)
        pct = c / total_final * 100 if total_final > 0 else 0
        log.info(f"  {src:<25} {c:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<25} {total_final:>10,}")

    log.info(f"\n  seniority_final level distribution (5-level, all rows):")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = seniority_final_level_counts.get(level, 0)
        pct = c / total_final * 100 if total_final > 0 else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    final_unknown = seniority_final_level_counts.get("unknown", 0)
    final_unknown_rate = final_unknown / total_final * 100 if total_final > 0 else 0
    log.info(
        f"\n  ** Stage 5 unknown rate: {final_unknown_rate:.1f}% "
        f"({final_unknown:,}/{total_final:,}) — Stage 10 LLM will fill these in **"
    )

    log.info(f"\n  seniority_final level distribution for SWE only (5-level):")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = swe_final_level_counts.get(level, 0)
        pct = c / total_swe * 100 if total_swe else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    log.info(f"\n  Derived seniority_3level distribution (all rows):")
    for val, count in seniority_3level_counts.most_common():
        log.info(f"    {val:<25} {count:>10,} ({count/written*100:.1f}%)")

    log.info(f"\n  Derived seniority_3level for SWE only:")
    for val, count in swe_seniority_3level_counts.most_common():
        pct = count / total_swe * 100 if total_swe else 0
        log.info(f"    {val:<25} {count:>10,} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage5():
    log.info("=" * 60)
    log.info("STAGE 5: Classification (SWE / SWE-adjacent / control + Seniority)")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage4_dedup.parquet"
    output_path = INTERMEDIATE_DIR / "stage5_classification.parquet"
    tmp_output_path = prepare_temp_output(output_path)

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    log.info(f"Input: {total_rows:,} rows")

    # Phase 1: Build SWE lookup (includes embedding computation)
    swe_lookup, control_lookup = build_swe_lookup(input_path)
    gc.collect()

    # Phase 2: Build seniority title lookup (strong rules only)
    seniority_title_lookup = build_seniority_title_lookup(input_path, swe_lookup, control_lookup)
    gc.collect()

    # Phase 3: Stream chunks, apply lookups, write output
    try:
        report_stats = streaming_write(
            input_path,
            tmp_output_path,
            swe_lookup,
            seniority_title_lookup,
            control_lookup,
        )
    except Exception:
        cleanup_temp_file(tmp_output_path)
        raise

    promote_temp_file(tmp_output_path, output_path)

    # Free lookups
    del swe_lookup, seniority_title_lookup, control_lookup
    gc.collect()

    # Phase 4: Report
    generate_report(input_path, output_path, report_stats)

    log.info(f"\nOutput: {output_path}")
    log.info(f"Rows written: {report_stats['written']:,}")
    log.info("Stage 5 complete.")


if __name__ == "__main__":
    run_stage5()
