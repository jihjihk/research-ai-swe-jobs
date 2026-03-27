#!/usr/bin/env python3
"""
Stage 5: Classification (SWE / SWE-adjacent / control + seniority imputation)  -- V2

Redesigned based on manual review of 100 postings and LLM classification
of 500 Tier 2 titles.

Changes from V1:
  - Seniority: explicit-signal classifier applied uniformly for
    `seniority_imputed`. `seniority_final` now applies a family-aware
    resolver for SWE-adjacent and control rows:
      strong title cues > native backfill > within-family title prior >
      weak title/description cues > unknown.
    YOE is extracted for contradiction flags only; it does NOT drive the enum.
  - SWE: Tier 2 now uses a curated title lookup artifact plus
    split thresholds (>= 0.85 = SWE, 0.70-0.85 = SWE_ADJACENT).

Memory-safe: builds lookup dicts from unique titles first, then streams
chunks through for the final write pass.  31GB RAM constraint.

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

from io_utils import cleanup_temp_file, prepare_temp_output, promote_temp_file

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
    r'\b(junior|jr\.?|new\s*grad|entry[- ]?level|early\s*career)\b', re.IGNORECASE
)
TITLE_ASSOCIATE = re.compile(r'\bassociate\b', re.IGNORECASE)
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

# Level numbers in SWE titles
TITLE_LEVEL_HIGH = re.compile(
    r'(?:engineer|developer|swe|sde)\s*(?:iii|iv|v|[3-9])\b', re.IGNORECASE
)
TITLE_LEVEL_SENIOR = re.compile(
    r'(?:engineer|developer|swe|sde)\s*ii\b', re.IGNORECASE
)
TITLE_LEVEL_ASSOCIATE = re.compile(
    r'(?:engineer|developer|swe|sde)\s*i\b', re.IGNORECASE
)

CONTROL_LEVEL_HIGH = re.compile(
    r'\b(?:accountant|analyst|engineer|nurse(?:\s*case\s*manager)?|nurse\s*practitioner)\s*(?:ii|iii|iv|v|[2-9])\b',
    re.IGNORECASE,
)
CONTROL_LEVEL_ENTRY = re.compile(
    r'\b(?:accountant|analyst|engineer)\s*i\b',
    re.IGNORECASE,
)
ADJACENT_LEVEL_HIGH = re.compile(
    r'\b(?:analyst|engineer|architect|developer|administrator|specialist)\s*(?:ii|iii|iv|v|[2-9])\b',
    re.IGNORECASE,
)
ADJACENT_LEVEL_ENTRY = re.compile(
    r'\b(?:analyst|engineer|developer|administrator|specialist)\s*i\b',
    re.IGNORECASE,
)

# Description YOE patterns for contradiction checks only.
# This extractor intentionally works from raw description text instead of
# description_core because Stage 3 can trim requirement-heavy sections.
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

# Description explicit seniority cues only. These must refer to the posting
# itself via role/job context or title-like role labels, not generic mentions
# of other senior people, support staff, or reporting lines.
ROLE_CONTEXT = r'(?:role|position|job|opening|opportunity|title)'
HIRING_CONTEXT = (
    r"(?:we(?:'re| are)?\s+hiring|"
    r"we(?:'re| are)?\s+looking\s+for|"
    r"we(?:'re| are)?\s+looking\s+to\s+hire|"
    r"we(?:'re| are)?\s+actively\s+looking\s+to\s+hire|"
    r"currently\s+looking\s+to\s+hire|"
    r"seeking|"
    r"join\s+us\s+as)"
)
DECLARATIVE_ROLE_PREFIX = (
    rf'(?:\b(?:this|the)\s+{ROLE_CONTEXT}\s+(?:is|will\s+be|would\s+be|is\s+for)\s+(?:an?\s+)?|'
    rf'\b(?:{ROLE_CONTEXT})\s*[:\-]\s*(?:an?\s+)?)'
)
HIRING_ROLE_PREFIX = rf'(?:\b(?:{HIRING_CONTEXT})\s+(?:an?\s+)?)'
ACTION_ROLE_PREFIX = r'(?:^|[\n.;:])\s*(?:as|serve\s+as|work\s+as)\s+(?:an?\s+)?'
ROLE_NOUN = (
    r'(?:engineer|developer|consultant|analyst|scientist|architect|manager|'
    r'administrator|coordinator|specialist|designer|assistant|officer|'
    r'representative|attorney|lawyer|accountant|therapist|nurse|physician|'
    r'pharmacist|technician|installer|guide|generalist|integrator|'
    r'superintendent|writer|editor|professional)'
)
ROLE_MODIFIER_WORD = (
    r'(?!of\b|to\b|the\b|a\b|an\b|and\b|with\b|for\b|as\b|well\b|or\b|'
    r'in\b|on\b|by\b)[\w/&+.-]+'
)
ROLE_TITLE_PART = rf'(?:{ROLE_MODIFIER_WORD}|&|and)'
ROLE_TITLE_TAIL = rf'(?:{ROLE_TITLE_PART}\s+){{0,4}}{ROLE_NOUN}\b'
ENTRY_TITLE = (
    rf'(?:entry[- ]?level|junior|jr\.?|new\s*grad(?:uate)?|intern(?:ship)?|'
    rf'co-?op|early\s*career)\s+{ROLE_TITLE_TAIL}'
)
ASSOCIATE_LABEL = r'associate(?!\s+(?:vice\s*president|vp|director|head|chief|dean)\b)'
ASSOCIATE_TITLE = (
    rf'(?:{ASSOCIATE_LABEL}(?:\s+{ROLE_TITLE_TAIL})?|'
    rf'(?:{ROLE_MODIFIER_WORD}\s+){{0,2}}{ASSOCIATE_LABEL}\b)'
)
MID_SENIOR_TITLE = (
    rf'(?:(?:senior\s+(?!level\b))|sr\.?\s+|staff\s+|principal\s+){ROLE_TITLE_TAIL}'
)
DIRECTOR_FUNCTION = (
    r'(?:engineering|operations|marketing|finance|communications|technology|'
    r'security|data|product|property|resources|sales|hr|people|talent)'
)
DIRECTOR_TITLE = (
    rf'(?:director(?:\s+of\s+(?:[\w/&+.-]+\s+){{0,3}}{DIRECTOR_FUNCTION})?|'
    rf'vp|vice\s*president(?:\s+of\s+(?:[\w/&+.-]+\s+){{0,3}}{DIRECTOR_FUNCTION})?|'
    rf'head\s+of\s+(?:[\w/&+.-]+\s+){{0,3}}{DIRECTOR_FUNCTION}|'
    r'chief\s+(?:officer|technology\s+officer|information\s+officer)|'
    r'cto|cio)'
)

DESC_ENTRY = re.compile(
    rf'(?:{DECLARATIVE_ROLE_PREFIX}{ENTRY_TITLE}|'
    rf'{HIRING_ROLE_PREFIX}{ENTRY_TITLE}|'
    rf'{ACTION_ROLE_PREFIX}{ENTRY_TITLE})',
    re.IGNORECASE,
)
DESC_ASSOCIATE = re.compile(
    rf'(?:{DECLARATIVE_ROLE_PREFIX}{ASSOCIATE_TITLE}|'
    rf'{HIRING_ROLE_PREFIX}{ASSOCIATE_TITLE}|'
    rf'{ACTION_ROLE_PREFIX}{ASSOCIATE_TITLE})|'
    r'\b(?:engineer|developer|swe|sde)\s*i\b',
    re.IGNORECASE,
)
DESC_MID_SENIOR = re.compile(
    rf'(?:{DECLARATIVE_ROLE_PREFIX}{MID_SENIOR_TITLE}|'
    rf'{HIRING_ROLE_PREFIX}{MID_SENIOR_TITLE}|'
    rf'{ACTION_ROLE_PREFIX}{MID_SENIOR_TITLE})|'
    r'\b(?:engineer|developer|swe|sde)\s*(?:ii|iii|iv|v|[2-9])\b',
    re.IGNORECASE,
)
DESC_DIRECTOR = re.compile(
    rf'(?:{DECLARATIVE_ROLE_PREFIX}{DIRECTOR_TITLE}|'
    rf'{HIRING_ROLE_PREFIX}{DIRECTOR_TITLE}|'
    rf'{ACTION_ROLE_PREFIX}{DIRECTOR_TITLE})',
    re.IGNORECASE,
)

TITLE_ONLY_SOURCES = {
    "title_keyword",
    "title_manager",
    "weak_title_level",
    "weak_title_associate",
}
STRONG_FINAL_SOURCES = {"title_keyword", "title_manager"}
TITLE_PRIOR_THRESHOLDS = {
    "swe": (0.90, 20),
    "adjacent": (0.85, 20),
    "control": (0.85, 20),
}


def family_role_hint(family: str):
    if family == "adjacent":
        return ADJACENT_ROLE_HINT
    if family == "control":
        return CONTROL_ROLE_HINT
    return None


def infer_seniority_family(
    title_raw: str,
    title_normalized,
    swe_lookup: dict,
    control_lookup: dict,
) -> str:
    raw_lower = str(title_raw).lower().strip() if pd.notna(title_raw) else ""
    normalized_key = (
        str(title_normalized).lower().strip()
        if pd.notna(title_normalized) and str(title_normalized).strip()
        else normalize_swe_title_key(raw_lower)
    )
    normalized_key = normalize_swe_title_key(normalized_key)
    if control_lookup.get(raw_lower, False):
        return "control"
    swe_tuple = swe_lookup.get(normalized_key, (False, False, 0.0, "unresolved"))
    if swe_tuple[1]:
        return "adjacent"
    if swe_tuple[0]:
        return "swe"
    return "other"


def normalize_title_prior_key(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower().strip())


def resolve_seniority_final(
    family: str,
    imputed_level: str,
    imputed_source: str,
    imputed_confidence: float,
    native_raw,
    title_prior,
):
    """
    Resolve the final seniority label used downstream.

    For SWE / SWE-adjacent / control rows, plain titles plus native metadata are
    often more informative than weak text-only cues. Strong title signals still
    win. Other families remain text-only for now.
    """
    if family not in {"swe", "adjacent", "control"}:
        return imputed_level, imputed_source, float(imputed_confidence)

    if imputed_source in STRONG_FINAL_SOURCES:
        return imputed_level, imputed_source, float(imputed_confidence)

    native = normalize_native(native_raw)
    if native is not None:
        return native, "native_backfill", 0.85

    if title_prior is not None and imputed_source == "unknown":
        prior_level, prior_share, _ = title_prior
        return prior_level, "title_prior", float(prior_share)

    if imputed_source != "unknown":
        return imputed_level, imputed_source, float(imputed_confidence)

    return "unknown", "unknown", 0.0


def classify_seniority(title: str, description: str, family: str = "other") -> tuple:
    """
    Multi-signal seniority classifier. Applied UNIFORMLY to all rows.
    Never prefers native labels -- uses them only for cross-validation.

    Returns (seniority_level, source, confidence).

    Seniority levels: entry, associate, mid-senior, director, unknown
    Source: title_keyword, title_manager, weak_title_level,
            weak_title_associate, description_explicit, unknown
    Confidence: 0.0-1.0
    """
    title_str = str(title).lower().strip() if pd.notna(title) else ""
    desc_str = str(description).lower() if pd.notna(description) else ""
    role_hint = family_role_hint(family)

    # --- Signal 1: Title keywords (confidence 0.95) ---

    if TITLE_INTERN.search(title_str):
        return ("entry", "title_keyword", 0.95)

    if TITLE_DIRECTOR.search(title_str):
        return ("director", "title_keyword", 0.95)

    if TITLE_SENIOR.search(title_str):
        return ("mid-senior", "title_keyword", 0.95)

    if role_hint and role_hint.search(title_str):
        if TITLE_MANAGERISH.search(title_str) or TITLE_LEAD_GENERAL.search(title_str):
            return ("mid-senior", "title_manager", 0.92)

    if TITLE_LEAD.search(title_str):
        return ("mid-senior", "title_keyword", 0.95)

    # Level numbers
    if TITLE_LEVEL_HIGH.search(title_str):
        return ("mid-senior", "weak_title_level", 0.85)
    if TITLE_LEVEL_SENIOR.search(title_str):
        return ("mid-senior", "weak_title_level", 0.85)
    if TITLE_LEVEL_ASSOCIATE.search(title_str):
        return ("associate", "weak_title_level", 0.80)

    if family == "control":
        if CONTROL_LEVEL_HIGH.search(title_str):
            return ("mid-senior", "weak_title_level", 0.80)
        if CONTROL_LEVEL_ENTRY.search(title_str):
            return ("associate", "weak_title_level", 0.70)

    if family == "adjacent":
        if ADJACENT_LEVEL_HIGH.search(title_str):
            return ("mid-senior", "weak_title_level", 0.80)
        if ADJACENT_LEVEL_ENTRY.search(title_str):
            return ("associate", "weak_title_level", 0.70)

    if TITLE_JUNIOR.search(title_str):
        return ("entry", "title_keyword", 0.95)

    if TITLE_ASSOCIATE.search(title_str):
        return ("associate", "weak_title_associate", 0.70)

    # --- Signal 2: explicit description role labels only (confidence 0.70) ---
    if desc_str:
        if DESC_DIRECTOR.search(desc_str):
            return ("director", "description_explicit", 0.70)
        if DESC_MID_SENIOR.search(desc_str):
            return ("mid-senior", "description_explicit", 0.70)
        if DESC_ASSOCIATE.search(desc_str):
            return ("associate", "description_explicit", 0.70)
        if DESC_ENTRY.search(desc_str):
            return ("entry", "description_explicit", 0.70)

    return ("unknown", "unknown", 0.0)


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
) -> dict:
    """
    Build lookup: raw title (lowered) -> (seniority, source, confidence)
    Only covers cases where the title alone is enough to decide.
    Description-based signals are applied row-by-row in the streaming pass.

    CRITICAL: Uses the raw 'title' column, NOT 'title_normalized', because
    title_normalized strips seniority prefixes (Senior, Sr, Lead, Staff, etc.)
    which is exactly what we need to classify on.
    """
    log.info("--- 5b. Building seniority title lookup ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    unique_titles = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title", "title_normalized"]):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            raw_title = getattr(row, "title")
            if pd.isna(raw_title):
                continue
            raw_lower = str(raw_title).lower().strip()
            if not raw_lower:
                continue
            # Derive family using the same swe_lookup path as the streaming
            # pass (normalize_swe_title_key on title_normalized) to avoid
            # the double-normalization bug in infer_seniority_family.
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

    # Apply title-only rules
    title_lookup = {}
    for t, family in unique_titles.items():
        seniority, source, confidence = classify_seniority(t, "", family=family)
        if source in TITLE_ONLY_SOURCES:
            title_lookup[t] = (seniority, source, confidence)

    resolved_counts = Counter(v[0] for v in title_lookup.values())
    log.info(f"  Title-based seniority resolved: {len(title_lookup):,}")
    for level, count in resolved_counts.most_common():
        log.info(f"    {level}: {count:,}")
    log.info(f"  Need description fallback: {len(unique_titles) - len(title_lookup):,}")

    elapsed = time.time() - t0
    log.info(f"  Seniority title lookup built in {elapsed:.1f}s")

    return title_lookup


def build_group_title_prior_lookup(
    input_path: Path,
    swe_lookup: dict,
    control_lookup: dict,
) -> dict:
    """
    Build conservative within-family priors for normalized titles using only
    rows that already carry native seniority.
    """
    log.info("--- 5b. Building family title-prior lookup ---")
    t0 = time.time()

    counts = Counter()
    pf = pq.ParquetFile(input_path)

    for batch in pf.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["title", "title_normalized", "seniority_native"],
    ):
        chunk = batch.to_pandas()
        for row in chunk.itertuples(index=False):
            native = normalize_native(getattr(row, "seniority_native"))
            if native is None:
                continue
            # Derive family using the same swe_lookup path as the streaming
            # pass to avoid the double-normalization bug.
            raw_title = getattr(row, "title")
            raw_lower_f = str(raw_title).lower().strip() if pd.notna(raw_title) else ""
            swe_key = normalize_swe_title_key(getattr(row, "title_normalized"))
            swe_tuple = swe_lookup.get(swe_key, (False, False, 0.0, "unresolved"))
            if control_lookup.get(raw_lower_f, False):
                family = "control"
            elif swe_tuple[1]:
                family = "adjacent"
            elif swe_tuple[0]:
                family = "swe"
            else:
                family = "other"
            if family not in TITLE_PRIOR_THRESHOLDS:
                continue
            title_key = normalize_title_prior_key(getattr(row, "title_normalized"))
            if not title_key:
                continue
            counts[(family, title_key, native)] += 1
        del chunk, batch
        gc.collect()

    grouped = {}
    for (family, title_key, native), count in counts.items():
        grouped.setdefault((family, title_key), []).append((native, count))

    lookup = {}
    kept_counts = Counter()
    for key, label_counts in grouped.items():
        label_counts.sort(key=lambda item: (-item[1], item[0]))
        total_n = sum(count for _, count in label_counts)
        top_label, top_count = label_counts[0]
        top_share = top_count / total_n if total_n else 0.0
        min_share, min_n = TITLE_PRIOR_THRESHOLDS[key[0]]
        if total_n >= min_n and top_share >= min_share:
            lookup[key] = (top_label, top_share, total_n)
            kept_counts[key[0]] += 1

    log.info("  Family title priors kept: %s", len(lookup))
    for family, count in kept_counts.items():
        log.info("    %s: %s", family, f"{count:,}")
    log.info("  Family title-prior lookup built in %.1fs", time.time() - t0)
    return lookup


# ---------------------------------------------------------------------------
# 5c. 3-level bucketing
# ---------------------------------------------------------------------------

SENIORITY_3LEVEL = {
    "entry": "junior",
    "associate": "mid",
    "mid-senior": "senior",
    "director": "senior",
    "unknown": "unknown",
}


def map_3level(seniority: str) -> str:
    return SENIORITY_3LEVEL.get(str(seniority).lower().strip(), "unknown")


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

# Normalize native seniority labels for comparison
NATIVE_NORMALIZE = {
    "mid senior": "mid-senior",
    "mid-senior level": "mid-senior",
    "mid-senior": "mid-senior",
    "entry level": "entry",
    "entry": "entry",
    "associate": "associate",
    "director": "director",
    "executive": "director",
    "internship": "entry",
    "intern": "entry",
}


def normalize_native(val):
    """Normalize LinkedIn's seniority label for comparison."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    return NATIVE_NORMALIZE.get(str(val).lower().strip(), str(val).lower().strip())


def cross_check(our_level: str, native_raw) -> str:
    """
    Compare our classification against native label.
    Returns: 'agrees', 'native_disagrees', 'no_native'
    """
    native = normalize_native(native_raw)
    if native is None:
        return "no_native"
    if native == our_level:
        return "agrees"
    return "native_disagrees"


# ---------------------------------------------------------------------------
# Streaming write pass
# ---------------------------------------------------------------------------

def streaming_write(
    input_path: Path,
    output_path: Path,
    swe_lookup: dict,
    seniority_title_lookup: dict,
    title_prior_lookup: dict,
    control_lookup: dict,
):
    """
    Stream chunks, apply lookups + explicit-description seniority fallback,
    write incrementally.
    """
    log.info("--- Streaming write pass ---")
    t0 = time.time()

    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    writer = None
    written = 0

    # Counters for reporting
    swe_tier_counts = Counter()
    seniority_source_counts = Counter()
    seniority_final_source_counts = Counter()
    seniority_level_counts = Counter()
    seniority_final_level_counts = Counter()
    cross_check_counts = Counter()
    swe_by_source_counts = Counter()
    adj_by_source_counts = Counter()
    total_by_source_counts = Counter()
    swe_only_tier_counts = Counter()
    adj_only_tier_counts = Counter()
    seniority_3level_counts = Counter()
    swe_final_level_counts = Counter()
    swe_seniority_3level_counts = Counter()
    disagreement_pairs = Counter()

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

        # ---- 5b: Seniority classification (multi-signal, uniform) ----
        # CRITICAL: Use raw 'title' column (not title_normalized) because
        # title_normalized strips seniority prefixes (Senior, Sr, Lead, etc.)
        title_lower = chunk["title"].fillna("").str.lower().str.strip()
        chunk["is_control"] = title_lower.map(lambda t: control_lookup.get(t, False)).astype(bool)

        # Enforce mutual exclusion: SWE > SWE_adjacent > Control
        chunk.loc[chunk["is_swe"], "is_control"] = False
        chunk.loc[chunk["is_swe"], "is_swe_adjacent"] = False
        chunk.loc[chunk["is_swe_adjacent"], "is_control"] = False

        title_results = title_lower.map(lambda t: seniority_title_lookup.get(t, None))
        title_prior_keys = chunk["title_normalized"].map(normalize_title_prior_key)

        seniority_levels = []
        seniority_sources = []
        seniority_confidences = []
        seniority_final_levels = []
        seniority_final_sources = []
        seniority_final_confidences = []

        description_text = chunk["description_core"].where(
            chunk["description_core"].notna(),
            chunk["description"],
        )
        yoe_text = chunk["description"].where(
            chunk["description"].notna(),
            description_text,
        )

        yoe_values = []
        yoe_min_values = []
        yoe_max_values = []
        yoe_match_counts = []
        yoe_resolution_rules = []
        yoe_mentions_json = []
        yoe_contradictions = []

        for idx in range(n):
            # Derive family from already-computed classification columns
            # (avoids double-normalization bug in infer_seniority_family)
            if chunk["is_control"].iloc[idx]:
                family = "control"
            elif chunk["is_swe_adjacent"].iloc[idx]:
                family = "adjacent"
            elif chunk["is_swe"].iloc[idx]:
                family = "swe"
            else:
                family = "other"
            tr = title_results.iloc[idx]
            if tr is not None:
                level, source, conf = tr
            else:
                title_val = title_lower.iloc[idx]
                desc_val = description_text.iloc[idx]
                level, source, conf = classify_seniority(
                    title_val,
                    desc_val if pd.notna(desc_val) else "",
                    family=family,
                )

            seniority_levels.append(level)
            seniority_sources.append(source)
            seniority_confidences.append(conf)

            title_prior = title_prior_lookup.get((family, title_prior_keys.iloc[idx]))
            final_level, final_source, final_conf = resolve_seniority_final(
                family=family,
                imputed_level=level,
                imputed_source=source,
                imputed_confidence=conf,
                native_raw=chunk["seniority_native"].iloc[idx],
                title_prior=title_prior,
            )
            seniority_final_levels.append(final_level)
            seniority_final_sources.append(final_source)
            seniority_final_confidences.append(final_conf)

            yoe_features = extract_yoe_features(yoe_text.iloc[idx])
            yoe_val = yoe_features["yoe_extracted"]
            yoe_values.append(yoe_val)
            yoe_min_values.append(yoe_features["yoe_min_extracted"])
            yoe_max_values.append(yoe_features["yoe_max_extracted"])
            yoe_match_counts.append(yoe_features["yoe_match_count"])
            yoe_resolution_rules.append(yoe_features["yoe_resolution_rule"])
            yoe_mentions_json.append(yoe_features["yoe_all_mentions_json"])
            yoe_contradictions.append(has_yoe_contradiction(final_level, yoe_val))

        chunk["seniority_imputed"] = seniority_levels
        chunk["seniority_source"] = seniority_sources
        chunk["seniority_confidence"] = np.array(seniority_confidences, dtype=np.float64)
        chunk["seniority_final"] = seniority_final_levels
        chunk["seniority_final_source"] = seniority_final_sources
        chunk["seniority_final_confidence"] = np.array(seniority_final_confidences, dtype=np.float64)
        chunk["yoe_extracted"] = np.array(yoe_values, dtype=np.float64)
        chunk["yoe_min_extracted"] = np.array(yoe_min_values, dtype=np.float64)
        chunk["yoe_max_extracted"] = np.array(yoe_max_values, dtype=np.float64)
        chunk["yoe_match_count"] = np.array(yoe_match_counts, dtype=np.int16)
        chunk["yoe_resolution_rule"] = yoe_resolution_rules
        chunk["yoe_all_mentions_json"] = yoe_mentions_json
        chunk["yoe_seniority_contradiction"] = np.array(yoe_contradictions, dtype=bool)

        # Normalize seniority_native
        chunk["seniority_native"] = chunk["seniority_native"].replace("", pd.NA)

        # Cross-validation
        chunk["seniority_cross_check"] = [
            cross_check(chunk["seniority_imputed"].iloc[i], chunk["seniority_native"].iloc[i])
            for i in range(n)
        ]

        # Count sources and levels
        src_vc = chunk["seniority_source"].value_counts()
        for src, count in src_vc.items():
            seniority_source_counts[src] += int(count)
        final_src_vc = chunk["seniority_final_source"].value_counts()
        for src, count in final_src_vc.items():
            seniority_final_source_counts[src] += int(count)

        level_vc = chunk["seniority_imputed"].value_counts()
        for level, count in level_vc.items():
            seniority_level_counts[level] += int(count)
        final_level_vc = chunk["seniority_final"].value_counts()
        for level, count in final_level_vc.items():
            seniority_final_level_counts[level] += int(count)
        for level, count in chunk.loc[chunk["is_swe"], "seniority_final"].value_counts().items():
            swe_final_level_counts[level] += int(count)

        cc_vc = chunk["seniority_cross_check"].value_counts()
        for cc, count in cc_vc.items():
            cross_check_counts[cc] += int(count)

        del title_results, seniority_levels, seniority_sources, seniority_confidences
        del seniority_final_levels, seniority_final_sources, seniority_final_confidences

        # ---- 5c: 3-level bucketing ----
        chunk["seniority_3level"] = chunk["seniority_final"].apply(map_3level)

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
        for level, count in chunk["seniority_3level"].value_counts().items():
            seniority_3level_counts[level] += int(count)
        for level, count in chunk.loc[chunk["is_swe"], "seniority_3level"].value_counts().items():
            swe_seniority_3level_counts[level] += int(count)

        disagree_mask = chunk["seniority_cross_check"] == "native_disagrees"
        disagree_chunk = pd.DataFrame({
            "native": chunk.loc[disagree_mask, "seniority_native"].apply(normalize_native),
            "ours": chunk.loc[disagree_mask, "seniority_imputed"],
        }).fillna("missing")
        for row in disagree_chunk.itertuples(index=False):
            disagreement_pairs[(row.native, row.ours)] += 1

        # ---- Force float64 on numeric columns for schema consistency ----
        chunk["swe_confidence"] = chunk["swe_confidence"].astype(np.float64)
        chunk["seniority_confidence"] = chunk["seniority_confidence"].astype(np.float64)
        for col in chunk.select_dtypes(include=["object"]).columns:
            chunk[col] = chunk[col].astype("string")

        # ---- Write ----
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
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
        "seniority_source_counts": seniority_source_counts,
        "seniority_final_source_counts": seniority_final_source_counts,
        "seniority_level_counts": seniority_level_counts,
        "seniority_final_level_counts": seniority_final_level_counts,
        "cross_check_counts": cross_check_counts,
        "swe_by_source_counts": swe_by_source_counts,
        "adj_by_source_counts": adj_by_source_counts,
        "total_by_source_counts": total_by_source_counts,
        "swe_only_tier_counts": swe_only_tier_counts,
        "adj_only_tier_counts": adj_only_tier_counts,
        "seniority_3level_counts": seniority_3level_counts,
        "swe_final_level_counts": swe_final_level_counts,
        "swe_seniority_3level_counts": swe_seniority_3level_counts,
        "disagreement_pairs": disagreement_pairs,
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
    seniority_source_counts = report_stats["seniority_source_counts"]
    seniority_final_source_counts = report_stats["seniority_final_source_counts"]
    seniority_level_counts = report_stats["seniority_level_counts"]
    seniority_final_level_counts = report_stats["seniority_final_level_counts"]
    cross_check_counts = report_stats["cross_check_counts"]
    swe_by_source_counts = report_stats["swe_by_source_counts"]
    adj_by_source_counts = report_stats["adj_by_source_counts"]
    total_by_source_counts = report_stats["total_by_source_counts"]
    swe_only_tier_counts = report_stats["swe_only_tier_counts"]
    adj_only_tier_counts = report_stats["adj_only_tier_counts"]
    seniority_3level_counts = report_stats["seniority_3level_counts"]
    swe_final_level_counts = report_stats["swe_final_level_counts"]
    swe_seniority_3level_counts = report_stats["swe_seniority_3level_counts"]
    disagreement_pairs = report_stats["disagreement_pairs"]

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
    log.info("\n--- 5b. Seniority Classification ---")
    log.info(f"\n  Seniority by source signal:")
    log.info(f"  {'Source':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    total_sen = sum(seniority_source_counts.values())
    for src in [
        "title_keyword",
        "title_manager",
        "weak_title_level",
        "weak_title_associate",
        "description_explicit",
        "unknown",
    ]:
        c = seniority_source_counts.get(src, 0)
        pct = c / total_sen * 100 if total_sen > 0 else 0
        log.info(f"  {src:<25} {c:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<25} {total_sen:>10,}")

    # Text-only seniority distribution
    log.info(f"\n  Text-imputed seniority level distribution (5-level):")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = seniority_level_counts.get(level, 0)
        pct = c / total_sen * 100 if total_sen > 0 else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    unknown_count = seniority_level_counts.get("unknown", 0)
    unknown_rate = unknown_count / total_sen * 100 if total_sen > 0 else 0
    log.info(f"\n  ** Unknown rate: {unknown_rate:.1f}% ({unknown_count:,}/{total_sen:,}) **")

    log.info(f"\n  Final seniority resolution source:")
    log.info(f"  {'Source':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    total_final = sum(seniority_final_source_counts.values())
    for src in [
        "title_keyword",
        "title_manager",
        "native_backfill",
        "title_prior",
        "weak_title_level",
        "weak_title_associate",
        "description_explicit",
        "unknown",
    ]:
        c = seniority_final_source_counts.get(src, 0)
        pct = c / total_final * 100 if total_final > 0 else 0
        log.info(f"  {src:<25} {c:>10,} {pct:>7.1f}%")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    log.info(f"  {'TOTAL':<25} {total_final:>10,}")

    log.info(f"\n  Final seniority level distribution (5-level, canonical):")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = seniority_final_level_counts.get(level, 0)
        pct = c / total_final * 100 if total_final > 0 else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    final_unknown = seniority_final_level_counts.get("unknown", 0)
    final_unknown_rate = final_unknown / total_final * 100 if total_final > 0 else 0
    log.info(f"\n  ** Final unknown rate: {final_unknown_rate:.1f}% ({final_unknown:,}/{total_final:,}) **")

    log.info(f"\n  Final seniority level distribution for SWE only (5-level, canonical):")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = swe_final_level_counts.get(level, 0)
        pct = c / total_swe * 100 if total_swe else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    # Supplemental 3-level derived view
    log.info(f"\n  Supplemental derived seniority_3level distribution:")
    for val, count in seniority_3level_counts.most_common():
        log.info(f"    {val:<25} {count:>10,} ({count/written*100:.1f}%)")

    log.info(f"\n  Supplemental derived seniority_3level for SWE only:")
    for val, count in swe_seniority_3level_counts.most_common():
        pct = count / total_swe * 100 if total_swe else 0
        log.info(f"    {val:<25} {count:>10,} ({pct:.1f}%)")

    # Cross-validation report
    log.info(f"\n--- Cross-validation: native vs classifier ---")
    log.info(f"  {'Check':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    total_cc = sum(cross_check_counts.values())
    for cc in ["agrees", "native_disagrees", "no_native"]:
        c = cross_check_counts.get(cc, 0)
        pct = c / total_cc * 100 if total_cc > 0 else 0
        log.info(f"  {cc:<25} {c:>10,} {pct:>7.1f}%")

    # Disagreement details
    if disagreement_pairs:
        total_disagree = sum(disagreement_pairs.values())
        log.info(f"\n  Disagreement breakdown ({total_disagree:,} rows):")
        for (native, ours), count in disagreement_pairs.most_common(20):
            log.info(f"    native={native:<12} ours={ours:<12} {count:>8,}")

    # Agreement rate where native exists
    total_with_native = total_cc - cross_check_counts.get("no_native", 0)
    if total_with_native > 0:
        agree = cross_check_counts.get("agrees", 0)
        log.info(f"\n  Agreement rate (where native exists): {agree/total_with_native:.1%} "
                 f"({agree:,}/{total_with_native:,})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage5():
    log.info("=" * 60)
    log.info("STAGE 5: Classification V2 (SWE / SWE-adjacent / control + Seniority)")
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

    # Phase 2: Build seniority title lookup
    seniority_title_lookup = build_seniority_title_lookup(input_path, swe_lookup, control_lookup)
    title_prior_lookup = build_group_title_prior_lookup(input_path, swe_lookup, control_lookup)
    gc.collect()

    # Phase 3: Stream chunks, apply lookups, write output
    try:
        report_stats = streaming_write(
            input_path,
            tmp_output_path,
            swe_lookup,
            seniority_title_lookup,
            title_prior_lookup,
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
