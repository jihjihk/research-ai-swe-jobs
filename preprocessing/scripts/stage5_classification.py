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
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

    value = re.sub(r"\b(senior|sr\.?|junior|jr\.?|lead|staff|principal|distinguished)\b", "", value)
    value = re.sub(r"\b(i{1,3}|iv|v)\b", "", value)
    value = re.sub(r"\b[1-5]\b", "", value)
    value = re.sub(r"\s*[-–—]\s*(remote|hybrid|onsite|on-site)\s*$", "", value, flags=re.IGNORECASE)
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
    r'founding\s*engineer|member\s*of\s*technical\s*staff|'
    r'product\s*engineer|systems?\s*engineer'
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
    r'network\s*engineer|hardware\s*engineer'
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
    r'system|computing|computer|cyber|security|'
    r'automation|qa\b|quality\s*assur|test|'
    r'blockchain|crypto|defi|smart.?contract|'
    r'python|java|javascript|typescript|react|node|'
    r'golang|rust|ruby|scala|kotlin|swift|'
    r'it\b|information\s*tech'
    r')\b'
)


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
    title_control = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title_normalized", "title"]):
        chunk = batch.to_pandas()
        for title_key in chunk["title_normalized"].map(normalize_swe_title_key).drop_duplicates():
            t = title_key
            if not t:
                continue
            if t not in title_swe:
                title_swe[t] = bool(SWE_INCLUDE.search(t)) and not bool(SWE_EXCLUDE.search(t))
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
                lookup[t] = (True, False, 0.90, "embedding_llm")
                llm_swe += 1
            elif classification == "SWE_ADJACENT":
                lookup[t] = (False, True, 0.85, "embedding_llm")
                llm_adjacent += 1
            else:  # NOT_SWE
                lookup[t] = (False, False, 0.90, "embedding_llm")
                llm_not_swe += 1
            llm_resolved += 1
        else:
            still_unresolved.append(t)

    log.info(f"  LLM lookup resolved: {llm_resolved:,} (SWE={llm_swe}, ADJ={llm_adjacent}, NOT={llm_not_swe})")
    log.info(f"  Still unresolved after LLM lookup: {len(still_unresolved):,}")

    del unresolved
    gc.collect()

    # Pre-filter for embedding candidates
    candidates = [t for t in still_unresolved if TECH_PREFILTER.search(str(t))]
    non_candidates = [t for t in still_unresolved if not TECH_PREFILTER.search(str(t))]
    log.info(f"  Embedding candidates (after tech pre-filter): {len(candidates):,}")
    log.info(f"  Non-candidates (clearly non-SWE, skipping): {len(non_candidates):,}")

    # Mark non-candidates immediately
    for t in non_candidates:
        lookup[t] = (False, False, 0.0, "unresolved")

    del title_swe, all_titles, still_unresolved, non_candidates
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

# Description YOE patterns for contradiction checks only
YOE_PATTERN = re.compile(
    r'(\d+)(?:\s*[-–]\s*\d+)?\+?\s*(?:years?|yrs?)\s*(?:of\s+)?'
    r'(?:experience|professional|relevant|proven|work|working)',
    re.IGNORECASE
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


def extract_min_yoe(description: str):
    """Extract the minimum reasonable YOE mention from text."""
    if pd.isna(description) or str(description).strip() == "":
        return np.nan

    matches = YOE_PATTERN.findall(str(description).lower())
    years = [int(y) for y in matches if 0 < int(y) <= 20]
    if not years:
        return np.nan
    return float(min(years))


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
            unique_titles[raw_lower] = infer_seniority_family(
                raw_title,
                getattr(row, "title_normalized"),
                swe_lookup,
                control_lookup,
            )
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
            family = infer_seniority_family(
                getattr(row, "title"),
                getattr(row, "title_normalized"),
                swe_lookup,
                control_lookup,
            )
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
    if output_path.exists():
        output_path.unlink()
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

        yoe_values = []
        yoe_contradictions = []

        for idx in range(n):
            family = infer_seniority_family(
                chunk["title"].iloc[idx],
                chunk["title_normalized"].iloc[idx],
                swe_lookup,
                control_lookup,
            )
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

            yoe_val = extract_min_yoe(description_text.iloc[idx])
            yoe_values.append(yoe_val)
            yoe_contradictions.append(has_yoe_contradiction(final_level, yoe_val))

        chunk["seniority_imputed"] = seniority_levels
        chunk["seniority_source"] = seniority_sources
        chunk["seniority_confidence"] = np.array(seniority_confidences, dtype=np.float64)
        chunk["seniority_final"] = seniority_final_levels
        chunk["seniority_final_source"] = seniority_final_sources
        chunk["seniority_final_confidence"] = np.array(seniority_final_confidences, dtype=np.float64)
        chunk["yoe_extracted"] = np.array(yoe_values, dtype=np.float64)
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
    swe_seniority_3level_counts = report_stats["swe_seniority_3level_counts"]
    disagreement_pairs = report_stats["disagreement_pairs"]

    # --- SWE classification report ---
    log.info("\n--- 5a. SWE Classification ---")
    total = sum(swe_tier_counts.values())

    log.info(f"  {'Tier':<25} {'Rows':>10} {'Pct':>8}")
    log.info(f"  {'-'*25} {'-'*10} {'-'*8}")
    for tier in ["regex", "embedding_llm", "embedding_high", "embedding_adjacent", "unresolved"]:
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

    # Seniority level distribution
    log.info(f"\n  Seniority level distribution:")
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

    log.info(f"\n  Final seniority level distribution:")
    for level in ["entry", "associate", "mid-senior", "director", "unknown"]:
        c = seniority_final_level_counts.get(level, 0)
        pct = c / total_final * 100 if total_final > 0 else 0
        log.info(f"    {level:<25} {c:>10,} ({pct:.1f}%)")

    final_unknown = seniority_final_level_counts.get("unknown", 0)
    final_unknown_rate = final_unknown / total_final * 100 if total_final > 0 else 0
    log.info(f"\n  ** Final unknown rate: {final_unknown_rate:.1f}% ({final_unknown:,}/{total_final:,}) **")

    # Seniority 3-level distribution
    log.info(f"\n  Seniority 3-level distribution:")
    for val, count in seniority_3level_counts.most_common():
        log.info(f"    {val:<25} {count:>10,} ({count/written*100:.1f}%)")

    # Seniority 3-level for SWE only
    log.info(f"\n  Seniority 3-level for SWE only:")
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
    report_stats = streaming_write(
        input_path,
        output_path,
        swe_lookup,
        seniority_title_lookup,
        title_prior_lookup,
        control_lookup,
    )

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
