#!/usr/bin/env python3
"""
Stage 5: Classification (SWE / SWE-adjacent / control + seniority imputation)  -- V2

Redesigned based on manual review of 100 postings and LLM classification
of 500 Tier 2 titles.

Changes from V1:
  - Seniority: explicit-signal classifier applied uniformly (never prefers native).
    Title keywords > explicit description signals > unknown.
    YOE is extracted for contradiction flags only; it does NOT drive the enum.
  - SWE: Tier 2 now uses LLM-validated title lookup (500 titles) plus
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

# ---------------------------------------------------------------------------
# 5a. SWE Classification -- Patterns
# ---------------------------------------------------------------------------

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

    # Load LLM title lookup
    tier2_llm = {}
    if TIER2_LOOKUP_PATH.exists():
        llm_df = pd.read_parquet(TIER2_LOOKUP_PATH)
        for _, row in llm_df.iterrows():
            # Normalize to lowercase for matching against title_normalized
            tier2_llm[str(row["title"]).lower().strip()] = row["llm_swe_classification"]
        log.info(f"  Loaded LLM title lookup: {len(tier2_llm)} entries")
        llm_vc = Counter(tier2_llm.values())
        for k, v in llm_vc.items():
            log.info(f"    {k}: {v}")
        del llm_df
    else:
        log.warning(f"  LLM title lookup not found at {TIER2_LOOKUP_PATH}")

    # Collect unique raw titles and classify directly from the title text.
    pf = pq.ParquetFile(input_path)
    title_swe = {}
    title_control = {}

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title"]):
        chunk = batch.to_pandas()
        for title in chunk["title"].dropna().astype(str).str.lower().str.strip().drop_duplicates():
            t = title
            if t not in title_swe:
                title_swe[t] = bool(SWE_INCLUDE.search(t)) and not bool(SWE_EXCLUDE.search(t))
                title_control[t] = bool(CONTROL_PATTERN.search(t))
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
        t_lower = str(t).lower().strip()
        if t_lower in tier2_llm:
            classification = tier2_llm[t_lower]
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

# Description YOE patterns for contradiction checks only
YOE_PATTERN = re.compile(
    r'(\d+)(?:\s*[-–]\s*\d+)?\+?\s*(?:years?|yrs?)\s*(?:of\s+)?'
    r'(?:experience|professional|relevant|proven|work|working)',
    re.IGNORECASE
)

# Description explicit seniority cues only
DESC_ENTRY = re.compile(
    r'\b(junior|jr\.?|intern(?:ship)?|co-?op|new\s*grad|entry[- ]?level|early\s*career)\b',
    re.IGNORECASE
)
DESC_ASSOCIATE = re.compile(
    r'(?:engineer|developer|swe|sde)\s*i\b|'
    r'\bassociate\s+(?:engineer|developer|software|qa|test|data|ml|ai|devops|'
    r'cloud|platform|backend|back[- ]?end|frontend|front[- ]?end|full[- ]?stack)\b',
    re.IGNORECASE
)
DESC_MID_SENIOR = re.compile(
    r'\b(senior|sr\.?|staff|principal)\b|'
    r'(?:engineer|developer|swe|sde)\s*(?:ii|iii|iv|v|[2-9])\b',
    re.IGNORECASE
)
DESC_DIRECTOR = re.compile(
    r'\b(director|vp|vice\s*president|head\s+of|chief|cto|cio)\b',
    re.IGNORECASE
)


def classify_seniority(title: str, description: str) -> tuple:
    """
    Multi-signal seniority classifier. Applied UNIFORMLY to all rows.
    Never prefers native labels -- uses them only for cross-validation.

    Returns (seniority_level, source, confidence).

    Seniority levels: entry, associate, mid-senior, director, unknown
    Source: title_keyword, title_level_number, description_explicit, unknown
    Confidence: 0.0-1.0
    """
    title_str = str(title).lower().strip() if pd.notna(title) else ""
    desc_str = str(description).lower() if pd.notna(description) else ""

    # --- Signal 1: Title keywords (confidence 0.95) ---

    if TITLE_INTERN.search(title_str):
        return ("entry", "title_keyword", 0.95)

    if TITLE_DIRECTOR.search(title_str):
        return ("director", "title_keyword", 0.95)

    if TITLE_SENIOR.search(title_str):
        return ("mid-senior", "title_keyword", 0.95)

    if TITLE_LEAD.search(title_str):
        return ("mid-senior", "title_keyword", 0.95)

    # Level numbers
    if TITLE_LEVEL_HIGH.search(title_str):
        return ("mid-senior", "title_level_number", 0.95)
    if TITLE_LEVEL_SENIOR.search(title_str):
        return ("mid-senior", "title_level_number", 0.95)
    if TITLE_LEVEL_ASSOCIATE.search(title_str):
        return ("associate", "title_level_number", 0.95)

    if TITLE_JUNIOR.search(title_str):
        return ("entry", "title_keyword", 0.95)

    if TITLE_ASSOCIATE.search(title_str):
        return ("associate", "title_keyword", 0.95)

    # --- Signal 2: explicit description cues only (confidence 0.70) ---
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


def build_seniority_title_lookup(input_path: Path) -> dict:
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
    unique_titles = set()

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=["title"]):
        chunk = batch.to_pandas()
        # Use raw title, lowered + stripped for dedup
        unique_titles.update(
            chunk["title"].dropna().str.lower().str.strip().unique()
        )
        del chunk, batch
        gc.collect()

    log.info(f"  Unique raw titles to classify: {len(unique_titles):,}")

    # Apply title-only rules
    title_lookup = {}
    for t in unique_titles:
        seniority, source, confidence = classify_seniority(t, "")
        if source.startswith("title_"):
            title_lookup[t] = (seniority, source, confidence)

    resolved_counts = Counter(v[0] for v in title_lookup.values())
    log.info(f"  Title-based seniority resolved: {len(title_lookup):,}")
    for level, count in resolved_counts.most_common():
        log.info(f"    {level}: {count:,}")
    log.info(f"  Need description fallback: {len(unique_titles) - len(title_lookup):,}")

    elapsed = time.time() - t0
    log.info(f"  Seniority title lookup built in {elapsed:.1f}s")

    return title_lookup


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
    control_lookup: dict,
):
    """
    Stream chunks, apply lookups + description-based seniority fallback,
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
    seniority_level_counts = Counter()
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
        swe_results = chunk["title_normalized"].map(
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
        title_results = title_lower.map(
            lambda t: seniority_title_lookup.get(t, None)
        )

        seniority_levels = []
        seniority_sources = []
        seniority_confidences = []

        description_text = chunk["description_core"].where(
            chunk["description_core"].notna(),
            chunk["description"],
        )

        yoe_values = []
        yoe_contradictions = []

        for idx in range(n):
            tr = title_results.iloc[idx]
            if tr is not None:
                seniority_levels.append(tr[0])
                seniority_sources.append(tr[1])
                seniority_confidences.append(tr[2])
            else:
                title_val = title_lower.iloc[idx]
                desc_val = description_text.iloc[idx]
                level, source, conf = classify_seniority(title_val, desc_val if pd.notna(desc_val) else "")
                seniority_levels.append(level)
                seniority_sources.append(source)
                seniority_confidences.append(conf)

            yoe_val = extract_min_yoe(description_text.iloc[idx])
            yoe_values.append(yoe_val)
            yoe_contradictions.append(has_yoe_contradiction(seniority_levels[-1], yoe_val))

        chunk["seniority_imputed"] = seniority_levels
        chunk["seniority_source"] = seniority_sources
        chunk["seniority_confidence"] = np.array(seniority_confidences, dtype=np.float64)
        chunk["yoe_extracted"] = np.array(yoe_values, dtype=np.float64)
        chunk["yoe_seniority_contradiction"] = np.array(yoe_contradictions, dtype=bool)

        # seniority_final = seniority_imputed (uniform classifier is canonical)
        chunk["seniority_final"] = chunk["seniority_imputed"]

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

        level_vc = chunk["seniority_imputed"].value_counts()
        for level, count in level_vc.items():
            seniority_level_counts[level] += int(count)

        cc_vc = chunk["seniority_cross_check"].value_counts()
        for cc, count in cc_vc.items():
            cross_check_counts[cc] += int(count)

        del title_results, seniority_levels, seniority_sources, seniority_confidences

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
        "seniority_level_counts": seniority_level_counts,
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
    seniority_level_counts = report_stats["seniority_level_counts"]
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
    for src in ["title_keyword", "title_level_number", "description_explicit", "unknown"]:
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
    seniority_title_lookup = build_seniority_title_lookup(input_path)
    gc.collect()

    # Phase 3: Stream chunks, apply lookups, write output
    report_stats = streaming_write(
        input_path,
        output_path,
        swe_lookup,
        seniority_title_lookup,
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
