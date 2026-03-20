#!/usr/bin/env python3
"""
Stage 6a: Company Name Standardization (V2)

Builds a canonical company name lookup table using tiered normalization:
  1. cleanco legal suffix stripping + text normalization + abbreviation expansion
  2. Exact normalized grouping
  3. Known alias overrides
  4. 4-char prefix blocking
  5. Composite fuzzy score (Jaro-Winkler + token_sort_ratio + ratio)
     with length-dependent thresholds and generic-word guards

Input:  intermediate/stage1_unified.parquet (company_name column only)
Output: intermediate/company_name_lookup.parquet
Log:    logs/stage6a_company_names.log
"""

import re
import logging
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from cleanco import basename as cleanco_basename

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

INPUT_PATH = INTERMEDIATE_DIR / "stage1_unified.parquet"
OUTPUT_PATH = INTERMEDIATE_DIR / "company_name_lookup.parquet"
LOG_PATH = LOG_DIR / "stage6a_company_names.log"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation expansion (applied before comparison)
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "intl": "international",
    "natl": "national",
    "ctr": "center",
    "dept": "department",
    "svc": "services",
    "svcs": "services",
    "mgmt": "management",
    "mfg": "manufacturing",
    "engr": "engineering",
    "engrg": "engineering",
    "sys": "systems",
    "assoc": "associates",
    "govt": "government",
    "univ": "university",
}

# Pre-compile abbreviation pattern: match whole words only
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in ABBREVIATIONS) + r')\b'
)

# ---------------------------------------------------------------------------
# Generic business words — used to guard against false merges
# ---------------------------------------------------------------------------
GENERIC_WORDS = {
    "health", "care", "solutions", "services", "group", "systems",
    "consulting", "staffing", "medical", "network", "digital",
    "financial", "management", "energy", "national", "american",
    "general", "united", "first", "professional", "advanced",
    "technology", "technologies", "tech", "global", "partners",
    "associates", "international", "resources", "insurance",
}

# ---------------------------------------------------------------------------
# Trailing/leading punctuation cleanup
# ---------------------------------------------------------------------------
TRAILING_PUNCT = re.compile(r"[,.\-\s]+$")
LEADING_PUNCT = re.compile(r"^[,.\-\s]+")

# ---------------------------------------------------------------------------
# Known aliases: manually curated merges for tricky cases
# Each key is the canonical name; values are known variants.
# ---------------------------------------------------------------------------
KNOWN_ALIASES = {
    "Amazon": [
        "amazon", "amazon.com", "amazon.com inc", "amazon.com inc.",
        "amazon web services", "aws", "amazon.com services",
    ],
    "Google": [
        "google", "alphabet", "alphabet inc", "alphabet inc.",
        "google llc", "google cloud", "google deepmind",
    ],
    "Meta": [
        "meta", "meta platforms", "meta platforms inc", "facebook",
        "facebook inc",
    ],
    "Microsoft": [
        "microsoft", "microsoft corporation", "microsoft corp",
    ],
    "JPMorgan Chase & Co.": [
        "jpmorgan chase", "jpmorganchase", "jpmorgan chase & co",
        "jpmorgan chase & co.", "jpmorgan chase and co", "jpmorganchaseco",
        "jp morgan chase", "jp morgan", "jpmc", "j.p. morgan",
        "j p morgan",
    ],
    "Apple": [
        "apple", "apple inc", "apple inc.",
    ],
    "IBM": [
        "ibm", "international business machines",
        "international business machines corp",
        "international business machines corporation",
    ],
    "Deloitte": [
        "deloitte consulting", "deloitte llp", "deloitte touche tohmatsu",
        "deloitte & touche",
    ],
    "PricewaterhouseCoopers": [
        "pwc", "pricewaterhousecoopers llp", "pricewaterhousecoopers",
    ],
    "Ernst & Young": [
        "ey", "ernst & young llp", "ernst and young",
    ],
    "McKinsey & Company": [
        "mckinsey", "mckinsey and company", "mckinsey & co",
    ],
    "Accenture": [
        "accenture llp", "accenture federal services",
    ],
    "Bank of America": [
        "bank of america corp", "bank of america corporation",
        "bofa securities",
    ],
    "Wells Fargo": [
        "wells fargo & company", "wells fargo bank",
    ],
    "Goldman Sachs": [
        "goldman sachs & co", "the goldman sachs group",
        "goldman sachs group",
    ],
    "Morgan Stanley": [
        "morgan stanley & co",
    ],
    "Citigroup": [
        "citibank", "citi",
    ],
}

# Names that should NOT be merged with similarly-named companies.
# Maps a normalized form to itself (prevents absorption into alias groups).
# These are checked during fuzzy matching to block false merges.
PROTECTED_NAMES = {
    "appleone", "applebees", "apple bank", "applegate",
    "apple roofing", "apple federal credit union",
    "apple hospitality", "apple leisure group",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def normalize_for_comparison(name: str) -> str:
    """Normalize a company name for grouping/comparison.

    Steps:
      1. Lowercase + strip
      2. Apply cleanco to strip legal suffixes (Inc, LLC, GmbH, etc.)
      3. Remove punctuation (keep & and -)
      4. Expand abbreviations
      5. Collapse whitespace
      6. Strip leading/trailing junk
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    s = name.strip()

    # Apply cleanco for legal suffix stripping
    s = cleanco_basename(s)

    # Lowercase after cleanco (cleanco is case-aware)
    s = s.lower().strip()

    # Replace common punctuation with space (keep & and -)
    s = re.sub(r"[''`]", "", s)           # smart quotes / apostrophes
    s = re.sub(r"[()[\]{}/\\]", " ", s)   # brackets
    s = re.sub(r"[.,;:!?\"#*]", " ", s)   # general punctuation

    # Normalize ampersand variants
    s = re.sub(r"\s+and\s+", " & ", s)

    # Expand abbreviations
    s = _ABBREV_PATTERN.sub(lambda m: ABBREVIATIONS[m.group(0)], s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Strip leading/trailing punctuation artifacts
    s = TRAILING_PUNCT.sub("", s)
    s = LEADING_PUNCT.sub("", s)

    return s.strip()


def blocking_key(normalized_name: str) -> str:
    """Generate a blocking key for grouping candidates.

    Uses first 4 characters (after stripping 'the ') to create coarse blocks.
    """
    s = normalized_name
    if s.startswith("the "):
        s = s[4:]
    return s[:4] if len(s) >= 4 else s


# ---------------------------------------------------------------------------
# Composite similarity scoring
# ---------------------------------------------------------------------------
def company_similarity(a: str, b: str) -> float:
    """Multi-metric company name similarity. Returns 0-100."""
    min_len = min(len(a), len(b))

    jw = JaroWinkler.similarity(a, b) * 100
    tsr = fuzz.token_sort_ratio(a, b)
    r = fuzz.ratio(a, b)

    # Weighted combination favoring Jaro-Winkler (prefix-biased)
    score = 0.4 * jw + 0.35 * tsr + 0.25 * r

    # Penalize short-name matches (< 5 chars)
    if min_len < 5:
        score *= 0.7

    return score


def get_threshold(name: str) -> float:
    """Return length-dependent similarity threshold."""
    n = len(name)
    if n >= 10:
        return 90.0
    elif n >= 5:
        return 93.0
    else:
        return 97.0


# Tokens that carry geographic or numeric identity — these differentiate
# branches/affiliates of the same parent organization.
_GEOGRAPHIC_INDICATOR_WORDS = {
    "of", "in", "at", "for", "the",
}

# Common organizational type words that appear in franchise/affiliate patterns
# like "University of X", "HCA Florida X Hospital", "Blue Cross ... of X"
_ORG_PREFIX_WORDS = {
    "university", "college", "hospital", "medical", "center", "school",
    "district", "community", "county", "city", "state", "church",
    "bank", "credit", "union", "association", "society", "foundation",
    "institute", "academy", "clinic",
}


def _is_near_plural(a: str, b: str) -> bool:
    """Check if two tokens differ only by a trailing 's' or 'es' (plural)."""
    if a == b:
        return True
    if a + "s" == b or b + "s" == a:
        return True
    if a + "es" == b or b + "es" == a:
        return True
    return False


def meaningful_overlap(a: str, b: str) -> bool:
    """Check if two normalized names should be allowed to merge.

    Returns True (allow merge) if the names appear to be variants of the same
    entity. Returns False (block merge) when:
    - The only shared tokens are generic business words, OR
    - The names follow a "shared-prefix + different-identifier" pattern
      (e.g., "University of Dayton" vs "University of Bath"), OR
    - One name is a subset of the other with an additional distinguishing token
      near an organizational word (e.g., "Northeastern State University" vs
      "Northeastern University")
    """
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    # If either name is a single token, skip the guard —
    # similarity score alone should decide.
    if len(a_tokens) <= 1 or len(b_tokens) <= 1:
        return True

    shared = a_tokens & b_tokens
    meaningful = shared - GENERIC_WORDS
    only_a = a_tokens - b_tokens
    only_b = b_tokens - a_tokens

    # Guard 1: No meaningful shared tokens at all
    if len(shared) > 0 and len(meaningful) == 0:
        return False

    # Early exit: if the only differences are plural forms, allow the merge.
    # "Community Health System" vs "Community Health Systems" should merge.
    if len(only_a) <= 1 and len(only_b) <= 1:
        if not only_a and not only_b:
            return True  # identical token sets
        if only_a and only_b:
            tok_a = next(iter(only_a))
            tok_b = next(iter(only_b))
            if _is_near_plural(tok_a, tok_b):
                return True
        elif only_a and not only_b:
            # b is a subset of a — extra token is just "s" variant?
            pass  # handled below
        elif only_b and not only_a:
            pass  # handled below

    has_org_prefix = bool(shared & _ORG_PREFIX_WORDS)

    # Guard 2: Location/branch differentiation pattern.
    # If names share tokens but differ in 1-2 tokens that look like
    # identifiers (location, number), block the merge.
    # "HCA Florida JFK Hospital" != "HCA Florida West Hospital"
    # "Trinity University" != "Trine University"
    if len(shared) >= 1 and len(only_a) >= 1 and len(only_b) >= 1:
        # Check if differing tokens contain a number (district/school pattern)
        diff_has_number = any(any(c.isdigit() for c in t) for t in only_a | only_b)

        if has_org_prefix and diff_has_number:
            return False

        # If shared portion includes org prefix AND both sides have differing
        # non-trivial tokens, block the merge. The differing tokens are the
        # distinguishing identity (location, specialty, etc.)
        if has_org_prefix and len(only_a) >= 1 and len(only_b) >= 1:
            # All differing tokens are short enough to be identifiers
            diff_tokens_short = all(len(t) <= 12 for t in only_a | only_b)
            if diff_tokens_short:
                return False

        # Catch: both sides have exactly 1 differing token of similar length
        # and the shared fraction is high — typically location variants
        if (len(only_a) == 1 and len(only_b) == 1 and len(shared) >= 3):
            diff_a = next(iter(only_a))
            diff_b = next(iter(only_b))
            # If the differing tokens are both >= 3 chars and not very similar,
            # they're probably different identifiers
            if (len(diff_a) >= 3 and len(diff_b) >= 3 and
                    fuzz.ratio(diff_a, diff_b) < 75):
                return False

    # Guard 3: Subset with distinguishing extra token.
    # "Northeastern State University" vs "Northeastern University" —
    # one is a strict subset of the other, but the extra token ("State")
    # is a meaningful differentiator when org prefix words are present.
    if has_org_prefix:
        if (len(only_a) == 0 and len(only_b) >= 1) or \
           (len(only_b) == 0 and len(only_a) >= 1):
            extra = only_a if only_a else only_b
            # If any extra token is a meaningful differentiator (not just
            # a generic word or trivial suffix), block the merge
            meaningful_extra = extra - GENERIC_WORDS - {"&", "of", "the", "in", "at", "for"}
            if meaningful_extra:
                # But allow if extra tokens are very short (likely typo/abbreviation)
                if any(len(t) >= 3 for t in meaningful_extra):
                    return False

    return True


# ---------------------------------------------------------------------------
# Build canonical lookup
# ---------------------------------------------------------------------------
def build_alias_index(aliases: dict) -> dict:
    """Build a reverse index: normalized_variant -> canonical_name."""
    index = {}
    for canonical, variants in aliases.items():
        canon_norm = normalize_for_comparison(canonical)
        index[canon_norm] = canonical
        for v in variants:
            v_norm = normalize_for_comparison(v)
            if v_norm:
                index[v_norm] = canonical
    return index


def is_protected(normalized_name: str) -> bool:
    """Check if a normalized name is in the protected list (should not be merged)."""
    return normalized_name in PROTECTED_NAMES


def build_lookup(names_series: pd.Series) -> pd.DataFrame:
    """Build the company name normalization lookup table.

    Strategy:
      1. Count frequency of each unique original name.
      2. Normalize all names (cleanco + abbreviation expansion) and group exact matches.
         Pick the most frequent original spelling as the canonical.
      3. Apply known aliases as overrides.
      4. Build blocking groups by 4-char prefix.
      5. Within each block, composite fuzzy-match with guards.
      6. Build output lookup table.

    Returns a DataFrame with columns: company_name, company_name_normalized
    """
    t0 = time.time()

    # Step 0: Drop nulls/empty, count frequencies
    names_series = names_series.dropna()
    names_series = names_series[names_series.str.strip().str.len() > 0]
    freq = names_series.value_counts()
    unique_names = freq.index.tolist()
    log.info(f"Unique non-empty company names: {len(unique_names):,}")

    # Step 1: Normalize each unique name
    log.info("Normalizing names (cleanco + abbreviation expansion)...")
    norm_map = {}  # original -> normalized
    for name in unique_names:
        norm_map[name] = normalize_for_comparison(name)

    # Step 2: Group by normalized form; pick most-frequent original as canonical
    log.info("Grouping exact normalized matches...")
    norm_groups = defaultdict(list)  # normalized -> [(original, count), ...]
    for name in unique_names:
        nf = norm_map[name]
        if nf:
            norm_groups[nf].append((name, freq[name]))
        else:
            # Empty after normalization — keep original as-is
            norm_groups[name.lower().strip()].append((name, freq[name]))

    # For each normalized group, pick the most frequent original as canonical label
    canonical_labels = {}  # normalized -> best original
    canonical_counts = {}  # normalized -> total count across all originals in group
    for nf, members in norm_groups.items():
        members.sort(key=lambda x: x[1], reverse=True)
        canonical_labels[nf] = members[0][0]  # most frequent spelling
        canonical_counts[nf] = sum(c for _, c in members)

    log.info(f"After exact-normalization grouping: {len(norm_groups):,} groups "
             f"(from {len(unique_names):,} unique names)")

    # Step 3: Apply known aliases BEFORE fuzzy matching
    log.info("Applying known alias overrides...")
    alias_index = build_alias_index(KNOWN_ALIASES)
    alias_merges = 0

    # Map: normalized_form -> merged_canonical_normalized_form
    merge_target = {}  # nf -> target_nf

    for nf in list(norm_groups.keys()):
        if nf in alias_index:
            target_canonical = alias_index[nf]
            target_nf = normalize_for_comparison(target_canonical)
            if target_nf != nf:
                merge_target[nf] = target_nf
                alias_merges += 1

    # Execute alias merges
    for source_nf, target_nf in merge_target.items():
        if source_nf in norm_groups:
            # Ensure target exists
            if target_nf not in norm_groups:
                norm_groups[target_nf] = []
                canonical_labels[target_nf] = alias_index.get(
                    target_nf, canonical_labels.get(source_nf, "")
                )
                canonical_counts[target_nf] = 0

            # Transfer members
            norm_groups[target_nf].extend(norm_groups.pop(source_nf))
            canonical_counts[target_nf] += canonical_counts.pop(source_nf, 0)

            # Re-pick canonical label from KNOWN_ALIASES (use the dict key)
            for canon_key in KNOWN_ALIASES:
                if normalize_for_comparison(canon_key) == target_nf:
                    canonical_labels[target_nf] = canon_key
                    break

    log.info(f"Alias merges applied: {alias_merges}")
    log.info(f"Groups after aliases: {len(norm_groups):,}")

    # Step 4: Fuzzy matching within blocking groups
    log.info("Building blocking groups for fuzzy matching (4-char prefix)...")
    blocks = defaultdict(list)  # block_key -> [normalized_form, ...]
    for nf in norm_groups:
        bk = blocking_key(nf)
        blocks[bk].append(nf)

    block_sizes = [len(v) for v in blocks.values()]
    log.info(f"Blocks: {len(blocks):,}, "
             f"max size: {max(block_sizes):,}, "
             f"median size: {int(np.median(block_sizes))}, "
             f"blocks > 100: {sum(1 for s in block_sizes if s > 100):,}")

    log.info("Running composite fuzzy matching within blocks...")
    fuzzy_merges = 0
    fuzzy_merge_log = []  # track merges for quality review
    fuzzy_blocked_log = []  # track blocked merges for review

    # Process blocks
    for bk, block_nfs in blocks.items():
        if len(block_nfs) <= 1:
            continue

        # Sort by total count descending — frequent names stay canonical
        block_nfs.sort(key=lambda nf: canonical_counts.get(nf, 0), reverse=True)

        # Use a list of surviving canonicals in this block
        surviving = []
        pending_merges = []  # (source_nf, target_nf, score)

        for nf in block_nfs:
            if not surviving:
                surviving.append(nf)
                continue

            # Skip protected names — they should never be absorbed
            if is_protected(nf):
                surviving.append(nf)
                continue

            # Get length-dependent threshold for this name
            threshold = get_threshold(nf)

            # Compare against all surviving canonicals in this block
            best_score = 0.0
            best_target = None
            for candidate in surviving:
                # Skip if candidate is protected
                if is_protected(candidate):
                    continue

                score = company_similarity(nf, candidate)
                if score > best_score:
                    best_score = score
                    best_target = candidate

            if best_score >= threshold and best_target is not None:
                # Generic word guard: check for meaningful token overlap
                if meaningful_overlap(nf, best_target):
                    pending_merges.append((nf, best_target, best_score))
                else:
                    # Blocked by generic word guard
                    if len(fuzzy_blocked_log) < 100:
                        fuzzy_blocked_log.append({
                            "source": canonical_labels.get(nf, nf),
                            "target": canonical_labels.get(best_target, best_target),
                            "score": best_score,
                            "reason": "generic-word-only overlap",
                        })
                    surviving.append(nf)
            else:
                surviving.append(nf)

        # Execute merges for this block
        for source_nf, target_nf, score in pending_merges:
            if source_nf in norm_groups and target_nf in norm_groups:
                # Log the merge for review
                if len(fuzzy_merge_log) < 500:
                    fuzzy_merge_log.append({
                        "source": canonical_labels.get(source_nf, source_nf),
                        "target": canonical_labels.get(target_nf, target_nf),
                        "score": score,
                        "source_count": canonical_counts.get(source_nf, 0),
                        "target_count": canonical_counts.get(target_nf, 0),
                    })

                norm_groups[target_nf].extend(norm_groups.pop(source_nf))
                canonical_counts[target_nf] += canonical_counts.pop(source_nf, 0)
                fuzzy_merges += 1

    log.info(f"Fuzzy merges: {fuzzy_merges:,}")
    log.info(f"Fuzzy merges blocked by guards: {len(fuzzy_blocked_log):,}")
    log.info(f"Final canonical groups: {len(norm_groups):,}")

    # Step 5: Build the output lookup table
    log.info("Building lookup table...")
    rows = []
    for nf, members in norm_groups.items():
        canon = canonical_labels.get(nf, nf)
        for original, count in members:
            rows.append({"company_name": original, "company_name_normalized": canon})

    # Also handle names that had empty normalization (edge cases)
    mapped_originals = {r["company_name"] for r in rows}
    for name in unique_names:
        if name not in mapped_originals:
            rows.append({"company_name": name, "company_name_normalized": name})

    lookup_df = pd.DataFrame(rows)

    elapsed = time.time() - t0
    log.info(f"Lookup table built in {elapsed:.1f}s: {len(lookup_df):,} rows")

    # Step 6: Log statistics
    log_statistics(lookup_df, norm_groups, canonical_labels, canonical_counts,
                   fuzzy_merge_log, fuzzy_blocked_log, unique_names)

    return lookup_df


# ---------------------------------------------------------------------------
# Statistics and quality logging
# ---------------------------------------------------------------------------
def log_statistics(
    lookup_df: pd.DataFrame,
    norm_groups: dict,
    canonical_labels: dict,
    canonical_counts: dict,
    fuzzy_merge_log: list,
    fuzzy_blocked_log: list,
    unique_names: list,
):
    """Log summary statistics, top groups, and sample fuzzy matches."""
    n_unique = len(unique_names)
    n_canonical = len(norm_groups)
    compression = n_unique / n_canonical if n_canonical > 0 else 0

    log.info("")
    log.info("=" * 70)
    log.info("STAGE 6a: Company Name Standardization V2 -- Summary")
    log.info("=" * 70)
    log.info(f"Total unique company names:   {n_unique:>10,}")
    log.info(f"Total canonical groups:       {n_canonical:>10,}")
    log.info(f"Compression ratio:            {compression:>10.2f}x")
    log.info(f"Names merged:                 {n_unique - n_canonical:>10,}")

    # Top 20 largest groups
    log.info("")
    log.info("-" * 70)
    log.info("Top 20 largest canonical groups (by total posting count)")
    log.info("-" * 70)

    # Sort groups by total count
    sorted_groups = sorted(
        norm_groups.items(),
        key=lambda kv: canonical_counts.get(kv[0], 0),
        reverse=True,
    )

    for rank, (nf, members) in enumerate(sorted_groups[:20], 1):
        canon = canonical_labels.get(nf, nf)
        total = canonical_counts.get(nf, 0)
        # Show up to 10 variants
        variants = sorted(members, key=lambda x: x[1], reverse=True)
        variant_strs = [f"{name} ({count:,})" for name, count in variants[:10]]
        if len(variants) > 10:
            variant_strs.append(f"... +{len(variants) - 10} more")
        log.info(f"\n  #{rank} {canon} (total: {total:,}, variants: {len(members)})")
        for vs in variant_strs:
            log.info(f"      {vs}")

    # Sample fuzzy matches for quality review
    log.info("")
    log.info("-" * 70)
    log.info("Sample fuzzy matches (for quality review)")
    log.info("-" * 70)

    if fuzzy_merge_log:
        merge_df = pd.DataFrame(fuzzy_merge_log)
        merge_df = merge_df.sort_values("score")

        # Near threshold (potentially risky)
        near_threshold = merge_df[merge_df["score"] <= 92].head(25)
        if len(near_threshold) > 0:
            log.info("\n  Near-threshold matches (lowest scores) -- REVIEW THESE:")
            for _, row in near_threshold.iterrows():
                log.info(f"    [{row['score']:5.1f}] '{row['source']}' ({row['source_count']:,}) "
                         f"-> '{row['target']}' ({row['target_count']:,})")

        # High confidence
        high_conf = merge_df[merge_df["score"] >= 97].head(20)
        if len(high_conf) > 0:
            log.info("\n  High-confidence matches (score >= 97):")
            for _, row in high_conf.iterrows():
                log.info(f"    [{row['score']:5.1f}] '{row['source']}' ({row['source_count']:,}) "
                         f"-> '{row['target']}' ({row['target_count']:,})")

        # Mid-range
        mid = merge_df[(merge_df["score"] > 92) & (merge_df["score"] < 97)].head(20)
        if len(mid) > 0:
            log.info("\n  Mid-range matches (score 92-97):")
            for _, row in mid.iterrows():
                log.info(f"    [{row['score']:5.1f}] '{row['source']}' ({row['source_count']:,}) "
                         f"-> '{row['target']}' ({row['target_count']:,})")
    else:
        log.info("  No fuzzy merges recorded.")

    # Blocked merges (generic word guard prevented these)
    log.info("")
    log.info("-" * 70)
    log.info("Blocked merges (generic-word guard prevented these)")
    log.info("-" * 70)

    if fuzzy_blocked_log:
        for entry in fuzzy_blocked_log[:30]:
            log.info(f"    [{entry['score']:5.1f}] '{entry['source']}' "
                     f"-> '{entry['target']}' -- {entry['reason']}")
    else:
        log.info("  No merges blocked by guards.")

    # Spot-check known companies
    log.info("")
    log.info("-" * 70)
    log.info("Spot-check: known company variants")
    log.info("-" * 70)

    spot_checks = [
        "JPMorgan Chase", "Amazon", "Google", "Microsoft",
        "Meta", "Deloitte", "Goldman Sachs", "Apple", "IBM",
        "Health",
    ]
    for company in spot_checks:
        matches = lookup_df[
            lookup_df["company_name_normalized"].str.contains(
                r'(?:^|\s)' + re.escape(company) + r'(?:\s|$)',
                case=False, na=False, regex=True,
            )
        ]
        if len(matches) > 0:
            top_norm = matches["company_name_normalized"].value_counts().head(5)
            for norm_name, cnt in top_norm.items():
                variants = matches[matches["company_name_normalized"] == norm_name]["company_name"].tolist()
                if len(variants) <= 5:
                    log.info(f"  {norm_name}: {variants}")
                else:
                    log.info(f"  {norm_name}: {variants[:5]} ... +{len(variants)-5} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("STAGE 6a: Company Name Standardization V2")
    log.info("=" * 70)
    log.info(f"Input:  {INPUT_PATH}")
    log.info(f"Output: {OUTPUT_PATH}")

    # Load only company_name column
    log.info("Loading company_name column from parquet...")
    t0 = time.time()
    table = pq.read_table(str(INPUT_PATH), columns=["company_name"])
    names = table.column("company_name").to_pandas()
    log.info(f"Loaded {len(names):,} rows in {time.time() - t0:.1f}s")
    log.info(f"Null/empty: {names.isna().sum():,}")

    # Build lookup
    lookup_df = build_lookup(names)

    # Save
    log.info(f"\nSaving lookup table to {OUTPUT_PATH}...")
    lookup_df.to_parquet(str(OUTPUT_PATH), index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    log.info(f"Saved: {len(lookup_df):,} rows, {size_mb:.1f} MB")

    # Verify: every original name should appear exactly once
    assert lookup_df["company_name"].nunique() == len(lookup_df), \
        "Duplicate original names in lookup!"
    log.info("Integrity check passed: no duplicate original names.")

    log.info("\nStage 6a V2 complete.")


if __name__ == "__main__":
    main()
