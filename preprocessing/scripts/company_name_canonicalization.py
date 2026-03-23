#!/usr/bin/env python3
"""
Utilities for canonicalizing effective company names in Stage 4.

The main pipeline no longer treats company-name normalization as a standalone
numbered stage. Stage 4 builds a lookup from `company_name_effective` and uses
that canonical label for posting-level deduplication.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from cleanco import basename as cleanco_basename
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


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

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ABBREVIATIONS) + r")\b"
)

GENERIC_WORDS = {
    "health", "care", "solutions", "services", "group", "systems",
    "consulting", "staffing", "medical", "network", "digital",
    "financial", "management", "energy", "national", "american",
    "general", "united", "first", "professional", "advanced",
    "technology", "technologies", "tech", "global", "partners",
    "associates", "international", "resources", "insurance",
}

TRAILING_PUNCT = re.compile(r"[,.\-\s]+$")
LEADING_PUNCT = re.compile(r"^[,.\-\s]+")

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

PROTECTED_NAMES = {
    "appleone", "applebees", "apple bank", "applegate",
    "apple roofing", "apple federal credit union",
    "apple hospitality", "apple leisure group",
}

_ORG_PREFIX_WORDS = {
    "university", "college", "hospital", "medical", "center", "school",
    "district", "community", "county", "city", "state", "church",
    "bank", "credit", "union", "association", "society", "foundation",
    "institute", "academy", "clinic",
}


def normalize_for_comparison(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""

    normalized = cleanco_basename(name.strip())
    normalized = normalized.lower().strip()
    normalized = re.sub(r"['`]", "", normalized)
    normalized = re.sub(r"[()[\]{}/\\]", " ", normalized)
    normalized = re.sub(r"[.,;:!?\"#*]", " ", normalized)
    normalized = re.sub(r"\s+and\s+", " & ", normalized)
    normalized = _ABBREV_PATTERN.sub(lambda match: ABBREVIATIONS[match.group(0)], normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = TRAILING_PUNCT.sub("", normalized)
    normalized = LEADING_PUNCT.sub("", normalized)
    return normalized.strip()


def blocking_key(normalized_name: str) -> str:
    value = normalized_name[4:] if normalized_name.startswith("the ") else normalized_name
    return value[:4] if len(value) >= 4 else value


def company_similarity(a: str, b: str) -> float:
    min_len = min(len(a), len(b))
    jw = JaroWinkler.similarity(a, b) * 100
    tsr = fuzz.token_sort_ratio(a, b)
    ratio = fuzz.ratio(a, b)
    score = 0.4 * jw + 0.35 * tsr + 0.25 * ratio
    if min_len < 5:
        score *= 0.7
    return score


def get_threshold(name: str) -> float:
    n = len(name)
    if n >= 10:
        return 90.0
    if n >= 5:
        return 93.0
    return 97.0


def _is_near_plural(a: str, b: str) -> bool:
    if a == b:
        return True
    if a + "s" == b or b + "s" == a:
        return True
    if a + "es" == b or b + "es" == a:
        return True
    return False


def meaningful_overlap(a: str, b: str) -> bool:
    a_tokens = set(a.split())
    b_tokens = set(b.split())

    shared = a_tokens & b_tokens
    meaningful = shared - GENERIC_WORDS
    only_a = a_tokens - b_tokens
    only_b = b_tokens - a_tokens

    if len(a_tokens) == 1 and len(b_tokens) == 1:
        return a == b
    if len(a_tokens) == 1 or len(b_tokens) == 1:
        return False
    if len(shared) <= 1 and min(len(a_tokens), len(b_tokens)) <= 2:
        return False
    if shared and not meaningful:
        return False

    if len(only_a) <= 1 and len(only_b) <= 1:
        if not only_a and not only_b:
            return True
        if only_a and only_b:
            tok_a = next(iter(only_a))
            tok_b = next(iter(only_b))
            if _is_near_plural(tok_a, tok_b):
                return True

    has_org_prefix = bool(shared & _ORG_PREFIX_WORDS)

    if shared and only_a and only_b:
        diff_has_number = any(any(ch.isdigit() for ch in token) for token in only_a | only_b)
        if has_org_prefix and diff_has_number:
            return False

        if has_org_prefix:
            diff_tokens_short = all(len(token) <= 12 for token in only_a | only_b)
            if diff_tokens_short:
                return False

        if len(only_a) == 1 and len(only_b) == 1 and len(shared) >= 3:
            diff_a = next(iter(only_a))
            diff_b = next(iter(only_b))
            if len(diff_a) >= 3 and len(diff_b) >= 3 and fuzz.ratio(diff_a, diff_b) < 75:
                return False

        if len(only_a) == 1 and len(only_b) == 1 and len(shared) <= 2:
            diff_a = next(iter(only_a))
            diff_b = next(iter(only_b))
            if len(diff_a) >= 4 and len(diff_b) >= 4 and fuzz.ratio(diff_a, diff_b) < 85:
                return False

    if has_org_prefix:
        if (not only_a and only_b) or (not only_b and only_a):
            extra = only_a if only_a else only_b
            meaningful_extra = extra - GENERIC_WORDS - {"&", "of", "the", "in", "at", "for"}
            if meaningful_extra and any(len(token) >= 3 for token in meaningful_extra):
                return False

    return True


def build_alias_index(aliases: dict[str, list[str]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, variants in aliases.items():
        index[normalize_for_comparison(canonical)] = canonical
        for variant in variants:
            normalized_variant = normalize_for_comparison(variant)
            if normalized_variant:
                index[normalized_variant] = canonical
    return index


def is_protected(normalized_name: str) -> bool:
    return normalized_name in PROTECTED_NAMES


def build_company_name_lookup(
    names_series: pd.Series,
    source_column: str = "company_name_effective",
) -> pd.DataFrame:
    names_series = names_series.dropna().astype(str)
    names_series = names_series[names_series.str.strip().str.len() > 0]

    if names_series.empty:
        return pd.DataFrame(
            columns=[source_column, "company_name_canonical", "company_name_canonical_method"]
        )

    freq = names_series.value_counts()
    unique_names = freq.index.tolist()
    norm_map = {name: normalize_for_comparison(name) for name in unique_names}

    norm_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    member_methods: dict[str, str] = {}

    for name in unique_names:
        normalized = norm_map[name]
        key = normalized if normalized else name.lower().strip()
        norm_groups[key].append((name, int(freq[name])))

    canonical_labels: dict[str, str] = {}
    canonical_counts: dict[str, int] = {}

    for normalized, members in norm_groups.items():
        members.sort(key=lambda item: item[1], reverse=True)
        canonical_labels[normalized] = members[0][0]
        canonical_counts[normalized] = sum(count for _, count in members)
        if len(members) == 1:
            member_methods[members[0][0]] = "passthrough"
            continue
        for original, _ in members:
            member_methods[original] = (
                "passthrough" if original == canonical_labels[normalized] else "exact_normalized"
            )

    alias_index = build_alias_index(KNOWN_ALIASES)
    merge_target: dict[str, str] = {}
    for normalized in list(norm_groups.keys()):
        if normalized in alias_index:
            target_canonical = alias_index[normalized]
            target_normalized = normalize_for_comparison(target_canonical)
            if target_normalized != normalized:
                merge_target[normalized] = target_normalized

    for source_normalized, target_normalized in merge_target.items():
        if source_normalized not in norm_groups:
            continue

        source_members = list(norm_groups[source_normalized])
        if target_normalized not in norm_groups:
            norm_groups[target_normalized] = []
            canonical_labels[target_normalized] = alias_index.get(
                target_normalized,
                canonical_labels.get(source_normalized, ""),
            )
            canonical_counts[target_normalized] = 0

        for original, _ in source_members:
            member_methods[original] = "alias"

        norm_groups[target_normalized].extend(norm_groups.pop(source_normalized))
        canonical_counts[target_normalized] += canonical_counts.pop(source_normalized, 0)

        for canonical_name in KNOWN_ALIASES:
            if normalize_for_comparison(canonical_name) == target_normalized:
                canonical_labels[target_normalized] = canonical_name
                break

    blocks: dict[str, list[str]] = defaultdict(list)
    for normalized in norm_groups:
        blocks[blocking_key(normalized)].append(normalized)

    for block_normalized_forms in blocks.values():
        if len(block_normalized_forms) <= 1:
            continue

        block_normalized_forms.sort(
            key=lambda normalized: canonical_counts.get(normalized, 0),
            reverse=True,
        )

        surviving: list[str] = []
        pending_merges: list[tuple[str, str]] = []

        for normalized in block_normalized_forms:
            if not surviving:
                surviving.append(normalized)
                continue

            if is_protected(normalized):
                surviving.append(normalized)
                continue

            threshold = get_threshold(normalized)
            best_score = 0.0
            best_target = None

            for candidate in surviving:
                if is_protected(candidate):
                    continue
                score = company_similarity(normalized, candidate)
                if score > best_score:
                    best_score = score
                    best_target = candidate

            if best_target is None or best_score < threshold:
                surviving.append(normalized)
                continue

            if meaningful_overlap(normalized, best_target):
                pending_merges.append((normalized, best_target))
            else:
                surviving.append(normalized)

        for source_normalized, target_normalized in pending_merges:
            if source_normalized not in norm_groups or target_normalized not in norm_groups:
                continue

            source_members = list(norm_groups[source_normalized])
            for original, _ in source_members:
                member_methods[original] = "fuzzy"

            norm_groups[target_normalized].extend(norm_groups.pop(source_normalized))
            canonical_counts[target_normalized] += canonical_counts.pop(source_normalized, 0)

    rows: list[dict[str, str]] = []
    for normalized, members in norm_groups.items():
        canonical = canonical_labels.get(normalized, normalized)
        for original, _ in members:
            rows.append(
                {
                    source_column: original,
                    "company_name_canonical": canonical,
                    "company_name_canonical_method": member_methods.get(original, "passthrough"),
                }
            )

    lookup_df = pd.DataFrame(rows)
    lookup_df = lookup_df.drop_duplicates(subset=[source_column], keep="first").reset_index(drop=True)
    return lookup_df.sort_values(source_column, kind="stable").reset_index(drop=True)
