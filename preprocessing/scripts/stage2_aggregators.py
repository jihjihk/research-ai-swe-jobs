#!/usr/bin/env python3
"""
Stage 2: Aggregator / Staffing Company Handling

Identifies and flags aggregator/staffing company postings.
Attempts to extract real employer from description.

Input:  intermediate/stage1_unified.parquet
Output: intermediate/stage2_aggregators.parquet (adds is_aggregator, real_employer columns)
"""

import re
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "preprocessing" / "intermediate"
LOG_DIR = PROJECT_ROOT / "preprocessing" / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stage2_aggregators.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregator/Staffing company list
# ---------------------------------------------------------------------------
AGGREGATORS = {
    # Job boards / aggregators
    "lensa", "jobs via dice", "dice", "jobot", "cybercoders",
    "hired", "ziprecruiter", "talentally", "ladders",
    "efinancialcareers", "ventureloop", "builtin", "jobs for humanity",
    "jobs via efinancialcareers", "clearancejobs", "clickjobs.io",
    # Staffing companies
    "actalent", "randstad", "robert half", "teksystems",
    "kforce", "insight global", "motion recruitment", "harnham",
    "apex systems", "modis", "experis", "manpowergroup",
    "kelly services", "adecco", "hays", "michael page",
    "collabera", "infosys bpm", "wipro", "tata consultancy",
    "tata consultancy services", "tata consultancy services (tcs)",
    "cognizant", "capgemini", "revature", "smoothstack",
    "recruiting from scratch", "aston carter", "solomon page",
    "creative circle", "applicantz", "aditi consulting",
    "ask consulting", "inspyr solutions", "net2source inc.",
    "net2source inc", "net2source",
    # Data annotation / crowdwork (suspected non-traditional SWE)
    "dataannotation", "data annotation", "appen", "scale ai",
    "remotasks", "clickworker",
}

# Patterns for fuzzy matching aggregator names
AGGREGATOR_PATTERNS = [
    re.compile(r'(?i)\blensa\b'),
    re.compile(r'(?i)\bjobs?\s+via\b'),
    re.compile(r'(?i)\bjobs?\s*via\s*dice\b'),
    re.compile(r'(?i)\bjobs?\s*via\s*efinancialcareers\b'),
    re.compile(r'(?i)\b(dice|dice\.com)\b'),
    re.compile(r'(?i)\bjobot\b'),
    re.compile(r'(?i)\bcybercoders\b'),
    re.compile(r'(?i)\bclickjobs(?:\.io)?\b'),
    re.compile(r'(?i)\bclearancejobs\b'),
    re.compile(r'(?i)\bziprecruiter\b'),
    re.compile(r'(?i)\bjobs?\s+for\s+humanity\b'),
    re.compile(r'(?i)\bactalent\b'),
    re.compile(r'(?i)\baston\s*carter\b'),
    re.compile(r'(?i)\brandstad\b'),
    re.compile(r'(?i)\brobert\s*half\b'),
    re.compile(r'(?i)\bteksystems\b'),
    re.compile(r'(?i)\bkforce\b'),
    re.compile(r'(?i)\binsight\s*global\b'),
    re.compile(r'(?i)\bapex\s*systems\b'),
    re.compile(r'(?i)\bsolomon\s+page\b'),
    re.compile(r'(?i)\bcreative\s+circle\b'),
    re.compile(r'(?i)\brecruiting\s+from\s+scratch\b'),
    re.compile(r'(?i)\bdata\s*annotation\b'),
    re.compile(r'(?i)\brevature\b'),
    re.compile(r'(?i)\bsmoothstack\b'),
    re.compile(r'(?i)\btalentally\b'),
    re.compile(r'(?i)\bharnham\b'),
    re.compile(r'(?i)\bmotion\s*recruitment\b'),
    re.compile(r'(?i)\baditi\s+consulting\b'),
    re.compile(r'(?i)\bask\s+consulting\b'),
    re.compile(r'(?i)\binspyr\s+solutions\b'),
    re.compile(r'(?i)\bnet2source(?:\s+inc\.?)?\b'),
    re.compile(r'(?i)\btata\s+consultancy(?:\s+services)?(?:\s+\(tcs\))?\b'),
    re.compile(r'(?i)\bapplicantz\b'),
]


def is_aggregator(company_name: str) -> bool:
    """Check if a company name matches the aggregator list."""
    if not isinstance(company_name, str):
        return False
    name_lower = company_name.strip().lower()
    # Exact match
    if name_lower in AGGREGATORS:
        return True
    # Pattern match
    for pat in AGGREGATOR_PATTERNS:
        if pat.search(company_name):
            return True
    return False


# Real employer extraction patterns
REAL_EMPLOYER_PATTERNS = [
    # "Our client, [Company Name], is..."
    re.compile(r'(?i)our\s+client[,:]?\s*([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+is|\s+has|\s+seeks|\s+needs|\.)'),
    # "on behalf of [Company Name]"
    re.compile(r'(?i)on\s+behalf\s+of\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+is|\s+we|\.)'),
    # "partnered with [Company Name]"
    re.compile(r'(?i)partner(?:ed|ing)\s+with\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:,|\s+to|\s+for|\.)'),
    # "working with [Company Name]"
    re.compile(r'(?i)working\s+with\s+([A-Z][A-Za-z0-9\s&.,-]+?)\s+(?:to\s+find|who\s+is|looking)'),
    # Lensa: "Lensa is the leading... This job is also available..." then company name
    re.compile(r'(?i)(?:job|position|role)\s+(?:is\s+)?(?:with|at|for)\s+([A-Z][A-Za-z0-9\s&.,-]+?)(?:\.|,|\s+in\b|\s+located)'),
]

BAD_EMPLOYER_TERMS = re.compile(
    r'(?i)\b('
    r'client|clients|company|companies|organization|team|role|position|opportunity|'
    r'seeking|looking|hiring|available|access|extension|contract|'
    r'engineer|developer|manager|director|architect|scientist|analyst|'
    r'specialist|consultant|administrator|technician|recruiter|designer|'
    r'located|location|remote|hybrid|onsite|site|week|weeks|sector|industry|'
    r'ownership|detail\s+oriented|direct\s+client|integrators'
    r')\b'
)

ORG_SUFFIX_HINT = re.compile(
    r'(?i)\b('
    r'inc|llc|ltd|corp|corporation|company|co|group|systems|solutions|'
    r'technologies|technology|labs|lab|health|partners|holdings|bank|'
    r'university|international|networks|software|services'
    r')\.?\b'
)

LEADING_FRAGMENT = re.compile(r'(?i)^(in|at|for|with|to|on|of|from)\b')


def _clean_employer_candidate(candidate: str) -> str:
    cleaned = candidate.strip(" \t\n\r-.,:;()[]{}\"'")
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def _looks_like_organization_name(candidate: str) -> bool:
    if LEADING_FRAGMENT.match(candidate):
        return False
    tokens = candidate.split()
    if len(tokens) == 0 or len(tokens) > 6:
        return False
    uppercase_tokens = sum(1 for token in tokens if any(ch.isupper() for ch in token))
    if ORG_SUFFIX_HINT.search(candidate):
        return True
    if len(tokens) == 1:
        return uppercase_tokens == 1 and tokens[0].isalpha()
    return uppercase_tokens >= 2


def extract_real_employer(description: str, company_name: str) -> str | None:
    """Try to extract the real employer from an aggregator posting description."""
    if not isinstance(description, str) or len(description) < 50:
        return None

    for pattern in REAL_EMPLOYER_PATTERNS:
        match = pattern.search(description[:2000])  # Only check first 2000 chars
        if match:
            employer = _clean_employer_candidate(match.group(1))
            # Precision-first validation: reject obvious fragments and role phrases.
            if (
                2 < len(employer) < 80
                and any(ch.isupper() for ch in employer)
                and _looks_like_organization_name(employer)
                and not BAD_EMPLOYER_TERMS.search(employer)
                and not is_aggregator(employer)
                and employer.lower() != company_name.lower()
            ):
                return employer
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_stage2():
    log.info("=" * 60)
    log.info("STAGE 2: Aggregator / Staffing Company Handling")
    log.info("=" * 60)

    input_path = INTERMEDIATE_DIR / "stage1_unified.parquet"
    log.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    log.info(f"  Loaded: {len(df):,} rows")

    # --- Flag aggregators ---
    log.info("Flagging aggregator companies...")
    df["is_aggregator"] = df["company_name"].apply(is_aggregator)
    agg_count = df["is_aggregator"].sum()
    log.info(f"  Aggregator postings: {agg_count:,} ({agg_count/len(df):.1%})")

    # Breakdown by source
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        agg_sub = sub["is_aggregator"].sum()
        log.info(f"    {src}: {agg_sub:,} / {len(sub):,} ({agg_sub/len(sub):.1%})")

    # Top aggregators
    agg_df = df[df["is_aggregator"]]
    log.info("\n  Top aggregator companies:")
    for company, count in agg_df["company_name"].value_counts().head(20).items():
        log.info(f"    {company}: {count:,}")

    # --- Extract real employers ---
    log.info("\nExtracting real employers from aggregator descriptions...")
    agg_mask = df["is_aggregator"]
    df["real_employer"] = None
    df.loc[agg_mask, "real_employer"] = df.loc[agg_mask].apply(
        lambda r: extract_real_employer(r["description"], r["company_name"]),
        axis=1,
    )
    extracted = df.loc[agg_mask, "real_employer"].notna().sum()
    log.info(f"  Real employer extracted: {extracted:,} / {agg_count:,} ({extracted/max(agg_count,1):.1%})")

    # Sample of extractions
    extracted_df = df.loc[agg_mask & df["real_employer"].notna(),
                          ["company_name", "real_employer", "title"]].head(20)
    log.info("\n  Sample real employer extractions:")
    for _, row in extracted_df.iterrows():
        log.info(f"    {row['company_name']} -> {row['real_employer']} ({row['title'][:50]})")

    # DataAnnotation remains a flagged special case for sensitivity analysis.
    da_count = df["company_name"].str.contains(r'(?i)data\s*annotation', na=False).sum()
    log.info(f"\n  DataAnnotation postings flagged: {da_count:,}")

    # --- Save ---
    output_path = INTERMEDIATE_DIR / "stage2_aggregators.parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    df = run_stage2()
    print(f"\nDone: {len(df):,} rows, {df['is_aggregator'].sum():,} aggregators flagged")
