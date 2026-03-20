#!/usr/bin/env python3
"""
Stage 2: Aggregator / Staffing Company Handling

Identifies and flags aggregator/staffing company postings.
Strips aggregator-specific boilerplate where possible.
Attempts to extract real employer from description.

Input:  intermediate/stage1_unified.parquet
Output: intermediate/stage2_aggregators.parquet (adds is_aggregator, real_employer columns)
"""

import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np

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
    "efinancialcareers", "ventureloop", "builtin",
    # Staffing companies
    "actalent", "randstad", "robert half", "teksystems",
    "kforce", "insight global", "motion recruitment", "harnham",
    "apex systems", "modis", "experis", "manpowergroup",
    "kelly services", "adecco", "hays", "michael page",
    "collabera", "infosys bpm", "wipro", "tata consultancy",
    "cognizant", "capgemini", "revature", "smoothstack",
    # Data annotation / crowdwork (suspected non-traditional SWE)
    "dataannotation", "data annotation", "appen", "scale ai",
    "remotasks", "clickworker",
}

# Patterns for fuzzy matching aggregator names
AGGREGATOR_PATTERNS = [
    re.compile(r'(?i)\blensa\b'),
    re.compile(r'(?i)\bjobs?\s*via\s*dice\b'),
    re.compile(r'(?i)\b(dice|dice\.com)\b'),
    re.compile(r'(?i)\bjobot\b'),
    re.compile(r'(?i)\bcybercoders\b'),
    re.compile(r'(?i)\bziprecruiter\b'),
    re.compile(r'(?i)\bactalent\b'),
    re.compile(r'(?i)\brandstad\b'),
    re.compile(r'(?i)\brobert\s*half\b'),
    re.compile(r'(?i)\bteksystems\b'),
    re.compile(r'(?i)\bkforce\b'),
    re.compile(r'(?i)\binsight\s*global\b'),
    re.compile(r'(?i)\bapex\s*systems\b'),
    re.compile(r'(?i)\bdata\s*annotation\b'),
    re.compile(r'(?i)\brevature\b'),
    re.compile(r'(?i)\bsmoothstack\b'),
    re.compile(r'(?i)\btalentally\b'),
    re.compile(r'(?i)\bharnham\b'),
    re.compile(r'(?i)\bmotion\s*recruitment\b'),
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


# ---------------------------------------------------------------------------
# Aggregator boilerplate strippers
# ---------------------------------------------------------------------------

# Lensa prepends self-description before actual job content
LENSA_BOILERPLATE = re.compile(
    r'(?is)^.*?lensa\s+(is|helps|partners|connects).*?(?=\n\n|\r\n\r\n|about\s+the\s+(role|position|job)|job\s+description|responsibilities|overview)',
    re.DOTALL
)

# Dice wraps in templates
DICE_BOILERPLATE = re.compile(
    r'(?is)^.*?(dice\s+(is|connects|helps)|posted\s+by\s+dice).*?(?=\n\n|\r\n\r\n|job\s+(summary|description)|about\s+the\s+role)',
    re.DOTALL
)

# Generic staffing boilerplate at the end
STAFFING_FOOTER = re.compile(
    r'(?is)(about\s+(actalent|randstad|robert\s+half|teksystems|kforce|insight\s+global|apex|harnham|revature|smoothstack).*?)$',
    re.DOTALL
)

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


def extract_real_employer(description: str, company_name: str) -> str | None:
    """Try to extract the real employer from an aggregator posting description."""
    if not isinstance(description, str) or len(description) < 50:
        return None

    for pattern in REAL_EMPLOYER_PATTERNS:
        match = pattern.search(description[:2000])  # Only check first 2000 chars
        if match:
            employer = match.group(1).strip()
            # Basic validation: not the aggregator itself, reasonable length
            if (len(employer) > 2 and len(employer) < 80
                    and not is_aggregator(employer)
                    and employer.lower() != company_name.lower()):
                return employer
    return None


def strip_aggregator_boilerplate(description: str, company_name: str) -> str:
    """Remove known aggregator boilerplate from descriptions."""
    if not isinstance(description, str):
        return description

    name_lower = (company_name or "").lower()

    if "lensa" in name_lower:
        description = LENSA_BOILERPLATE.sub("", description)
    elif "dice" in name_lower:
        description = DICE_BOILERPLATE.sub("", description)

    # Strip staffing company "About Us" footers
    description = STAFFING_FOOTER.sub("", description)

    return description.strip()


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
        swe_count = agg_df.loc[agg_df["company_name"] == company, "is_swe"].sum()
        log.info(f"    {company}: {count:,} total, {swe_count:,} SWE")

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

    # --- Strip aggregator boilerplate ---
    log.info("\nStripping aggregator boilerplate...")
    df["description_pre_agg_strip"] = df.loc[agg_mask, "description_length"].copy()
    df.loc[agg_mask, "description"] = df.loc[agg_mask].apply(
        lambda r: strip_aggregator_boilerplate(r["description"], r["company_name"]),
        axis=1,
    )
    # Update description length
    df["description_length"] = df["description"].str.len()

    # Report length changes for aggregator postings
    if agg_count > 0:
        pre_len = df.loc[agg_mask, "description_pre_agg_strip"]
        post_len = df.loc[agg_mask, "description_length"]
        # Only report where pre_len was set
        valid_mask = agg_mask & pre_len.notna()
        if valid_mask.sum() > 0:
            change = (post_len[valid_mask] - pre_len[valid_mask])
            log.info(f"  Boilerplate stripped: median change = {change.median():.0f} chars")

    # Drop helper column
    df.drop(columns=["description_pre_agg_strip"], inplace=True, errors="ignore")

    # --- SWE aggregator analysis ---
    swe_agg = df[df["is_swe"] & df["is_aggregator"]]
    swe_total = df["is_swe"].sum()
    log.info(f"\n  SWE postings from aggregators: {len(swe_agg):,} / {swe_total:,} ({len(swe_agg)/swe_total:.1%})")

    # DataAnnotation special analysis
    da_mask = df["company_name"].str.contains(r'(?i)data\s*annotation', na=False)
    da_count = da_mask.sum()
    da_swe = (da_mask & df["is_swe"]).sum()
    log.info(f"\n  DataAnnotation: {da_count:,} total, {da_swe:,} SWE")
    if da_swe > 0:
        da_titles = df.loc[da_mask & df["is_swe"], "title"].value_counts().head(5)
        log.info("    Top SWE titles from DataAnnotation:")
        for title, count in da_titles.items():
            log.info(f"      {title}: {count:,}")

    # --- Save ---
    output_path = INTERMEDIATE_DIR / "stage2_aggregators.parquet"
    df.to_parquet(output_path, index=False)
    log.info(f"\nSaved to {output_path}")

    return df


if __name__ == "__main__":
    df = run_stage2()
    print(f"\nDone: {len(df):,} rows, {df['is_aggregator'].sum():,} aggregators flagged")
