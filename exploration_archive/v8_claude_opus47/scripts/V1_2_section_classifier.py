"""V1.2 - Independent section classifier re-derivation.

Build my own regex-based section classifier from scratch. Do NOT read T13.
Sample 2,000 postings per period-source, classify sections, compute median chars.
Compare to T13's reported numbers (within 20% tolerance).
"""
import duckdb
import re
import pandas as pd
import numpy as np
import json
from pathlib import Path

np.random.seed(42)

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ Section classifier (mine, independent of T13) ============
# Strategy: headers are lines like "Responsibilities:", "Requirements", "What You'll Do", "Benefits"
# We split by line breaks, match headers to canonical categories, then accumulate chars until next header.

# Category keyword patterns (applied to header text, not body)
SECTION_PATTERNS = {
    "role_summary": [
        r"\babout (?:the|this) (?:role|position|job|opportunity)\b",
        r"\bthe (?:role|position|opportunity)\b",
        r"\brole (?:overview|description|summary)\b",
        r"\bposition (?:overview|description|summary)\b",
        r"\b(?:role|position|job) summary\b",
        r"\bjob description\b",
        r"\bsummary\b$",
        r"\boverview\b$",
        r"\bwhat (?:is|are) (?:the|this) (?:role|position|opportunity|team)\b",
        r"\bwho we['']?re looking for\b",
        r"\bwho we look for\b",
        r"\babout (?:you|the candidate)\b",
    ],
    "responsibilities": [
        r"\bresponsibilities\b",
        r"\bwhat you['']?(?:ll| will) (?:do|be doing|build|work on|own|accomplish)\b",
        r"\bwhat you['']?ll work on\b",
        r"\bday[- ]to[- ]day\b",
        r"\b(?:your |the )?(?:key )?(?:duties|tasks|accountabilities)\b",
        r"\bkey responsibilities\b",
        r"\bessential (?:functions|duties)\b",
        r"\bin this role,? you['']?ll\b",
        r"\byou will\b",
        r"\bwhat you['']?ll (?:own|deliver|ship)\b",
        r"\bday to day\b",
        r"\bjob duties\b",
        r"\bprimary responsibilities\b",
        r"\byour role\b",
        r"\byour responsibilities\b",
        r"\bthe work\b$",
        r"\bwhat the (?:role|job) entails\b",
    ],
    "requirements": [
        r"\brequirements?\b",
        r"\bqualifications?\b",
        r"\brequired (?:qualifications|skills|experience)\b",
        r"\bminimum (?:qualifications|requirements|experience)\b",
        r"\bmust[- ]haves?\b",
        r"\bbasic qualifications\b",
        r"\bmandatory (?:qualifications|skills)\b",
        r"\bskills (?:and|&) (?:experience|qualifications)\b",
        r"\bskills required\b",
        r"\bwhat (?:we[''']?re|we are) looking for\b",
        r"\bwhat you['']?(?:ll| will) bring\b",
        r"\bwhat you(?:ll| will) need\b",
        r"\bwho you are\b",
        r"\byou (?:have|bring|are|possess)\b",
        r"\bthe ideal candidate\b",
        r"\bcandidate (?:profile|qualifications)\b",
        r"\brequired skills\b",
        r"\bjob requirements\b",
        r"\beducation (?:and|&) experience\b",
        r"\byour (?:qualifications|profile|skills)\b",
    ],
    "preferred": [
        r"\bpreferred\b",
        r"\bnice[- ]to[- ]haves?\b",
        r"\bbonus (?:points|skills)?\b",
        r"\b(?:a )?plus\b",
        r"\bdesirable (?:qualifications|skills)?\b",
        r"\bdesired (?:qualifications|skills)\b",
        r"\boptional (?:qualifications|skills)\b",
        r"\badditional (?:qualifications|skills)\b",
        r"\bextra credit\b",
        r"\beven better if\b",
        r"\bbonus if you\b",
    ],
    "benefits": [
        r"\bbenefits\b",
        r"\bperks\b",
        r"\bwhat we offer\b",
        r"\bwhat we['']?ll offer\b",
        r"\bwhat we['']?re offering\b",
        r"\bcompensation\b",
        r"\bsalary\b",
        r"\btotal (?:rewards|compensation)\b",
        r"\bwe offer\b",
        r"\bour benefits\b",
        r"\bpay range\b",
        r"\bwhy (?:work (?:with|for) us|join us|us)\b",
        r"\bwhy (?:you['']?ll|you will) love\b",
        r"\bemployee benefits\b",
        r"\brewards\b$",
    ],
    "about_company": [
        r"\babout (?:us|the company|our (?:company|team|mission))\b",
        r"\bwho we are\b",
        r"\bour (?:mission|vision|values|story|team|company|culture|impact)\b",
        r"\bcompany (?:overview|description)\b",
        r"\bwhy (?:the company|\w+ )?exists\b",  # e.g. "why stripe exists"
        r"\bour (?:commitment|approach)\b",
        r"\bmission\b$",
        r"\bvision\b$",
        r"\bculture\b$",
    ],
    "legal": [
        r"\bequal (?:employment |opportunity )?(?:opportunity|employer)\b",
        r"\beeo\b",
        r"\bnon[- ]discrimination\b",
        r"\ban equal opportunity\b",
        r"\bdiversity (?:and|&) inclusion\b",
        r"\baccommodations?\b",
        r"\breasonable accommodation\b",
        r"\bada\b$",
        r"\bdisclosur",
        r"\be-verify\b",
        r"\bbackground check\b",
        r"\bauthorized to work\b",
        r"\bsponsorship\b",
        r"\bvisa\b",
        r"\bcalifornia (?:applicants|consumer|privacy)\b",
        r"\bnotice to (?:applicants|candidates)\b",
        r"\bprivacy (?:notice|policy)\b",
        r"\bat[- ]will employment\b",
        r"\baffirmative action\b",
        r"\bdrug[- ]free\b",
    ],
}


def classify_header(header_text):
    """Return the category of a header line, or None if not a header."""
    if not header_text or not header_text.strip():
        return None
    h = header_text.strip().lower()
    # Strip trailing punctuation
    h = re.sub(r"[:;.?!\-—–*•\s]+$", "", h)
    h = re.sub(r"^[*•\-—–\s]+", "", h)
    # Strip markdown headings
    h = re.sub(r"^#+\s*", "", h)
    # Heuristic: header should be short (<~100 chars)
    if len(h) > 120:
        return None
    for category, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, h, flags=re.IGNORECASE):
                return category
    return None


def is_header_line(line):
    """Heuristic: a line is a header if it's short, ends with : or is all-caps/title-case, and classifies."""
    s = line.strip()
    if not s:
        return False
    # Strip bullet / number leads
    s_stripped = re.sub(r"^[*•\-—–\s\d+\.\)]+", "", s)
    if len(s_stripped) > 120:
        return False
    # Header cues: ends with :, or is all-caps, or is markdown heading, or is a bolded line
    is_colon = s.rstrip().endswith(":")
    is_heading = s.lstrip().startswith("#")
    is_short = len(s_stripped) <= 80
    is_titlecase = (s_stripped and
                    sum(1 for c in s_stripped if c.isupper()) / max(len(s_stripped), 1) > 0.15
                    and s_stripped == s_stripped.title())
    return is_colon or is_heading or (is_short and is_titlecase)


def classify_posting(text):
    """Split text into sections. Return dict of category -> char count."""
    if not text:
        return {}
    # Normalize newlines
    lines = text.split("\n")
    sections = {k: 0 for k in SECTION_PATTERNS}
    sections["unclassified"] = 0
    current = "unclassified"
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if is_header_line(line):
            cat = classify_header(s)
            if cat is not None:
                current = cat
                continue  # don't count header itself
        sections[current] += len(s)
    return sections


# ==== Inline TDD asserts ====
def tests():
    # Test header classification
    assert classify_header("Responsibilities:") == "responsibilities"
    assert classify_header("What You'll Do") == "responsibilities"
    assert classify_header("Minimum Qualifications") == "requirements"
    assert classify_header("Nice to have") == "preferred"
    assert classify_header("Benefits") == "benefits"
    assert classify_header("About the Company") == "about_company"
    assert classify_header("Equal Opportunity Employer") == "legal"
    assert classify_header("What We Offer") == "benefits"
    assert classify_header("Who You Are") == "requirements"
    assert classify_header("Role Overview") == "role_summary"
    assert classify_header("The Team") is None  # too generic
    assert classify_header("") is None

    # Test classify_posting
    sample = """
Role Overview:
We are a fast-growing tech company building AI products.

What You'll Do:
- Build backend services in Python
- Collaborate with the ML team

Requirements:
- 5+ years experience
- Python, Go expertise

Nice to Have:
- Kubernetes experience

Benefits:
- Competitive salary
- Health insurance

About Us:
We're a Series B startup.

Equal Opportunity Employer:
We value diversity.
"""
    result = classify_posting(sample)
    # Check that each section got characters
    assert result["role_summary"] > 0, f"role_summary = {result['role_summary']}"
    assert result["responsibilities"] > 0, f"responsibilities = {result['responsibilities']}"
    assert result["requirements"] > 0, f"requirements = {result['requirements']}"
    assert result["preferred"] > 0, f"preferred = {result['preferred']}"
    assert result["benefits"] > 0, f"benefits = {result['benefits']}"
    assert result["about_company"] > 0, f"about_company = {result['about_company']}"
    assert result["legal"] > 0, f"legal = {result['legal']}"
    print("All tests passed.")


def main():
    tests()

    con = duckdb.connect()
    # Load 2,000 from each of: arshkon 2024-04, asaniczka 2024-01, scraped 2026-03+04
    print("Loading samples...")
    # Use description_raw (full untouched text) since section headers are preserved there.
    q = """
    SELECT uid, source, period, description, seniority_final
    FROM '/home/jihgaboot/gabor/job-research/data/unified.parquet'
    WHERE source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok' AND is_swe = true
      AND description IS NOT NULL AND description != ''
    """
    df = con.execute(q).df()
    print(f"  loaded {len(df)} rows")

    # Sample 2000 per period-source
    samples = []
    for key, sub in df.groupby(["source", "period"]):
        sz = min(2000, len(sub))
        s = sub.sample(n=sz, random_state=42)
        samples.append(s)
    df_sample = pd.concat(samples, ignore_index=True)
    print(f"  sampled {len(df_sample)} rows across groups")

    # Classify
    print("Classifying...")
    all_results = []
    for idx, row in df_sample.iterrows():
        sects = classify_posting(row["description"])
        result = {"uid": row["uid"], "source": row["source"], "period": row["period"],
                  "seniority_final": row["seniority_final"], **sects}
        all_results.append(result)
    res = pd.DataFrame(all_results)
    print(res.head())

    # Compute per-group metrics
    print("\n=== Mean chars by section, per period (pooled across seniority) ===")
    groups = res.groupby(["source", "period"])
    section_cols = list(SECTION_PATTERNS.keys()) + ["unclassified"]
    means = groups[section_cols].mean().round(1)
    print(means.to_string())

    # Pooled 2024 vs scraped 2026
    res["period_group"] = res["period"].apply(lambda p: "2024" if p.startswith("2024") else "2026")
    print("\n=== Mean chars by section, 2024 vs 2026 ===")
    group_means = res.groupby("period_group")[section_cols].mean().round(1)
    print(group_means.to_string())

    # Total chars and share
    res["total"] = res[section_cols].sum(axis=1)
    # Compute per-posting share
    for s in section_cols:
        res[f"{s}_share"] = res[s] / res["total"].replace(0, np.nan)
    share_cols = [f"{s}_share" for s in section_cols]
    share_means = res.groupby("period_group")[share_cols].mean().round(3)
    print("\n=== Share-of-total by section, 2024 vs 2026 ===")
    print(share_means.to_string())

    # Delta table
    delta = group_means.loc["2026"] - group_means.loc["2024"]
    pct_change = (delta / group_means.loc["2024"]).round(3)
    print("\n=== Delta (2026 - 2024) in mean chars ===")
    for s in section_cols:
        d = delta[s]
        p24 = group_means.loc["2024", s]
        p26 = group_means.loc["2026", s]
        pc = pct_change[s]
        print(f"  {s:20s}  2024={p24:>7.1f}  2026={p26:>7.1f}  delta={d:>+7.1f}  pct={pc:+.1%}")

    # Specifically check requirements-section shrinkage
    req24 = group_means.loc["2024", "requirements"]
    req26 = group_means.loc["2026", "requirements"]
    req_delta = req26 - req24
    req_pct = req_delta / req24

    # T13 reported pooled-2024 req = 1,308; scraped req = 1,059; delta = -249; -19%.
    T13_REQ_24 = 1308
    T13_REQ_26 = 1059
    T13_REQ_DELTA = -249
    T13_REQ_PCT = -0.19

    req_24_agree = abs(req24 - T13_REQ_24) / T13_REQ_24
    req_26_agree = abs(req26 - T13_REQ_26) / T13_REQ_26

    print(f"\nT13 reported: req24={T13_REQ_24}, req26={T13_REQ_26}, delta={T13_REQ_DELTA} ({T13_REQ_PCT:.0%})")
    print(f"Mine: req24={req24:.1f}, req26={req26:.1f}, delta={req_delta:.1f} ({req_pct:.1%})")
    print(f"Req24 deviation: {req_24_agree:.1%}; Req26 deviation: {req_26_agree:.1%}")
    # Within 20% -> verified
    verified = (req_24_agree < 0.20) and (req_26_agree < 0.20) and (req_pct < 0)  # still shrank
    print(f"V1.2 verified (within 20%): {verified}")

    # Save results
    res.to_csv(OUT_DIR / "V1_2_section_classifications.csv", index=False)
    summary = {
        "T13_reported": {"req24": T13_REQ_24, "req26": T13_REQ_26, "delta": T13_REQ_DELTA, "pct": T13_REQ_PCT},
        "mine": {
            "req24": float(req24), "req26": float(req26),
            "delta": float(req_delta), "pct": float(req_pct),
        },
        "agreement": {
            "req24_deviation": float(req_24_agree),
            "req26_deviation": float(req_26_agree),
            "req_still_shrank": bool(req_pct < 0),
            "verified_within_20pct": bool(verified),
        },
        "full_means": group_means.to_dict(),
        "full_shares": share_means.to_dict(),
    }
    with open(OUT_DIR / "V1_2_section_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Also entry vs mid-senior specifically (T13 reports entry -9%, senior -22%)
    res_sen = res[res["seniority_final"].isin(["entry", "associate", "mid-senior", "director"])].copy()
    res_sen["jsen"] = res_sen["seniority_final"].apply(
        lambda s: "entry" if s in ("entry", "associate") else "senior"
    )
    grp = res_sen.groupby(["period_group", "jsen"])["requirements"].mean()
    print("\n=== Requirements chars by (period, entry vs senior) ===")
    print(grp)
    if ("2024", "entry") in grp.index and ("2026", "entry") in grp.index:
        en24 = grp.loc[("2024", "entry")]
        en26 = grp.loc[("2026", "entry")]
        sn24 = grp.loc[("2024", "senior")]
        sn26 = grp.loc[("2026", "senior")]
        print(f"Entry: {en24:.1f} -> {en26:.1f} ({(en26-en24)/en24:.1%})")
        print(f"Senior: {sn24:.1f} -> {sn26:.1f} ({(sn26-sn24)/sn24:.1%})")
        summary["entry_senior"] = {
            "entry_24": float(en24), "entry_26": float(en26),
            "entry_pct": float((en26 - en24) / en24),
            "senior_24": float(sn24), "senior_26": float(sn26),
            "senior_pct": float((sn26 - sn24) / sn24),
        }
        with open(OUT_DIR / "V1_2_section_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {OUT_DIR / 'V1_2_section_summary.json'}")


if __name__ == "__main__":
    main()
