"""
T13 Section classifier for SWE job postings (reusable by T12 and other agents).

Handles BOTH text formats:
  - Kaggle (arshkon/asaniczka): HTML-stripped, limited section markers (e.g., "Responsibilities:").
  - Scraped: preserves markdown headers (e.g., "**Responsibilities** |", "**Minimum Qualifications**").

Public API:
    classify_sections(text) -> list[dict]
        Returns a list of segments [{section, start, end, text, char_len}, ...]
        covering the entire input, non-overlapping, spanning 0 to len(text).

    section_char_proportions(text) -> dict[section_name, int]
        Aggregated char counts per section for a single posting.

    SECTIONS: list of canonical section names (order used for reporting).

Assertions at bottom of file validate behaviour on sample strings from BOTH sources.
Run this file directly to execute the assertions: ./.venv/bin/python exploration/scripts/T13_section_classifier.py
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple

# Canonical section labels (reporting order)
SECTIONS = [
    "role_summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal",
    "unclassified",
]

# ---------------------------------------------------------------------------
# Header pattern spec
# ---------------------------------------------------------------------------
# Each entry maps canonical section -> list of regex header patterns.
# Patterns must match the full header token. They will be compiled with
# re.IGNORECASE | re.MULTILINE and anchored to appear at a line start OR
# within a line after a bullet/hyphen/pipe/asterisk delimiter.
#
# Design rules (test with both formats):
#   - Scraped headers often look like "**Responsibilities** |" or "**Responsibilities:**"
#   - Kaggle headers often look like "Responsibilities:" or bolded with no markdown.
#   - Order matters: more-specific (preferred/nice-to-have) must be listed BEFORE
#     more-generic (requirements/qualifications) because the matcher picks the
#     FIRST pattern that matches the candidate header text.
# ---------------------------------------------------------------------------

_HEADER_SPECS: List[Tuple[str, str]] = [
    # Preferred / Nice-to-have — must come before requirements
    ("preferred", r"preferred(?:\s+qualifications?| skills?| experience)?"),
    ("preferred", r"nice[\s\-]*to[\s\-]*have(?:s)?"),
    ("preferred", r"bonus(?:\s+points?| skills?)?"),
    ("preferred", r"plus(?:es)?(?:\s+skills?)?"),
    ("preferred", r"desired(?:\s+qualifications?| skills?| experience)?"),
    ("preferred", r"additional(?:\s+qualifications?| skills?)?"),

    # Requirements / Qualifications
    ("requirements", r"(?:minimum|basic|required|key|core|mandatory|standard|general)\s+qualifications?"),
    ("requirements", r"qualifications?(?:\s+(?:and|&)\s+(?:skills?|experience))?"),
    ("requirements", r"(?:job|position|role|key|core|technical|general)\s+requirements?"),
    ("requirements", r"requirements?"),
    ("requirements", r"what\s+you(?:'|\u2019)?ll\s+need"),
    ("requirements", r"what\s+we(?:'|\u2019)?re\s+looking\s+for"),
    ("requirements", r"what\s+we\s+need"),
    ("requirements", r"who\s+you\s+are"),
    ("requirements", r"(?:key\s+|technical\s+|required\s+)?skills?(?:\s+(?:and|&)\s+(?:experience|qualifications?))?"),
    ("requirements", r"experience"),
    ("requirements", r"education(?:\s+(?:and|&)\s+experience)?"),
    ("requirements", r"must[\s\-]*haves?"),
    ("requirements", r"you\s+(?:have|bring|will\s+have)"),

    # Responsibilities / Duties
    ("responsibilities", r"(?:key\s+|core\s+|main\s+|primary\s+|your\s+)?responsibilities"),
    ("responsibilities", r"(?:key\s+|primary\s+|essential\s+|your\s+)?duties(?:\s+include[s]?)?"),
    ("responsibilities", r"what\s+you(?:'|\u2019)?ll\s+do"),
    ("responsibilities", r"what\s+you(?:'|\u2019)?ll\s+be\s+doing"),
    ("responsibilities", r"what\s+you\s+will\s+(?:do|be\s+doing)"),
    ("responsibilities", r"in\s+this\s+role(?:[,:]?\s*you(?:'|\u2019)?(?:ll|\s*will)(?:\s+\w+)?)?"),
    ("responsibilities", r"day[\s\-]*to[\s\-]*day"),
    ("responsibilities", r"your\s+(?:role|impact|mission)"),
    ("responsibilities", r"tasks\s+include[s]?(?:\s+but\s+not\s+limited\s+to)?"),
    ("responsibilities", r"job\s+(?:description|duties|responsibilities)"),
    ("responsibilities", r"position\s+(?:description|duties|responsibilities)"),
    ("responsibilities", r"role\s+(?:description|overview)"),

    # Benefits / Compensation / Perks
    ("benefits", r"benefits(?:\s+(?:and|&)\s+perks?)?"),
    ("benefits", r"perks(?:\s+(?:and|&)\s+benefits?)?"),
    ("benefits", r"compensation(?:\s+(?:and|&)\s+benefits?)?"),
    ("benefits", r"(?:what\s+)?(?:we\s+|you(?:'|\u2019)?ll\s+get|you\s+get)?\s*offer"),
    ("benefits", r"(?:our\s+)?(?:total\s+)?rewards?(?:\s+package)?"),
    ("benefits", r"pay(?:\s+range| rate| scale| and benefits)?"),
    ("benefits", r"salary(?:\s+range)?"),

    # About company
    ("about_company", r"about\s+(?:us|the\s+company|our\s+company|the\s+team|our\s+team)"),
    ("about_company", r"company\s+(?:overview|description|profile)"),
    ("about_company", r"who\s+we\s+are"),
    ("about_company", r"our\s+(?:story|mission|vision|culture|values)"),
    ("about_company", r"why\s+join(?:\s+us)?"),
    ("about_company", r"why\s+(?:work|you(?:'|\u2019)?ll\s+love)(?:\s+(?:here|for\s+us|with\s+us))?"),

    # Legal / EEO
    ("legal", r"equal\s+(?:opportunity|employment)(?:\s+(?:employer|statement))?"),
    ("legal", r"e\.?e\.?o\.?(?:\s+statement| policy)?"),
    ("legal", r"diversity(?:\s+(?:and|&)\s+inclusion|\s+statement)?"),
    ("legal", r"affirmative\s+action"),
    ("legal", r"at\s+\w+,\s+we\s+are\s+an\s+equal"),
    ("legal", r"(?:disability|accommodation)\s+(?:statement|notice)"),
    ("legal", r"we\s+value\s+equal\s+opportunity"),
    ("legal", r"e-verify"),

    # Role summary / job summary (last — catches very generic openings)
    ("role_summary", r"(?:job|position|role|general|general\s+role)\s+summary"),
    ("role_summary", r"about\s+(?:this\s+role|the\s+role|the\s+position|the\s+job)"),
    ("role_summary", r"position\s+(?:title|overview|summary)"),
    ("role_summary", r"the\s+role"),
    ("role_summary", r"(?:job|role|position|general)\s+overview"),
    ("role_summary", r"overview"),
    ("role_summary", r"summary"),
]

# Compile header patterns. We match a header at the start of a line (optionally
# preceded by markdown emphasis, bullets, numbering, or whitespace), followed
# by a terminator such as ":", "|", "**", end of line, or a bullet char.
#
# The full line-level regex captures optional leading markers, the header text,
# and any trailing marker. We scan with finditer and each match gives us a
# span. After collecting all spans we sort by start position and segment the
# text.

_LEAD = r"(?:[\s\*#>\-\u2022\u25AA\u25CF0-9\.\)\(]*)"   # noqa: E501 (leading bullets/markdown/numbering)
_TRAIL = r"(?:\*\*)?\s*(?:[:\|\u2013\u2014\-]\s*(?:\*\*)?)?\s*$"  # : | -- etc.

_compiled: List[Tuple[str, "re.Pattern[str]"]] = []
for section, body in _HEADER_SPECS:
    pattern = rf"^{_LEAD}(?:\*\*)?\s*(?:{body})\s*(?:\*\*)?\s*(?:[:\|\u2013\u2014\-]|$)"
    _compiled.append((section, re.compile(pattern, re.IGNORECASE | re.MULTILINE)))

# Run-together header patterns (arshkon/asaniczka HTML-stripped text often
# has headers merged with the next word, e.g. "ResponsibilitiesDevelop quality
# software"). We match these conservatively: the header word must start with a
# capital or line-start, and be followed directly by an uppercase letter that
# starts the next clause. We only accept a fixed whitelist of robust headers
# to avoid matching English prose.
_RUN_TOGETHER_SPECS: List[Tuple[str, str]] = [
    ("responsibilities", r"Responsibilities"),
    ("responsibilities", r"Duties"),
    ("requirements", r"Qualifications"),
    ("requirements", r"Requirements"),
    ("requirements", r"Required\s*Skills"),
    ("requirements", r"Required\s*Experience"),
    ("preferred", r"Preferred\s*Qualifications"),
    ("preferred", r"Preferred\s*Skills"),
    ("preferred", r"Nice\s*to\s*Have"),
    ("benefits", r"Benefits"),
    ("benefits", r"What\s*We\s*Offer"),
]
# Require the header to start at a line boundary or after whitespace, AND be
# followed immediately by an uppercase letter (run-together) OR one of :|.
# This catches "ResponsibilitiesDevelop..." but not "responsibilities include".
_compiled_rt: List[Tuple[str, "re.Pattern[str]"]] = []
for section, body in _RUN_TOGETHER_SPECS:
    pattern = rf"(?:^|\n)\s*({body})(?=[A-Z][a-z])"
    _compiled_rt.append((section, re.compile(pattern, re.MULTILINE)))

# Inline header patterns (for HTML-stripped kaggle text where all whitespace
# collapses and headers like "Job Description:" appear mid-sentence). We
# require the header to be preceded by a space or start of string and be
# followed by `:`. Only a strict whitelist is allowed.
_INLINE_SPECS: List[Tuple[str, str]] = [
    ("role_summary", r"Job\s+Description"),
    ("role_summary", r"Position\s+Summary"),
    ("role_summary", r"Role\s+Description"),
    ("role_summary", r"Role\s+Summary"),
    ("role_summary", r"Overview"),
    ("responsibilities", r"(?:Key\s+|Primary\s+|Core\s+|Main\s+|Your\s+)?Responsibilities"),
    ("responsibilities", r"Duties\s+Include[s]?"),
    ("responsibilities", r"Tasks\s+Include[s]?"),
    ("responsibilities", r"What\s+You(?:'|\u2019)?ll\s+Do"),
    ("responsibilities", r"What\s+You\s+Will\s+Do"),
    ("requirements", r"(?:Minimum|Basic|Required|Key|Core|Mandatory)\s+Qualifications?"),
    ("requirements", r"(?:Minimum|Basic|Required|Key|Core|Mandatory)\s+Skills?"),
    ("requirements", r"(?:Minimum|Basic|Required|Key|Core|Mandatory)\s+Requirements?"),
    ("requirements", r"Qualifications?"),
    ("requirements", r"Requirements?"),
    ("requirements", r"Required"),
    ("requirements", r"Must\s+Have"),
    ("preferred", r"Preferred\s+Qualifications?"),
    ("preferred", r"Preferred\s+Skills?"),
    ("preferred", r"Nice\s+to\s+Have"),
    ("preferred", r"Bonus\s+Points?"),
    ("preferred", r"Bonus\s+Skills?"),
    ("benefits", r"Benefits"),
    ("benefits", r"Compensation"),
    ("benefits", r"Perks"),
    ("about_company", r"About\s+Us"),
    ("about_company", r"About\s+the\s+Company"),
    ("legal", r"Equal\s+Opportunity\s+Employer"),
    ("legal", r"EEO"),
]
_compiled_inline: List[Tuple[str, "re.Pattern[str]"]] = []
for section, body in _INLINE_SPECS:
    # Require preceding whitespace or start-of-string, followed by ":" or " |"
    pattern = rf"(?:^|(?<=\s))({body})\s*[:\|]"
    _compiled_inline.append((section, re.compile(pattern)))


# Standalone header filter: the matched line should be a header, not prose.
# We accept matches whose content up to a line break is short (<= 80 chars) and
# whose matched text does not contain sentence-like punctuation inside the header.
_MAX_HEADER_LINE = 90


def _find_header_spans(text: str) -> List[Tuple[int, int, str]]:
    """Find all section header matches in text. Returns list of (start, end, section)."""
    spans: List[Tuple[int, int, str]] = []
    if not text:
        return spans
    seen: set = set()
    for section, pat in _compiled:
        for m in pat.finditer(text):
            start = m.start()
            end = m.end()
            # expand end to end of line for the segment boundary
            nl = text.find("\n", end)
            line_end = nl if nl != -1 else len(text)
            line_text = text[start:line_end]
            # Reject if line is too long to be a header (likely prose with leading word)
            if len(line_text) > _MAX_HEADER_LINE:
                # Exception: scraped headers can be "**Responsibilities** | * bullet..."
                # In that case the pipe separates header from inline bullet content.
                # We accept if there is a "|" or "**" within the first 60 chars and treat
                # the boundary as the end of the header token itself.
                if "|" not in line_text[:60] and "**" not in line_text[:60]:
                    continue
            key = (start, section)
            if key in seen:
                continue
            seen.add(key)
            spans.append((start, end, section))
    # Also match run-together headers (arshkon HTML-stripped format)
    for section, pat in _compiled_rt:
        for m in pat.finditer(text):
            # group 1 is the header word itself
            start = m.start(1)
            end = m.end(1)
            key = (start, section)
            if key in seen:
                continue
            seen.add(key)
            spans.append((start, end, section))
    # Inline headers (e.g. "... Job Description: As a ..." in HTML-stripped text).
    # Only accepted if no higher-quality header at the same position exists.
    for section, pat in _compiled_inline:
        for m in pat.finditer(text):
            start = m.start(1)
            end = m.end(0)
            key = (start, section)
            if key in seen:
                continue
            # Check if the match is preceded by a header already (within 5 chars)
            overlap = any(abs(start - s) < 5 for s, _, _ in spans)
            if overlap:
                continue
            seen.add(key)
            spans.append((start, end, section))
    # Deduplicate: if multiple patterns matched the same start, keep the FIRST
    # spec order (which is our priority ordering). We already iterate in priority
    # order, so the first seen wins.
    best: Dict[int, Tuple[int, int, str]] = {}
    for start, end, section in spans:
        if start not in best:
            best[start] = (start, end, section)
    out = sorted(best.values(), key=lambda x: x[0])
    return out


def classify_sections(text: str) -> List[Dict]:
    """Segment text into section-labeled regions.

    Returns a list of dicts: [{section, start, end, text, char_len}, ...].
    Segments are non-overlapping and cover the whole input. Regions before the
    first header are labeled 'role_summary' if they contain any content; the
    unclassified label is used only for segments between headers that we could
    not categorize (rare)."""
    if not text:
        return []
    headers = _find_header_spans(text)
    segments: List[Dict] = []
    if not headers:
        return [{
            "section": "unclassified",
            "start": 0,
            "end": len(text),
            "text": text,
            "char_len": len(text),
        }]
    # Region before first header
    first_start = headers[0][0]
    if first_start > 0:
        pre = text[:first_start]
        if pre.strip():
            segments.append({
                "section": "role_summary",
                "start": 0,
                "end": first_start,
                "text": pre,
                "char_len": len(pre),
            })
    for i, (h_start, h_end, section) in enumerate(headers):
        # Segment spans from header start to next header start
        next_start = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        seg_text = text[h_start:next_start]
        segments.append({
            "section": section,
            "start": h_start,
            "end": next_start,
            "text": seg_text,
            "char_len": len(seg_text),
        })
    return segments


def section_char_proportions(text: str) -> Dict[str, int]:
    """Return a dict mapping section name -> total char count for this posting.

    Every value in SECTIONS is present (default 0). Proportions can be computed
    downstream by dividing by sum() or len(text)."""
    counts = {s: 0 for s in SECTIONS}
    for seg in classify_sections(text):
        counts[seg["section"]] = counts.get(seg["section"], 0) + seg["char_len"]
    return counts


# ---------------------------------------------------------------------------
# Inline assertions — run when executed as a script.
# ---------------------------------------------------------------------------
def _run_assertions() -> None:
    # --- scraped markdown format ---
    scraped_a = (
        "As a Software Engineer, you will design and build systems.\n"
        "You'll collaborate with cross-functional teams.\n"
        "**Minimum Qualifications** | * Bachelor's degree\n"
        "* 2+ years of programming experience\n"
        "**Responsibilities** | * Architect software\n"
        "* Develop drivers\n"
        "**Preferred Qualifications** | * Master's degree\n"
        "* 5+ years experience\n"
    )
    counts = section_char_proportions(scraped_a)
    assert counts["role_summary"] > 0, f"scraped_a role_summary: {counts}"
    assert counts["requirements"] > 0, f"scraped_a requirements: {counts}"
    assert counts["responsibilities"] > 0, f"scraped_a responsibilities: {counts}"
    assert counts["preferred"] > 0, f"scraped_a preferred: {counts}"

    # --- scraped with benefits and EEO ---
    scraped_b = (
        "**About This Role**\n"
        "Wells Fargo is seeking a Lead Software Engineer.\n"
        "**In This Role, You Will**\n"
        "* Lead complex technology initiatives\n"
        "* Design pipelines\n"
        "**Required Qualifications:** | * 5+ years experience\n"
        "* Proficient in SQL and Python\n"
        "**Desired Qualifications:** | * GCP experience\n"
        "**Pay Range**\n"
        "$150,000 - $200,000\n"
        "**We Value Equal Opportunity**\n"
        "Wells Fargo is an equal opportunity employer.\n"
    )
    counts_b = section_char_proportions(scraped_b)
    assert counts_b["about_company"] > 0 or counts_b["role_summary"] > 0, f"scraped_b about/role: {counts_b}"
    assert counts_b["responsibilities"] > 0, f"scraped_b resp: {counts_b}"
    assert counts_b["requirements"] > 0, f"scraped_b req: {counts_b}"
    assert counts_b["preferred"] > 0, f"scraped_b pref: {counts_b}"
    assert counts_b["benefits"] > 0, f"scraped_b benefits: {counts_b}"
    assert counts_b["legal"] > 0, f"scraped_b legal: {counts_b}"

    # --- kaggle arshkon format (limited markers) ---
    kaggle_a = (
        "Gfk is seeking a lead software engineer.\n"
        "We are looking for a developer.\n"
        "Key Responsibilities:\n"
        "Lead developers on the team to meet product deliverables.\n"
        "Coach junior developers.\n"
        "Qualifications:\n"
        "Hands-on Java development.\n"
        "5+ years experience.\n"
        "Preferred Skills:\n"
        "Kubernetes knowledge.\n"
    )
    counts_k = section_char_proportions(kaggle_a)
    assert counts_k["role_summary"] > 0, f"kaggle_a role_summary: {counts_k}"
    assert counts_k["responsibilities"] > 0, f"kaggle_a responsibilities: {counts_k}"
    assert counts_k["requirements"] > 0, f"kaggle_a requirements: {counts_k}"
    assert counts_k["preferred"] > 0, f"kaggle_a preferred: {counts_k}"

    # --- kaggle asaniczka with benefits ---
    kaggle_b = (
        "As a Senior Software Engineer, you will develop applications.\n"
        "Responsibilities:\n"
        "Build services.\n"
        "Debug issues.\n"
        "Requirements:\n"
        "5+ years experience.\n"
        "Python proficiency.\n"
        "Benefits:\n"
        "Health, dental, 401k.\n"
        "Equal Opportunity Employer:\n"
        "We are an equal opportunity employer.\n"
    )
    counts_kb = section_char_proportions(kaggle_b)
    assert counts_kb["role_summary"] > 0, f"kaggle_b role_summary: {counts_kb}"
    assert counts_kb["responsibilities"] > 0, f"kaggle_b responsibilities: {counts_kb}"
    assert counts_kb["requirements"] > 0, f"kaggle_b requirements: {counts_kb}"
    assert counts_kb["benefits"] > 0, f"kaggle_b benefits: {counts_kb}"
    assert counts_kb["legal"] > 0, f"kaggle_b legal: {counts_kb}"

    # --- edge cases ---
    assert classify_sections("") == []
    # No headers at all
    plain = "We are looking for a software engineer with strong Python skills."
    segs_plain = classify_sections(plain)
    assert len(segs_plain) == 1
    assert segs_plain[0]["section"] == "unclassified"

    # Priority test: "preferred qualifications" should land in preferred, not requirements
    pri = "**Preferred Qualifications** | * Master's degree\n"
    counts_pri = section_char_proportions(pri)
    assert counts_pri["preferred"] > 0 and counts_pri["requirements"] == 0, f"priority: {counts_pri}"

    # Priority: "minimum qualifications" should land in requirements
    min_pri = "**Minimum Qualifications** | * Bachelor's degree\n"
    counts_min = section_char_proportions(min_pri)
    assert counts_min["requirements"] > 0 and counts_min["preferred"] == 0, f"min priority: {counts_min}"

    print("All section classifier assertions passed.")


if __name__ == "__main__":
    _run_assertions()
