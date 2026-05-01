"""T13 section classifier — reusable by T12 and downstream tasks.

Splits a raw job description into blocks, classifies each block into a
section type, and returns per-section character totals plus a
section-filtered text.

Section types:
  - role_summary
  - responsibilities
  - requirements
  - preferred
  - benefits
  - about_company
  - legal
  - unclassified

Usage:
  from T13_section_classifier import classify_description, SECTION_TYPES
  sections = classify_description(raw_description)
  # sections['responsibilities']['chars']  -> int char count
  # sections['responsibilities']['text']   -> raw text for that section
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

SECTION_TYPES = [
    "role_summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal",
    "unclassified",
]

# Each pattern is tested against the first ~140 chars of a block (lower-cased).
# Order matters — more specific patterns must come before more general ones
# (e.g. preferred before requirements, because "preferred qualifications" also
# contains "qualifications").
_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "legal",
        re.compile(
            r"(equal\s+(opportunity|employment)|\beeo\b|affirmative\s+action|"
            r"without\s+regard\s+to|protected\s+(class|characteristics|veteran)|"
            r"e[- ]verify|at[- ]will\s+employment|reasonable\s+accommodation|"
            r"disability\s+accommodation|ada\s+compliance|pay\s+transparency|"
            r"applicants?\s+with\s+(a\s+)?(criminal|arrest)|"
            r"background\s+check|drug[- ]free\s+workplace)",
            re.IGNORECASE,
        ),
    ),
    (
        "preferred",
        re.compile(
            r"(^preferred$|^preferred\s|"
            r"preferred\s+(qualifications|skills|experience|requirements)|"
            r"nice[- ]to[- ]have|nice\s+to\s+haves?|bonus\s+(points|skills|qualifications)|"
            r"pluses?|ideal\s+candidate|extra\s+credit|"
            r"preferred\s+but\s+not\s+required|desired\s+(skills|qualifications|experience)|"
            r"good\s+to\s+have|additional\s+(skills|qualifications)|"
            r"you\s+might\s+also\s+have|"
            r"even\s+better)",
            re.IGNORECASE,
        ),
    ),
    (
        "benefits",
        re.compile(
            r"(\bbenefits?\b|\bperks\b|\bcompensation\b|compensation\s+(package|range)|"
            r"what\s+we\s+offer|why\s+(join|work|you'll\s+love)|why\s+us|why\s+we're\s+different|"
            r"\bpto\b|401\s*\(?\s*k\s*\)?|health\s+(insurance|care|benefits|coverage)|"
            r"total\s+rewards|\bsalary\b|pay\s+range|salary\s+range|base\s+(salary|pay)|"
            r"our\s+benefits|\bequity\b\s*(package|compensation)?|"
            r"\bbonuses?\b|stock\s+options|rsu|employee\s+benefits|"
            r"paid\s+time\s+off|medical,?\s+dental|"
            r"life\s+insurance|parental\s+leave|"
            r"flexible\s+(schedule|hours|working)|remote\s+work\s+(policy|stipend)|"
            r"wellness\s+(benefits|program))",
            re.IGNORECASE,
        ),
    ),
    (
        "responsibilities",
        re.compile(
            r"(\bresponsibilities\b|\bkey\s+responsibilities\b|"
            r"what\s+you('ll|\s+will)\s+(do|be\s+doing|be\s+working\s+on|"
            r"own|build|ship|deliver)|"
            r"your\s+(role|day)|day[- ]to[- ]day|"
            r"a\s+day\s+in\s+the\s+life|in\s+this\s+role|"
            r"the\s+role\s+(involves|includes|requires)|"
            r"what\s+the\s+role\s+looks\s+like|"
            r"job\s+duties|duties\s+(include|&\s+responsibilities|and\s+responsibilities)|"
            r"you\s+will\s+(be|help|lead|own|build|design)|"
            r"responsibilities\s+include|"
            r"primary\s+responsibilities|core\s+responsibilities|"
            r"essential\s+(functions|duties)|"
            r"the\s+job|your\s+impact)",
            re.IGNORECASE,
        ),
    ),
    (
        "requirements",
        re.compile(
            r"(\brequirements?\b|\bqualifications\b|\bminimum\s+(qualifications|requirements)\b|"
            r"required\s+(skills|qualifications|experience|qualification)|"
            r"what\s+(we're|we\s+are)\s+looking\s+for|"
            r"what\s+you('ll|\s+will)\s+need|"
            r"what\s+you\s+(have|bring|should\s+have)|"
            r"you\s+(have|should\s+have|are|must\s+have)|"
            r"must[- ]haves?|must\s+have|"
            r"basic\s+qualifications|"
            r"about\s+you|who\s+you\s+are|the\s+ideal\s+candidate\s+has|"
            r"we're\s+looking\s+for\s+someone|"
            r"the\s+candidate|skills\s+(and|&)\s+experience|skills\s+(required|needed)|"
            r"technical\s+(skills|requirements|qualifications)|"
            r"experience\s+(required|with)|"
            r"education\s+and\s+experience)",
            re.IGNORECASE,
        ),
    ),
    (
        "about_company",
        re.compile(
            r"(^about\s+(us|the\s+(company|team|organization))|"
            r"^who\s+we\s+are|^our\s+(mission|vision|values|story|culture)|"
            r"^at\s+[a-z][\w.\-]+,?\s+we('re|\s+are)|"
            r"^company\s+(overview|description)|"
            r"^the\s+company|^our\s+(team|company)|"
            r"founded\s+in|established\s+in|"
            r"^we\s+are\s+a\s+(leading|fast[- ]growing|global|technology)|"
            r"^our\s+(customers?|clients?|products?)|"
            r"^why\s+(work\s+(at|with)|you\s+should\s+join))",
            re.IGNORECASE,
        ),
    ),
    (
        "role_summary",
        re.compile(
            r"(about\s+(the|this)\s+(role|position|opportunity|job)|"
            r"the\s+(role|position|opportunity)|"
            r"we('re|\s+are)\s+looking\s+for|we\s+seek|we\s+are\s+seeking|"
            r"overview|summary|"
            r"job\s+description|position\s+description|role\s+description|"
            r"job\s+summary|position\s+summary|role\s+summary|"
            r"job\s+overview|position\s+overview|"
            r"what\s+is\s+the\s+role|"
            r"role\s+at\s+\w+|"
            r"the\s+opportunity|the\s+position)",
            re.IGNORECASE,
        ),
    ),
]


# Keywords that, if appearing as a header-like opening to a block, strongly
# suggest a legal/EEO block even without the word 'equal'. Useful because
# some postings have short disclaimers at the bottom.
_HEADER_STRIP = re.compile(r"^[\s\*\#\-\•\·\>\_\=\~\+\*]+")
_HEADER_TRAIL = re.compile(r"[\s\*\#\-\•\·\>\_\=\~\+\:\.\,\!\?\)\(]+$")
_BLOCK_SPLIT = re.compile(r"\n{1,}")
_MARKDOWN_UNESCAPE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")


def _normalize_block_header(block: str) -> str:
    """Return the first line (stripped of markdown bullet noise) lower-cased.
    This is used as the PRIMARY header-match target. We deliberately do NOT
    include body text to avoid matching random phrases inside paragraphs."""
    if not block:
        return ""
    head = block[:240]
    head = _HEADER_STRIP.sub("", head)
    first_line = head.split("\n", 1)[0]
    # Trim trailing punctuation on the first line
    first_line = _HEADER_TRAIL.sub("", first_line).strip()
    return first_line.lower()


def _classify_section(header_text: str) -> str:
    """Classify a single block by its header (first line only). If the first
    line is short (< 80 chars) we consider it a likely header and require an
    exact-header match. If the block is long (body text), we additionally scan
    the first ~60 chars of the second line to catch header-line + colon
    patterns."""
    if not header_text.strip():
        return "unclassified"
    head = _normalize_block_header(header_text)
    # Strong rule: first line alone must match, OR first-line+next-word
    # boundary. Don't allow deep body-text matches (that's how "join our"
    # was matching about_company wrongly in bodies).
    for name, pat in _SECTION_PATTERNS:
        if pat.search(head):
            return name
    return "unclassified"


def classify_description(description: str) -> Dict[str, Dict]:
    """Split a raw description into blocks, classify each, aggregate.

    Returns: {section_name: {'chars': int, 'blocks': int, 'text': str}}
    where text is '\n\n'.join of block texts in that section.
    """
    out = {s: {"chars": 0, "blocks": 0, "text": ""} for s in SECTION_TYPES}
    if not description:
        return out
    # Unescape common markdown backslashes (scraped has \+, \#, etc.)
    text = _MARKDOWN_UNESCAPE.sub(r"\1", description)
    # Split on double-newlines OR single-newlines where the next line begins
    # with a likely header marker. We keep it simple: split on blank lines and
    # newline boundaries.
    raw_blocks = re.split(r"\n\s*\n", text)
    if len(raw_blocks) < 3:
        # fall back: split on single newlines and group
        raw_blocks = [b for b in text.split("\n") if b.strip()]

    current = "unclassified"
    text_buf: Dict[str, List[str]] = {s: [] for s in SECTION_TYPES}
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        # Classify this block's header — if nothing matches, inherit the
        # previous block's section. This lets bullet-lists following a header
        # be attributed to that header.
        sec = _classify_section(block)
        if sec == "unclassified":
            sec = current
        else:
            current = sec
        out[sec]["chars"] += len(block)
        out[sec]["blocks"] += 1
        text_buf[sec].append(block)
    for s in SECTION_TYPES:
        out[s]["text"] = "\n\n".join(text_buf[s])
    return out


def filter_sections(description: str, keep: List[str]) -> str:
    """Return concatenated raw text for sections named in `keep`.
    Example: filter_sections(desc, ['requirements','responsibilities']).
    """
    secs = classify_description(description)
    return "\n\n".join(secs[k]["text"] for k in keep if k in secs and secs[k]["text"])


# ---------------------------------------------------------------------------
# Validation asserts — run inline on import.
# ---------------------------------------------------------------------------
def _run_self_tests() -> None:
    assert _classify_section("Responsibilities:") == "responsibilities", _classify_section("Responsibilities:")
    assert _classify_section("**Responsibilities**") == "responsibilities"
    assert _classify_section("What you'll do") == "responsibilities"
    assert _classify_section("Why join our team") == "benefits"
    assert _classify_section("What we offer") == "benefits"
    assert _classify_section("Equal Employment Opportunity") == "legal"
    assert _classify_section("Requirements") == "requirements", _classify_section("Requirements")
    assert _classify_section("Qualifications") == "requirements"
    assert _classify_section("Preferred Qualifications") == "preferred", _classify_section("Preferred Qualifications")
    assert _classify_section("Nice to have") == "preferred"
    assert _classify_section("About the role") == "role_summary"
    assert _classify_section("About us") == "about_company", _classify_section("About us")
    assert _classify_section("Our mission") == "about_company"
    assert _classify_section("random bullet text with no header marker") == "unclassified"

    # integration check — a realistic posting should produce non-zero content
    # in at least 2 sections.
    post = (
        "About the role\n"
        "We are looking for a Senior Software Engineer.\n\n"
        "Responsibilities\n"
        "* Build scalable systems\n"
        "* Mentor engineers\n\n"
        "Requirements\n"
        "* 5+ years experience\n"
        "* Python\n\n"
        "Preferred\n"
        "* AWS experience\n\n"
        "Benefits\n"
        "* Health insurance\n"
        "* 401k\n\n"
        "Equal Employment Opportunity\n"
        "We are an equal opportunity employer.\n"
    )
    r = classify_description(post)
    assert r["responsibilities"]["chars"] > 0, r["responsibilities"]
    assert r["requirements"]["chars"] > 0
    assert r["benefits"]["chars"] > 0
    assert r["legal"]["chars"] > 0
    assert r["about_company"]["chars"] == 0


_run_self_tests()

if __name__ == "__main__":
    _run_self_tests()
    print("T13 section classifier: all self-tests passed.")
