"""Section classifier for job descriptions.

Handles both HTML-stripped kaggle text (flat prose) and scraped markdown text
(bold headers + bullets). Returns per-posting character counts per section.

Sections:
    summary      - Role summary / About the role
    responsibilities - Responsibilities / What you'll do
    requirements - Requirements / Qualifications / Must have
    preferred    - Preferred / Nice-to-have / Bonus
    benefits     - Benefits / Perks / Compensation / Salary / 401k / PTO
    about_company - About the company
    legal        - EEO / Equal opportunity / Legal / Sponsorship
    unclassified - Anything not matched

Approach:
    1. Detect explicit section headers (markdown bold **Header** or ALL-CAPS
       or "Header:" patterns). Split text by header positions.
    2. For each chunk, classify by matching the preceding header to a
       canonical section label using a regex vocabulary.
    3. If no headers found (e.g. arshkon flat text), fall back to
       sentence-level classification by keyword density.

Public API:
    classify_sections(text: str) -> Dict[str, int]
        Returns {section_name: char_count}.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

SECTIONS = [
    "summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal",
    "unclassified",
]

# Canonical header vocabulary — regex alternatives per section, case-insensitive.
_HEADER_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("preferred", re.compile(
        r"(preferred|nice\s*to\s*have|bonus|plus|pluses|a\s*plus|good\s*to\s*have|"
        r"desired\s*(skills|qualifications)?|not\s*required|additional\s*qualifications|extra\s*credit)",
        re.I)),
    ("requirements", re.compile(
        r"(requirements?|qualifications|minimum\s*qualifications|basic\s*qualifications|"
        r"required\s*(skills|experience|qualifications)?|must\s*have|what\s*(you'?ll|you)\s*(need|bring)|"
        r"what\s*we'?re\s*looking\s*for|skills?\s*(and|&)?\s*(experience|qualifications)?|"
        r"who\s*you\s*are|your\s*(qualifications|experience|background|profile))",
        re.I)),
    ("responsibilities", re.compile(
        r"(responsibilities|duties|key\s*responsibilities|what\s*you'?ll\s*do|"
        r"the\s*role|role\s*overview|role\s*description|day\s*to\s*day|"
        r"what\s*you\s*will\s*do|job\s*description|job\s*duties|"
        r"your\s*(responsibilities|role|impact|mission)|in\s*this\s*role)",
        re.I)),
    ("benefits", re.compile(
        r"(benefits|perks|compensation|salary|pay|what\s*we\s*offer|"
        r"we\s*offer|why\s*(join|work)|our\s*benefits|total\s*rewards|"
        r"health\s*(benefits|insurance)|401\s*k|pto|paid\s*time\s*off|"
        r"wellness|retirement|equity|bonus\s*structure)",
        re.I)),
    ("about_company", re.compile(
        r"(about\s*(us|the\s*company|our\s*company|our\s*team|\w+)|"
        r"company\s*(overview|description|background|profile)|who\s*we\s*are|"
        r"our\s*(mission|values|culture|story|vision))",
        re.I)),
    ("legal", re.compile(
        r"(equal\s*(employment\s*)?opportunity|eeo|eoe|affirmative\s*action|"
        r"diversity|inclusion|sponsorship|visa|e-?verify|background\s*check|"
        r"americans?\s*with\s*disabilities|ada|gdpr|privacy\s*notice|"
        r"ccpa|accommodation|legal\s*notice|disclaimer)",
        re.I)),
    ("summary", re.compile(
        r"(summary|overview|introduction|about\s*this\s*(role|position|job)|"
        r"position\s*(overview|summary)|role\s*summary|job\s*summary|"
        r"we\s*are\s*looking\s*for|we\s*are\s*hiring|we'?re\s*hiring)",
        re.I)),
]

# For fallback keyword-density classification of sentences.
_KEYWORDS: Dict[str, List[str]] = {
    "responsibilities": [
        "responsible for", "you will", "you'll ", "will be", "design and",
        "develop and", "build and", "maintain", "collaborate with", "own the",
        "drive the", "lead the", "deliver", "implement",
    ],
    "requirements": [
        "years of experience", "years experience", "yoe", "bachelor", "master",
        "phd", "degree in", "proficiency in", "proficient in", "experience with",
        "experience in", "knowledge of", "familiarity with", "must have",
        "required", "proven", "strong understanding",
    ],
    "preferred": [
        "nice to have", "bonus", "a plus", "preferred", "good to have",
        "additional", "desirable",
    ],
    "benefits": [
        "401k", "401(k)", "pto", "paid time off", "medical", "dental", "vision",
        "health insurance", "equity", "stock options", "rsus", "salary range",
        "compensation", "bonus", "wellness", "parental leave", "retirement",
        "flexible hours", "remote work", "work from home", "per year", "$",
    ],
    "about_company": [
        "our mission", "our team", "our company", "founded in", "headquartered",
        "our culture", "our values", "our vision", "customers", "millions of",
        "billions of",
    ],
    "legal": [
        "equal opportunity", "equal employment", "affirmative action", "eeo",
        "visa sponsor", "e-verify", "americans with disabilities", "background check",
        "without regard to", "race", "religion", "gender", "sexual orientation",
        "protected veteran", "disability status",
    ],
}

# Heuristic header detection.
# Matches markdown bold: **Header** — allow anywhere (scraped JDs rarely use
# real line breaks; headers sit inline after prose).
_MD_HEADER = re.compile(r"(?:\*\*|__)\s*([^\n*]{2,80}?)\s*(?:\*\*|__)")
# Matches ALL-CAPS line headers (>= 3 chars, mostly uppercase, end-of-line).
_CAPS_HEADER = re.compile(r"(?:^|\n)\s*([A-Z][A-Z \t/&,'-]{2,60}[A-Z])\s*[:|]?\s*(?=\n|$)", re.M)
# Matches "Header:" inline (Title Case, 2-6 words, followed by colon then content).
_COLON_HEADER = re.compile(
    r"(?:^|\n|\. )\s*([A-Z][A-Za-z][A-Za-z &/'-]{2,50}?)\s*:\s+",
    re.M,
)


def _classify_header(header_text: str) -> str:
    """Map a header string to a canonical section name."""
    h = header_text.strip().lower()
    for name, pat in _HEADER_PATTERNS:
        if pat.search(h):
            return name
    return "unclassified"


def _find_header_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, canonical_section) for detected headers.

    Only retains headers whose text maps to a known canonical section.
    """
    spans: List[Tuple[int, int, str]] = []
    for pat in (_MD_HEADER, _CAPS_HEADER, _COLON_HEADER):
        for m in pat.finditer(text):
            canon = _classify_header(m.group(1))
            if canon != "unclassified":
                spans.append((m.start(), m.end(), canon))
    spans.sort(key=lambda x: x[0])
    # Deduplicate overlapping: keep earliest-start longest match.
    dedup: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, name in spans:
        if s >= last_end:
            dedup.append((s, e, name))
            last_end = e
    return dedup


def _classify_sentence(sent: str) -> str:
    """Score a sentence against keyword categories; return best section."""
    s = sent.lower()
    best = ("unclassified", 0)
    for name, kws in _KEYWORDS.items():
        score = sum(1 for kw in kws if kw in s)
        if score > best[1]:
            best = (name, score)
    return best[0] if best[1] > 0 else "unclassified"


def classify_sections(text: str) -> Dict[str, int]:
    """Return character counts per section for a posting."""
    result = {s: 0 for s in SECTIONS}
    if not text:
        return result
    for name, chunk in _iter_section_chunks(text):
        result[name] += len(chunk)
    return result


def extract_sections(text: str) -> Dict[str, str]:
    """Return concatenated text per section for a posting.

    Same algorithm as classify_sections, but collects substrings instead of
    character counts. T12 uses this to run term analyses on core content only.
    """
    result: Dict[str, str] = {s: "" for s in SECTIONS}
    if not text:
        return result
    buckets: Dict[str, list] = {s: [] for s in SECTIONS}
    for name, chunk in _iter_section_chunks(text):
        if chunk:
            buckets[name].append(chunk)
    for s in SECTIONS:
        result[s] = "\n".join(buckets[s])
    return result


def _iter_section_chunks(text: str):
    """Yield (section_name, chunk_text) tuples for the classifier.

    Shared between classify_sections and extract_sections to keep the
    splitting/fallback logic in one place.
    """
    total = len(text)
    headers = _find_header_spans(text)

    if headers:
        if headers[0][0] > 20:
            yield ("summary", text[: headers[0][0]])
        for i, (s, e, name) in enumerate(headers):
            chunk_start = e
            chunk_end = headers[i + 1][0] if i + 1 < len(headers) else total
            if chunk_end > chunk_start:
                yield (name, text[chunk_start:chunk_end])
        return

    # Fallback: sentence-level keyword classification for flat prose.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for sent in sentences:
        if not sent.strip():
            continue
        name = _classify_sentence(sent)
        yield (name, sent)


if __name__ == "__main__":
    # Small smoke test.
    sample_md = (
        "We are looking for an engineer to join our team.\n"
        "**Responsibilities**\n"
        "* Build services\n"
        "* Design systems\n"
        "**Requirements**\n"
        "* 5 years of experience in Python\n"
        "* Bachelor degree\n"
        "**Nice to Have**\n"
        "* Rust experience\n"
        "**Benefits**\n"
        "* 401k, dental, vision\n"
    )
    print(classify_sections(sample_md))

    sample_flat = (
        "Responsible for building scalable services. You will collaborate with "
        "engineers and drive the roadmap. Requirements include 5 years of "
        "experience with Python, bachelor degree in computer science. "
        "We offer 401k, medical, dental, paid time off. Equal opportunity employer."
    )
    print(classify_sections(sample_flat))
