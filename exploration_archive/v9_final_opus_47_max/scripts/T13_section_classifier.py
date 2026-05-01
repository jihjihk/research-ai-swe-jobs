r"""
T13 Section Classifier
======================

Regex-based section classifier for job description text.

Handles three format families observed in unified.parquet (SWE LinkedIn corpus):

  1. **scraped** (2026) — markdown bold headers (``**Requirements**``, ``**Key
     Responsibilities:**``), bullets as ``*`` at start of line, escaped punctuation
     (``5\+``, ``\&``). Blank lines between sections.

  2. **kaggle_asaniczka** (2024) — plain capitalized headers on their own line
     (``Responsibilities``, ``Required Skills``) or a header followed by ``|``
     pipe before content (``Requirements: | ...``). Occasional run-together
     text fragments.

  3. **kaggle_arshkon** (2024) — colon-terminated headers (``Responsibilities:``),
     often at start of a line but sometimes mid-paragraph (``Qualifications:``
     followed by inline content). Occasional concatenated text where the
     header is fused to the first sentence
     (``Company DescriptionNextCreator is ...``).

Public API
----------

``classify_sections(text: str) -> dict[str, int]``
    Returns a dict with character counts per canonical section label and a
    ``total`` key. Unclassified characters go to ``unclassified``. The sum of
    per-section counts equals ``total``.

``SECTION_LABELS`` — tuple of canonical labels used by the classifier.

Canonical section labels (8):
    - ``summary`` (Role summary / About the role / Position overview)
    - ``responsibilities`` (Responsibilities / What you will do / Duties / Impact)
    - ``requirements`` (Requirements / Qualifications / What you need / Skills)
    - ``preferred`` (Preferred qualifications / Nice to have / Bonus / Plus)
    - ``benefits`` (Benefits / Perks / Compensation / Pay)
    - ``about_company`` (About us / Who we are / Our story / Company)
    - ``legal`` (EEO / Equal opportunity / Legal / Affirmative / Disclosures)
    - ``unclassified`` (text before first header or in unparsed gaps)

Design notes
------------

The classifier walks the text line by line. Each line either matches one of
the header patterns (triggering a section switch) or is accumulated to the
currently active section's character count. A pre-pass normalizes markdown
bold markers and strips bullet glyphs so that header detection is uniform
across formats. Text before any header is ``unclassified`` ("lead text"); this
captures the prelude many postings have before any structured section.

For arshkon-style fused headers (``Company DescriptionNextCreator is ...``)
the classifier also applies an inline rescue pass that looks for a canonical
header phrase followed by a capital letter, and splits the header off. This
is conservative — it only splits on a curated list of phrases with high
precision.

Hygiene
-------

- Tests live at the bottom of this module and run as ``python
  T13_section_classifier.py`` (exit 0 if all asserts pass).
- The module writes no files and has no side-effects on import.

"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

SECTION_LABELS: Tuple[str, ...] = (
    "summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal",
    "unclassified",
)

# ---------------------------------------------------------------------------
# Header phrase catalog
# ---------------------------------------------------------------------------
#
# Each entry is ``(label, phrase_pattern)`` where ``phrase_pattern`` is a regex
# fragment (no anchors, no flags). The classifier assembles these into
# compiled line-level patterns.
#
# Order matters: more specific phrases first. For example, "Preferred
# Qualifications" must be checked before "Qualifications". We enforce this by
# sorting each label's entries by phrase length (descending) and by labeling
# the preferred/about/legal lists first in the search loop.
#
# Phrases are case-insensitive.

_HEADER_PHRASES: List[Tuple[str, str]] = [
    # ------------------ PREFERRED (must be checked before requirements) -----
    ("preferred", r"preferred\s+qualifications?"),
    ("preferred", r"preferred\s+(?:skills?|experiences?|requirements?)"),
    ("preferred", r"education\s+(?:and|&)\s+preferred\s+qualifications?"),
    ("preferred", r".*(?:and|&)\s+preferred\s+qualifications?"),
    ("preferred", r"nice[\s-]to[\s-]have"),
    ("preferred", r"nice\s+to\s+haves?"),
    ("preferred", r"bonus\s+points?"),
    ("preferred", r"bonus\s+(?:skills?|qualifications?)"),
    ("preferred", r"(?:it[’'`]s\s+)?(?:a\s+)?plus(?:\s+if)?"),
    ("preferred", r"desired\s+(?:skills?|qualifications?|experience|certifications?)"),
    ("preferred", r"additional\s+(?:skills?|qualifications?|considerations?)"),
    ("preferred", r"ideal\s+(?:candidate|qualifications?|experience)"),
    ("preferred", r"extra\s+credit"),
    # ------------------ REQUIREMENTS ---------------------------------------
    ("requirements", r"basic\s+qualifications?"),
    ("requirements", r"minimum\s+qualifications?"),
    ("requirements", r"minimum\s+requirements?"),
    ("requirements", r"required\s+qualifications?"),
    ("requirements", r"required\s+skills?"),
    ("requirements", r"required\s+experience"),
    ("requirements", r"required\s+(?:knowledge|knowledge\s+and\s+experiences?)"),
    ("requirements", r"key\s+(?:qualifications?|requirements?|skills?)"),
    ("requirements", r"core\s+(?:qualifications?|requirements?|skills?)"),
    ("requirements", r"qualifications?\s+(?:and|&)\s+skills?"),
    ("requirements", r"skills?\s+(?:and|&)\s+qualifications?"),
    ("requirements", r"skills?\s+(?:and|&)\s+experience"),
    ("requirements", r"experience\s+(?:and|&)\s+skills?"),
    ("requirements", r"technical\s+(?:skills?|qualifications?|requirements?)"),
    ("requirements", r"must[\s-]haves?"),
    ("requirements", r"what\s+you[’'`]?ll?\s+(?:need|bring|have)"),
    ("requirements", r"what\s+we[’'`]?re?\s+looking\s+for"),
    ("requirements", r"what\s+we\s+want"),
    ("requirements", r"who\s+you\s+are"),
    ("requirements", r"about\s+you"),
    ("requirements", r"qualifications?"),
    ("requirements", r"requirements?"),
    ("requirements", r"eligibility"),
    ("requirements", r"education(?:\s+(?:and|&)\s+(?:preferred\s+)?qualifications?)?"),
    ("requirements", r"education\s+(?:and|&)\s+experience"),
    ("requirements", r"required\s+education"),
    ("requirements", r"education\s+requirements?"),
    ("requirements", r"desired\s+certifications?"),
    # ------------------ RESPONSIBILITIES ----------------------------------
    ("responsibilities", r"key\s+responsibilities"),
    ("responsibilities", r"primary\s+responsibilities"),
    ("responsibilities", r"core\s+responsibilities"),
    ("responsibilities", r"main\s+responsibilities"),
    ("responsibilities", r"essential\s+(?:functions?|duties|responsibilities)"),
    ("responsibilities", r"job\s+(?:duties|responsibilities|functions?)"),
    ("responsibilities", r"duties\s+(?:and|&)\s+responsibilities"),
    ("responsibilities", r"roles?\s+(?:and|&)\s+responsibilities"),
    ("responsibilities", r"responsibilities\s+(?:and|&)\s+(?:impact|duties)"),
    ("responsibilities", r"what\s+you[’'`]?ll?\s+(?:do|be\s+doing|own|work\s+on)"),
    ("responsibilities", r"what\s+you\s+will\s+(?:do|be\s+doing)"),
    ("responsibilities", r"what\s+you\s+will\s+be\s+responsible\s+for"),
    ("responsibilities", r"what\s+the\s+role\s+(?:does|involves|looks\s+like)"),
    ("responsibilities", r"a\s+day\s+in\s+the\s+life"),
    ("responsibilities", r"day[\s-]to[\s-]day"),
    ("responsibilities", r"your\s+(?:role|mission|impact|day|responsibilities)"),
    ("responsibilities", r"the\s+(?:impact\s+you[’'`]?ll?\s+make|opportunity|impact)"),
    ("responsibilities", r"impact\s+you[’'`]?ll?\s+(?:make|have)"),
    ("responsibilities", r"you\s+will"),
    ("responsibilities", r"responsibilities"),
    ("responsibilities", r"duties"),
    # ------------------ SUMMARY -------------------------------------------
    ("summary", r"role\s+(?:overview|description|summary)"),
    ("summary", r"job\s+(?:overview|summary|description)"),
    ("summary", r"position\s+(?:overview|summary|description)"),
    ("summary", r"about\s+(?:the|this)\s+role"),
    ("summary", r"about\s+(?:the|this)\s+(?:position|opportunity|job)"),
    ("summary", r"opportunity"),
    ("summary", r"the\s+opportunity"),
    ("summary", r"overview"),
    ("summary", r"summary"),
    ("summary", r"description"),
    # ------------------ BENEFITS ------------------------------------------
    ("benefits", r"benefits?\s+(?:and|&)\s+(?:perks|compensation|pay)"),
    ("benefits", r"compensation\s+(?:and|&)\s+benefits?"),
    ("benefits", r"perks?\s+(?:and|&)\s+benefits?"),
    ("benefits", r"what\s+we\s+offer"),
    ("benefits", r"what\s+you[’'`]?ll?\s+(?:get|enjoy)"),
    ("benefits", r"why\s+(?:work|join)\s+(?:us|with\s+us|here)"),
    ("benefits", r"why\s+you[’'`]?ll?\s+love\s+(?:it|working)"),
    ("benefits", r"our\s+benefits?"),
    ("benefits", r"total\s+rewards?"),
    ("benefits", r"pay\s+(?:range|transparency|rate|details)"),
    ("benefits", r"salary\s+(?:range|details)"),
    ("benefits", r"compensation\s+(?:range|details)?"),
    ("benefits", r"benefits?"),
    ("benefits", r"perks?"),
    # ------------------ ABOUT COMPANY -------------------------------------
    ("about_company", r"about\s+(?:us|the\s+company|our\s+company)"),
    ("about_company", r"who\s+we\s+are"),
    ("about_company", r"our\s+(?:story|mission|vision|values|team|company)"),
    ("about_company", r"the\s+company"),
    ("about_company", r"company\s+(?:overview|description|profile|background)"),
    ("about_company", r"about\s+[a-z0-9][\w&.,' \-]{2,30}(?=[:\n])"),  # e.g. "About Acme Corp:"
    # ------------------ LEGAL ---------------------------------------------
    ("legal", r"equal\s+(?:employment\s+)?opportunity(?:\s+employer|\s+statement)?"),
    ("legal", r"eeo\s+(?:statement|policy|policy\s+statement)?"),
    ("legal", r"(?:eeo|e\.e\.o\.)"),
    ("legal", r"affirmative\s+action"),
    ("legal", r"non[\s-]?discrimination"),
    ("legal", r"diversity\s+(?:and|&)\s+inclusion\s+statement"),
    ("legal", r"pay\s+transparency\s+(?:notice|statement)"),
    ("legal", r"(?:california|ccpa)\s+(?:residents?|privacy)"),
    ("legal", r"accommodation(?:s)?\s+(?:request|statement|notice)?"),
    ("legal", r"reasonable\s+accommodation"),
    ("legal", r"disability\s+accommodation"),
    ("legal", r"disclosure(?:\s+statement)?"),
    ("legal", r"export\s+control"),
    ("legal", r"legal\s+(?:notice|disclosure|disclaimer)"),
    ("legal", r"e[\-\s]?verify"),
    ("legal", r"background\s+check"),
    ("legal", r"employment\s+authorization"),
    ("legal", r"visa\s+(?:sponsorship|status)"),
    ("legal", r"notice\s+to\s+(?:applicants|candidates)"),
]


# ---------------------------------------------------------------------------
# Precompiled patterns
# ---------------------------------------------------------------------------
#
# Line-level detection: the entire (stripped + normalized) line must be the
# header. We allow trailing ``:``, ``:—``, or empty. Leading bullet glyphs and
# asterisk-bold markers are stripped by the normalizer before matching.

_LINE_HEADER_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        label,
        re.compile(
            rf"^{phrase}(?:\s*[:\-—–]\s*.*)?$",
            re.IGNORECASE,
        ),
    )
    for label, phrase in _HEADER_PHRASES
]

# Also allow a short-header-contains match: if the normalized line is short
# (< 80 chars) AND contains a known header phrase as a contiguous substring,
# attribute to that label. This handles compound headers like
# "Role Overview And Core Responsibilities" and "Required Knowledge And
# Experiences". We enforce that the phrase match covers a substantial
# fraction of the line (>=50%) to avoid false positives on mid-sentence use.
_CONTAINS_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        label,
        re.compile(
            rf"\b{phrase}\b",
            re.IGNORECASE,
        ),
    )
    for label, phrase in _HEADER_PHRASES
]


# Inline rescue pass: looks for header phrase immediately followed by a
# capital letter or digit (arshkon fused-header style). We require the phrase
# to occur at the start of the text or right after a period/newline.
_INLINE_HEADER_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        label,
        re.compile(
            rf"(?:^|(?<=[\.\n]))\s*({phrase})(?=[A-Z0-9])",
            re.IGNORECASE,
        ),
    )
    for label, phrase in _HEADER_PHRASES
]


# Bullet / bold / asterisk stripping for the normalizer. We replace
# leading markdown bold markers (``**X**``) with their inner text, then drop
# leading bullet glyphs (``*``, ``-``, ``•``, ``–``) and normalize the line.
_LEADING_BULLET = re.compile(r"^\s*(?:[\*\-•–]+\s+)+")
_MARKDOWN_BOLD = re.compile(r"\*\*([^\*\n]{1,80})\*\*")
_WRAPPING_BOLD_LINE = re.compile(r"^\s*\*\*\s*([^\*\n]{1,80})\s*\*\*\s*:?\s*$")
_ESCAPED_PUNCT = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")


def _normalize_for_header_detection(line: str) -> str:
    """Return a version of ``line`` suitable for header regex matching.

    Does NOT mutate the original text used for character accounting.
    """
    s = line.strip()
    # Undo scraped-markdown escape on punctuation before matching.
    s = _ESCAPED_PUNCT.sub(r"\1", s)
    # If the entire line is **Header**, peel the bold markers.
    m = _WRAPPING_BOLD_LINE.match(s)
    if m:
        s = m.group(1).strip()
    else:
        # Remove inline bold markers but keep the text.
        s = _MARKDOWN_BOLD.sub(r"\1", s)
    # Drop leading bullet glyphs.
    s = _LEADING_BULLET.sub("", s)
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    # Drop trailing punctuation that decorates headers.
    s = s.rstrip(":—–- ").strip()
    # Finally drop residual colons/dashes/pipes that separate from inline content.
    s = re.sub(r"[:\|]\s*.*$", "", s).strip()
    return s


def _match_header(norm_line: str) -> str | None:
    """Return the canonical label for ``norm_line`` if it matches a header."""
    if not norm_line:
        return None
    # Guard: very long lines are unlikely to be headers. Most real headers are
    # short ( < 60 chars).
    if len(norm_line) > 80:
        return None
    # Exact-line match first (strictest, highest precision).
    for label, pat in _LINE_HEADER_PATTERNS:
        if pat.match(norm_line):
            return label
    # Compound-header fallback: if a known phrase appears as a substring and
    # covers at least 40% of the normalized line, accept it. Preferred labels
    # are checked first by virtue of ordering in ``_HEADER_PHRASES`` so
    # "Preferred Qualifications" inside "Additional Preferred Qualifications"
    # resolves to preferred, not requirements.
    for label, pat in _CONTAINS_PATTERNS:
        m = pat.search(norm_line)
        if m and (m.end() - m.start()) / max(len(norm_line), 1) >= 0.40:
            return label
    return None


def classify_sections(text: str) -> Dict[str, int]:
    """Classify the text into section character counts.

    Parameters
    ----------
    text : str
        Job description text, raw or LLM-cleaned.

    Returns
    -------
    dict[str, int]
        Keys: the 8 ``SECTION_LABELS`` entries plus ``total``. Values are
        non-negative integers; per-section counts sum to ``total`` which
        equals ``len(text)``.
    """
    counts: Dict[str, int] = {label: 0 for label in SECTION_LABELS}

    if not text:
        counts["total"] = 0
        return counts

    n = len(text)

    # ------------------------------------------------------------------
    # Pass 1. Collect header events as (start, end, label).
    #
    # A header event is the character span that belongs to a header. Anything
    # between two consecutive header events (or before the first one) is
    # assigned to the section introduced by the preceding event (or
    # ``unclassified`` for the lead text).
    #
    # We collect events from two sources:
    #   (a) line-level: full-line headers after normalization
    #   (b) inline rescue: fused headers like "Company DescriptionThe ..."
    # Line-level wins on overlap because it spans more characters (the
    # header line) and because the normalized-line match is higher precision.
    # ------------------------------------------------------------------

    events: List[Tuple[int, int, str]] = []

    # ---- (a) line-level headers ----
    line_start = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = line_start + len(raw_line)
        content = raw_line.strip()
        if content:
            norm = _normalize_for_header_detection(raw_line)
            hit = _match_header(norm)
            if hit is not None:
                events.append((line_start, line_end, hit))
        line_start = line_end

    # ---- (b) inline rescue for fused headers ----
    # Only apply inline rescue where the character position is NOT already
    # claimed by a line-level header.
    claimed: List[Tuple[int, int]] = [(s, e) for s, e, _ in events]

    def overlaps_claimed(s: int, e: int) -> bool:
        for cs, ce in claimed:
            if s < ce and e > cs:
                return True
        return False

    for label, pat in _INLINE_HEADER_PATTERNS:
        for m in pat.finditer(text):
            # Group 1 is the header phrase without surrounding whitespace.
            hs, he = m.span(1)
            if overlaps_claimed(hs, he):
                continue
            events.append((hs, he, label))
            claimed.append((hs, he))

    # Sort events by start position and resolve overlaps (prefer earlier-starting,
    # then longer span).
    events.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    dedup: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, label in events:
        if s >= last_end:
            dedup.append((s, e, label))
            last_end = e
    events = dedup

    # ------------------------------------------------------------------
    # Pass 2. Walk the text and attribute characters.
    # ------------------------------------------------------------------
    current = "unclassified"
    cursor = 0
    for s, e, label in events:
        if s > cursor:
            counts[current] += s - cursor
        current = label
        counts[current] += e - s
        cursor = e
    if cursor < n:
        counts[current] += n - cursor

    counts["total"] = sum(counts[k] for k in SECTION_LABELS)
    return counts


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_tests() -> None:
    # 1. Simple scraped markdown case
    t1 = (
        "**Role Overview**\n"
        "We build things.\n"
        "\n"
        "**Responsibilities**\n"
        "* Design stuff\n"
        "* Ship things\n"
        "\n"
        "**Requirements**\n"
        "* 5 years of experience\n"
        "* Python\n"
        "\n"
        "**Benefits**\n"
        "Health and dental.\n"
    )
    c1 = classify_sections(t1)
    assert c1["responsibilities"] > 0, c1
    assert c1["requirements"] > 0, c1
    assert c1["benefits"] > 0, c1
    assert c1["summary"] > 0, c1
    assert c1["total"] == len(t1), (c1, len(t1))
    assert sum(c1[k] for k in SECTION_LABELS) == c1["total"]

    # 2. Asaniczka-style plain headers + pipe separators
    t2 = (
        "Job Description\n"
        "We need a developer.\n"
        "Responsibilities:\n"
        "Write code\n"
        "Test code\n"
        "Required Skills:\n"
        "Python, SQL\n"
        "Education & Preferred Qualifications\n"
        "BS in CS\n"
    )
    c2 = classify_sections(t2)
    assert c2["responsibilities"] > 0, c2
    assert c2["requirements"] > 0, c2
    assert c2["preferred"] > 0, c2  # Education & Preferred Qualifications
    assert c2["summary"] > 0, c2  # "Job Description"

    # 3. Arshkon-style fused header: inline rescue
    t3 = (
        "Company DescriptionNextCreator is a content creation startup.\n"
        "Role DescriptionThis is a remote unpaid internship role.\n"
        "QualificationsStrong knowledge of computer science.\n"
    )
    c3 = classify_sections(t3)
    assert c3["requirements"] > 0, c3  # "Qualifications" fused case
    assert c3["summary"] > 0, c3  # "Role Description" fused case
    assert c3["about_company"] > 0 or c3["summary"] > 0, c3

    # 4. Unclassified when no headers
    t4 = "Just some text without any headers. " * 20
    c4 = classify_sections(t4)
    assert c4["unclassified"] == c4["total"]

    # 5. Preferred vs Requirements precedence
    t5 = (
        "Qualifications:\n"
        "Python, 5 years.\n"
        "Preferred Qualifications:\n"
        "AWS, Docker.\n"
    )
    c5 = classify_sections(t5)
    assert c5["requirements"] > 0, c5
    assert c5["preferred"] > 0, c5
    # Preferred section should NOT collapse into requirements.
    assert c5["preferred"] >= len("Preferred Qualifications:\n")

    # 6. Legal section
    t6 = (
        "Responsibilities:\nDo things.\n"
        "Equal Opportunity Employer\n"
        "We consider all qualified applicants.\n"
    )
    c6 = classify_sections(t6)
    assert c6["legal"] > 0, c6

    # 7. Escaped punctuation in scraped text
    t7 = "**Requirements**\n* 5\\+ years\n* Python\n"
    c7 = classify_sections(t7)
    assert c7["requirements"] > 0, c7
    assert c7["summary"] == 0, c7

    # 8. Benefits / Pay Range
    t8 = "**Pay Range**\n$100k - $200k\n**About Us**\nWe are cool.\n"
    c8 = classify_sections(t8)
    assert c8["benefits"] > 0, c8
    assert c8["about_company"] > 0, c8

    # 9. Lead text goes to unclassified
    t9 = (
        "We are a growing startup.\n"
        "Responsibilities:\nDo things.\n"
    )
    c9 = classify_sections(t9)
    assert c9["unclassified"] > 0, c9
    assert c9["responsibilities"] > 0, c9
    # The lead text length should match
    lead = "We are a growing startup.\n"
    assert c9["unclassified"] == len(lead), (c9, len(lead))

    # 10. Totals always add up
    for t in (t1, t2, t3, t4, t5, t6, t7, t8, t9):
        c = classify_sections(t)
        assert sum(c[k] for k in SECTION_LABELS) == c["total"] == len(t), (c, len(t))

    print("OK — all T13 section classifier asserts pass.")


if __name__ == "__main__":
    _run_tests()
