"""T33 simple-regex alternative section classifier.

Intentionally simpler than T13_section_classifier. Designed as a sensitivity
probe: detect "requirements-ish" headers with straightforward regexes and
attribute a line + its immediate bullet/plain-text successors until another
header-ish line appears. Goal: if a T13 finding flips under this classifier,
the finding is classifier-sensitive.

Public API
----------
``simple_classify(text: str) -> dict[str, int]``
    Returns char counts for the canonical labels {requirements, responsibilities,
    benefits, other, total}. The ``other`` bucket absorbs everything that is
    not attributed to one of the named sections (summary, preferred, about,
    legal, unclassified in T13 parlance).

Design notes
------------
- Pattern set is deliberately narrow: only the most common header stems for
  each of the three labels are matched. If a posting uses unusual headers we
  lose recall — that is the point of a sensitivity classifier.
- Header detection is per-line, after stripping markdown bold and bullet
  glyphs. Headers may have trailing ``:``; otherwise require them to stand
  alone on a line.
- No inline-rescue or fused-header detection. That is T13's responsibility
  and is exactly the kind of pattern we want to contrast against.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


SIMPLE_LABELS: Tuple[str, ...] = (
    "requirements",
    "responsibilities",
    "benefits",
    "other",
)


_BULLET = re.compile(r"^\s*(?:[\*\-•–]+\s+)+")
_BOLD_WRAP = re.compile(r"^\s*\*\*\s*([^\*\n]{1,80})\s*\*\*\s*:?\s*$")
_BOLD_INLINE = re.compile(r"\*\*([^\*\n]{1,80})\*\*")
_ESC = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")


# ---- simple header regex set (case-insensitive, anchored to full line) ----
_SIMPLE_HEADERS: List[Tuple[str, re.Pattern[str]]] = [
    ("requirements", re.compile(r"^(?:requirements?|qualifications?|required\s+(?:qualifications?|skills?|experience)|minimum\s+qualifications?|basic\s+qualifications?|must[\s-]haves?|what\s+you(?:['’]?ll|[\s\-]will)?\s+(?:need|bring|have)|what\s+we['’\s]*re?\s+looking\s+for|who\s+you\s+are|about\s+you|education\s+requirements?|skills?\s*(?:and|&)\s*qualifications?)\s*:?\s*$", re.IGNORECASE)),
    ("responsibilities", re.compile(r"^(?:responsibilities|(?:key|primary|core|main)\s+responsibilities|duties(?:\s*(?:and|&)\s*responsibilities)?|roles?\s*(?:and|&)\s*responsibilities|essential\s+(?:functions?|duties|responsibilities)|what\s+you(?:['’]?ll|[\s\-]will)?\s+(?:do|be\s+doing|own)|your\s+(?:role|mission|impact)|a\s+day\s+in\s+the\s+life)\s*:?\s*$", re.IGNORECASE)),
    ("benefits", re.compile(r"^(?:benefits?|perks?|benefits?\s*(?:and|&)\s*perks?|perks?\s*(?:and|&)\s*benefits?|compensation(?:\s*(?:and|&)\s*benefits?)?|pay\s+(?:range|transparency|rate|details)|salary\s+(?:range|details)|total\s+rewards?|what\s+we\s+offer|why\s+(?:work|join)\s+(?:us|with\s+us|here)|our\s+benefits?)\s*:?\s*$", re.IGNORECASE)),
]


def _normalize_line(line: str) -> str:
    s = line.strip()
    s = _ESC.sub(r"\1", s)
    m = _BOLD_WRAP.match(s)
    if m:
        s = m.group(1).strip()
    else:
        s = _BOLD_INLINE.sub(r"\1", s)
    s = _BULLET.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(":—–- ").strip()
    # Drop any trailing colon-inline content so "Requirements: 5+ yrs..." collapses to header only.
    s = re.sub(r"[:\|]\s*.*$", "", s).strip()
    return s


def simple_classify(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in SIMPLE_LABELS}
    if not text:
        counts["total"] = 0
        return counts
    n = len(text)
    current = "other"
    line_start = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = line_start + len(raw_line)
        stripped = raw_line.strip()
        if stripped:
            norm = _normalize_line(raw_line)
            new_label: str | None = None
            if 0 < len(norm) <= 80:
                for label, pat in _SIMPLE_HEADERS:
                    if pat.match(norm):
                        new_label = label
                        break
            if new_label is not None:
                current = new_label
        counts[current] += line_end - line_start
        line_start = line_end
    counts["total"] = sum(counts[k] for k in SIMPLE_LABELS)
    return counts


def _run_tests() -> None:
    # 1. Simple requirements + responsibilities + benefits
    t = (
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
    c = simple_classify(t)
    assert c["requirements"] > 0, c
    assert c["responsibilities"] > 0, c
    assert c["benefits"] > 0, c
    assert c["total"] == len(t), (c, len(t))

    # 2. Plain "Qualifications:" header (asaniczka)
    t2 = (
        "Job Description\n"
        "We need a developer.\n"
        "Qualifications:\n"
        "Python 5+\n"
    )
    c2 = simple_classify(t2)
    assert c2["requirements"] > 0, c2

    # 3. No headers -> everything other
    t3 = "Just some text without any headers. " * 20
    c3 = simple_classify(t3)
    assert c3["other"] == c3["total"]

    # 4. Totals add up
    for tt in (t, t2, t3):
        cc = simple_classify(tt)
        assert sum(cc[k] for k in SIMPLE_LABELS) == cc["total"] == len(tt)

    print("OK — simple classifier asserts pass.")


if __name__ == "__main__":
    _run_tests()
