#!/usr/bin/env python3
"""T13 linguistic and structural evolution analysis.

Memory posture:
- Reads the shared cleaned-text artifact in Arrow batches.
- Uses DuckDB only with a 4GB memory limit and one thread.
- Does not materialize data/unified.parquet wholesale.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import textstat


ROOT = Path(__file__).resolve().parents[2]
SHARED_TEXT = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
UNIFIED = ROOT / "data/unified.parquet"
TABLE_DIR = ROOT / "exploration/tables/T13"
FIG_DIR = ROOT / "exploration/figures/T13"

SECTION_ARTIFACT = TABLE_DIR / "section_text_by_uid.parquet"

SECTION_TYPES = [
    "role_summary",
    "responsibilities",
    "requirements",
    "preferred",
    "benefits",
    "about_company",
    "legal_eeo",
    "unclassified",
]

SECTION_LABELS = {
    "role_summary": "Role summary",
    "responsibilities": "Responsibilities",
    "requirements": "Requirements",
    "preferred": "Preferred",
    "benefits": "Benefits",
    "about_company": "About company",
    "legal_eeo": "Legal/EEO",
    "unclassified": "Unclassified",
}


HEADING_PATTERNS: Sequence[Tuple[str, Sequence[re.Pattern[str]]]] = [
    (
        "legal_eeo",
        [
            re.compile(r"\b(equal opportunity|eeo|eoe|nondiscrimination|non-discrimination)\b", re.I),
            re.compile(r"\b(accommodation|reasonable accommodation|background check|drug screen)\b", re.I),
            re.compile(r"\b(work authorization|sponsorship|visa sponsorship|right to work)\b", re.I),
            re.compile(r"\b(pay transparency|privacy notice|applicant privacy)\b", re.I),
        ],
    ),
    (
        "benefits",
        [
            re.compile(r"\b(benefits?|perks?|compensation|salary|pay range|base pay|total rewards)\b", re.I),
            re.compile(r"\b(what we offer|rewards package|employee rewards|paid time off)\b", re.I),
        ],
    ),
    (
        "preferred",
        [
            re.compile(r"\b(preferred|nice[- ]?to[- ]?have|bonus points?|desired|plus|extra credit)\b", re.I),
            re.compile(r"\b(preferred qualifications|preferred skills|preferred experience)\b", re.I),
        ],
    ),
    (
        "requirements",
        [
            re.compile(r"\b(requirements?|qualifications?|minimum qualifications|basic qualifications)\b", re.I),
            re.compile(r"\b(required skills?|what you(?:'|’)ll need|what you will need|must have)\b", re.I),
            re.compile(r"\b(skills and experience|experience and skills|who you are)\b", re.I),
        ],
    ),
    (
        "responsibilities",
        [
            re.compile(r"\b(responsibilities|duties|what you(?:'|’)ll do|what you will do)\b", re.I),
            re.compile(r"\b(key responsibilities|essential functions|accountabilities|day[- ]to[- ]day)\b", re.I),
            re.compile(r"\b(the role|role responsibilities|your impact|in this role)\b", re.I),
        ],
    ),
    (
        "about_company",
        [
            re.compile(r"\b(about us|about the company|about our company|company overview)\b", re.I),
            re.compile(r"\b(who we are|our mission|our company|life at|our story)\b", re.I),
        ],
    ),
    (
        "role_summary",
        [
            re.compile(r"\b(about the role|role overview|position summary|job summary|overview|summary)\b", re.I),
            re.compile(r"\b(introduction|opportunity|job description)\b", re.I),
        ],
    ),
]


TONE_PATTERNS: Mapping[str, Sequence[re.Pattern[str]]] = {
    "imperative": [
        re.compile(r"\byou\s+will\b", re.I),
        re.compile(r"\byou(?:'|’)ll\b", re.I),
        re.compile(r"\bmust\b", re.I),
        re.compile(r"\bshould\b", re.I),
        re.compile(r"\brequired\s+to\b", re.I),
        re.compile(r"\bwill\s+be\s+responsible\s+for\b", re.I),
    ],
    "inclusive": [
        re.compile(r"\bwe\b", re.I),
        re.compile(r"\bour\b", re.I),
        re.compile(r"\bour\s+team\b", re.I),
        re.compile(r"\bjoin\s+(?:us|our\s+team)\b", re.I),
        re.compile(r"\byou(?:'|’)ll\s+join\b", re.I),
        re.compile(r"\bwe(?:'|’)re\b|\bwe\s+are\b", re.I),
    ],
    "passive_formal": [
        re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+"
            r"(?:[a-z]{3,}ed|required|expected|provided|included|considered|performed|managed)\b",
            re.I,
        ),
    ],
    "direct_address": [
        re.compile(r"\byou\b", re.I),
        re.compile(r"\byour\b", re.I),
        re.compile(r"\byou(?:'|’)ll\b", re.I),
    ],
    "marketing": [
        re.compile(r"\b(exciting|innovative|cutting[- ]edge|world[- ]class|fast[- ]paced)\b", re.I),
        re.compile(r"\b(passionate|mission[- ]driven|best[- ]in[- ]class|impactful|dynamic)\b", re.I),
        re.compile(r"\b(disrupt|revolutioni[sz]e|transformative|industry[- ]leading)\b", re.I),
    ],
}


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#./'-]*")


def assert_regex_edges() -> None:
    """A small TDD guard for section and tone regex behavior."""

    text = "Responsibilities\nBuild APIs\nRequirements\nPython\nBenefits\n401k"
    sections, has_headings, _ = classify_sections(text)
    assert has_headings
    assert "Build APIs" in sections["responsibilities"]
    assert "Python" in sections["requirements"]
    assert "401k" in sections["benefits"]

    text = "**Nice-to-have:** Kubernetes\nEqual Opportunity Employer\nNo discrimination"
    sections, _, _ = classify_sections(text)
    assert "Kubernetes" in sections["preferred"]
    assert "No discrimination" in sections["legal_eeo"]

    text = "Plain unheaded paragraph about a job. Another sentence."
    sections, has_headings, _ = classify_sections(text)
    assert not has_headings
    assert sections["unclassified"].startswith("Plain unheaded")

    counts = count_tone_markers("You'll join our team. You will build. This is required to ship.")
    assert counts["imperative_count"] >= 3
    assert counts["inclusive_count"] >= 2
    assert counts["direct_address_count"] >= 2


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=1")
    return con


def clean_heading_candidate(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^[#>*\-\u2022\s]+", "", s)
    s = re.sub(r"[*_`]+", "", s)
    s = s.strip()
    return s


INLINE_HEADING_RE = re.compile(
    r"^\s*(?:[#>*\-\u2022]+\s*)?(?P<head>[A-Za-z0-9&/ .,'’()\-]{2,70})\s*[:\-–]\s*(?P<body>.*)$"
)

FLATTENED_PREFIX_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    (
        "responsibilities",
        re.compile(
            r"^(?P<head>responsibilities|key responsibilities|duties|what you(?:'|’)ll do|what you will do)"
            r"\b(?P<body>.*)$",
            re.I,
        ),
    ),
    (
        "requirements",
        re.compile(
            r"^(?P<head>requirements|qualifications|minimum qualifications|basic qualifications|required skills|"
            r"must[- ]?have|what you(?:'|’)ll need|what you will need)\b(?P<body>.*)$",
            re.I,
        ),
    ),
    (
        "preferred",
        re.compile(
            r"^(?P<head>preferred qualifications|preferred skills|nice[- ]?to[- ]?have|desired|bonus points?)"
            r"\b(?P<body>.*)$",
            re.I,
        ),
    ),
    (
        "benefits",
        re.compile(r"^(?P<head>benefits|perks|compensation|salary|pay range|what we offer)\b(?P<body>.*)$", re.I),
    ),
    (
        "about_company",
        re.compile(r"^(?P<head>about us|about the company|who we are|our mission|company overview)\b(?P<body>.*)$", re.I),
    ),
    (
        "legal_eeo",
        re.compile(r"^(?P<head>equal opportunity employer|equal opportunity|eeo|eoe)\b(?P<body>.*)$", re.I),
    ),
]

RESPONSIBILITY_CUES = re.compile(
    r"\b(build|design|develop|implement|own|lead|manage|support|analy[sz]e|collaborate|deliver|create|"
    r"maintain|participate|drive|architect|review|test|debug|deploy)\b",
    re.I,
)
REQUIREMENT_CUES = re.compile(
    r"\b(years?|experience|degree|bachelor|master|phd|proficiency|expertise|skills?|knowledge|"
    r"hands[- ]on|familiar|python|java|javascript|typescript|aws|azure|kubernetes|c\+\+|\.net|sql)\b",
    re.I,
)


def flattened_prefix_is_plausible(section_type: str, body: str) -> bool:
    body = body[:350]
    if not body:
        return True
    if section_type == "responsibilities":
        return bool(RESPONSIBILITY_CUES.search(body))
    if section_type in {"requirements", "preferred"}:
        return bool(REQUIREMENT_CUES.search(body))
    return True


def classify_heading(line: str) -> Tuple[str | None, str]:
    candidate = clean_heading_candidate(line)
    body = ""
    inline = INLINE_HEADING_RE.match(candidate)
    if inline:
        candidate = inline.group("head").strip()
        body = inline.group("body").strip()

    compact = re.sub(r"\s+", " ", candidate).strip(" :-–")
    if not compact:
        return None, body

    for section_type, pattern in FLATTENED_PREFIX_PATTERNS:
        match = pattern.match(compact)
        if match:
            flattened_body = match.group("body").strip(" :-–")
            final_body = flattened_body or body
            if flattened_prefix_is_plausible(section_type, final_body):
                return section_type, final_body

    # Long prose lines are rarely headings unless an inline heading was captured.
    if len(compact) > 95 and not inline:
        return None, body

    for section_type, patterns in HEADING_PATTERNS:
        if any(pattern.search(compact) for pattern in patterns):
            return section_type, body
    return None, body


def insert_heading_breaks(text: str) -> str:
    # Scraped LinkedIn often preserves markdown; historical Kaggle often flattens it.
    headings = [
        "Key Responsibilities",
        "Responsibilities",
        "What You'll Do",
        "What You Will Do",
        "Minimum Qualifications",
        "Basic Qualifications",
        "Required Skills",
        "What You'll Need",
        "What You Will Need",
        "Must-have",
        "Requirements",
        "Qualifications",
        "Preferred Qualifications",
        "Preferred Skills",
        "Nice-to-have",
        "Benefits",
        "Compensation",
        "About Us",
        "About the Company",
        "About the Role",
        "Equal Opportunity Employer",
        "EEO",
    ]
    escaped = "|".join(re.escape(h) for h in headings)
    text = re.sub(rf"(?<!^)(?<!\n)(\b(?:{escaped})\b\s*:)", r"\n\1", text, flags=re.I)
    # LLM-cleaned text often flattens headings by removing punctuation. Add conservative
    # breaks before likely heading tokens; classify_heading then applies plausibility checks.
    text = re.sub(rf"(?<!^)(?<!\n)(\b(?:{escaped})\b)", r"\n\1", text, flags=re.I)
    return text


def classify_sections(text: str | None) -> Tuple[Dict[str, str], bool, List[str]]:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = insert_heading_breaks(text)
    sections: Dict[str, List[str]] = {section: [] for section in SECTION_TYPES}
    current = "role_summary"
    seen_heading = False
    detected_headings: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        section_type, inline_body = classify_heading(line)
        if section_type:
            current = section_type
            seen_heading = True
            detected_headings.append(section_type)
            if inline_body:
                sections[current].append(inline_body)
            continue
        sections[current].append(line)

    if not seen_heading:
        # Avoid pretending unsectioned text has a role-summary section.
        unclassified = "\n".join(sections["role_summary"]).strip()
        sections = {section: [] for section in SECTION_TYPES}
        sections["unclassified"] = [unclassified] if unclassified else []

    joined = {section: "\n".join(parts).strip() for section, parts in sections.items()}
    return joined, seen_heading, detected_headings


def count_regexes(patterns: Sequence[re.Pattern[str]], text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in patterns)


def count_tone_markers(text: str | None) -> Dict[str, int]:
    text = "" if text is None else str(text)
    return {f"{name}_count": count_regexes(patterns, text) for name, patterns in TONE_PATTERNS.items()}


def corpus_label(source: str, period: str) -> str:
    if source == "scraped":
        return "scraped_2026"
    if source == "kaggle_arshkon":
        return "arshkon_2024"
    if source == "kaggle_asaniczka":
        return "asaniczka_2024"
    return f"{source}_{period}"


def tokens_first_1000(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall((text or "")[:1000])]


def type_token_ratio_first_1000(text: str) -> float:
    toks = tokens_first_1000(text)
    if not toks:
        return math.nan
    return len(set(toks)) / len(toks)


def readability_metrics(text: str) -> Dict[str, float]:
    text = text or ""
    try:
        return {
            "fk_grade": float(textstat.flesch_kincaid_grade(text)),
            "reading_ease": float(textstat.flesch_reading_ease(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
            "avg_sentence_length": float(textstat.avg_sentence_length(text)),
            "type_token_ratio_first_1000": float(type_token_ratio_first_1000(text)),
            "lexicon_count": float(textstat.lexicon_count(text, removepunct=True)),
            "syllable_count": float(textstat.syllable_count(text)),
        }
    except Exception:
        return {
            "fk_grade": math.nan,
            "reading_ease": math.nan,
            "gunning_fog": math.nan,
            "avg_sentence_length": math.nan,
            "type_token_ratio_first_1000": math.nan,
            "lexicon_count": math.nan,
            "syllable_count": math.nan,
        }


def section_row(row: Mapping[str, object], text_key: str = "description_cleaned") -> Dict[str, object]:
    text = "" if row.get(text_key) is None else str(row.get(text_key))
    sections, has_headings, detected = classify_sections(text)
    tone_counts = count_tone_markers(text)
    char_len = len(text)
    section_chars = {section: len(sections[section]) for section in SECTION_TYPES}
    req_resp_text = "\n\n".join(
        part for part in [sections["responsibilities"], sections["requirements"], sections["preferred"]] if part
    )

    out: Dict[str, object] = {
        "uid": row.get("uid"),
        "text_source": row.get("text_source"),
        "source": row.get("source"),
        "period": row.get("period"),
        "corpus": corpus_label(str(row.get("source")), str(row.get("period"))),
        "seniority_final": row.get("seniority_final"),
        "seniority_3level": row.get("seniority_3level"),
        "is_aggregator": bool(row.get("is_aggregator")),
        "company_name_canonical": row.get("company_name_canonical"),
        "yoe_extracted": row.get("yoe_extracted"),
        "swe_classification_tier": row.get("swe_classification_tier"),
        "seniority_final_source": row.get("seniority_final_source"),
        "char_len": char_len,
        "has_explicit_sections": has_headings,
        "detected_heading_types": ";".join(detected),
        "section_type_count": len(set(detected)),
        "requirements_responsibilities_text": req_resp_text,
    }
    for section, chars in section_chars.items():
        out[f"chars_{section}"] = chars
        out[f"prop_{section}"] = chars / char_len if char_len else math.nan
    out["chars_core_req_resp_pref"] = (
        section_chars["responsibilities"] + section_chars["requirements"] + section_chars["preferred"]
    )
    out["chars_boilerplate_benefits_about_legal"] = (
        section_chars["benefits"] + section_chars["about_company"] + section_chars["legal_eeo"]
    )
    out["prop_core_req_resp_pref"] = out["chars_core_req_resp_pref"] / char_len if char_len else math.nan
    out["prop_boilerplate_benefits_about_legal"] = (
        out["chars_boilerplate_benefits_about_legal"] / char_len if char_len else math.nan
    )
    out.update(tone_counts)
    return out


def section_schema() -> pa.Schema:
    fields: List[pa.Field] = [
        pa.field("uid", pa.string()),
        pa.field("text_source", pa.string()),
        pa.field("source", pa.string()),
        pa.field("period", pa.string()),
        pa.field("corpus", pa.string()),
        pa.field("seniority_final", pa.string()),
        pa.field("seniority_3level", pa.string()),
        pa.field("is_aggregator", pa.bool_()),
        pa.field("company_name_canonical", pa.string()),
        pa.field("yoe_extracted", pa.float64()),
        pa.field("swe_classification_tier", pa.string()),
        pa.field("seniority_final_source", pa.string()),
        pa.field("char_len", pa.int32()),
        pa.field("has_explicit_sections", pa.bool_()),
        pa.field("detected_heading_types", pa.string()),
        pa.field("section_type_count", pa.int16()),
        pa.field("requirements_responsibilities_text", pa.string()),
    ]
    for section in SECTION_TYPES:
        fields.append(pa.field(f"chars_{section}", pa.int32()))
        fields.append(pa.field(f"prop_{section}", pa.float64()))
    fields.extend(
        [
            pa.field("chars_core_req_resp_pref", pa.int32()),
            pa.field("chars_boilerplate_benefits_about_legal", pa.int32()),
            pa.field("prop_core_req_resp_pref", pa.float64()),
            pa.field("prop_boilerplate_benefits_about_legal", pa.float64()),
        ]
    )
    for name in TONE_PATTERNS:
        fields.append(pa.field(f"{name}_count", pa.int32()))
    return pa.schema(fields)


def write_rows(writer: pq.ParquetWriter, rows: List[Dict[str, object]], schema: pa.Schema) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)
    rows.clear()


def build_section_artifact() -> None:
    query = f"""
        SELECT uid, description_cleaned, text_source, source, period, seniority_final,
               seniority_3level, is_aggregator, company_name_canonical, yoe_extracted,
               swe_classification_tier, seniority_final_source
        FROM read_parquet('{SHARED_TEXT.as_posix()}')
        WHERE description_cleaned IS NOT NULL
    """
    schema = section_schema()
    con = connect()
    reader = con.execute(query).fetch_record_batch(rows_per_batch=2048)
    writer = pq.ParquetWriter(SECTION_ARTIFACT, schema=schema, compression="zstd")
    buffer: List[Dict[str, object]] = []
    try:
        for batch in reader:
            data = batch.to_pylist()
            for row in data:
                buffer.append(section_row(row))
                if len(buffer) >= 1024:
                    write_rows(writer, buffer, schema)
        write_rows(writer, buffer, schema)
    finally:
        writer.close()
        con.close()


def write_classifier_metadata() -> None:
    data = {
        "section_types": SECTION_TYPES,
        "core_text_for_T12": ["responsibilities", "requirements", "preferred"],
        "heading_patterns": {
            section: [pattern.pattern for name, patterns in HEADING_PATTERNS if name == section for pattern in patterns]
            for section in SECTION_TYPES
        },
        "tone_patterns": {name: [pattern.pattern for pattern in patterns] for name, patterns in TONE_PATTERNS.items()},
        "notes": [
            "Pre-heading text is role_summary only when at least one explicit heading is detected.",
            "Unheaded descriptions are classified as unclassified.",
            "T12 should filter to text_source='llm' for boilerplate-sensitive corpus comparisons.",
        ],
    }
    (TABLE_DIR / "section_classifier_patterns.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_text_coverage() -> None:
    con = connect()
    df = con.execute(
        f"""
        SELECT source, period, text_source, count(*) AS n,
               sum(CASE WHEN is_aggregator THEN 1 ELSE 0 END) AS aggregators,
               count(DISTINCT company_name_canonical) AS companies
        FROM read_parquet('{SHARED_TEXT.as_posix()}')
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """
    ).fetchdf()
    df.to_csv(TABLE_DIR / "text_source_coverage.csv", index=False)
    con.close()


def union_corpus_cte(table_path: str) -> str:
    return f"""
    WITH base AS (
      SELECT *,
             CASE
               WHEN source='scraped' THEN 'scraped_2026'
               WHEN source='kaggle_arshkon' THEN 'arshkon_2024'
               WHEN source='kaggle_asaniczka' THEN 'asaniczka_2024'
               ELSE source || '_' || period
             END AS corpus_group
      FROM read_parquet('{table_path}')
      WHERE text_source='llm' AND char_len > 0
    ),
    unioned AS (
      SELECT * FROM base
      UNION ALL
      SELECT * REPLACE('pooled_2024' AS corpus_group)
      FROM base
      WHERE source IN ('kaggle_arshkon', 'kaggle_asaniczka')
    )
    """


def write_section_tables() -> None:
    con = connect()
    table_path = SECTION_ARTIFACT.as_posix()

    section_selects = []
    for section in SECTION_TYPES:
        section_selects.append(
            f"""
            SELECT corpus_group, '{section}' AS section_type, count(*) AS n,
                   avg(chars_{section}) AS mean_chars,
                   median(chars_{section}) AS median_chars,
                   avg(prop_{section}) AS mean_row_prop,
                   median(prop_{section}) AS median_row_prop,
                   sum(chars_{section})::DOUBLE / nullif(sum(char_len), 0) AS char_weighted_prop,
                   sum(CASE WHEN chars_{section} > 0 THEN 1 ELSE 0 END)::DOUBLE / count(*) AS posting_presence
            FROM unioned
            GROUP BY 1
            """
        )
    query = union_corpus_cte(table_path) + "\n" + "\nUNION ALL\n".join(section_selects)
    section_comp = con.execute(query).fetchdf()
    section_comp["section_label"] = section_comp["section_type"].map(SECTION_LABELS)
    section_comp.to_csv(TABLE_DIR / "section_composition_by_corpus.csv", index=False)

    metrics_expr = ",\n".join(
        [f"avg(chars_{section}) AS mean_chars_{section}" for section in SECTION_TYPES]
        + [f"median(chars_{section}) AS median_chars_{section}" for section in SECTION_TYPES]
        + [
            "avg(char_len) AS mean_char_len",
            "median(char_len) AS median_char_len",
            "avg(prop_core_req_resp_pref) AS mean_prop_core_req_resp_pref",
            "avg(prop_boilerplate_benefits_about_legal) AS mean_prop_boilerplate",
            "avg(CASE WHEN has_explicit_sections THEN 1 ELSE 0 END) AS explicit_section_share",
        ]
    )
    means = con.execute(union_corpus_cte(table_path) + f"SELECT corpus_group, count(*) n, {metrics_expr} FROM unioned GROUP BY 1").fetchdf()
    means.to_csv(TABLE_DIR / "section_metrics_wide_by_corpus.csv", index=False)

    # Length decomposition: changes in mean section character counts.
    rows = []
    means_idx = means.set_index("corpus_group")
    for baseline in ["pooled_2024", "arshkon_2024", "asaniczka_2024"]:
        if baseline not in means_idx.index or "scraped_2026" not in means_idx.index:
            continue
        delta_total = means_idx.loc["scraped_2026", "mean_char_len"] - means_idx.loc[baseline, "mean_char_len"]
        for section in SECTION_TYPES:
            delta = means_idx.loc["scraped_2026", f"mean_chars_{section}"] - means_idx.loc[baseline, f"mean_chars_{section}"]
            rows.append(
                {
                    "baseline": baseline,
                    "comparison": "scraped_2026",
                    "section_type": section,
                    "section_label": SECTION_LABELS[section],
                    "baseline_mean_chars": means_idx.loc[baseline, f"mean_chars_{section}"],
                    "scraped_mean_chars": means_idx.loc["scraped_2026", f"mean_chars_{section}"],
                    "delta_mean_chars": delta,
                    "delta_total_mean_chars": delta_total,
                    "share_of_total_growth": delta / delta_total if delta_total else math.nan,
                }
            )
    decomp = pd.DataFrame(rows)
    decomp.to_csv(TABLE_DIR / "length_growth_section_decomposition.csv", index=False)

    section_by_seniority = con.execute(
        union_corpus_cte(table_path)
        + """
        SELECT corpus_group, seniority_3level, count(*) AS n,
               avg(char_len) AS mean_char_len,
               median(char_len) AS median_char_len,
               avg(prop_core_req_resp_pref) AS mean_prop_core_req_resp_pref,
               avg(prop_boilerplate_benefits_about_legal) AS mean_prop_boilerplate,
               avg(chars_requirements) AS mean_chars_requirements,
               avg(chars_responsibilities) AS mean_chars_responsibilities,
               avg(chars_preferred) AS mean_chars_preferred,
               avg(chars_benefits + chars_about_company + chars_legal_eeo) AS mean_chars_boilerplate
        FROM unioned
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    section_by_seniority.to_csv(TABLE_DIR / "section_composition_by_corpus_seniority.csv", index=False)
    con.close()


def write_tone_tables() -> None:
    con = connect()
    table_path = SECTION_ARTIFACT.as_posix()
    tone_exprs = []
    for name in TONE_PATTERNS:
        tone_exprs.extend(
            [
                f"avg({name}_count * 1000.0 / nullif(char_len, 0)) AS mean_{name}_per_1k",
                f"median({name}_count * 1000.0 / nullif(char_len, 0)) AS median_{name}_per_1k",
                f"sum({name}_count)::DOUBLE / nullif(sum(char_len), 0) * 1000 AS pooled_{name}_per_1k",
                f"sum(CASE WHEN {name}_count > 0 THEN 1 ELSE 0 END)::DOUBLE / count(*) AS share_any_{name}",
            ]
        )
    tone = con.execute(
        union_corpus_cte(table_path)
        + f"""
        SELECT corpus_group, 'all' AS spec, count(*) AS n, {','.join(tone_exprs)}
        FROM unioned
        GROUP BY 1
        UNION ALL
        SELECT corpus_group, 'exclude_aggregators' AS spec, count(*) AS n, {','.join(tone_exprs)}
        FROM unioned
        WHERE NOT is_aggregator
        GROUP BY 1
        ORDER BY 1,2
        """
    ).fetchdf()
    tone.to_csv(TABLE_DIR / "tone_metrics_by_corpus.csv", index=False)

    tone_seniority = con.execute(
        union_corpus_cte(table_path)
        + f"""
        SELECT corpus_group, seniority_3level, count(*) AS n, {','.join(tone_exprs)}
        FROM unioned
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    tone_seniority.to_csv(TABLE_DIR / "tone_metrics_by_corpus_seniority.csv", index=False)
    con.close()


def write_readability_tables() -> None:
    con = connect()
    # Bounded, deterministic, seniority-aware sample: at most 500 rows per source-period-seniority cell.
    query = f"""
    SELECT uid, description_cleaned, text_source, source, period, seniority_final, seniority_3level,
           is_aggregator, company_name_canonical, yoe_extracted, swe_classification_tier
    FROM (
      SELECT *,
             row_number() OVER (
               PARTITION BY source, period, coalesce(seniority_3level, 'unknown')
               ORDER BY hash(uid)
             ) AS rn
      FROM read_parquet('{SHARED_TEXT.as_posix()}')
      WHERE text_source='llm' AND length(description_cleaned) >= 100
    )
    WHERE rn <= 500
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=1024)
    rows: List[Dict[str, object]] = []
    for batch in reader:
        for row in batch.to_pylist():
            metrics = readability_metrics(row["description_cleaned"])
            rows.append(
                {
                    "uid": row["uid"],
                    "source": row["source"],
                    "period": row["period"],
                    "corpus": corpus_label(row["source"], row["period"]),
                    "seniority_final": row["seniority_final"],
                    "seniority_3level": row["seniority_3level"],
                    "is_aggregator": bool(row["is_aggregator"]),
                    **metrics,
                }
            )
    con.close()
    sample = pd.DataFrame(rows)
    sample.to_parquet(TABLE_DIR / "readability_sample_metrics.parquet", index=False)

    unioned = pd.concat(
        [
            sample,
            sample[sample["source"].isin(["kaggle_arshkon", "kaggle_asaniczka"])].assign(corpus="pooled_2024"),
        ],
        ignore_index=True,
    )
    metric_cols = [
        "fk_grade",
        "reading_ease",
        "gunning_fog",
        "avg_sentence_length",
        "type_token_ratio_first_1000",
        "lexicon_count",
        "syllable_count",
    ]
    agg = (
        unioned.groupby(["corpus", "seniority_3level"], dropna=False)[metric_cols]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns.to_flat_index()]
    agg.to_csv(TABLE_DIR / "readability_by_corpus_seniority.csv", index=False)

    agg_corpus = unioned.groupby(["corpus"], dropna=False)[metric_cols].agg(["count", "mean", "median"]).reset_index()
    agg_corpus.columns = ["_".join([str(x) for x in col if x]) for col in agg_corpus.columns.to_flat_index()]
    agg_corpus.to_csv(TABLE_DIR / "readability_by_corpus.csv", index=False)


def junior_flags_expr() -> Mapping[str, str]:
    return {
        "J1": "seniority_final = 'entry'",
        "J2": "seniority_final IN ('entry', 'associate')",
        "J3": "yoe_extracted <= 2",
        "J4": "yoe_extracted <= 3",
    }


def senior_flags_expr() -> Mapping[str, str]:
    return {
        "S1": "seniority_final IN ('mid-senior', 'director')",
        "S4": "yoe_extracted >= 5",
    }


def write_entry_panel_tables() -> None:
    con = connect()
    path = SECTION_ARTIFACT.as_posix()
    metrics = {
        "mean_char_len": "avg(char_len)",
        "mean_prop_core_req_resp_pref": "avg(prop_core_req_resp_pref)",
        "mean_prop_boilerplate": "avg(prop_boilerplate_benefits_about_legal)",
        "pooled_imperative_per_1k": "sum(imperative_count)::DOUBLE / nullif(sum(char_len), 0) * 1000",
        "pooled_inclusive_per_1k": "sum(inclusive_count)::DOUBLE / nullif(sum(char_len), 0) * 1000",
        "pooled_marketing_per_1k": "sum(marketing_count)::DOUBLE / nullif(sum(char_len), 0) * 1000",
        "explicit_section_share": "avg(CASE WHEN has_explicit_sections THEN 1 ELSE 0 END)",
    }
    metric_sql = ", ".join([f"{expr} AS {name}" for name, expr in metrics.items()])
    rows = []
    for j_name, j_expr in junior_flags_expr().items():
        for s_name, s_expr in senior_flags_expr().items():
            for group_name, expr in [(j_name, j_expr), (s_name, s_expr)]:
                q = (
                    union_corpus_cte(path)
                    + f"""
                    SELECT '{j_name}' AS junior_definition,
                           '{s_name}' AS senior_comparator,
                           '{group_name}' AS group_definition,
                           corpus_group,
                           count(*) AS n,
                           {metric_sql}
                    FROM unioned
                    WHERE {expr}
                    GROUP BY 1,2,3,4
                    """
                )
                rows.append(con.execute(q).fetchdf())
    panel = pd.concat(rows, ignore_index=True)
    panel.to_csv(TABLE_DIR / "entry_vs_senior_structure_tone_panel.csv", index=False)
    con.close()


def summarize_raw_same_row_sensitivity() -> None:
    """Compare cleaned LLM text with raw descriptions for the same LLM-covered rows."""

    con = connect()
    query = f"""
    SELECT s.uid, u.description AS raw_description, 'raw_same_llm_rows' AS text_source,
           s.source, s.period, s.seniority_final, s.seniority_3level, s.is_aggregator,
           s.company_name_canonical, s.yoe_extracted, s.swe_classification_tier,
           s.seniority_final_source
    FROM read_parquet('{SHARED_TEXT.as_posix()}') s
    JOIN read_parquet('{UNIFIED.as_posix()}') u USING (uid)
    WHERE s.text_source='llm'
      AND u.description IS NOT NULL
    """
    reader = con.execute(query).fetch_record_batch(rows_per_batch=1024)
    aggregates: MutableMapping[Tuple[str, str], MutableMapping[str, float]] = defaultdict(lambda: defaultdict(float))
    for batch in reader:
        for row in batch.to_pylist():
            row = dict(row)
            row["description_cleaned"] = row.pop("raw_description")
            out = section_row(row)
            key = (out["corpus"], "raw_same_llm_rows")
            agg = aggregates[key]
            agg["n"] += 1
            agg["total_chars"] += out["char_len"]
            for section in SECTION_TYPES:
                agg[f"chars_{section}"] += out[f"chars_{section}"]
            for name in TONE_PATTERNS:
                agg[f"{name}_count"] += out[f"{name}_count"]
            agg["explicit_section_n"] += 1 if out["has_explicit_sections"] else 0
    con.close()

    rows = []
    for (corp, spec), agg in aggregates.items():
        total = agg["total_chars"]
        row = {
            "corpus": corp,
            "text_variant": spec,
            "n": int(agg["n"]),
            "mean_char_len": total / agg["n"] if agg["n"] else math.nan,
            "explicit_section_share": agg["explicit_section_n"] / agg["n"] if agg["n"] else math.nan,
        }
        for section in SECTION_TYPES:
            row[f"char_weighted_prop_{section}"] = agg[f"chars_{section}"] / total if total else math.nan
        for name in TONE_PATTERNS:
            row[f"pooled_{name}_per_1k"] = agg[f"{name}_count"] / total * 1000 if total else math.nan
        rows.append(row)

    raw_df = pd.DataFrame(rows)

    # Add cleaned primary rows with the same summary shape.
    con = connect()
    cleaned_exprs = [
        "count(*) AS n",
        "avg(char_len) AS mean_char_len",
        "avg(CASE WHEN has_explicit_sections THEN 1 ELSE 0 END) AS explicit_section_share",
    ]
    for section in SECTION_TYPES:
        cleaned_exprs.append(f"sum(chars_{section})::DOUBLE / nullif(sum(char_len), 0) AS char_weighted_prop_{section}")
    for name in TONE_PATTERNS:
        cleaned_exprs.append(f"sum({name}_count)::DOUBLE / nullif(sum(char_len), 0) * 1000 AS pooled_{name}_per_1k")
    cleaned = con.execute(
        union_corpus_cte(SECTION_ARTIFACT.as_posix())
        + f"""
        SELECT corpus_group AS corpus, 'llm_cleaned' AS text_variant, {','.join(cleaned_exprs)}
        FROM unioned
        GROUP BY 1
        """
    ).fetchdf()
    con.close()
    pd.concat([cleaned, raw_df], ignore_index=True).to_csv(TABLE_DIR / "text_source_sensitivity_same_rows.csv", index=False)


def write_validation_samples() -> None:
    con = connect()
    rows = []
    # Deterministic contexts for major tone families. These are audit samples, not automated precision claims.
    for family in ["imperative", "inclusive", "passive_formal", "marketing"]:
        count_col = f"{family}_count"
        sample = con.execute(
            f"""
            SELECT uid, source, period, requirements_responsibilities_text, {count_col}
            FROM read_parquet('{SECTION_ARTIFACT.as_posix()}')
            WHERE text_source='llm' AND {count_col} > 0
            ORDER BY hash(uid)
            LIMIT 30
            """
        ).fetchdf()
        for _, row in sample.iterrows():
            text = str(row["requirements_responsibilities_text"] or "")
            rows.append(
                {
                    "family": family,
                    "uid": row["uid"],
                    "source": row["source"],
                    "period": row["period"],
                    "count": row[count_col],
                    "context_preview": re.sub(r"\s+", " ", text[:300]),
                    "manual_precision_claim": "not_claimed_context_sample_only",
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "tone_marker_context_samples.csv", index=False)
    con.close()


def plot_section_composition() -> None:
    comp = pd.read_csv(TABLE_DIR / "section_composition_by_corpus.csv")
    corp_order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    comp = comp[comp["corpus_group"].isin(corp_order)]
    pivot = comp.pivot(index="corpus_group", columns="section_type", values="char_weighted_prop").reindex(corp_order)
    colors = {
        "role_summary": "#4C78A8",
        "responsibilities": "#F58518",
        "requirements": "#54A24B",
        "preferred": "#B279A2",
        "benefits": "#E45756",
        "about_company": "#72B7B2",
        "legal_eeo": "#FF9DA6",
        "unclassified": "#9D755D",
    }
    ax = pivot[SECTION_TYPES].plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5),
        color=[colors[s] for s in SECTION_TYPES],
        width=0.74,
    )
    ax.set_ylabel("Share of cleaned-text characters")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend([SECTION_LABELS[s] for s in SECTION_TYPES], bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Description Section Composition")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "section_composition_stacked.png", dpi=150)
    plt.close()


def plot_readability() -> None:
    df = pd.read_csv(TABLE_DIR / "readability_by_corpus.csv")
    corp_order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    df = df[df["corpus"].isin(corp_order)].set_index("corpus").reindex(corp_order)
    metrics = [
        ("fk_grade_mean", "FK grade"),
        ("gunning_fog_mean", "Gunning fog"),
        ("avg_sentence_length_mean", "Sentence length"),
        ("type_token_ratio_first_1000_mean", "Type-token ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(df.index, df[col], color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "readability_metrics.png", dpi=150)
    plt.close()


def plot_tone() -> None:
    df = pd.read_csv(TABLE_DIR / "tone_metrics_by_corpus.csv")
    corp_order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    df = df[(df["spec"] == "all") & (df["corpus_group"].isin(corp_order))].set_index("corpus_group").reindex(corp_order)
    cols = [
        ("pooled_imperative_per_1k", "Imperative"),
        ("pooled_inclusive_per_1k", "Inclusive"),
        ("pooled_passive_formal_per_1k", "Passive/formal"),
        ("pooled_marketing_per_1k", "Marketing"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(corp_order))
    width = 0.2
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for idx, (col, label) in enumerate(cols):
        ax.bar(x + (idx - 1.5) * width, df[col], width=width, label=label, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(corp_order, rotation=20)
    ax.set_ylabel("Matches per 1,000 cleaned-text chars")
    ax.set_title("Tone Marker Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tone_marker_density.png", dpi=150)
    plt.close()


def plot_entry_panel() -> None:
    df = pd.read_csv(TABLE_DIR / "entry_vs_senior_structure_tone_panel.csv")
    df = df[
        (df["senior_comparator"] == "S1")
        & (df["corpus_group"].isin(["pooled_2024", "scraped_2026"]))
        & (df["group_definition"].isin(["J1", "J2", "J3", "J4"]))
    ]
    pivot = df.pivot(index="group_definition", columns="corpus_group", values="mean_prop_core_req_resp_pref").reindex(
        ["J1", "J2", "J3", "J4"]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(pivot.index))
    ax.bar(x - 0.18, pivot["pooled_2024"], width=0.36, label="Pooled 2024", color="#4C78A8")
    ax.bar(x + 0.18, pivot["scraped_2026"], width=0.36, label="Scraped 2026", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Mean req/resp/preferred share")
    ax.set_title("Junior Panel Core-Section Share")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "entry_panel_core_section_share.png", dpi=150)
    plt.close()


def write_figures() -> None:
    plot_section_composition()
    plot_readability()
    plot_tone()
    plot_entry_panel()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    assert_regex_edges()
    write_classifier_metadata()
    write_text_coverage()
    build_section_artifact()
    write_section_tables()
    write_tone_tables()
    write_readability_tables()
    write_entry_panel_tables()
    summarize_raw_same_row_sensitivity()
    write_validation_samples()
    write_figures()
    print(f"Wrote T13 outputs to {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    main()
