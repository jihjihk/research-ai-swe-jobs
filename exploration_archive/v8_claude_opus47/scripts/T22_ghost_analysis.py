"""T22 main analysis — ghost & aspirational requirements forensics.

Produces:
  * per-posting ghost indicators
  * prevalence by period x seniority (T30 panel J1/J2/J3/J4, S1/S4)
  * AI ghostiness test (aspiration ratio AI vs non-AI, ghost_assessment_llm
    cross-tab with AI-mention strict)
  * aggregator vs direct
  * industry patterns
  * 20 most ghost-like entry-level postings (for examples table)
  * validated_mgmt_patterns.json
"""

import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path("/home/jihgaboot/gabor/job-research")
sys.path.insert(0, str(REPO / "exploration/scripts"))
from T22_patterns import (  # noqa: E402
    AI_STRICT_REGEX,
    AI_BROAD_REGEX,
    AI_TOOL_REGEX,
    AI_DOMAIN_REGEX,
    ASPIRATION_REGEX,
    FIRM_REGEX,
    ORG_SCOPE_REGEX,
    MGMT_STRICT_REGEX,
    DEGREE_NO_REGEX,
    DEGREE_MS_REGEX,
)

ART = REPO / "exploration/artifacts/T22"
SHARED = REPO / "exploration/artifacts/shared"
ART.mkdir(parents=True, exist_ok=True)


def load_data() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE post AS
        SELECT
            u.uid,
            u.source,
            u.period,
            u.is_aggregator,
            u.seniority_final,
            u.seniority_native,
            u.company_name_canonical,
            u.company_industry,
            u.yoe_extracted,
            u.ghost_job_risk,
            u.ghost_assessment_llm,
            u.llm_classification_coverage,
            u.description,
            u.description_length,
            u.title,
            u.title_normalized
        FROM '""" + str(REPO / "data/unified.parquet") + """' u
        WHERE u.source_platform='linkedin'
          AND u.is_english=true
          AND u.date_flag='ok'
          AND u.is_swe=true
        """
    )
    # Join with cleaned text (text_source='llm' only)
    con.execute(
        f"""
        CREATE TABLE txt AS
        SELECT uid, description_cleaned, text_source
        FROM '{SHARED / "swe_cleaned_text.parquet"}'
        WHERE text_source='llm'
        """
    )
    con.execute(
        """
        CREATE TABLE merged AS
        SELECT p.*, t.description_cleaned
        FROM post p
        LEFT JOIN txt t USING (uid)
        """
    )
    return con


def compute_features(con: duckdb.DuckDBPyConnection) -> None:
    """Compute per-posting features using regex over description_cleaned
    (LLM-frame), falling back to description when cleaned is NULL."""

    # Pull needed columns into python for regex work
    rows = con.execute(
        """
        SELECT uid, source, period, is_aggregator, seniority_final, seniority_native,
               company_name_canonical, company_industry, yoe_extracted, ghost_job_risk,
               ghost_assessment_llm, llm_classification_coverage, description_cleaned,
               description, description_length, title
        FROM merged
        """
    ).fetchall()
    print(f"Loaded {len(rows):,} rows for feature computation.")

    asp = re.compile(ASPIRATION_REGEX, re.IGNORECASE)
    firm = re.compile(FIRM_REGEX, re.IGNORECASE)
    org = re.compile(ORG_SCOPE_REGEX, re.IGNORECASE)
    mgmt = re.compile(MGMT_STRICT_REGEX, re.IGNORECASE)
    ai_strict = re.compile(AI_STRICT_REGEX, re.IGNORECASE)
    ai_broad = re.compile(AI_BROAD_REGEX, re.IGNORECASE)
    ai_tool = re.compile(AI_TOOL_REGEX, re.IGNORECASE)
    ai_domain = re.compile(AI_DOMAIN_REGEX, re.IGNORECASE)
    deg_no = re.compile(DEGREE_NO_REGEX, re.IGNORECASE)
    deg_ms = re.compile(DEGREE_MS_REGEX, re.IGNORECASE)
    # tech heuristic: count common tech keywords (keep simple)
    tech_regex = re.compile(
        r"\b(python|java|javascript|typescript|go|rust|c\+\+|c#|ruby|kotlin|swift|scala|php|sql|react|angular|vue|nextjs|node|django|flask|spring|\.net|rails|fastapi|express|aws|azure|gcp|kubernetes|docker|terraform|jenkins|postgres|postgresql|mysql|mongodb|redis|kafka|spark|snowflake|databricks|elasticsearch|tensorflow|pytorch|sklearn|pandas|numpy|huggingface|langchain|rag|copilot|cursor|chatgpt|claude|langgraph)\b",
        re.IGNORECASE,
    )

    # sentence splitter (rough but fine for count purposes)
    sent_splitter = re.compile(r"[.!?]\s+|\n+|\*|•")

    out = []
    for r in rows:
        (
            uid,
            source,
            period,
            is_agg,
            sen_final,
            sen_native,
            company,
            industry,
            yoe,
            ghost_risk,
            ghost_llm,
            llm_cov,
            desc_clean,
            desc_raw,
            desc_len,
            title,
        ) = r
        text = desc_clean if desc_clean else (desc_raw or "")
        text_l = text.lower() if text else ""

        tech_count = len(set(m.group(0).lower() for m in tech_regex.finditer(text_l)))
        org_count = len(org.findall(text_l))
        kitchen_sink = tech_count * org_count

        asp_total = len(asp.findall(text_l))
        firm_total = len(firm.findall(text_l))
        aspiration_ratio = asp_total / firm_total if firm_total > 0 else (float("inf") if asp_total > 0 else 0.0)

        mgmt_count = len(mgmt.findall(text_l))

        ai_strict_bin = 1 if ai_strict.search(text_l) else 0
        ai_broad_bin = 1 if ai_broad.search(text_l) else 0
        ai_tool_bin = 1 if ai_tool.search(text_l) else 0
        ai_domain_bin = 1 if ai_domain.search(text_l) else 0

        # Per-sentence: aspiration counts within AI vs non-AI sentences.
        ai_asp_local = 0
        ai_firm_local = 0
        non_ai_asp_local = 0
        non_ai_firm_local = 0
        if ai_strict_bin == 1 or ai_broad_bin == 1:
            # Use broad pattern for AI-context sentence detection (more matches
            # in the distant-tail sentences), but the AI-ghostiness question is
            # STRICT+BROAD.
            for sent in sent_splitter.split(text_l):
                s = sent.strip()
                if not s:
                    continue
                is_ai_sent = bool(ai_broad.search(s))
                asp_here = len(asp.findall(s))
                firm_here = len(firm.findall(s))
                if is_ai_sent:
                    ai_asp_local += asp_here
                    ai_firm_local += firm_here
                else:
                    non_ai_asp_local += asp_here
                    non_ai_firm_local += firm_here
        else:
            # No AI presence at all
            non_ai_asp_local = asp_total
            non_ai_firm_local = firm_total

        ai_aspiration_ratio = (
            ai_asp_local / ai_firm_local if ai_firm_local > 0 else (float("inf") if ai_asp_local > 0 else 0.0)
        )
        non_ai_aspiration_ratio = (
            non_ai_asp_local / non_ai_firm_local
            if non_ai_firm_local > 0
            else (float("inf") if non_ai_asp_local > 0 else 0.0)
        )

        # YOE-scope mismatch (J1): entry+yoe>=5 OR entry+>=3 org terms
        yoe_val = yoe if yoe is not None else np.nan
        is_entry = sen_final == "entry"
        yoe_mismatch = int(
            is_entry
            and (
                (not np.isnan(yoe_val) and yoe_val >= 5)
                or org_count >= 3
            )
        )

        # Credential impossibility flags
        contradiction = 0
        # 1. entry + 10+ YOE
        if is_entry and not np.isnan(yoe_val) and yoe_val >= 10:
            contradiction = 1
        # 2. both "no degree required" and "MS/PhD required"
        if deg_no.search(text_l) and deg_ms.search(text_l):
            contradiction = 1

        out.append(
            {
                "uid": uid,
                "source": source,
                "period": period,
                "is_aggregator": bool(is_agg) if is_agg is not None else False,
                "seniority_final": sen_final,
                "seniority_native": sen_native,
                "company": company,
                "industry": industry,
                "yoe_extracted": float(yoe_val) if not np.isnan(yoe_val) else None,
                "ghost_job_risk": ghost_risk,
                "ghost_assessment_llm": ghost_llm,
                "llm_classification_coverage": llm_cov,
                "desc_length": int(desc_len) if desc_len else 0,
                "title": title,
                "tech_count": tech_count,
                "org_scope_count": org_count,
                "kitchen_sink": kitchen_sink,
                "aspiration_count": asp_total,
                "firm_count": firm_total,
                "aspiration_ratio": aspiration_ratio if np.isfinite(aspiration_ratio) else float("inf"),
                "mgmt_count": mgmt_count,
                "ai_strict_bin": ai_strict_bin,
                "ai_broad_bin": ai_broad_bin,
                "ai_tool_bin": ai_tool_bin,
                "ai_domain_bin": ai_domain_bin,
                "ai_asp_count": ai_asp_local,
                "ai_firm_count": ai_firm_local,
                "non_ai_asp_count": non_ai_asp_local,
                "non_ai_firm_count": non_ai_firm_local,
                "ai_aspiration_ratio": ai_aspiration_ratio if np.isfinite(ai_aspiration_ratio) else float("inf"),
                "non_ai_aspiration_ratio": non_ai_aspiration_ratio if np.isfinite(non_ai_aspiration_ratio) else float("inf"),
                "yoe_mismatch": yoe_mismatch,
                "credential_contradiction": contradiction,
            }
        )

    # write features parquet
    cols = list(out[0].keys())
    table = pa.Table.from_pydict({c: [r[c] for r in out] for c in cols})
    pq.write_table(table, ART / "T22_features.parquet")
    print(f"Wrote {ART / 'T22_features.parquet'} ({len(out):,} rows).")


def main() -> None:
    con = load_data()
    compute_features(con)


if __name__ == "__main__":
    main()
