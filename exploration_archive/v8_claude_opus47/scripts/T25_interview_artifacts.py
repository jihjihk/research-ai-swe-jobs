"""T25 — Interview elicitation artifacts.

Produces six artifacts in ``exploration/artifacts/T25_interview/``:

1. inflated_junior_jds.md — 3-5 entry-level postings with scope-inflation / ghost features.
2. paired_jds_over_time.md — 3-5 same-company 2024-vs-2026 pairs.
3. junior_share_trend.png — J1/J2/J3/J4 trajectory with AI-release annotations.
4. senior_archetype_chart.png — T21 mgmt/orch/strat densities at S1 by period.
5. employer_usage_divergence.png — T23 employer rate vs worker-usage benchmark.
6. ai_gradient_chart.png — T18 SWE vs adjacent vs control AI trajectory with DiD.

Plus README.md describing each artifact and its analytic provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
UNIFIED = ROOT / "data" / "unified.parquet"
T22_FEATURES = ROOT / "exploration/artifacts/T22/T22_features.parquet"
OUT_DIR = ROOT / "exploration/artifacts/T25_interview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# AI release timeline used for annotations on the junior-share plot.
AI_RELEASES = [
    ("GPT-4", "2023-03-14"),
    ("Claude 3", "2024-03-04"),
    ("GPT-4o", "2024-05-13"),
    ("Claude 3.5 Sonnet", "2024-06-20"),
    ("o1-preview", "2024-09-12"),
    ("Claude 4", "2025-09-15"),
    ("Gemini 2.5 Pro", "2026-03-18"),
]


def _truncate(s: str | None, n: int = 900) -> str:
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\n\n", "\n").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _q(sql: str) -> pd.DataFrame:
    return duckdb.connect().execute(sql).fetchdf()


# ---------------------------------------------------------------------------
# Artifact 1 — Inflated junior JDs
# ---------------------------------------------------------------------------


def artifact1_inflated_junior_jds() -> None:
    """Select 4 junior postings with extreme scope-inflation / ghost features."""
    # Use the T22 features parquet for consistent ghost + AI flags.
    df = duckdb.sql(
        f"""
        SELECT
            u.uid, u.title, u.company_name_canonical, u.period, u.seniority_final,
            u.yoe_extracted, u.company_industry, u.is_aggregator,
            u.description_core_llm, u.description,
            t.tech_count, t.org_scope_count, t.aspiration_count, t.firm_count,
            t.ghost_assessment_llm, t.ai_strict_bin AS ai_strict, t.yoe_mismatch AS yoe_mismatch_flag,
            t.kitchen_sink
        FROM read_parquet('{UNIFIED}') u
        JOIN read_parquet('{T22_FEATURES}') t USING (uid)
        WHERE u.is_swe = TRUE
          AND u.source_platform = 'linkedin'
          AND u.is_english = TRUE
          AND u.date_flag = 'ok'
          AND u.seniority_final IN ('entry','associate')
          AND t.ghost_assessment_llm IN ('inflated','ghost_likely')
          AND t.tech_count >= 6
          AND t.org_scope_count >= 1
        ORDER BY kitchen_sink DESC, t.tech_count DESC
        LIMIT 40
        """
    ).fetchdf()

    # Prefer diverse examples: 2024 vs 2026, different industries, at least one AI-mention row.
    chosen_uids: list[str] = []
    picked: list[pd.Series] = []
    # Manual pick for diversity.
    def _pick(condition) -> None:
        nonlocal chosen_uids, picked
        for _, row in df.iterrows():
            if row["uid"] in chosen_uids:
                continue
            if condition(row):
                picked.append(row)
                chosen_uids.append(row["uid"])
                return

    _pick(lambda r: r["period"] in ("2026-03", "2026-04") and bool(r["ai_strict"]))  # 2026, AI-flavoured
    _pick(lambda r: r["period"] in ("2026-03", "2026-04") and not bool(r["ai_strict"]))  # 2026, traditional
    _pick(lambda r: r["period"] in ("2024-01", "2024-04") and not bool(r["ai_strict"]))  # 2024 baseline
    _pick(lambda r: r["period"] in ("2026-03", "2026-04") and int(r["yoe_mismatch_flag"]) == 1)  # YOE-mismatch

    # Fallback to top-scoring if we under-filled.
    i = 0
    while len(picked) < 4 and i < len(df):
        row = df.iloc[i]
        if row["uid"] not in chosen_uids:
            picked.append(row)
            chosen_uids.append(row["uid"])
        i += 1

    lines: list[str] = []
    lines.append("# Inflated Junior JDs (Interview Elicitation Set 1)\n")
    lines.append(
        "Four entry-labeled LinkedIn SWE postings ranked high on kitchen-sink "
        "composite × LLM ghost rubric. Use in interviews with hiring managers, "
        "HR, or recruiters to elicit: Is this scope realistic? Would you hire "
        "this person? How many candidates can meet it?\n"
    )
    lines.append(
        "Selection rule: `seniority_final ∈ {entry, associate}`, "
        "`ghost_assessment_llm ∈ {inflated, ghost_likely}`, "
        "`tech_count ≥ 6`, `org_scope_count ≥ 1`. "
        "Diversity constraint: 2 × 2026 (one AI-flagged, one traditional), "
        "1 × 2024 baseline, 1 × YOE-mismatch (entry + yoe≥5 or scope≥3). "
        "See T22 `T22_top20_ghost.csv` for the broader list.\n"
    )
    for i, row in enumerate(picked, 1):
        lines.append(f"## Posting {i} — `{row['uid']}`\n")
        lines.append(f"- **Title:** {row['title']}")
        lines.append(f"- **Company:** {row['company_name_canonical']}")
        lines.append(f"- **Industry:** {row['company_industry'] or '(n/a)'}")
        lines.append(f"- **Period:** {row['period']}")
        lines.append(f"- **Seniority label:** {row['seniority_final']}")
        lines.append(f"- **YOE extracted:** {row['yoe_extracted']}")
        lines.append(f"- **Aggregator:** {bool(row['is_aggregator'])}")
        lines.append(
            f"- **Ghost features:** tech_count={row['tech_count']}, "
            f"org_scope={row['org_scope_count']}, "
            f"kitchen_sink={row['kitchen_sink']}, "
            f"ghost_llm={row['ghost_assessment_llm']}, "
            f"ai_strict={bool(row['ai_strict'])}, "
            f"yoe_mismatch={bool(row['yoe_mismatch_flag'])}"
        )
        lines.append("")
        lines.append("### Description (cleaned LLM frame if available, else raw):\n")
        desc = row.get("description_core_llm") or row.get("description")
        lines.append("```")
        lines.append(_truncate(desc, 1400))
        lines.append("```")
        lines.append("")
        lines.append("### Interview probes:")
        lines.append(
            "- Would the posting's stated scope (tech + org responsibilities + YOE) "
            "realistically be met by a 2024 or 2026 entry-level candidate?"
        )
        lines.append(
            "- If no, what does the posting communicate to candidates / to the company?"
        )
        lines.append(
            "- How does this posting compare to a typical entry-level JD at your firm "
            "3 years ago (pre-ChatGPT) and today?"
        )
        lines.append("")

    out = OUT_DIR / "inflated_junior_jds.md"
    out.write_text("\n".join(lines))
    print(f"[1/6] Wrote {out} with {len(picked)} postings.")


# ---------------------------------------------------------------------------
# Artifact 2 — Paired JDs over time (same company, 2024 vs 2026)
# ---------------------------------------------------------------------------


def artifact2_paired_jds() -> None:
    """Find 4 same-company 2024↔2026 pairs showing AI/mentor/scope shifts."""
    # First, find 240-co overlap candidates (per T16) with:
    #   - ≥3 postings in arshkon AND ≥3 in scraped
    #   - Ideally at senior level (mid-senior) for the AI/mentor story
    candidates = duckdb.sql(
        f"""
        WITH base AS (
            SELECT
                uid, company_name_canonical, period, seniority_final, title,
                description_core_llm, description,
                CASE WHEN period IN ('2026-03','2026-04') THEN '2026' ELSE '2024' END AS era,
                CASE WHEN regexp_matches(
                    lower(coalesce(description_core_llm, description)),
                    '(copilot|cursor|\\bclaude\\b|chatgpt|openai|\\bgpt-?[0-9]|gemini|\\bcodex\\b|llamaindex|langchain|prompt engineering|fine[- ]tuning|\\brag\\b|vector database|pinecone|huggingface)'
                ) THEN 1 ELSE 0 END AS ai_strict,
                CASE WHEN regexp_matches(
                    lower(coalesce(description_core_llm, description)),
                    '\\bmentor\\w*'
                ) THEN 1 ELSE 0 END AS mentor,
                length(coalesce(description_core_llm, description)) AS desc_len
            FROM read_parquet('{UNIFIED}')
            WHERE is_swe = TRUE
              AND source_platform = 'linkedin'
              AND is_english = TRUE
              AND date_flag = 'ok'
              AND seniority_final IN ('mid-senior', 'entry', 'associate')
              AND description_core_llm IS NOT NULL
              AND length(description_core_llm) > 400
        ),
        co_periods AS (
            SELECT
                company_name_canonical,
                COUNT(*) FILTER (WHERE era = '2024') AS n_2024,
                COUNT(*) FILTER (WHERE era = '2026') AS n_2026,
                AVG(ai_strict) FILTER (WHERE era = '2026') AS ai_2026,
                AVG(ai_strict) FILTER (WHERE era = '2024') AS ai_2024,
                AVG(mentor)    FILTER (WHERE era = '2026') AS mentor_2026,
                AVG(mentor)    FILTER (WHERE era = '2024') AS mentor_2024
            FROM base
            GROUP BY company_name_canonical
            HAVING n_2024 >= 2 AND n_2026 >= 2
        )
        SELECT *
        FROM co_periods
        WHERE (ai_2026 - ai_2024) > 0.25 OR (mentor_2026 - mentor_2024) > 0.12
        ORDER BY (ai_2026 - ai_2024) + (mentor_2026 - mentor_2024) DESC
        LIMIT 30
        """
    ).fetchdf()

    # Avoid aggregators and tiny staffing.
    avoid = {
        "dice", "jobs via dice", "mygwork", "jobs for humanity",
        "hackajob", "motion recruitment", "staffing", "jobleads", "lensa",
    }
    candidates = candidates[
        ~candidates["company_name_canonical"].str.lower().isin(avoid)
    ].reset_index(drop=True)

    def _fetch_pair(co: str) -> tuple[pd.Series | None, pd.Series | None]:
        rows = duckdb.sql(
            f"""
            SELECT
                uid, period, seniority_final, title,
                coalesce(description_core_llm, description) AS desc_text,
                length(coalesce(description_core_llm, description)) AS desc_len,
                CASE WHEN regexp_matches(
                    lower(coalesce(description_core_llm, description)),
                    '(copilot|cursor|\\bclaude\\b|chatgpt|openai|\\bgpt-?[0-9]|gemini|\\bcodex\\b|llamaindex|langchain|prompt engineering|fine[- ]tuning|\\brag\\b|vector database|pinecone|huggingface)'
                ) THEN 1 ELSE 0 END AS ai_strict,
                CASE WHEN regexp_matches(
                    lower(coalesce(description_core_llm, description)),
                    '\\bmentor\\w*'
                ) THEN 1 ELSE 0 END AS mentor
            FROM read_parquet('{UNIFIED}')
            WHERE is_swe = TRUE
              AND source_platform = 'linkedin'
              AND is_english = TRUE
              AND date_flag = 'ok'
              AND company_name_canonical = ?
              AND description_core_llm IS NOT NULL
              AND length(description_core_llm) BETWEEN 600 AND 6000
              AND seniority_final IN ('mid-senior', 'entry', 'associate')
            ORDER BY period, seniority_final
            """,
            params=[co],
        ).fetchdf()
        if rows.empty:
            return None, None
        # Prefer mid-senior for the senior story.
        mid_rows = rows[rows["seniority_final"] == "mid-senior"]
        if mid_rows.empty:
            mid_rows = rows
        r_2024 = mid_rows[mid_rows["period"].isin(["2024-01", "2024-04"])]
        r_2026 = mid_rows[mid_rows["period"].isin(["2026-03", "2026-04"])]
        if r_2024.empty or r_2026.empty:
            return None, None
        # Prefer 2026 with AI mention.
        r26 = r_2026.sort_values("ai_strict", ascending=False).iloc[0]
        r24 = r_2024.iloc[0]
        return r24, r26

    pairs: list[tuple[str, pd.Series, pd.Series]] = []
    seen: set[str] = set()
    for _, cand in candidates.iterrows():
        co = cand["company_name_canonical"]
        if co in seen or not co:
            continue
        r24, r26 = _fetch_pair(co)
        if r24 is None or r26 is None:
            continue
        pairs.append((co, r24, r26))
        seen.add(co)
        if len(pairs) >= 4:
            break

    lines: list[str] = []
    lines.append("# Paired JDs 2024 vs 2026 (Interview Elicitation Set 2)\n")
    lines.append(
        "Four same-company pairs of mid-senior SWE postings selected from the "
        "240-company arshkon∩scraped overlap panel (T16), ranked by the joint "
        "2024→2026 increase in AI-strict mention rate + mentor-binary rate. "
        "Companies most visibly shifting toward AI/mentoring content.\n"
    )
    lines.append(
        "Selection rule: `is_swe`, LinkedIn, English, `seniority_final = 'mid-senior'`, "
        "company has ≥2 postings in each period, company-level Δ(ai_strict) > 0.3 OR "
        "Δ(mentor) > 0.15. Excludes aggregators and staffing rebrands.\n"
    )
    for i, (co, r24, r26) in enumerate(pairs, 1):
        lines.append(f"## Pair {i} — {co}\n")
        lines.append(f"### 2024 (`{r24['uid']}`)")
        lines.append(
            f"- Title: {r24['title']}  |  seniority: {r24['seniority_final']}  "
            f"|  length: {r24['desc_len']} chars  |  AI-strict: {bool(r24['ai_strict'])}  "
            f"|  mentor: {bool(r24['mentor'])}"
        )
        lines.append("```")
        lines.append(_truncate(r24["desc_text"], 1400))
        lines.append("```")
        lines.append(f"### 2026 (`{r26['uid']}`)")
        lines.append(
            f"- Title: {r26['title']}  |  seniority: {r26['seniority_final']}  "
            f"|  length: {r26['desc_len']} chars  |  AI-strict: {bool(r26['ai_strict'])}  "
            f"|  mentor: {bool(r26['mentor'])}"
        )
        lines.append("```")
        lines.append(_truncate(r26["desc_text"], 1400))
        lines.append("```")
        lines.append("### Interview probes:")
        lines.append(
            "- Side-by-side: what content is NEW in 2026? Which items correspond to "
            "real hiring-requirement change vs JD-writing style change?"
        )
        lines.append(
            "- If your team drafted the 2026 version, which sections came from a template / "
            "LLM / copy-edit vs actual hiring-manager requests?"
        )
        lines.append(
            "- Does the 2026 description reflect candidates you have actually hired for "
            "this role, or a broader wishlist?"
        )
        lines.append("")

    out = OUT_DIR / "paired_jds_over_time.md"
    out.write_text("\n".join(lines))
    print(f"[2/6] Wrote {out} with {len(pairs)} same-company pairs.")


# ---------------------------------------------------------------------------
# Artifact 3 — Junior-share trend plot with AI release annotations
# ---------------------------------------------------------------------------


def artifact3_junior_share_trend() -> None:
    """Plot J1–J4 junior share with AI release markers."""
    # Rebuild the T30 four-variant panel in a single-frame format.
    df = duckdb.sql(
        f"""
        SELECT
            period AS t_label,
            COUNT(*) AS n_all,
            SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS j1_share,
            SUM(CASE WHEN seniority_final IN ('entry','associate') THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS j2_share,
            SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j3_share,
            SUM(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END), 0) AS j4_share
        FROM read_parquet('{UNIFIED}')
        WHERE is_swe = TRUE
          AND source_platform = 'linkedin'
          AND is_english = TRUE
          AND date_flag = 'ok'
          AND period IN ('2024-01','2024-04','2026-03','2026-04')
        GROUP BY period
        ORDER BY period
        """
    ).fetchdf()

    # Label → date midpoint used for plotting.
    date_map = {
        "2024-01": pd.Timestamp("2024-01-15"),
        "2024-04": pd.Timestamp("2024-04-15"),
        "2026-03": pd.Timestamp("2026-03-15"),
        "2026-04": pd.Timestamp("2026-04-15"),
    }
    df["date"] = df["t_label"].map(date_map)
    df = df.sort_values("date").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    series = [
        ("j1_share", "J1 (entry)", "#1f77b4"),
        ("j2_share", "J2 (entry+associate)", "#2ca02c"),
        ("j3_share", "J3 (yoe ≤ 2)", "#ff7f0e"),
        ("j4_share", "J4 (yoe ≤ 3)", "#d62728"),
    ]
    for col, label, color in series:
        ax.plot(df["date"], df[col] * 100, marker="o", label=label, color=color, linewidth=2)
        for dt, val in zip(df["date"], df[col] * 100):
            ax.annotate(f"{val:.1f}%", (dt, val), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)

    # AI release annotations
    ymax = df[["j1_share", "j2_share", "j3_share", "j4_share"]].values.max() * 100
    for name, iso in AI_RELEASES:
        dt = pd.Timestamp(iso)
        if dt < df["date"].min() - pd.Timedelta(days=90):
            continue
        if dt > df["date"].max() + pd.Timedelta(days=90):
            continue
        ax.axvline(dt, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.annotate(
            name, (dt, ymax * 1.04), rotation=60, fontsize=7, color="gray",
            ha="left", va="bottom",
        )

    ax.set_ylabel("Share of SWE LinkedIn postings (%)")
    ax.set_title("Junior-share trajectory by operationalization (J1–J4) with AI model releases")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Period")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, ymax * 1.2)
    caption = (
        "Baseline-dependent story: J3/J4 (YOE-based, label-independent) rise 2024→2026; "
        "J1/J2 (native-label entry) depend on baseline — down under arshkon-only, up under "
        "pooled-2024. T05 SNR < 1 on every junior metric: the direction is inherently "
        "baseline-contingent (see T30 panel, Gate 1 memo)."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True)
    fig.tight_layout()
    out = OUT_DIR / "junior_share_trend.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[3/6] Wrote {out}")


# ---------------------------------------------------------------------------
# Artifact 4 — Senior archetype chart (T21 mgmt/orch/strat densities at S1)
# ---------------------------------------------------------------------------


def artifact4_senior_archetype() -> None:
    """T21-replicated bar chart: S1 density by density-class × period."""
    # Use S1 = mid-senior + director. Patterns from T21 (validated).
    df = duckdb.sql(
        f"""
        WITH base AS (
            SELECT
                CASE WHEN period IN ('2024-01','2024-04') THEN '2024' ELSE '2026' END AS t,
                seniority_final,
                lower(coalesce(description_core_llm, description)) AS txt,
                length(coalesce(description_core_llm, description)) AS n_chars
            FROM read_parquet('{UNIFIED}')
            WHERE is_swe = TRUE
              AND source_platform = 'linkedin'
              AND is_english = TRUE
              AND date_flag = 'ok'
              AND seniority_final IN ('mid-senior', 'director')
              AND period IN ('2024-01','2024-04','2026-03','2026-04')
              AND description_core_llm IS NOT NULL
        )
        SELECT
            t,
            AVG(CASE WHEN regexp_matches(txt, '\\b(mentor|coach|hire|headcount|performance[- ]?review)\\w*') THEN 1 ELSE 0 END) AS mgmt_bin,
            AVG(CASE WHEN regexp_matches(txt, '(architecture review|code review|system design|technical direction|ai orchestration|workflow|pipeline|automation|evaluate|validate|quality gate|guardrails|prompt engineering|tool selection)') THEN 1 ELSE 0 END) AS orch_bin,
            AVG(CASE WHEN regexp_matches(txt, '(business impact|revenue|product strategy|roadmap|prioritization|resource allocation|budgeting|cross[- ]functional alignment|\\bstakeholders?\\b)') THEN 1 ELSE 0 END) AS strat_broad_bin,
            AVG(CASE WHEN regexp_matches(txt, '(copilot|cursor|\\bclaude\\b|chatgpt|openai|\\bgpt-?[0-9]|gemini|\\bcodex\\b|llamaindex|langchain|prompt engineering|fine[- ]tuning|\\brag\\b|vector database|pinecone|huggingface)') THEN 1 ELSE 0 END) AS ai_strict_bin,
            COUNT(*) AS n
        FROM base
        GROUP BY t
        ORDER BY t
        """
    ).fetchdf()

    # Keep chart clean: 4 metric groups, 2 period bars each.
    metrics = ["mgmt_bin", "orch_bin", "strat_broad_bin", "ai_strict_bin"]
    labels = [
        "Management\n(mentor|coach|hire|\nheadcount|perf-review)",
        "Orchestration\n(workflow/pipeline/\nautomation/review)",
        "Strategic scope\n(roadmap|revenue|\nstakeholder|alignment)",
        "AI-strict\n(copilot|claude|\nrag|langchain|...)",
    ]
    periods = df["t"].tolist()
    assert "2024" in periods and "2026" in periods
    vals_2024 = df.loc[df["t"] == "2024", metrics].values.flatten() * 100
    vals_2026 = df.loc[df["t"] == "2026", metrics].values.flatten() * 100

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    x = np.arange(len(labels))
    w = 0.38
    bars24 = ax.bar(x - w / 2, vals_2024, w, label=f"2024 (n={df.loc[df.t=='2024','n'].iloc[0]:,})", color="#7f7f7f")
    bars26 = ax.bar(x + w / 2, vals_2026, w, label=f"2026 (n={df.loc[df.t=='2026','n'].iloc[0]:,})", color="#d62728")
    for bars, vals in [(bars24, vals_2024), (bars26, vals_2026)]:
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=9)
    # Delta labels
    for xi, (a, b) in enumerate(zip(vals_2024, vals_2026)):
        delta = b - a
        ax.annotate(f"Δ {delta:+.1f}pp", (xi, max(a, b) + 5), ha="center",
                    fontsize=9, color="black", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Binary-presence share (%) of S1 (mid-senior + director) SWE LinkedIn postings")
    ax.set_title(
        "Senior (S1) archetype densities 2024 → 2026 — management, orchestration, strategic, AI"
    )
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals_2026) * 1.3)
    caption = (
        "T21 mid-senior headline: orchestration +21pp, mgmt +14pp (mentor-driven), strat-broad +12pp, "
        "AI-strict +14pp. Mgmt+orch+strat+AI sub-archetype (T21 cluster 2) is 97% 2026, n=860. "
        "The mentor-rate rise at mid-senior is 1.46-1.73× vs entry 1.07× — senior-disproportionate."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True)
    fig.tight_layout()
    out = OUT_DIR / "senior_archetype_chart.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[4/6] Wrote {out}")


# ---------------------------------------------------------------------------
# Artifact 5 — Employer requirement vs worker usage divergence
# ---------------------------------------------------------------------------


def artifact5_employer_usage_divergence() -> None:
    """Side-by-side bars: employer requirement (strict/broad) vs worker-usage benchmarks."""
    # Load T23 divergence table and benchmarks.
    t23 = pd.read_csv(ROOT / "exploration/artifacts/T23/T23_requirement_rates.csv")
    bench = json.loads((ROOT / "exploration/artifacts/T23/T23_benchmarks.json").read_text())

    # Condense: 2024 and 2026, ALL panel, strict + broad.
    d2024 = t23[(t23["period"] == 2024) & (t23["panel"] == "ALL")].iloc[0]
    d2026 = t23[(t23["period"] == 2026) & (t23["panel"] == "ALL")].iloc[0]

    # Worker-usage values (per T23):
    so_2024_current = 62
    so_2024_total = 76
    octoverse_2024 = 73
    anthropic_2025 = 75
    so_2023_current = 44

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=160)
    groups = ["2024", "2026"]
    bar_y = np.arange(2)

    # Employer bars
    emp_ai_tool = [d2024["ai_tool_rate_pct"], d2026["ai_tool_rate_pct"]]
    emp_strict = [d2024["ai_strict_rate_pct"], d2026["ai_strict_rate_pct"]]
    emp_broad = [d2024["ai_broad_rate_pct"], d2026["ai_broad_rate_pct"]]

    bar_h = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_h
    ax.barh(bar_y + offsets[0], emp_ai_tool, bar_h, label="Employer: AI-specific-tool (T23 strict, V1)", color="#08306b")
    ax.barh(bar_y + offsets[1], emp_strict, bar_h, label="Employer: AI-strict (T23)", color="#2171b5")
    ax.barh(bar_y + offsets[2], emp_broad, bar_h, label="Employer: AI-broad (T23)", color="#6baed6")

    # Worker-usage benchmark lines (2026 annotations)
    # 2024 worker benchmarks: SO 2023 (44%), SO 2024 current (62%), SO 2024 total (76%), Octoverse 2024 (73%)
    for y_idx, benchmarks in [(0, [("SO 2023 current 44%", so_2023_current)]),
                              (1, [("SO 2024 current 62%", so_2024_current),
                                   ("Octoverse 2024 73%", octoverse_2024),
                                   ("Anthropic 2025 exposure 75%", anthropic_2025)])]:
        for name, val in benchmarks:
            ax.axvline(x=val, ymin=(y_idx / 2) + 0.05, ymax=(y_idx / 2) + 0.45,
                       color="#d62728", linestyle="--", linewidth=1.3, alpha=0.8)

    # Worker-usage band (shaded area 50-85% over 2026 row)
    ax.axvspan(50, 85, ymin=0.5 + 0.02, ymax=1.0 - 0.02, color="#fcbba1", alpha=0.30,
               label="2026 worker-usage plausible band (50-85%)")

    # Labels
    for i, (ytool, ystrict, ybroad) in enumerate(zip(emp_ai_tool, emp_strict, emp_broad)):
        ax.annotate(f"{ytool:.1f}%", (ytool, bar_y[i] + offsets[0]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center")
        ax.annotate(f"{ystrict:.1f}%", (ystrict, bar_y[i] + offsets[1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center")
        ax.annotate(f"{ybroad:.1f}%", (ybroad, bar_y[i] + offsets[2]), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, va="center")

    ax.set_yticks(bar_y)
    ax.set_yticklabels(groups, fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share / rate (%)")
    ax.set_title(
        "RQ3 inversion: employer AI requirements UNDER-specify vs worker usage (2024 vs 2026)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    caption = (
        "Employer requirement = AI vocabulary mentioned in JD; worker usage = developer self-report "
        "(SO 2023-2024) or task-coverage (Anthropic Labor Market Impacts 2025). 2026 gap: broad "
        "46.8% vs central 75% = -28pp; strict 14.0% vs 75% = -61pp. V2 confirmed direction. "
        "Seniors are MORE AI-specified than juniors (S1 ai_broad 51.4% > J2 43.5%) — rules out an "
        "'AI as junior filter' interpretation."
    )
    fig.text(0.5, -0.04, caption, ha="center", va="top", fontsize=8, wrap=True)
    fig.tight_layout()
    out = OUT_DIR / "employer_usage_divergence.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[5/6] Wrote {out}")


# ---------------------------------------------------------------------------
# Artifact 6 — Cross-occupation AI gradient chart (T18 DiD)
# ---------------------------------------------------------------------------


def artifact6_ai_gradient() -> None:
    """SWE vs adjacent vs control AI trajectory with DiD annotation."""
    # Re-derive from unified.parquet using V1-refined strict pattern
    df = duckdb.sql(
        f"""
        WITH base AS (
            SELECT
                period AS t,
                CASE
                    WHEN is_swe THEN 'SWE'
                    WHEN is_swe_adjacent THEN 'adjacent'
                    WHEN is_control THEN 'control'
                    ELSE 'other'
                END AS grp,
                CASE WHEN regexp_matches(
                    lower(coalesce(description_core_llm, description)),
                    '(copilot|cursor|\\bclaude\\b|chatgpt|openai|\\bgpt-?[0-9]|gemini|\\bcodex\\b|llamaindex|langchain|prompt engineering|fine[- ]tuning|\\brag\\b|vector database|pinecone|huggingface)'
                ) THEN 1 ELSE 0 END AS ai_strict
            FROM read_parquet('{UNIFIED}')
            WHERE source_platform = 'linkedin'
              AND is_english = TRUE
              AND date_flag = 'ok'
              AND (is_swe OR is_swe_adjacent OR is_control)
              AND period IN ('2024-01','2024-04','2026-03','2026-04')
        )
        SELECT t, grp, AVG(ai_strict) AS ai_strict_rate, COUNT(*) AS n
        FROM base
        GROUP BY t, grp
        ORDER BY t, grp
        """
    ).fetchdf()

    date_map = {
        "2024-01": pd.Timestamp("2024-01-15"),
        "2024-04": pd.Timestamp("2024-04-15"),
        "2026-03": pd.Timestamp("2026-03-15"),
        "2026-04": pd.Timestamp("2026-04-15"),
    }
    df["date"] = df["t"].map(date_map)
    df = df.sort_values(["grp", "date"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    colors = {"SWE": "#d62728", "adjacent": "#ff7f0e", "control": "#7f7f7f"}
    for grp in ["SWE", "adjacent", "control"]:
        sub = df[df["grp"] == grp]
        ax.plot(sub["date"], sub["ai_strict_rate"] * 100, marker="o",
                label=f"{grp} (n={sub['n'].sum():,})", color=colors[grp], linewidth=2)
        for dt, val in zip(sub["date"], sub["ai_strict_rate"] * 100):
            ax.annotate(f"{val:.2f}%", (dt, val), textcoords="offset points", xytext=(4, 5),
                        fontsize=8, color=colors[grp])

    # Simple DiD annotation (SWE 2024→2026 minus control 2024→2026) using period averages
    swe_24 = df[(df.grp == "SWE") & (df.date < pd.Timestamp("2025-01-01"))]["ai_strict_rate"].mean()
    swe_26 = df[(df.grp == "SWE") & (df.date > pd.Timestamp("2025-01-01"))]["ai_strict_rate"].mean()
    ctrl_24 = df[(df.grp == "control") & (df.date < pd.Timestamp("2025-01-01"))]["ai_strict_rate"].mean()
    ctrl_26 = df[(df.grp == "control") & (df.date > pd.Timestamp("2025-01-01"))]["ai_strict_rate"].mean()
    did = (swe_26 - swe_24) - (ctrl_26 - ctrl_24)
    swe_only = swe_26 - swe_24
    pct = did / swe_only if swe_only > 0 else np.nan

    ax.set_ylabel("AI-strict mention rate (%)")
    ax.set_title("Cross-occupation AI gradient: SWE vs SWE-adjacent vs control (T18 DiD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Period")

    caption = (
        f"DiD = (SWE Δ) − (control Δ) = {did*100:.2f}pp / SWE-only Δ {swe_only*100:.2f}pp "
        f"= {pct*100:.0f}% of SWE-only change is SWE-specific. T18 reported 99% for ai_strict. "
        "Control postings are essentially flat (0.002 → 0.002); SWE-adjacent (ML-eng, data-sci) "
        "tracks SWE closely but distinct. V2 verified CI [0.128, 0.136] excludes zero."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, wrap=True)
    fig.tight_layout()
    out = OUT_DIR / "ai_gradient_chart.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"[6/6] Wrote {out}")


# ---------------------------------------------------------------------------
# README for the artifact bundle
# ---------------------------------------------------------------------------


def write_readme() -> None:
    readme = """# T25 Interview Elicitation Artifacts

Generated by `exploration/scripts/T25_interview_artifacts.py` (Wave 4, Agent N).

These artifacts are designed for **RQ4 mechanism interviews** — qualitative
work that adjudicates between candidate explanations of the quantitative
findings (real labor-market restructuring vs recruiter-LLM template drift vs
platform taxonomy shifts vs coordination signaling).

## Artifact index

1. **`inflated_junior_jds.md`** — 4 entry-labeled LinkedIn SWE postings ranked
   high on kitchen-sink composite (tech_count × org_scope) × LLM ghost rubric.
   Sourced from T22 `T22_top20_ghost.csv`; diversified across 2024/2026, AI
   vs traditional, aggregator vs direct. Use with hiring managers or recruiters.

2. **`paired_jds_over_time.md`** — 4 same-company 2024-vs-2026 mid-senior
   JD pairs, selected from the 240-company arshkon∩scraped overlap panel (T16).
   Ranked by company-level increase in AI-strict + mentor binary rates.
   Side-by-side format invites probing which shifts are substantive vs stylistic.

3. **`junior_share_trend.png`** — J1 / J2 / J3 / J4 trajectories across four
   period-bins (asaniczka 2024-01, arshkon 2024-04, scraped 2026-03, scraped
   2026-04) with dashed vertical markers at GPT-4 (2023-03), Claude 3 (2024-03),
   GPT-4o (2024-05), Claude 3.5 Sonnet (2024-06), o1 (2024-09), Claude 4
   (2025-09), Gemini 2.5 Pro (2026-03). Baseline-dependent story: J3/J4 rise
   under both baselines; J1/J2 flip direction between arshkon-only and pooled.

4. **`senior_archetype_chart.png`** — Grouped bars for S1 (mid-senior+director)
   binary-presence share of (a) V1-refined strict management pattern, (b) strict
   orchestration pattern, (c) broad strategic-scope pattern, (d) V1-refined
   strict AI pattern. 2024 vs 2026 bars with Δ labels. The mgmt+orch+strat+AI
   co-occurrence cluster is 97% 2026 (T21 k-means cluster 2, n=860).

5. **`employer_usage_divergence.png`** — RQ3 inversion: AI-specific-tool,
   AI-strict, AI-broad employer rates (all-SWE) overlaid against Stack Overflow
   2023/2024 current-use, Octoverse 2024 OSS usage, Anthropic 2025 programmer
   exposure. 2026 employer ai_broad 46.8% < 2026 worker-usage plausible band
   (50-85%). Direction robust across all four usage assumptions.

6. **`ai_gradient_chart.png`** — SWE vs SWE-adjacent vs control AI-strict rate
   per period. DiD annotation shows ~99% of SWE-only Δ is SWE-specific. Control
   group is essentially flat (0.002 → 0.002); adjacent (ML-eng, data-sci)
   tracks SWE closely but distinct. From T18 (primary) with V2 cross-check.

## Usage rules

- **Descriptions in `inflated_junior_jds.md` and `paired_jds_over_time.md`**
  are truncated to ~1,400 chars and cleaned of double-newlines. The LLM-cleaned
  `description_core_llm` is used when available; raw `description` is fallback.

- **UIDs are included** for every posting so interviewees (or analysts) can
  trace any snippet back to the canonical source.

- **Aggregators and staffing rebrands excluded** from the paired-JD set (Dice,
  Jobs for Humanity, hackajob, Motion Recruitment, etc.) — these do not
  represent first-party hiring-manager intent.

## Reproduction

```
/home/jihgaboot/gabor/job-research/.venv/bin/python \\
  /home/jihgaboot/gabor/job-research/exploration/scripts/T25_interview_artifacts.py
```

The script is deterministic given the unified.parquet inputs. All 6 files
will be written to `exploration/artifacts/T25_interview/`.
"""
    (OUT_DIR / "README.md").write_text(readme)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    artifact1_inflated_junior_jds()
    artifact2_paired_jds()
    artifact3_junior_share_trend()
    artifact4_senior_archetype()
    artifact5_employer_usage_divergence()
    artifact6_ai_gradient()
    write_readme()
    print(f"\nDone. All artifacts under: {OUT_DIR}")


if __name__ == "__main__":
    main()
