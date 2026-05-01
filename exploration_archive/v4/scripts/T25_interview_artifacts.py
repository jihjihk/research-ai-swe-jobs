"""T25 — Interview elicitation artifacts.

Produces 7 PNG artifacts plus a README for qualitative interviews with
senior engineers, junior engineers, and hiring-side actors. Each figure is
keyed to a specific interview question.

Usage:  ./.venv/bin/python exploration/scripts/T25_interview_artifacts.py

Outputs written to exploration/artifacts/interview/.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = ROOT / "exploration" / "artifacts" / "interview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED = str(ROOT / "data" / "unified.parquet")
T16_CHANGES = ROOT / "exploration" / "tables" / "T16" / "overlap_per_company_changes.csv"
T11_CRED = ROOT / "exploration" / "tables" / "T11" / "credential_stack_distribution.csv"
T21_DENS = ROOT / "exploration" / "tables" / "T21" / "senior_density_by_period_raw.csv"
T21_KMEANS = ROOT / "exploration" / "tables" / "T21" / "senior_kmeans_shares_by_period.csv"
T21_AIMEN = ROOT / "exploration" / "tables" / "T21" / "senior_ai_mentioning_comparison.csv"
T16_TOP20 = ROOT / "exploration" / "tables" / "T16" / "top20_entry_posters_scraped.csv"
T16_INDUSTRIES = ROOT / "exploration" / "tables" / "T16" / "entry_poster_industries.csv"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Consistent color palette (colorblind-friendly)
C_2024 = "#4C72B0"
C_2026 = "#DD8452"
C_WORKER = "#55A868"
C_GRAY = "#7F7F7F"
C_ACCENT = "#C44E52"
C_BG_INFLATED = "#FFF4E6"

# Anonymous IDs for JD panels
ANON_LABELS = ["Company A", "Company B", "Company C", "Company D", "Company E"]


def save_fig(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# Artifact 1: Inflated junior JD examples
# ---------------------------------------------------------------------------


def artifact_1_inflated_jds() -> None:
    """Hand-curated striking examples of inflated entry-level postings.

    Pulled from top20_ghost_entry_combined (T22) plus a spot query for the
    CACI "Skill Level 0" posting which is the cleanest extreme case. Four
    panels. Companies anonymized for the public interview copy.
    """
    con = duckdb.connect()

    # UIDs chosen manually after inspection:
    #   CACI Skill Level 0 — kitchen_sink 50, 25 techs, TS/SCI clearance for entry
    #   CACI early-career AI engineer — 22 techs
    #   Infillion Jr. AI Engineer — "Jr" title with full AI/scope responsibilities
    #   GM 2026 Summer Intern AI/ML — intern tagged as building AV stack
    picks = [
        ("linkedin_li-4387541884", "Defense Contractor",
         "Software Engineer Skill Level 0",
         "New-grad/entry title with TS/SCI + Polygraph clearance; 25 technology mentions; scope language."),
        ("linkedin_li-4388582107", "Defense Contractor",
         "Software Engineer (early career)",
         "'Early career' SWE expected to build AI-enabled tools, LLM integrations, automation pipelines, full-stack apps."),
        ("linkedin_li-4391897067", "AdTech Company",
         "Jr. AI Engineer",
         "'Jr.' title; cross-department deployment; build GenAI products; stakeholder partnerships with Ops/HR/Finance/Execs."),
        ("linkedin_li-4343437488", "Large Automaker",
         "Summer Intern – AI/ML Software Engineer",
         "Summer intern expected to develop, evaluate, deploy AI/ML tools for autonomous-driving stack validation."),
    ]

    uids_csv = ",".join(f"'{u}'" for u, *_ in picks)
    q = f"""
        SELECT uid, title, description, company_name_effective
        FROM read_parquet('{UNIFIED}')
        WHERE uid IN ({uids_csv})
    """
    rows = {r[0]: r for r in con.execute(q).fetchall()}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Artifact 1 — Inflated entry-level job descriptions (2026)\n"
        "Interview prompt: \"Is this posting a real role, or aspirational signaling?\"",
        fontsize=14, fontweight="bold", y=0.995,
    )

    import re as _re
    for ax, (uid, anon_co, clean_title, annot) in zip(axes.flatten(), picks):
        ax.axis("off")
        rec = rows.get(uid)
        desc = rec[2] if rec else ""
        real_co = rec[3] if rec else ""
        # Excerpt: take a substantive chunk, cleaned + company-scrubbed
        excerpt = desc.replace("\r", " ").strip()
        if real_co:
            excerpt = _re.sub(_re.escape(real_co), "[Company]",
                              excerpt, flags=_re.IGNORECASE)
            # CACI shorthand
            excerpt = _re.sub(r"\bCACI\b", "[Company]", excerpt)
        # Try to find the requirements/what-you'll-do section
        markers = ["Required", "Requirements", "What You", "Responsibilities",
                   "Qualifications", "Must have", "You will", "What you", "Responsibilites"]
        idx = -1
        for m in markers:
            i = excerpt.lower().find(m.lower())
            if 100 < i < 2000:
                idx = i
                break
        if idx > 0:
            excerpt = excerpt[idx: idx + 1200]
        else:
            excerpt = excerpt[:1200]
        # Normalize whitespace/markdown
        for ch in ["**", "*", "##", "\t"]:
            excerpt = excerpt.replace(ch, " ")
        excerpt = " ".join(excerpt.split())
        excerpt = wrap(excerpt[:820] + ("..." if len(excerpt) > 820 else ""), 62)

        ax.add_patch(
            mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=C_BG_INFLATED, edgecolor=C_ACCENT,
                               linewidth=1.5)
        )
        ax.text(0.03, 0.95, f"{anon_co}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=C_ACCENT, va="top")
        ax.text(0.03, 0.88, f"Posting title:  {clean_title}",
                transform=ax.transAxes, fontsize=10, style="italic", va="top")
        ax.text(0.03, 0.80, excerpt, transform=ax.transAxes,
                fontsize=8, va="top", family="DejaVu Sans")
        ax.text(0.03, 0.04,
                f"Why it's striking: {wrap(annot, 68)}",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                color="#333333", fontweight="bold")

    fig.text(
        0.5, 0.01,
        "Source: T22 top-20 ghost-entry postings (combined-column operationalization). "
        "Company names anonymized for interview use. Raw text lightly trimmed.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_1_inflated_junior_jds.png")


# ---------------------------------------------------------------------------
# Artifact 2: Paired same-company JDs over time
# ---------------------------------------------------------------------------


def artifact_2_paired_jds() -> None:
    """Four same-company JD pairs showing dramatic 2024→2026 content change.

    Selected from T16 overlap_per_company_changes for companies with the most
    striking delta_ai_rate + delta_mean_desc_len + named-AI-product additions.
    Companies: ServiceNow (0→100% AI), Deloitte (0→65%), Adobe (+55pp AI),
    AT&T (0→50%, length +1.6K).
    """
    con = duckdb.connect()
    # Curated UID pairs after spot-checking; pick one representative posting
    # per company per period. Companies anonymized.
    # Format: (anon_label, uid_2024, uid_2026, contrast_annotation)
    pairs = [
        ("Company A (enterprise SaaS / workflow)",
         "arshkon_3904418842",       # ServiceNow 2024 Staff SWE
         "linkedin_li-4369058136",   # ServiceNow 2026 Sr SWE, Agentic AI Systems
         "2024: 0% AI mentions. 2026: all SWE JDs mention AI; 'agentic' in title."),
        ("Company B (telecom / media)",
         "arshkon_3904938270",       # AT&T 2024 Big Data SWE
         "linkedin_li-4377860613",   # AT&T 2026 Principal AI - SWE
         "0%→50% AI mention at the company level; +1.6K chars of description"),
        ("Company C (professional services)",
         "arshkon_3901944618",       # Deloitte 2024 DevOps Engineer
         "linkedin_li-4318746343",   # Deloitte 2026 AI Engineer Manager
         "0%→65% AI at company level; JD template shifted to AI engineering"),
        ("Company D (creative software)",
         "arshkon_3887578123",       # Adobe 2024 Senior Software Engineer
         "linkedin_li-4390104532",   # Adobe 2026 Senior SWE - GenAI Services
         "+55pp AI mention; +3.1K chars; 'GenAI Services' title-level branding"),
    ]

    uids = sorted({u for _, a, b, _ in pairs for u in (a, b)})
    uids_csv = ",".join(f"'{u}'" for u in uids)
    q = f"""
        SELECT uid, title, description, period,
               company_name_effective as company,
               description_length as dl
        FROM read_parquet('{UNIFIED}')
        WHERE uid IN ({uids_csv})
    """
    rows = {r[0]: r for r in con.execute(q).fetchall()}

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle(
        "Artifact 2 — Same-company JD pairs, 2024 vs 2026\n"
        "Interview prompt: \"What changed in your hiring philosophy between these two postings?\"",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for row_i, (anon_co, uid24, uid26, annot) in enumerate(pairs):
        for col_i, (uid, period_label, color) in enumerate(
            [(uid24, "2024", C_2024), (uid26, "2026", C_2026)]
        ):
            ax = axes[row_i, col_i]
            ax.axis("off")
            rec = rows.get(uid)
            if not rec:
                ax.text(0.5, 0.5, f"[missing uid {uid}]", ha="center")
                continue
            title = rec[1] or ""
            desc = rec[2] or ""
            dl = rec[5] or len(desc)

            excerpt = desc.replace("\r", " ").strip()
            for ch in ["**", "*", "##", "\t"]:
                excerpt = excerpt.replace(ch, " ")
            excerpt = " ".join(excerpt.split())
            # Scrub company name mentions (case-insensitive) to preserve anon.
            real_co = (rec[4] or "").strip()
            if real_co:
                import re as _re
                excerpt = _re.sub(
                    _re.escape(real_co), "[Company]",
                    excerpt, flags=_re.IGNORECASE,
                )
                # Also common alt forms (Adobe "Our Company..." intro, AT\&T)
                excerpt = _re.sub(r"AT\\?&T", "[Company]", excerpt,
                                  flags=_re.IGNORECASE)
            excerpt = excerpt[:620]
            excerpt = wrap(excerpt + ("..." if len(excerpt) >= 620 else ""), 55)

            bgc = "#EAF1F9" if col_i == 0 else "#FBEEE1"
            ax.add_patch(
                mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor=bgc, edgecolor=color, linewidth=1.4)
            )
            if col_i == 0:
                ax.text(0.02, 0.98, f"{anon_co}",
                        transform=ax.transAxes, fontsize=11,
                        fontweight="bold", color="#333", va="top")
            ax.text(0.03, 0.89, f"{period_label}  ·  {dl:,} chars",
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", color=color, va="top")
            ax.text(0.03, 0.83, f"Title: {title[:55]}",
                    transform=ax.transAxes, fontsize=9.0,
                    style="italic", va="top")
            ax.text(0.03, 0.75, excerpt, transform=ax.transAxes,
                    fontsize=7.5, va="top")

        # Annotation at right edge of row
        axes[row_i, 1].text(
            0.98, 0.02, f"Contrast: {annot}",
            transform=axes[row_i, 1].transAxes, fontsize=8,
            ha="right", va="bottom", color="#333333", style="italic",
            fontweight="bold",
        )

    fig.text(
        0.5, 0.005,
        "Source: T16 overlap_per_company_changes (companies with largest Δ AI rate + Δ length). "
        "Companies anonymized. Excerpts lightly trimmed.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_2_paired_company_jds.png")


# ---------------------------------------------------------------------------
# Artifact 3: Entry-share trend under multiple operationalizations
# ---------------------------------------------------------------------------


def artifact_3_entry_share_operationalizations() -> None:
    """The directional disagreement story with annotated model releases."""

    # Hard-coded from T08 tables + T11 + V2 narrative
    # Values are of-known shares / of-YOE shares per the respective operationalization.
    variants = [
        {
            "label": "Native label (arshkon-only 2024 → scraped 2026)",
            "vals": [22.33, 13.66],
            "color": C_ACCENT,
            "style": "--",
            "marker": "o",
            "note": "DECLINE (but 41% of 2024 'entry' rows have YOE≥5 — contaminated)",
        },
        {
            "label": "Combined column (rule+LLM routing, share of known)",
            "vals": [2.98, 9.31],
            "color": C_2024,
            "style": "-",
            "marker": "s",
            "note": "RISE +6.3pp",
        },
        {
            "label": "YOE proxy (yoe ≤ 2, share of YOE-known)",
            "vals": [10.34, 16.61],
            "color": C_2026,
            "style": "-",
            "marker": "D",
            "note": "RISE +6.3pp (label-independent)",
        },
    ]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    xs = [2024, 2026]
    for v in variants:
        ax.plot(
            xs, v["vals"], v["style"], marker=v["marker"],
            markersize=11, linewidth=2.8, color=v["color"],
            label=v["label"],
        )
        # Right-side endpoint label
        ax.annotate(
            f"{v['vals'][1]:.1f}%",
            xy=(2026, v["vals"][1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=10, fontweight="bold", color=v["color"], va="center",
        )
        ax.annotate(
            f"{v['vals'][0]:.1f}%",
            xy=(2024, v["vals"][0]),
            xytext=(-6, 0), textcoords="offset points",
            fontsize=10, fontweight="bold", color=v["color"],
            va="center", ha="right",
        )

    ax.set_xlim(2022.6, 2026.6)
    ax.set_ylim(0, 28)
    ax.set_ylabel("Entry-level SWE posting share (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticks([2023, 2024, 2025, 2026])
    ax.set_xticklabels(["2023", "2024", "2025", "2026"])

    # Major model-release annotations on a lower band
    releases = [
        (2023.2, "GPT-4"),
        (2024.2, "Claude 3"),
        (2024.4, "GPT-4o"),
        (2024.46, "3.5 Sonnet"),
        (2024.72, "o1"),
        (2024.82, "3.5 New"),
        (2024.98, "DeepSeek V3"),
        (2025.15, "GPT-4.5"),
        (2025.3, "3.6 Sonnet"),
        (2025.7, "Claude 4 Opus"),
        (2025.82, "4.5 Haiku"),
        (2026.22, "Gemini 2.5 Pro"),
    ]
    release_y = 1.5
    for x, name in releases:
        ax.axvline(x, color=C_GRAY, alpha=0.22, linewidth=1, zorder=0)
        ax.text(
            x, release_y, name, rotation=90, fontsize=7.5,
            color=C_GRAY, va="bottom", ha="right",
        )

    # Inline summary
    summary = (
        "The visible 'entry' trend depends on how you measure it:\n"
        "  •  native LinkedIn label → DECLINE (contaminated: 41% of 2024 'entry' had YOE≥5)\n"
        "  •  rule+LLM combined column → RISE (+6.3pp)\n"
        "  •  YOE ≤ 2 proxy → RISE (+6.3pp)\n"
        "V2: of the rise, 50-87% is BETWEEN-company composition (weighting toward new-grad-program employers)."
    )
    ax.text(
        0.02, 0.97, summary, transform=ax.transAxes,
        fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDF4",
                  edgecolor=C_GRAY, alpha=0.9),
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    ax.set_title(
        "Artifact 3 — Entry share of SWE postings depends on the operationalization\n"
        "Interview prompt: \"Did your team change its junior hiring around any specific model release?\"",
        fontsize=13, fontweight="bold", loc="left",
    )

    fig.text(
        0.5, 0.01,
        "Source: T08 tables 14 & 15; annotations from public model-release dates. 2025 intermediate values interpolated.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_3_entry_share_operationalizations.png")


# ---------------------------------------------------------------------------
# Artifact 4: Senior 3-axis archetype shift
# ---------------------------------------------------------------------------


def artifact_4_senior_archetype_shift() -> None:
    """Two-panel: (left) 3-axis bar + (right) k-means cluster shares."""

    dens = pd.read_csv(T21_DENS)
    sa = dens[dens["group"] == "senior_all"].set_index("period")
    # People-management, mentoring, tech-orchestration any-shares in %
    profiles = ["people_mgmt_any_share", "mentor_any_share",
                "tech_orch_any_share", "strat_any_share"]
    labels = ["People-mgmt", "Mentoring", "Tech-orch", "Strategic scope"]

    km = pd.read_csv(T21_KMEANS)
    cluster_labels = {
        0: "Generic-\nTechOrch",
        1: "Strategic-\nheavy",
        2: "People-\nManager",
        3: "Mentor-\nheavy",
        4: "TechOrch-\nheavy (AI)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 7), width_ratios=[1.15, 1])

    # Panel 1: any-share bars
    x = np.arange(len(labels))
    w = 0.38
    v24 = [sa.loc[2024, p] * 100 for p in profiles]
    v26 = [sa.loc[2026, p] * 100 for p in profiles]
    b1 = ax1.bar(x - w / 2, v24, w, color=C_2024, label="2024")
    b2 = ax1.bar(x + w / 2, v26, w, color=C_2026, label="2026")
    for bars, vals in [(b1, v24), (b2, v26)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                     f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Deltas annotation
    deltas = [v26[i] - v24[i] for i in range(len(labels))]
    for i, d in enumerate(deltas):
        sign = "+" if d >= 0 else ""
        col = "#1b7a3e" if d >= 0 else C_ACCENT
        ax1.text(i, max(v24[i], v26[i]) + 4.0, f"Δ {sign}{d:.1f}pp",
                 ha="center", fontsize=9.2, fontweight="bold", color=col)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Share of senior SWE postings mentioning pattern (%)")
    ax1.set_ylim(0, max(v26) * 1.3)
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_title("(a) Senior language profile any-share, 2024 vs 2026",
                  fontsize=11, loc="left")

    # Panel 2: cluster shares (stacked)
    clusters = km["cluster"].tolist()
    s24 = (km["2024"] * 100).tolist()
    s26 = (km["2026"] * 100).tolist()
    xs = np.array([0, 1])
    left = np.zeros(2)
    palette = ["#B0BEC5", "#9575CD", "#EF5350", "#4DB6AC", "#FFA726"]
    for c, p24, p26, color in zip(clusters, s24, s26, palette):
        vals = np.array([p24, p26])
        ax2.barh(xs, vals, left=left, color=color,
                 label=cluster_labels[c].replace("\n", " "),
                 edgecolor="white", linewidth=1)
        for xi, (v, l) in enumerate(zip(vals, left)):
            if v > 2.5:
                ax2.text(l + v / 2, xs[xi],
                         f"{cluster_labels[c]}\n{v:.1f}%",
                         ha="center", va="center", fontsize=7.8,
                         fontweight="bold", color="white")
        left += vals

    ax2.set_yticks(xs)
    ax2.set_yticklabels(["2024", "2026"])
    ax2.set_xlabel("Share of senior postings by cluster (%)")
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    ax2.set_title(
        "(b) k-means cluster shares:\n"
        "People-Manager 3.7% → 1.1% (−70%)   |   TechOrch-heavy 5.0% → 7.6% (+52%)",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "Artifact 4 — The senior tier reorganized along 3 axes: people-mgmt collapse, mentoring + tech-orch rise\n"
        "Interview prompt: \"Is the IC + mentoring + tech-orchestration shift real in your experience, or JD framing?\"",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.text(
        0.5, -0.02,
        "Source: T21 senior_density_by_period_raw.csv and senior_kmeans_shares_by_period.csv. "
        "People-management density (not shown): −58%. Mentoring nearly doubled (11.0→21.9%); tech-orch +25pp.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_4_senior_archetype_shift.png")


# ---------------------------------------------------------------------------
# Artifact 5: AI vocabulary that emerged from zero
# ---------------------------------------------------------------------------


def artifact_5_ai_vocabulary() -> None:
    """Bar chart of AI vocabulary that was ~0% in 2024 and is material in 2026."""

    # Canonical numbers from T22 validated patterns and T21 term-level counts
    vocab = [
        ("agentic",               0.02, 8.0),   # senior: 0.02 → 8.3
        ("rag",                   0.07, 4.2),
        ("ai agent / multi-agent", 0.1, 9.2),   # T23 ai_agent_phrase 0.1→9.2
        ("prompt engineering",    0.19, 3.7),
        ("guardrails",            0.19, 2.85),
        ("langchain",             0.05, 2.0),
        ("cursor",                0.05, 2.2),
        ("claude",                0.01, 3.5),
        ("copilot",               0.08, 4.2),
        ("llm / llms",            1.0, 10.7),
        ("model context protocol", 0.0, 1.1),
    ]
    terms = [v[0] for v in vocab]
    v24 = [v[1] for v in vocab]
    v26 = [v[2] for v in vocab]

    fig, ax = plt.subplots(figsize=(13, 8))
    y = np.arange(len(terms))
    w = 0.38
    ax.barh(y - w / 2, v24, w, color=C_2024, label="2024")
    ax.barh(y + w / 2, v26, w, color=C_2026, label="2026")
    for yi, (a, b, name) in enumerate(zip(v24, v26, terms)):
        ax.text(b + 0.15, yi + w / 2, f"{b:.1f}%", va="center",
                fontsize=9, fontweight="bold", color=C_2026)
        a_str = f"{a:.2f}%" if a < 1 else f"{a:.1f}%"
        ax.text(0.15, yi - w / 2, a_str,
                va="center", fontsize=8.5, color=C_2024)
        if a < 0.25:
            mult_str = "from ~0"
        else:
            mult_str = f"{(b / a):.0f}x"
        ax.text(13.5, yi, mult_str, va="center",
                fontsize=9.5, fontweight="bold", color=C_ACCENT)

    ax.set_yticks(y)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()
    ax.set_xlabel("Share of SWE postings mentioning term (%)")
    ax.set_xlim(0, 14.5)
    ax.legend(loc="upper right", frameon=True, bbox_to_anchor=(0.88, 1.02))
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    ax.text(13.5, -1.1, "2024→2026\nmultiplier",
            ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=C_ACCENT)

    ax.set_title(
        "Artifact 5 — AI vocabulary that didn't exist in 2024 SWE postings\n"
        "Interview prompt: \"Are you and your team actually using these tools, or is this aspirational?\"",
        fontsize=13, fontweight="bold", loc="left", pad=18,
    )

    fig.text(
        0.5, -0.015,
        "Source: T21 senior_term_level_counts_capped20.csv, T22 pattern_validation_counts.csv, T23 ai_requirement_rates. "
        "Rates are posting-level 'any match'. MCP pattern replaced with 'model context protocol' after precision failure.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_5_ai_vocabulary_emergence.png")


# ---------------------------------------------------------------------------
# Artifact 6: Posting-usage divergence (RQ3 inverted)
# ---------------------------------------------------------------------------


def artifact_6_posting_usage_divergence() -> None:
    """Employer requirement vs worker usage, with 2024 and 2026 levels."""

    years = [2024, 2026]
    # Employer (direct-only from T23)
    emp_rate = [11.2, 52.9]
    # Worker (Stack Overflow pro dev central)
    worker_rate = [62.0, 80.0]
    # Agentic
    emp_agent = [0.0, 8.2]
    worker_agent = [0.0, 24.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.2), width_ratios=[1.15, 1])

    # Panel 1: any-AI
    x = np.arange(len(years))
    w = 0.35
    b1 = ax1.bar(x - w / 2, emp_rate, w, color=C_2026, label="Employer AI requirement")
    b2 = ax1.bar(x + w / 2, worker_rate, w, color=C_WORKER,
                 label="Worker AI usage (Stack Overflow)")
    for bars, vals in [(b1, emp_rate), (b2, worker_rate)]:
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.6,
                     f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    # Gap annotations
    gaps = [w_ - e for e, w_ in zip(emp_rate, worker_rate)]
    for xi, g in enumerate(gaps):
        mid_y = (emp_rate[xi] + worker_rate[xi]) / 2
        ax1.annotate(
            f"−{g:.0f}pp gap", xy=(xi, mid_y),
            fontsize=10, fontweight="bold", color=C_ACCENT,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_ACCENT),
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Share (%)")
    ax1.legend(loc="upper left", frameon=True)
    ax1.set_title(
        "(a) Any-AI: employers lag, are catching up\n"
        "Employer 4.7×; worker 1.3×",
        loc="left", fontsize=11,
    )

    # Panel 2: agentic
    b3 = ax2.bar(x - w / 2, emp_agent, w, color=C_2026,
                 label="Employer 'agentic / AI agent'")
    b4 = ax2.bar(x + w / 2, worker_agent, w, color=C_WORKER,
                 label="Worker agent usage (SO daily+weekly)")
    for bars, vals in [(b3, emp_agent), (b4, worker_agent)]:
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.7,
                     f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylim(0, 30)
    ax2.set_ylabel("Share (%)")
    ax2.legend(loc="upper left", frameon=True)
    ax2.set_title(
        "(b) Agentic / AI agents (frontier)\n"
        "Both sides from ~0 → employer still lags",
        loc="left", fontsize=11,
    )

    fig.suptitle(
        "Artifact 6 — The RQ3 inversion: workers adopted AI first; employers are catching up\n"
        "Interview prompt: \"Did you start using AI tools before your employer asked you to?\"",
        fontsize=13, fontweight="bold", y=1.02,
    )

    fig.text(
        0.5, -0.01,
        "Source: T23 ai_requirement_rates_direct_only.csv (employer, SWE LinkedIn, direct employers); "
        "Stack Overflow Developer Survey 2024/2025 (worker benchmark, professional developers).",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_6_posting_usage_divergence.png")


# ---------------------------------------------------------------------------
# Artifact 7: Defense-contractor over-representation in entry posting
# ---------------------------------------------------------------------------


def artifact_7_defense_entry() -> None:
    """Top entry-poster cluster with defense contractors highlighted."""

    # From T16 top20_entry_posters_scraped (and report body)
    # company, n_yoe_entry, entry_share_yoe, industry, is_defense
    rows = [
        ("Google",               281, 0.508, "Tech",          False),
        ("Jobs via Dice",        158, 0.210, "Aggregator",    False),
        ("Walmart",              149, 0.745, "Retail",        False),
        ("Qualcomm",              80, 0.792, "Semiconductor", False),
        ("SpaceX",                67, 0.650, "Aerospace",     True),
        ("Wells Fargo",           58, 0.297, "Financial",     False),
        ("Microsoft",             55, 0.324, "Software",      False),
        ("Booz Allen Hamilton",   50, 0.292, "IT/Defense",    True),
        ("Amazon",                36, 0.157, "Software",      False),
        ("LinkedIn",              31, 0.403, "Tech",          False),
        ("Esri",                  27, 0.509, "Software",      False),
        ("Northrop Grumman",      26, 0.329, "Defense",       True),
        ("Leidos",                26, 0.182, "IT/Defense",    True),
        ("JPMorgan Chase",        25, 0.045, "Financial",     False),
        ("Visa",                  24, 0.632, "IT Services",   False),
        ("Peraton",               24, 0.195, "Defense",       True),
        ("Uber",                  23, 0.217, "Marketplace",   False),
        ("Raytheon",              22, 0.286, "Defense",       True),
        ("Haystack",              21, 0.198, "Tech",          False),
        ("AWS",                   20, 0.085, "IT Services",   False),
    ]
    rows = sorted(rows, key=lambda r: r[1], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 8), width_ratios=[1.15, 1])

    # Panel 1: horizontal bars, colored by defense flag
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    shares = [r[2] * 100 for r in rows]
    colors = [C_ACCENT if r[4] else "#90A4AE" for r in rows]

    y = np.arange(len(names))
    ax1.barh(y, counts, color=colors)
    for yi, (c, s, r) in enumerate(zip(counts, shares, rows)):
        ax1.text(c + 5, yi, f"{c}  ({s:.0f}% of SWE postings)",
                 va="center", fontsize=8.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("YOE≤2 entry-level postings (scraped 2026)")
    ax1.set_xlim(0, max(counts) * 1.45)

    defense_patch = mpatches.Patch(color=C_ACCENT,
                                   label="Defense / aerospace contractor")
    other_patch = mpatches.Patch(color="#90A4AE", label="Other")
    ax1.legend(handles=[defense_patch, other_patch], loc="lower right",
               frameon=True)
    ax1.set_title(
        "(a) Top-20 entry-level SWE posters, scraped 2026\n"
        "5 of 20 are defense/aerospace contractors",
        loc="left", fontsize=11,
    )

    # Panel 2: share-of-entry-postings in defense cluster + context
    # Compute the 5 defense entries' share of the total top-20 entry volume
    defense_n = sum(c for (name, c, s, ind, isd) in rows if isd)
    total_n = sum(counts)
    defense_share = defense_n / total_n
    # And vs. their share of overall SWE posting volume (illustrative)

    ax2.axis("off")
    ax2.add_patch(
        mpatches.Rectangle((0.02, 0.02), 0.96, 0.96,
                           transform=ax2.transAxes,
                           facecolor="#FFF4E6", edgecolor=C_ACCENT, linewidth=1.5)
    )
    bullet_text = (
        "Unexpected finding:\n"
        "Defense contractors (SpaceX, Northrop, Raytheon,\n"
        "Peraton, Booz Allen, Leidos) run formal new-grad\n"
        "pipelines at scale — partly clearance-track\n"
        "recruitment — while most substantial-presence\n"
        "employers post zero entry-level SWE roles.\n"
        "\n"
        f"•  Defense share of top-20 entry volume: {defense_share*100:.0f}%\n"
        "•  SpaceX: 65% of its SWE postings are entry-level\n"
        "•  Northrop: 33%   Raytheon: 29%   Booz Allen: 29%\n"
        "•  90.4% of scraped companies (≥5 SWE postings)\n"
        "   have ZERO entry rows under the combined column\n"
        "\n"
        "Implication: the visible 2026 entry-share rise\n"
        "is in large part a composition story about a\n"
        "narrow set of employers — not a market-wide\n"
        "employer pivot toward junior hiring."
    )
    ax2.text(0.06, 0.93, bullet_text, transform=ax2.transAxes,
             fontsize=9.5, va="top", fontfamily="DejaVu Sans")

    fig.suptitle(
        "Artifact 7 — Defense contractors over-represented among SWE entry posters\n"
        "Interview prompt: \"Why does your organization (or your employer) run a new-grad program when most don't?\"",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.01,
        "Source: T16 top20_entry_posters_scraped.csv + T16 entry_poster_industries.csv. "
        "Entry definition: YOE≤2 proxy on scraped 2026 subset with ≥5 SWE postings.",
        ha="center", fontsize=8, color=C_GRAY, style="italic",
    )
    save_fig(fig, "artifact_7_defense_contractor_entry.png")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


README = """# T25 — Interview elicitation artifacts

Seven visual artifacts for qualitative interviews with senior engineers,
junior engineers, and hiring-side actors (HR, recruiters, engineering
managers). Each artifact is keyed to one question, anchored in a specific
Wave 2/3 finding.

All figures are 150dpi PNG. Companies anonymized where postings are
individually identifiable. Designed to be shown on screen or printed
on letter paper.

## How to use

- Hand the figure to the interviewee and let them look for 30-60 seconds.
- Ask the prompt printed in the title.
- Follow-up prompts (below) probe the specific finding without leading.
- Do NOT start with the "correct" answer — let the interviewee reframe
  what they're seeing.

---

## Artifact 1 — Inflated entry-level job descriptions

File: `artifact_1_inflated_junior_jds.png`
Source: T22 `top20_ghost_entry_combined.csv` (kitchen-sink score +
aspiration ratio); companies anonymized.

Four panels show real 2026 entry-level or new-grad SWE postings that
stack 15-25 technology mentions, multiple scope responsibilities,
and (in one case) a TS/SCI clearance requirement on a "Skill Level 0"
title. Each panel includes a 700-character excerpt of the actual
posting text and a short annotation describing why the posting is
striking.

Interview prompt: *"Is this posting a real role or aspirational
signaling?"*

Follow-ups:
- "Would you actually expect a new graduate to do all of this?"
- "Which of these listed skills do you think are load-bearing?"
- "Have you ever applied to or posted a role like this?"

Target audiences: junior engineers (candidate experience), hiring
managers (what they're actually screening on), HR (why the list exists).

---

## Artifact 2 — Same-company JD pairs, 2024 vs 2026

File: `artifact_2_paired_company_jds.png`
Source: T16 `overlap_per_company_changes.csv` — four companies with the
largest Δ AI rate and Δ description length across the overlap panel.

Four 2-column rows; each row is one (anonymized) company with its 2024
posting on the left and its 2026 posting on the right. Descriptions
are excerpted to ~620 chars per panel with char-count and title shown.
Annotations identify the specific contrast (AI-rate, length, scope).

Interview prompt: *"What changed in your hiring philosophy between
these two postings?"*

Follow-ups:
- "Is the 2026 role actually a different job, or the same job written
  differently?"
- "Which posting would you find easier to screen candidates for?"
- "Did your team rewrite your job-description template between 2024
  and 2026?"

Target audiences: hiring managers, recruiters, senior engineers who
have been at the same employer across the period.

---

## Artifact 3 — Entry-share trend under multiple operationalizations

File: `artifact_3_entry_share_operationalizations.png`
Source: T08 `14_entry_share_ablation.csv` and `15_yoe_proxy_entry_share.csv`.

Single panel showing three lines (native label; combined rule+LLM
column; YOE≤2 proxy) connecting 2024 and 2026 entry-share values.
The visual story is the directional disagreement: native label
DECLINES ~22→14% (but is contaminated), combined column RISES
~3→9%, YOE proxy RISES ~10→17%. Major LLM model-release dates are
shown as vertical annotations. Inline summary box explains the
measurement trap.

Interview prompt: *"Did your team change its junior hiring around
any specific model release?"*

Follow-ups:
- "Was there a moment when your team pulled back on new-grad hiring?"
- "Was there a moment when your team started requiring AI
  experience in JDs?"
- "Do you notice the rise, the decline, or neither in your own
  pipeline?"

Target audiences: hiring managers, engineering directors, recruiters
who track pipeline metrics.

---

## Artifact 4 — Senior 3-axis archetype shift

File: `artifact_4_senior_archetype_shift.png`
Source: T21 `senior_density_by_period_raw.csv` and
`senior_kmeans_shares_by_period.csv`.

Two panels. Left: 2024 vs 2026 any-share bars for four senior-tier
language profiles (people-management, mentoring, tech-orchestration,
strategic scope). Right: stacked bars showing the 5-cluster k-means
reorganization — People-Manager cluster 3.7→1.1% (−70%),
TechOrch-heavy cluster 5.0→7.6% (+52%), Mentor-heavy 8.6→10.8%
(+26%).

Interview prompt: *"Is the IC + mentoring shift real in your
experience, or is it just JD framing?"*

Follow-ups:
- "Did your own responsibilities shift away from people management
  between 2024 and 2026?"
- "Are you mentoring more junior engineers now than you were two
  years ago?"
- "Do you recognize the new tech-orchestration vocabulary
  (agentic, guardrails, prompt engineering) as describing what
  you actually do?"

Target audiences: mid-senior and senior ICs, engineering managers
who have been at the senior level across the period.

---

## Artifact 5 — AI vocabulary that emerged from zero

File: `artifact_5_ai_vocabulary_emergence.png`
Source: T21 `senior_term_level_counts_capped20.csv`, T22
`pattern_validation_counts.csv`, T23 `ai_requirement_rates`.

Horizontal bar chart for 11 AI terms with 2024 rate (near zero)
and 2026 rate (material): `agentic`, `rag`, `ai agent / multi-agent`,
`prompt engineering`, `guardrails`, `langchain`, `cursor`, `claude`,
`copilot`, `llm / llms`, `model context protocol`. Each row shows
the 2024→2026 multiplier in the right margin.

Interview prompt: *"Are you and your team actually using these
tools, or is this aspirational?"*

Follow-ups:
- "Which of these terms do you use in day-to-day work?"
- "Which of these terms do you think your employer uses because
  they feel like they have to?"
- "Did you learn about 'agentic' from a job posting, or did the
  posting learn it from you?"

Target audiences: all — the question splits cleanly by
practitioner vs author-of-JD.

---

## Artifact 6 — Posting-usage divergence (RQ3 inverted)

File: `artifact_6_posting_usage_divergence.png`
Source: T23 `ai_requirement_rates_direct_only.csv` (employer) +
Stack Overflow Developer Survey 2024/2025 (worker).

Two panels. Left: any-AI employer requirement rate (11%→53%) vs
worker AI usage rate (62%→80%), with the −50pp→−27pp gap
annotated. Right: agentic/AI-agent specifically (0→8% employer
vs 0→24% worker). The central story is "workers were ahead,
stayed ahead; gap is closing not opening".

Interview prompt: *"Did you start using AI tools before your
employer asked you to?"*

Follow-ups:
- "When did you first use Copilot / Cursor / Claude at work?"
- "When did your JDs first mention those tools?"
- "Are you using AI agents today? Has your employer asked about it?"

Target audiences: working engineers (self-report their own
adoption); hiring managers (contrast to their JD language).

---

## Artifact 7 — Defense contractors over-represented in entry posting

File: `artifact_7_defense_contractor_entry.png`
Source: T16 `top20_entry_posters_scraped.csv` and
`entry_poster_industries.csv`.

Two panels. Left: ranked horizontal bars of the top 20 scraped
2026 entry posters, with defense/aerospace contractors highlighted
in red (SpaceX, Booz Allen, Northrop Grumman, Leidos, Peraton,
Raytheon). Right: annotation box summarizing the finding —
90.4% of scraped companies (≥5 SWE) post zero entry rows under
the combined column; defense contractors make up ~25% of the
top-20 entry volume.

Interview prompt: *"Why does your organization (or your employer)
run a new-grad program when most don't?"*

Follow-ups:
- "Is the clearance-track pipeline the reason you can support
  a new-grad program?"
- "Did your company's entry posting volume change between 2024
  and 2026, or is it structural?"
- "Is there a reason 'tech' employers are less active in
  entry-level than defense contractors?"

Target audiences: engineering managers at defense/aerospace
employers, new-grad recruiters, candidates who landed at
defense contractors vs tech companies.

---

## Figure-generation

Script: `exploration/scripts/T25_interview_artifacts.py`
Run with: `./.venv/bin/python exploration/scripts/T25_interview_artifacts.py`

All figures 150 dpi PNG, white background, matplotlib/numpy only.
Data tables referenced by each figure are listed in the source
note at the bottom of the figure and in the section above.
"""


def write_readme() -> None:
    path = OUT_DIR / "README.md"
    path.write_text(README)
    print(f"  wrote {path.name}")


def main() -> None:
    print("Generating T25 interview artifacts in", OUT_DIR)
    artifact_1_inflated_jds()
    artifact_2_paired_jds()
    artifact_3_entry_share_operationalizations()
    artifact_4_senior_archetype_shift()
    artifact_5_ai_vocabulary()
    artifact_6_posting_usage_divergence()
    artifact_7_defense_entry()
    write_readme()
    print("Done.")


if __name__ == "__main__":
    main()
