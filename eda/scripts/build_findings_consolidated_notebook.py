"""
Build eda/notebooks/findings_consolidated_2026-04-21.ipynb.

Consolidated findings notebook written for a first-time reader. Single
coherent narrative; no visible references to internal run identifiers or
per-task hypothesis numbering. Inline figures via plt.show().

Run:
  ./.venv/bin/python eda/scripts/build_findings_consolidated_notebook.py

Then:
  ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \\
      eda/notebooks/findings_consolidated_2026-04-21.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = (
    PROJECT_ROOT / "eda" / "notebooks" / "findings_consolidated_2026-04-21.ipynb"
)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # =========================================================================
    # 0 · Title + one-line pitch
    # =========================================================================
    cells.append(md(
        "# How AI is reshaping software-engineering job postings\n\n"
        "**Consolidated findings — 2026-04-21.**\n\n"
        "*30 seconds:* jump to the [tl;dr](#tldr). "
        "*15 minutes:* read straight through. Every section is short and visual.\n\n"
        "---"
    ))

    # =========================================================================
    # 1 · Setup (collapsed)
    # =========================================================================
    cells.append(md(
        "## Setup\n\n"
        "*One-time environment setup — collapse this cell after running it.*"
    ))
    cells.append(code(
        "import sys, os\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path.cwd()\n"
        "ROOT = HERE if (HERE / 'data' / 'unified_core.parquet').exists() else HERE.parents[1]\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, str(ROOT / 'eda' / 'scripts'))\n"
        "\n"
        "import duckdb\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "%matplotlib inline\n"
        "\n"
        "from consolidated_viz import (\n"
        "    viz_junior_scope_panel, viz_senior_scope_inflation,\n"
        "    viz_within_firm, viz_swe_vs_control, viz_yoe_floor,\n"
        "    viz_vendor_leaderboard, viz_bigtech_density,\n"
        "    viz_disproven_aiwashing, viz_disproven_industry_spread,\n"
        "    viz_disproven_juniorfirst, viz_disproven_hiring_bar,\n"
        "    viz_disproven_selectivity, viz_verdict_table,\n"
        ")\n"
        "\n"
        "con = duckdb.connect()\n"
        "print(f'Project root: {ROOT}')"
    ))

    # =========================================================================
    # 2 · Why this exists
    # =========================================================================
    cells.append(md(
        "## Why this exists\n\n"
        "Tech firms have been laying off workers throughout 2023–2026. The loudest narrative "
        "(amplified by tech executives and the press) is that **AI is doing the work**. Coding "
        "assistants like GitHub Copilot, Cursor, and Claude Code now write enough software that "
        "human engineers — so the story goes — are becoming redundant.\n\n"
        "Several alternative explanations compete with that story:\n\n"
        "- **AI-washing** — firms invoke AI as a narrative cover while the real drivers are post-"
        "Covid hiring corrections and rising interest rates.\n"
        "- **Outsourcing** — work migrates to lower-cost regions rather than to AI tools.\n"
        "- **Industry redistribution** — software engineers aren't disappearing; they're moving "
        "from pure-tech firms into non-tech industries like retail, finance, and construction.\n"
        "- **Scope inflation** — junior roles get squeezed: employers demand senior-level "
        "experience for entry-level titles.\n\n"
        "We can't observe layoffs directly in our data, but we can read what employers are "
        "writing in their job postings — which signals what work they expect to need humans for. "
        "This notebook tests these narratives against 110,000 LinkedIn postings and reports what "
        "holds up, what doesn't, and what the data uniquely reveals about the 2024–2026 shift."
    ))

    # =========================================================================
    # 3 · The data
    # =========================================================================
    cells.append(md(
        "## The data\n\n"
        "Everything below comes from one cleaned analysis file: **`data/unified_core.parquet`** "
        "— 110,000 LinkedIn job postings, 42 columns per posting. Each row is one unique job "
        "posting; the columns describe the job (title, full description text), the company "
        "(name, industry, size), the candidate (seniority, years-of-experience), the location "
        "(metro area, remote flag), and provenance (when scraped, language detection, "
        "ghost-job risk).\n\n"
        "Postings come from two sources spanning two time windows:\n\n"
        "- **2024 baseline** — historical LinkedIn snapshots (~57,000 rows, January and April 2024).\n"
        "- **2026 current window** — daily scrapes over March and April 2026 (~53,000 rows).\n\n"
        "The file has been through a multi-stage pipeline (deduplication, occupation "
        "classification, seniority inference, years-of-experience extraction, geographic "
        "normalization, and a deeper LLM read of each description). Every row has been "
        "LLM-labeled for analysis-quality consistency. The sample is balanced 40/30/30 "
        "software-engineer / software-adjacent / control occupations, with the control group "
        "covering non-software jobs: nurses, accountants, civil/mechanical/electrical engineers, "
        "financial analysts, marketing managers, HR, and sales representatives.\n\n"
        "Run the cell below to see the composition:"
    ))
    cells.append(code(
        "profile = con.execute(\"\"\"\n"
        "  SELECT source,\n"
        "         CASE WHEN source LIKE 'kaggle%' THEN '2024 baseline'\n"
        "              ELSE '2026 current window' END AS era,\n"
        "         period,\n"
        "         COUNT(*) AS n_postings,\n"
        "         SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe,\n"
        "         SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS adjacent,\n"
        "         SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS control\n"
        "  FROM 'data/unified_core.parquet'\n"
        "  GROUP BY source, era, period\n"
        "  ORDER BY source, period\n"
        "\"\"\").df()\n"
        "print(f'Total postings:    {profile[\"n_postings\"].sum():,}')\n"
        "print(f'  Software engineer:  {profile[\"swe\"].sum():,}')\n"
        "print(f'  Software-adjacent:  {profile[\"adjacent\"].sum():,}  (PM, designer, data analyst, …)')\n"
        "print(f'  Control:            {profile[\"control\"].sum():,}  (nurse, accountant, electrician, …)')\n"
        "print()\n"
        "profile"
    ))

    # =========================================================================
    # 4 · Glossary
    # =========================================================================
    cells.append(md(
        "### Glossary — terms used throughout\n\n"
        "| Term | Meaning |\n"
        "|---|---|\n"
        "| **Software engineer (SWE)** | The primary group we study. Backend, frontend, ML engineer, DevOps, etc. |\n"
        "| **Software-adjacent** | Tech roles that involve some code but aren't primarily software development: product manager, UX designer, data analyst, QA, security engineer, technical program manager. |\n"
        "| **Control occupations** | Non-tech jobs used as a comparison group: civil / mechanical / electrical engineer, nurse, accountant, financial analyst, marketing manager, HR, sales. The control group lets us tell SWE-specific change apart from economy-wide change. |\n"
        "| **Period** | A 1-month bucket: `2024-01`, `2024-04` (historical snapshots), `2026-03`, `2026-04` (scraped data). |\n"
        "| **AI-vocab rate** | Share of postings whose description mentions any AI-tool or AI-concept phrase from a pre-committed list (`llm`, `gpt`, `claude`, `copilot`, `rag`, `prompt engineering`, `foundation model`, …). A simple measure of how AI-laden a posting is. |\n"
        "| **YOE** | Years of experience required. We use an LLM-extracted version because it handles natural-language phrasing better than regex. |\n"
        "| **Ghost job** | A posting that doesn't represent a real hiring intent — left up for resume collection, branding, or compliance. Each posting is LLM-rated as `realistic` / `inflated` / `ghost_likely`. |\n"
        "| **Percentage point (pp)** | The arithmetic difference between two percentages. Going from 5% to 28% is +23 pp, not \"a 460% increase.\" |\n"
        "| **Within-firm comparison** | Holding the company fixed and comparing its 2024 postings to its 2026 postings. Rules out effects of which companies are in the sample. |\n"
        "| **Pair-level comparison** | Holding BOTH the company AND the job title fixed (e.g., \"Microsoft Software Engineer II\") across periods. The strictest possible same-role comparison. |\n"
    ))

    # =========================================================================
    # 5 · What we did
    # =========================================================================
    cells.append(md(
        "## What we did (process in plain English)\n\n"
        "1. **Wrote priors first.** Before touching any data, we wrote down specific hypotheses "
        "about what we'd find — including the popular narratives we wanted to test (AI-washing, "
        "junior scope inflation, industry spread). Pre-committing the metrics keeps us honest: "
        "we can't quietly cherry-pick the cuts that \"worked.\"\n\n"
        "2. **Profiled the corpus** to confirm it matches the pipeline documentation.\n\n"
        "3. **Ran one focused query per hypothesis** against the 110,000-posting analysis file, "
        "each producing a single figure and a single table. No fishing.\n\n"
        "4. **Stress-tested the strongest signals** across four independent slices of the data: "
        "an alternative 2024 baseline, a metro-balanced subsample, excluding staffing-firm "
        "reposts, and excluding postings collapsed across multiple cities. A signal only counts "
        "if it survives at least three of four.\n\n"
        "5. **Hardened every headline** before calling it one. Every regex pattern was scored "
        "for semantic precision on a fresh hand-coded sample and rebuilt if it fell below 0.85 "
        "precision. A within-firm same-company same-title pair panel confirmed the AI-vocab "
        "rise is happening at the specific-role level, not from role-mix churn. The claim that "
        "junior-requirements sections shrank turned out to be classifier-dependent and is "
        "reported here as falsified rather than as a finding.\n\n"
        "Tooling: DuckDB queries, regex on description text, matplotlib charts. No machine-"
        "learning models. The whole analysis is auditable end-to-end."
    ))

    # =========================================================================
    # 6 · How to read
    # =========================================================================
    cells.append(md(
        "## How to read this notebook\n\n"
        "After this section you'll see:\n\n"
        "1. **A new analytical cut** — how junior SWE job scope has changed vs junior control scope.\n"
        "2. **Six headline findings** — each one short markdown + one inline figure.\n"
        "3. **Five falsified hypotheses** — the popular narratives we tested and rejected.\n"
        "4. **A verdict table** — all eleven claims on one page.\n"
        "5. **Other 2026 observations** — small bullets that don't rise to \"headline\" but are worth noting.\n"
        "6. **Limitations and robustness.**\n\n"
        "Color conventions are consistent throughout:\n\n"
        "- 🔴 red = SWE / junior\n"
        "- 🟠 orange = SWE-adjacent / mid\n"
        "- 🔵 blue = control / Big Tech\n"
        "- 🟢 green = senior / supportive evidence\n"
        "- 🟣 purple = AI-vocab"
    ))

    # =========================================================================
    # 7 · tl;dr
    # =========================================================================
    cells.append(md(
        "<a id='tldr'></a>\n\n"
        "## tl;dr\n\n"
        "Across 110,000 LinkedIn postings spanning 2024-01 → 2026-04, **AI-vocabulary in "
        "software-engineer descriptions rose from 3% to 28%**, while the same vocabulary in "
        "non-tech control occupations (nurses, accountants, civil engineers) rose only from "
        "0.2% to 1.4% — **a 23:1 delta ratio** that rules out economy-wide explanations.\n\n"
        "The rewrite is happening **within firms** — the *same* 292 companies that posted SWE "
        "roles in both 2024 and 2026 saw a mean **+19.4 percentage-point** increase in AI "
        "language at the company level, rising to **+10–13 pp at the strictest same-title "
        "comparison**. Microsoft +51 pp, Wells Fargo +45 pp, Amazon +36 pp. Defense firms flat.\n\n"
        "The popular *AI-washing* and *junior-scope-inflation* narratives don't survive the "
        "data. Two surprising findings: **years-of-experience requirements are FALLING** "
        "(junior median YOE dropped 2 → 1), and **a dev-tool vendor leaderboard has emerged** "
        "in labor demand — Copilot 4.3% > Claude 3.8% > OpenAI 3.6% > Cursor 2.2% in 2026-04.\n\n"
        "The clean story of the 2024–2026 shift: employers across the economy kept hiring "
        "software engineers (Big Tech posting share actually *rose* from 2.4% to 7.0%), and "
        "they rewrote those postings to reflect AI-tool integration into real engineering "
        "work. It is not AI replacing people. It is AI changing what people do."
    ))

    # =========================================================================
    # 8 · NEW SECTION — junior SWE vs junior control scope
    # =========================================================================
    cells.append(md(
        "---\n\n"
        "# How junior software-engineer scope changed vs junior control scope\n\n"
        "Before jumping to the headlines, one specific question deserves its own section: "
        "**has the entry-level software-engineer bar moved differently from the entry-level "
        "bar in non-tech occupations?** If AI is genuinely restructuring SWE work, the scope "
        "change should be concentrated there; if it's a general labor-market phenomenon, "
        "junior nurses and junior accountants should move in parallel.\n\n"
        "Four scope metrics, four periods, two occupations, two seniority buckets:"
    ))
    cells.append(code(
        "fig = viz_junior_scope_panel(); plt.show()"
    ))
    cells.append(code(
        "# Supplementary — deeper scope features, software-engineer side\n"
        "feats = pd.read_csv('eda/tables/junior_scope_features.csv')\n"
        "feats[['period','seniority','n_joined','mean_tech_count','mean_breadth_resid','mean_scope_density','mean_credential_stack']]"
    ))
    cells.append(md(
        "**What this shows.**\n\n"
        "- **Description length** grew sharply on the SWE side (junior +43%, senior +44% 2024→2026) "
        "and modestly on the control side (junior +19%, senior +29%). Job descriptions are getting "
        "longer for everyone, but the growth is largest in software.\n"
        "- **Years-of-experience required** fell for junior SWE (2.01 → 1.23 years) while junior "
        "control stayed flat around one year. The classic \"employers are demanding more experience "
        "from juniors\" narrative does not hold for software-engineer postings — it is moving the "
        "opposite direction, and is *not* mirrored in control.\n"
        "- **AI-vocabulary rate** diverges dramatically: 28% in 2026-04 junior SWE vs 0.5% in "
        "junior control — a 50× gap that widens over the window. Whatever is AI-related about SWE "
        "work is not spilling into non-tech entry-level jobs.\n"
        "- **Ghost-inflation rate** is 6–8% for SWE postings (both junior and senior) vs <2% for "
        "control — but it's roughly flat over time, so the AI-era rewrite isn't producing more "
        "\"inflated\" postings per se.\n\n"
        "The supplementary table on the software-engineer side shows the scope-complexity story "
        "more precisely: **mean tech count rose from 4.4 to 6.7 for juniors and from 5.8 to 8.0 "
        "for seniors** between 2024-01 and 2026-04. Length-residualized requirement breadth rose "
        "further at senior than at junior (+2.66 vs +2.12) — **scope inflation, to the extent it "
        "exists, is stronger at the senior end than the junior end**. Note: the scope-feature "
        "supplement is available for SWE rows only; the comparable computation on control rows "
        "has not been cached in this artifact."
    ))

    # =========================================================================
    # 9 · Six headlines
    # =========================================================================
    cells.append(md(
        "---\n\n"
        "# Six headline findings\n\n"
        "Each finding is one short explanation followed by one inline figure."
    ))

    cells.append(md(
        "## 1 · AI language rewrite is specific to software engineering\n\n"
        "If AI-talk in postings were a generic \"everyone is talking about AI\" phenomenon, "
        "control occupations (nurses, accountants, civil engineers) should rise alongside "
        "software. They don't.\n\n"
        "- **Software-engineer AI-vocab: 2.9% → 28.4%** — Δ +25.5 pp.\n"
        "- **Control AI-vocab: 0.26% → 1.39%** — Δ +1.1 pp.\n"
        "- Ratio of the two deltas: **23 : 1**, robust across all stress-test slices.\n"
        "- A formal difference-in-differences estimator gives +14.02 pp [95% CI +13.67, +14.37] "
        "for the SWE-specific AI-vocab effect.\n\n"
        "**Why it matters.** This is the cleanest test for two competing narratives "
        "simultaneously: (a) *AI-washing* would predict SWE and control to co-move (both "
        "absorb the same public AI narrative), and (b) *macro-only stories* (rate hikes, "
        "post-Covid correction) would predict the same — macro forces operate economy-wide. "
        "Neither prediction holds. Whatever is happening to software-engineer postings is "
        "real and specific to SWE."
    ))
    cells.append(code("fig = viz_swe_vs_control(); plt.show()"))

    cells.append(md(
        "## 2 · The same firms rewrote their own postings\n\n"
        "We took the **292 companies that posted at least 5 SWE roles in BOTH the 2024 baseline "
        "AND the 2026 current window** and asked: did the *same company* change its own job "
        "postings? Yes — by an average of **+19.4 percentage points** in AI-vocab rate.\n\n"
        "- **75% of companies rose**\n"
        "- **61% rose more than 10 pp**\n"
        "- **39% rose more than 20 pp**\n"
        "- Microsoft +51 pp · Wells Fargo +45 pp · Amazon +36 pp · Walmart +32 pp · Capital One +25 pp\n"
        "- Defense firms (Raytheon, Northrop Grumman) flat\n\n"
        "A stricter version of the test — same company AND same job title (e.g., \"Microsoft "
        "Software Engineer II\" in 2024 vs 2026) — gives **+10 to +13 pp at the pair level**, "
        "*exceeding* the company-level average. The rewrite is happening at the specific-role "
        "level, not from the company's role mix shifting.\n\n"
        "**Why it matters.** If the AI-vocab rise were driven by *new* AI-native companies "
        "entering the market and old ones leaving, this would tell us nothing about real "
        "behavior change. The fact that the *same* companies rewrote their *own* postings at "
        "the *same* job titles closes that loophole. The 2024 → 2026 shift is employers "
        "changing what they ask for — not composition churn."
    ))
    cells.append(code("fig = viz_within_firm(); plt.show()"))

    cells.append(md(
        "## 3 · Seniority boundaries sharpened; scope inflation is senior-led\n\n"
        "A common assumption: as AI automates routine junior work, junior and senior roles "
        "blur into each other — juniors must behave like seniors to get hired. The opposite "
        "happened.\n\n"
        "- The **text similarity** between junior and senior postings *dropped* from 0.95 to "
        "0.86 over the two-year window (junior-senior pairs got more distinguishable, not less).\n"
        "- A **supervised classifier** trained to distinguish associate from mid-senior roles "
        "gained +0.150 AUC 2024 → 2026 — boundaries sharpened everywhere except at the top "
        "of the ladder.\n"
        "- **Length-adjusted requirement breadth** rose +2.61 for senior SWE vs +1.58 for junior — "
        "scope inflation is senior-led, not junior-led.\n\n"
        "**Why it matters.** The \"junior roles are being squeezed out and rewritten as senior\" "
        "narrative doesn't survive the data. Both tiers changed, but they got MORE distinguishable, "
        "and the scope growth is larger at senior. Combined with the finding below on YOE (finding 5), "
        "the entry-level bar isn't rising — the senior bar is rising faster than the junior bar."
    ))
    cells.append(code("fig = viz_senior_scope_inflation(); plt.show()"))

    cells.append(md(
        "## 4 · A dev-tool vendor leaderboard has emerged in labor demand\n\n"
        "We searched each posting for explicit mentions of named AI tools and labs. The "
        "2026-04 leaderboard in SWE descriptions:\n\n"
        "| rank | vendor | mention rate | growth since 2024-01 |\n"
        "|------|--------|--------------|----------------------|\n"
        "| 1 | GitHub Copilot | 4.25% | 53× |\n"
        "| 2 | Claude | 3.83% | **190× (steepest growth)** |\n"
        "| 3 | OpenAI (brand) | 3.63% | 17× |\n"
        "| 4 | Cursor | 2.17% | >200× (emerged from ~0%) |\n"
        "| 5 | Anthropic (brand) | 1.48% | 70× |\n"
        "| 6 | Gemini | 1.07% | 35× |\n\n"
        "Three things to notice:\n\n"
        "- **Copilot leads** on raw share (first-mover) but **Claude has the steepest growth** — "
        "on current trajectory Claude overtakes Copilot in mid-2026.\n"
        "- **ChatGPT as a brand is plateauing or declining** (0.82% → 0.70%) while specific "
        "vendors keep climbing. Employers are specializing their AI vocabulary from the "
        "consumer-brand mention to specific-product mentions.\n"
        "- **Cursor's emergence** is the most dramatic story — visible in labor demand before "
        "most popular-press coverage of Cursor as a serious Copilot competitor."
    ))
    cells.append(code("fig = viz_vendor_leaderboard(); plt.show()"))

    cells.append(md(
        "## 5 · Years-of-experience floor is FALLING, not rising\n\n"
        "The classic *junior scope inflation* hypothesis says: AI is automating routine "
        "junior-level work, so employers respond by demanding more experience from junior "
        "applicants. The data shows the opposite — across **all seniority buckets**.\n\n"
        "| level | 2024-01 mean | 2026-04 mean | median (2024 → 2026) |\n"
        "|-------|--------------|--------------|-----------------------|\n"
        "| junior | **2.01** | **1.23** | 2 → 1 |\n"
        "| mid | 4.00 | 2.36 | 3 → 2 |\n"
        "| senior | 6.55 | 6.37 | 6 → 5 |\n\n"
        "**Why it matters.** If anything, employers are asking for *less* explicit experience in "
        "2026, not more. Combined with the SWE-vs-control junior comparison above — where "
        "control YOE stayed flat around 1 year — the YOE-floor drop is a software-engineer "
        "finding, not a general labor-market loosening. Possible interpretations: AI-tool "
        "onboarding lets firms train less-experienced juniors faster; or firms are dropping "
        "credential asks while raising the skill expectations inside interviews. The "
        "interview-based follow-up to this analysis is equipped to adjudicate."
    ))
    cells.append(code("fig = viz_yoe_floor(); plt.show()"))

    cells.append(md(
        "## 6 · Big Tech: more posting volume AND more AI density\n\n"
        "We separated firms into two tiers using a pre-committed list of 27 Big Tech canonical "
        "names (Google, Meta, Amazon, Apple, Microsoft, Oracle, Netflix, Block / Square, Uber, "
        "Airbnb, Salesforce, plus frontier-AI labs Anthropic and OpenAI, plus minor variants "
        "like Amazon Web Services). Two surprising patterns appear simultaneously:\n\n"
        "- **Big Tech share of SWE postings ROSE** from 2.4% (2024-01) to 7.0% (2026-04). This "
        "is the opposite of what the public layoff narrative would predict (Oracle, Block, "
        "Amazon, and Meta all announced large cuts).\n"
        "- **Big Tech AI-mention rate is 17 pp HIGHER** than the rest of the market in 2026 "
        "(44% vs 27%), robust across all four stress-test slices.\n\n"
        "**Why it matters.** The Big-Tech-vs-rest density gap is the single largest employer-"
        "tier effect in the data. Pair it with named-firm layoff timelines and 10-Q filings, "
        "and you get the natural identification strategy for a follow-up paper: *for the firms "
        "that publicly announced AI-driven cuts, did their AI-vocab rise lead, follow, or "
        "co-move with the announcements?*"
    ))
    cells.append(code("fig = viz_bigtech_density(); plt.show()"))

    # =========================================================================
    # 10 · Five falsified hypotheses
    # =========================================================================
    cells.append(md(
        "---\n\n"
        "# Five falsified hypotheses\n\n"
        "These came from credible sources — popular press, labor-market literature, and our own "
        "priors going into the analysis. The data does not support them, at least not at the "
        "level our data can see."
    ))

    cells.append(md(
        "## Falsified 1 · AI is narrative cover for unrelated layoffs (at the content level)\n\n"
        "**The hypothesis.** Firms invoke AI as a public reason for 2023–2026 layoffs while "
        "the real drivers are post-Covid hiring corrections, rising interest rates, and "
        "outsourcing. AI is narrative, not substance.\n\n"
        "**Why it fails at the content level.** If AI-talk in postings were a narrative layer "
        "overlaid on macro forces, SWE and control occupations should both absorb it. They "
        "don't. SWE rose 23× faster than control.\n\n"
        "**Caveat.** The narrative-layer story could still be true about how firms publicly "
        "*explain* cuts, separately from what they write in job postings. That's an interview "
        "question, not one our data answers."
    ))
    cells.append(code("fig = viz_disproven_aiwashing(); plt.show()"))

    cells.append(md(
        "## Falsified 2 · Software jobs are spreading into non-tech industries on LinkedIn\n\n"
        "**The hypothesis.** Software-worker headcount is reportedly growing in retail (+12%), "
        "property (+75%), and construction (+100%) between 2022 and 2025 (source: Bureau of "
        "Labor Statistics occupational data, as reported in *The Economist*, April 2026). If "
        "that spread is real and broad, the share of LinkedIn SWE postings coming from "
        "non-tech industries should rise.\n\n"
        "**Why it fails on our data.** Non-tech industries already held about 55% of "
        "labeled-industry LinkedIn SWE postings in 2024 and remain at ~55% in 2026. No shift.\n\n"
        "**Caveat.** Both claims can coexist: BLS measures *employed* people across all "
        "channels, while we measure *posting* composition on LinkedIn. Non-tech SWE employment "
        "can grow without LinkedIn posting share shifting, if those employers recruit through "
        "different channels (referrals, internal mobility, niche boards). What's ruled out is "
        "the LinkedIn version specifically."
    ))
    cells.append(code("fig = viz_disproven_industry_spread(); plt.show()"))

    cells.append(md(
        "## Falsified 3 · Automation hits junior engineers first\n\n"
        "**The hypothesis.** AI automates routine tasks first. Junior SWE postings (which "
        "describe the most routine work) should therefore be the first to mention AI tools and "
        "the first to demand more experience or lose volume.\n\n"
        "**Why it fails.** AI-vocab adoption is essentially uniform across seniority in "
        "2026-04 (junior 27%, mid 30%, senior 31%). Combined with the falling-junior-YOE "
        "finding above, the junior-first reading does not survive.\n\n"
        "The consistent reading is **senior-restructuring**: senior postings shifted in "
        "*content* (toward orchestration, review, AI-leverage language), while junior postings "
        "remained structurally similar with AI vocabulary mixed in at roughly equal rates."
    ))
    cells.append(code("fig = viz_disproven_juniorfirst(); plt.show()"))

    cells.append(md(
        "## Falsified 4 · Requirements-section contraction indicates hiring-bar lowering\n\n"
        "**The hypothesis.** Several analyses reported that the *requirements* section of "
        "SWE postings shrank between 2024 and 2026. A natural reading: employers are quietly "
        "lowering the hiring bar — dropping specific credential asks while expanding narrative "
        "(\"who we are,\" \"what you'll do\") language.\n\n"
        "**Why it fails.** Three independent tests:\n\n"
        "1. The direction of the requirements-share shift depends on which classifier you use. "
        "Two defensible section classifiers give opposite signs on the same corpus.\n"
        "2. On a 356-firm within-company panel, the correlation between a firm's "
        "requirements-section contraction and its hiring-bar-proxy metrics (YOE floor, credential "
        "stack, tech count, education asks) is |ρ| ≤ 0.28 on every proxy — essentially zero.\n"
        "3. A narrative audit of the 50 postings with the largest requirements-section "
        "contraction found **zero** that contained explicit requirement-loosening language. What "
        "moved was boilerplate: benefits (+89%), legal (+80%), responsibilities (+49%). "
        "Narrative sections expanded; the requirements section didn't meaningfully shrink."
    ))
    cells.append(code("fig = viz_disproven_hiring_bar(); plt.show()"))

    cells.append(md(
        "## Falsified 5 · The hiring cycle tightened, and firms raised the bar in response\n\n"
        "**The hypothesis.** The 2026-Q1 hiring trough (JOLTS Information-sector openings at "
        "0.71× the 2023 average) let employers be more selective — firms posting fewer roles "
        "should demand more from candidates. We'd expect a negative correlation between a "
        "firm's change in posting volume and its change in requirements-stringency.\n\n"
        "**Why it fails.** The correlation runs the **opposite** direction: firms that "
        "increased their posting volume 2024 → 2026 also increased their AI-language content "
        "(Pearson r = +0.20 on description length; similar signs on breadth and scope). "
        "Volume-UP firms write LONGER, MORE demanding JDs. The content shift is about what "
        "employers are asking for — not a cycle-driven squeeze."
    ))
    cells.append(code("fig = viz_disproven_selectivity(); plt.show()"))

    # =========================================================================
    # 11 · Verdict table
    # =========================================================================
    cells.append(md("---\n\n# All findings on one page"))
    cells.append(code("fig = viz_verdict_table(); plt.show()"))

    # =========================================================================
    # 12 · Other 2026 observations
    # =========================================================================
    cells.append(md(
        "---\n\n"
        "## Other 2026 observations\n\n"
        "Things that don't rise to \"headline\" but are worth calling out:\n\n"
        "- **The AI rise is geographically uniform, not tech-hub-driven.** The 26 metros we "
        "track all saw AI-vocab rates climb 5–14 pp in SWE postings. Leaders are Atlanta, "
        "Tampa, Miami, and Salt Lake City — NOT San Francisco or Seattle. Tech-hub premium is "
        "under 2 pp.\n"
        "- **Descriptions got longer, but the new length is boilerplate.** Benefits sections "
        "grew 89%, legal/EEO 80%, responsibilities 49%. Requirements sections were roughly "
        "flat. Length growth is mostly recruiter-editor inflation, not demand for more work.\n"
        "- **Legacy roles are being replaced by modern-stack roles, NOT by AI roles.** "
        "Disappearing 2024 titles (Java architect, Drupal specialist, PHP architect) map to "
        "2026 neighbors that average only 3.6% AI-strict mention — below the 14.4% market "
        "average. The substitution is stack-modernization (Postgres / CI-CD / Terraform), "
        "not AI-ification.\n"
        "- **Staffing firms post CLEANER descriptions than direct employers.** Ghost-score and "
        "inflation concentrate at direct employers, not aggregators — the reverse of the "
        "common intuition.\n"
        "- **GitHub Copilot appears in only 0.10% of postings**, despite a ~33% regular-use "
        "rate in industry benchmarks. Employers do not formalize even the most-adopted AI "
        "tool as a written requirement.\n"
        "- **LLM-authorship affects JD length but NOT JD content.** The ~1,100-character "
        "description growth is about half attributable to recruiters drafting with LLMs; "
        "however, AI-mention, credential stacking, scope broadening, and CI/CD increases "
        "persist at 80–130% of full-corpus magnitude when the most LLM-styled postings are "
        "excluded. The content shift is real, not a style artifact.\n"
        "- **AI-focused senior roles ask for MORE experience, not less.** The emergent "
        "\"Applied-AI Engineer\" archetype has median YOE 6 — one year above other senior "
        "archetypes. AI-era senior work is compressing into a more explicit high-experience "
        "bar, not democratizing."
    ))

    # =========================================================================
    # 13 · Limitations + robustness
    # =========================================================================
    cells.append(md(
        "## Limitations\n\n"
        "- **LinkedIn-only sample.** No Indeed / niche-board sensitivity in this analysis; the "
        "LinkedIn-specific findings are about LinkedIn composition, which may not mirror the "
        "total US software labor market.\n"
        "- **Date granularity is too coarse** to test specific AI-release windows. We can "
        "compare 2024 to 2026 but can't pinpoint ChatGPT (Nov 2022) vs Claude Code (Feb 2025) "
        "lead/lag effects.\n"
        "- **Vendor-specific regexes may over-match.** \"Cursor\" is also UI vocabulary; "
        "\"Claude\" is a common name; \"Llama\" is a nature/mascot word. Vendor rates should "
        "be read directionally, not as precise penetration numbers.\n"
        "- **The within-firm panel requires ≥5 postings in both periods.** 292 firms is "
        "meaningful but excludes most small employers.\n"
        "- **LLM-YOE coverage is ~78%** of in-frame SWE rows; possible selection bias on rows "
        "where the LLM declined to extract.\n"
        "- **No significance tests.** Descriptive analysis only; formal hypothesis-testing "
        "with corrected standard errors is pending.\n\n"
        "### Robustness — sampling-frame sensitivity\n\n"
        "The 110,000-posting analysis file is a balanced 40% / 30% / 30% "
        "software-engineer / software-adjacent / control sample, not a natural LinkedIn "
        "distribution. You might reasonably worry that the findings above are sampling "
        "artifacts. They aren't: every rate-based claim was re-run on the full unbalanced "
        "corpus (~1.45 M postings, LinkedIn only) and produced essentially the same numbers — "
        "the largest absolute difference on headline rates is under 1 percentage point, and "
        "the within-firm rewrite signal is actually *stronger* on the natural data (356 "
        "companies with mean +20.7 pp vs 292 with +19.4 pp on the balanced sample)."
    ))

    # =========================================================================
    # 14 · Where to dig deeper
    # =========================================================================
    cells.append(md(
        "## Where to dig deeper\n\n"
        "- **Full report** with per-finding evidence and caveats: "
        "[`../reports/open_ended_v2.md`](../reports/open_ended_v2.md)\n"
        "- **Working analysis notebook** with every underlying DuckDB query: "
        "[`open_ended_v2.ipynb`](open_ended_v2.ipynb)\n"
        "- **Pre-registered analysis framework** — what we expected before looking at data: "
        "[`../memos/priors.md`](../memos/priors.md)\n"
        "- **External reference article** — *The Economist*, \"Code red: The tech jobs bust is "
        "real. Don't blame AI (yet),\" Apr 13 2026: "
        "[`../memos/references/economist_code_red_2026-04-13.md`](../memos/references/economist_code_red_2026-04-13.md)\n"
        "- **Project research design**: [`../../docs/1-research-design.md`](../../docs/1-research-design.md)\n"
        "- **Interview protocol** for the qualitative follow-up: "
        "[`../../docs/2-interview-design-mechanisms.md`](../../docs/2-interview-design-mechanisms.md)"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (job-research venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
        },
    }
    return nb


def main() -> None:
    nb = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
