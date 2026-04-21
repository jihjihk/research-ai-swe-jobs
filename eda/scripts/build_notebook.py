"""
Build eda/notebooks/open_ended_v2.ipynb from explicit cell definitions.

The notebook drives the full v2 analysis on `data/unified_core.parquet`:
 - Phase A corpus profile (quick DuckDB queries)
 - Phase B rerun of S1, S3, S10, S11, Sv on core
 - Phase B new scans S12 (YOE), S13 (vendors), S14 (AI×ghost),
   S15 (control drivers), S16 (survival), S17 (within-firm panel)
 - Headlines table
 - Recommendations

Run:
  ./.venv/bin/python eda/scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "eda" / "notebooks" / "open_ended_v2.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Open-ended EDA v2 — analysis on `data/unified_core.parquet`\n\n"
        "Clean notebook version of the open-ended senior-DS EDA. Primary data "
        "file switched from `data/unified.parquet` (1.45M × 96) to "
        "`data/unified_core.parquet` (110k × 42, LinkedIn-only, all rows inside "
        "the Stage 9 balanced LLM-frame, with `is_control` populated in all four "
        "periods).\n\n"
        "Hypothesis set extended to **H1–H13**. Full priors in "
        "[`../memos/priors.md`](../memos/priors.md).\n\n"
        "**How to read this notebook:** each scan writes a figure to "
        "`eda/figures/` and a CSV to `eda/tables/`. The code here is a thin "
        "driver — all scan logic lives in `eda/scripts/{scans,core_scans}.py`. "
        "Execute top-to-bottom to regenerate all artifacts."
    ))

    cells.append(md("## 0. Setup"))
    cells.append(code(
        "import sys, os\n"
        "from pathlib import Path\n"
        "\n"
        "# Resolve project root\n"
        "HERE = Path.cwd()\n"
        "ROOT = HERE if (HERE / 'data' / 'unified_core.parquet').exists() else HERE.parents[1]\n"
        "assert (ROOT / 'data' / 'unified_core.parquet').exists(), f'unified_core.parquet not found at {ROOT}'\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, str(ROOT / 'eda' / 'scripts'))\n"
        "\n"
        "import duckdb\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "pd.set_option('display.max_columns', 40)\n"
        "pd.set_option('display.width', 200)\n"
        "pd.set_option('display.float_format', lambda x: f'{x:.4f}' if isinstance(x, float) else x)\n"
        "\n"
        "from scans import AI_VOCAB_PATTERN, NEW_AI_TITLE_PATTERN, BIG_TECH_CANONICAL\n"
        "from core_scans import (\n"
        "    CORE_PATH, CORE_OBS_PATH, CORE_DEFAULT_FILTER, VENDOR_PATTERNS,\n"
        "    rerun_s1_core, rerun_s3_core, rerun_s10_core, rerun_s11_core, rerun_sv_core,\n"
        "    scan_s12, scan_s13, scan_s14, scan_s15, scan_s16, scan_s17,\n"
        ")\n"
        "\n"
        "con = duckdb.connect()\n"
        "print('Environment ready.')\n"
        "print(f'Core file:    data/{CORE_PATH.name}')\n"
        "print(f'Core obs:     data/{CORE_OBS_PATH.name}')"
    ))

    cells.append(md("## 1. Hypothesis recap (H1–H13)"))
    cells.append(md(
        "Full priors in `eda/memos/priors.md`. Short form:\n\n"
        "**Macro-narrative (from v1):**\n"
        "- **H1** AI-washing — AI is narrative cover for macro layoffs. *Falsified on content* if SWE vs control diverge.\n"
        "- **H2** AI creates new job types — emergence in titles.\n"
        "- **H3** Non-AI macro (Covid-binge + rate hikes + outsourcing) — Economist central claim.\n"
        "- **H4** Industry spread to non-tech — Economist tech-skills-everywhere.\n"
        "- **H5** Junior-first *vs* senior-restructuring within SWE.\n"
        "- **H6** Big Tech vs rest — reaction differs by tier.\n"
        "- **H7** SWE vs control divergence — direct SWE-specificity test.\n\n"
        "**New in v2 (from unified_core exploration):**\n"
        "- **H8** YOE floor is FALLING, not rising (counter scope-inflation).\n"
        "- **H9** Dev-tool vendor labor-market leaderboard is measurable (Copilot / Claude / OpenAI / Cursor hierarchy).\n"
        "- **H10** AI mention is NOT a ghost-job signal.\n"
        "- **H11** Non-SWE AI-spread is niche-specific (finance + power/nuclear engineering), not broad.\n"
        "- **H12** Posting survival differs by tier / AI-tag.\n"
        "- **H13** Within-firm AI rewrite is real — same firm, 2024 → 2026, did they rewrite their own postings?"
    ))

    cells.append(md("## 2. Phase A — corpus profile on core"))
    cells.append(code(
        "profile = con.execute(f\"\"\"\n"
        "  SELECT source, source_platform, period,\n"
        "         COUNT(*) AS n,\n"
        "         SUM(CASE WHEN is_swe THEN 1 ELSE 0 END) AS swe,\n"
        "         SUM(CASE WHEN is_swe_adjacent THEN 1 ELSE 0 END) AS adjacent,\n"
        "         SUM(CASE WHEN is_control THEN 1 ELSE 0 END) AS control\n"
        "  FROM '{CORE_PATH}'\n"
        "  GROUP BY 1,2,3 ORDER BY 1,2,3\n"
        "\"\"\").df()\n"
        "profile['total_check'] = profile['swe'] + profile['adjacent'] + profile['control']\n"
        "print('Core row count:', profile['n'].sum())\n"
        "profile"
    ))
    cells.append(code(
        "coverage = con.execute(f\"\"\"\n"
        "  SELECT period, analysis_group, llm_classification_coverage,\n"
        "         llm_extraction_coverage, COUNT(*) AS n\n"
        "  FROM '{CORE_PATH}'\n"
        "  GROUP BY 1,2,3,4 ORDER BY 1,2,3,4\n"
        "\"\"\").df()\n"
        "coverage_pivot = coverage.pivot_table(\n"
        "    index=['period','analysis_group'],\n"
        "    columns='llm_classification_coverage',\n"
        "    values='n', fill_value=0, aggfunc='sum'\n"
        ")\n"
        "coverage_pivot"
    ))

    cells.append(md(
        "## 3. Phase B — re-run finalists on core\n\n"
        "Five v1 finalists re-executed against unified_core. Writes to "
        "`eda/tables/{S1,S3,S10,S11,Sv}_core_*.csv` and matching PNGs."
    ))
    cells.append(code(
        "s1 = rerun_s1_core(con); s1"
    ))
    cells.append(code(
        "s3 = rerun_s3_core(con); s3"
    ))
    cells.append(code(
        "s10 = rerun_s10_core(con); s10"
    ))
    cells.append(code(
        "s11 = rerun_s11_core(con); s11"
    ))
    cells.append(code(
        "sv = rerun_sv_core(con); sv"
    ))

    cells.append(md(
        "## 4. Phase B — new scans S12–S17 (H8–H13)"
    ))
    cells.append(md("### S12 (H8) — YOE floor trajectory"))
    cells.append(code(
        "s12 = scan_s12(con)\n"
        "display(s12)"
    ))
    cells.append(md(
        "**Read:** junior mean YOE 2.01 → 1.23 (mean) and 2 → 1 (median) across "
        "2024→2026. Senior median dropped 6 → 5. Classic scope-inflation at the "
        "YOE bar is falsified on LLM-YOE."
    ))

    cells.append(md("### S13 (H9) — Vendor labor-market leaderboard"))
    cells.append(code(
        "s13 = scan_s13(con)\n"
        "s13"
    ))
    cells.append(code(
        "# Stand-alone 2026-04 leaderboard\n"
        "pd.read_csv('eda/tables/S13_vendor_leaderboard_2026_04.csv')"
    ))
    cells.append(md(
        "**Read:** in 2026-04 SWE postings, Copilot leads at 4.25%, Claude "
        "second at 3.83%, OpenAI third at 3.63%. Cursor emerged from ~0% to "
        "2.17% in two years. ChatGPT as a brand is PLATEAUING (0.82% → 0.70%) "
        "while Claude/OpenAI/Anthropic continue climbing."
    ))

    cells.append(md("### S14 (H10) — AI-mention × ghost rate"))
    cells.append(code(
        "s14 = scan_s14(con)\n"
        "s14"
    ))
    cells.append(md(
        "**Read:** AI-mentioning SWE postings are *slightly less* inflated than "
        "non-AI postings (4.5% vs 5.6% in 2026-04). The 'AI buzzword = ghost "
        "job' narrative is not supported by our LLM ghost assessment."
    ))

    cells.append(md("### S15 (H11) — Control AI-rise occupational drivers"))
    cells.append(code(
        "s15 = scan_s15(con)\n"
        "# Rank families by 2026-04 AI-rate\n"
        "latest = s15[s15['period']=='2026-04'].sort_values('ai_rate', ascending=False)\n"
        "latest"
    ))
    cells.append(md(
        "**Read:** the control-tier AI rise concentrates in finance/accounting "
        "and electrical/nuclear engineering. Nursing, HR, sales, marketing are "
        "essentially untouched. Reframes H4 from 'tech skills everywhere' to "
        "'AI language is reaching finance and utility engineering; service-"
        "occupation control still absent.'"
    ))

    cells.append(md("### S16 (H12) — Posting survival"))
    cells.append(code(
        "s16 = scan_s16(con)\n"
        "s16"
    ))
    cells.append(md(
        "**Read:** in 2026-03 (month with complete observation window), "
        "AI-mentioning SWE postings stay alive 0.9 days longer on average "
        "(5.27 vs 4.36 days). Control AI-mentioning postings show a larger "
        "gap (7.5 vs 2.6 days) but the sample is small. Small positive "
        "signal consistent with either higher demand or slower filling. "
        "2026-04 is still ongoing and not directly comparable."
    ))

    cells.append(md("### S17 (H13) — Within-firm AI rewrite overlap panel"))
    cells.append(code(
        "s17 = scan_s17(con)\n"
        "print('Panel size:', len(s17), 'companies')\n"
        "print(f\"Mean delta 2026−2024: {s17['delta'].mean()*100:.2f}pp\")\n"
        "print(f\"Median delta:         {s17['delta'].median()*100:.2f}pp\")\n"
        "print(f\"% companies with rise:    {(s17['delta']>0).mean()*100:.1f}%\")\n"
        "print(f\"% with rise >10pp:        {(s17['delta']>0.10).mean()*100:.1f}%\")\n"
        "print(f\"% with rise >20pp:        {(s17['delta']>0.20).mean()*100:.1f}%\")\n"
        "print()\n"
        "print('Top 15 risers:')\n"
        "s17.sort_values('delta', ascending=False).head(15)"
    ))
    cells.append(md(
        "**Read:** on the 292-company overlap panel (companies with ≥5 SWE "
        "postings in BOTH 2024 Kaggle and 2026 scraped), the *same companies* "
        "rewrote their postings. Mean within-firm delta = **+19.4pp**; 75% of "
        "companies rose; 61% rose more than 10pp; 39% rose more than 20pp. "
        "Microsoft +51pp, Wells Fargo +45pp, Amazon +36pp. Defense firms "
        "(Raytheon, Northrop Grumman) flat or near-flat.\n\n"
        "**This rules out between-firm composition as the sole driver of "
        "the 2024→2026 AI rewrite.** The rewrite is happening INSIDE the same "
        "firms."
    ))

    cells.append(md("## 5. Headlines — v2 verdict table"))
    cells.append(code(
        "verdicts = pd.DataFrame([\n"
        "    ('H1  AI-washing',                'against (content-level)',\n"
        "     'S11 SWE+25pp vs control+1.2pp — 21× delta ratio'),\n"
        "    ('H2  new AI job types',          'supported',\n"
        "     'S3 new-AI-title share 1.6% → 8.3% (5×)'),\n"
        "    ('H3  non-AI macro (content)',    'against',\n"
        "     'S11 SWE-specificity rules out economy-wide cause'),\n"
        "    ('H4  industry spread',           'NOT on LinkedIn',\n"
        "     'S8 non-tech share flat ~55% 2024→2026'),\n"
        "    ('H5  junior-first vs senior',    '(b) senior consistent; (a) falsified',\n"
        "     'S1 AI-vocab uniform across levels; S12 junior YOE FELL'),\n"
        "    ('H6  Big Tech vs rest',          'split',\n"
        "     'S10 BT AI 44% vs rest 27% (diff 17pp); BT SHARE ROSE not fell'),\n"
        "    ('H7  SWE-vs-control',            'strongly supports SWE-specificity',\n"
        "     'S11 on balanced core with real 2024 control baseline'),\n"
        "    ('H8  YOE floor falling',         'supported',\n"
        "     'S12 junior mean YOE 2.01 → 1.23, median 2 → 1'),\n"
        "    ('H9  vendor leaderboard',        'supported (Copilot > Claude > OpenAI > Cursor)',\n"
        "     'S13 2026-04: Copilot 4.25%, Claude 3.83%, OpenAI 3.63%'),\n"
        "    ('H10 AI mention ≠ ghost job',    'supported',\n"
        "     'S14 AI inflated rate 4.5% vs non-AI 5.6%'),\n"
        "    ('H11 control spread is niche',   'supported',\n"
        "     'S15 finance + electrical/nuclear dominate; service occupations flat'),\n"
        "    ('H12 posting survival',          'directional only',\n"
        "     'S16 AI postings live 0.9 days longer in 2026-03 (small n for control)'),\n"
        "    ('H13 within-firm AI rewrite',    'strongly supported',\n"
        "     'S17 mean +19.4pp, 75% of 292 companies rose'),\n"
        "], columns=['hypothesis', 'verdict', 'primary evidence'])\n"
        "verdicts"
    ))

    cells.append(md(
        "## 6. Recommended next research moves (ranked)\n\n"
        "1. **Big-Tech-stratified RQ1 paper.** The 17pp BT-vs-rest AI-density "
        "gap (S10) and the within-firm rise (S17) are strongest at Big Tech; "
        "pair with named-firm layoff timelines to test whether AI-density "
        "co-moves with announcements.\n"
        "2. **Vendor-specificity paper (S13).** Nobody has published a labor-"
        "demand vendor-share table for dev tools. The hierarchy (Copilot > "
        "Claude > OpenAI > Cursor, with ChatGPT plateauing) is entirely new.\n"
        "3. **YOE-floor inversion paper (S12).** Rule-based YOE in v8 rose "
        "compositionally; LLM-YOE declines across all seniority buckets. "
        "Contradicts the popular 'scope inflation' narrative. Worth calling "
        "out explicitly with the methodological note on rule-vs-LLM YOE.\n"
        "4. **Within-firm rewrite mechanism (S17) → RQ4 interviews.** v8 "
        "established cross-seniority rewriting; S17 localizes it to the "
        "within-firm channel. Interview script can ask: 'who inside your "
        "firm rewrote the JDs between 2024 and 2026?' Connects employer-side "
        "content to the ghost-vs-real debate (RQ4)."
    ))

    cells.append(md(
        "## 7. Limitations (v2)\n\n"
        "- `unified_core.parquet` is LinkedIn-only by construction — no Indeed "
        "sensitivity.\n"
        "- Control rows in 2024 come from Kaggle (asaniczka + arshkon) matched "
        "against Stage-5 `is_control` classification; the 2026 control rows "
        "come from explicit scraper query tiers. The *classification criterion* "
        "is consistent but the *recruitment surface* differs.\n"
        "- `unified_core_observations.parquet` has no scrape-cadence for Kaggle "
        "sources — S16 posting-survival is 2026 only.\n"
        "- Vendor-specific regexes match bare words ('cursor', 'claude', "
        "'gemini') — some false positives from unrelated uses (cursor as a UI "
        "element, claude as a person's name). Rates should be read directionally.\n"
        "- Within-firm panel (S17) requires ≥5 SWE postings in both periods; "
        "smaller firms are excluded. 292 firms is meaningful but not "
        "exhaustive.\n"
        "- H8 YOE finding depends on `yoe_min_years_llm` coverage (~80% of "
        "in-frame SWE rows). Selection bias possible: rows where LLM refused "
        "to extract YOE might differ systematically.\n"
        "- No significance tests. Descriptive only."
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
