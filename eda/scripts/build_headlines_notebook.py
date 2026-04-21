"""
Build eda/notebooks/headlines.ipynb — a polished, presentation-quality
notebook listing the 5 headline findings + 3 falsified hypotheses, with
inline figures that travel with the notebook (no external file deps).

Run:
  ./.venv/bin/python eda/scripts/build_headlines_notebook.py

Then execute end-to-end with:
  ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      eda/notebooks/headlines.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "eda" / "notebooks" / "headlines.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# SWE Labor-Market EDA — Headlines\n\n"
        "Polished summary of the open-ended senior-DS EDA on "
        "`data/unified_core.parquet`. Five headline findings + three "
        "falsified hypotheses, each with an inline visualization.\n\n"
        "**Working artifacts** (the analysis behind these visuals):\n"
        "- Priors: [`../memos/priors.md`](../memos/priors.md) (H1–H13)\n"
        "- Full report: [`../reports/open_ended_v2.md`](../reports/open_ended_v2.md)\n"
        "- Working notebook: [`open_ended_v2.ipynb`](open_ended_v2.ipynb)\n\n"
        "Figures here read from the CSVs already produced in `eda/tables/`. "
        "To regenerate, re-run `eda/scripts/core_scans.py` first.\n\n"
        "---\n\n"
        "## tl;dr\n\n"
        "Across 110k LinkedIn SWE/adjacent/control postings spanning 2024-01 → "
        "2026-04, AI-vocabulary in SWE descriptions rose **3% → 28%** while "
        "control occupations rose only **0.2% → 1.4%** (a 23:1 delta ratio). "
        "The rewrite is **within-firm** — same companies, mean +19.4pp on a "
        "292-co overlap panel — not driven by between-firm composition. The "
        "popular *AI-washing* and *junior-scope-inflation* narratives don't "
        "survive the data. New artifacts: a dev-tool vendor leaderboard "
        "(Copilot 4.3% > Claude 3.8% > OpenAI 3.6% > Cursor 2.2%) and a "
        "counter-narrative finding that **YOE floors are FALLING, not rising**."
    ))

    cells.append(md("## Setup"))
    cells.append(code(
        "import sys, os\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path.cwd()\n"
        "ROOT = HERE if (HERE / 'data' / 'unified_core.parquet').exists() else HERE.parents[1]\n"
        "os.chdir(ROOT)\n"
        "sys.path.insert(0, str(ROOT / 'eda' / 'scripts'))\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "%matplotlib inline\n"
        "\n"
        "from headlines_viz import (\n"
        "    viz_within_firm, viz_swe_vs_control, viz_yoe_floor,\n"
        "    viz_vendor_leaderboard, viz_bigtech_density,\n"
        "    viz_disproven_aiwashing, viz_disproven_industry_spread,\n"
        "    viz_disproven_juniorfirst, viz_verdict_table,\n"
        ")"
    ))

    cells.append(md(
        "---\n\n"
        "# Top 5 headlines (what holds up)"
    ))

    cells.append(md(
        "## 1. Within-firm AI rewrite is real and large\n\n"
        "On a panel of **292 companies that posted SWE roles in BOTH 2024 and "
        "2026**, the same firm's AI-vocab rate rose by an average of **+19.4 "
        "percentage points**. 75% of firms rose; 61% rose more than 10pp; 39% "
        "rose more than 20pp. Microsoft +51pp, Wells Fargo +45pp, Amazon +36pp. "
        "Defense firms (Raytheon, Northrop Grumman) flat.\n\n"
        "**Why it matters:** rules out between-firm composition churn as the "
        "dominant driver of the 2024→2026 AI rewrite. The rewrite is happening "
        "*inside* the same companies."
    ))
    cells.append(code("fig = viz_within_firm(); plt.show()"))

    cells.append(md(
        "## 2. AI rewriting is SWE-specific (not economy-wide)\n\n"
        "AI-vocabulary in SWE postings rose **+25.5pp** 2024-01 → 2026-04, "
        "while control occupations rose only **+1.1pp**. A 23:1 delta ratio.\n\n"
        "**Why it matters:** this is the cleanest test of H1 (AI as narrative "
        "cover for layoffs) and H3 (non-AI macro story like rate hikes / "
        "post-Covid correction). If either were the true cause, SWE and "
        "control would move together. They don't. The rewrite is something "
        "real and SWE-specific. Independently replicates v8's T18 finding."
    ))
    cells.append(code("fig = viz_swe_vs_control(); plt.show()"))

    cells.append(md(
        "## 3. YOE floors are FALLING, not rising (counter scope-inflation)\n\n"
        "The classic *junior scope inflation* hypothesis predicts that junior "
        "postings ask for more years of experience over time. LLM-extracted "
        "YOE shows the opposite — across **all seniority buckets**.\n\n"
        "- Junior mean YOE: **2.01 → 1.23** years; median 2 → 1.\n"
        "- Senior mean YOE: 6.55 → 6.37; median 6 → 5.\n"
        "- Mid mean YOE: 4.00 → 2.36.\n\n"
        "**Why it matters:** the popular scope-inflation framing is rejected at "
        "the YOE bar. If anything employers are asking for *less* experience. "
        "Note v8's rule-based YOE rose compositionally — LLM-YOE (cleaner per "
        "preprocessing-schema.md) tells the opposite story."
    ))
    cells.append(code("fig = viz_yoe_floor(); plt.show()"))

    cells.append(md(
        "## 4. Dev-tool vendor labor-market leaderboard (first published)\n\n"
        "2026-04 vendor mention rates in SWE postings:\n\n"
        "| rank | vendor    | rate   | growth since 2024-01 |\n"
        "|------|-----------|--------|----------------------|\n"
        "| 1    | Copilot   | 4.25%  | 53× |\n"
        "| 2    | Claude    | 3.83%  | **190×** |\n"
        "| 3    | OpenAI    | 3.63%  | 17× |\n"
        "| 4    | Cursor    | 2.17%  | >200× |\n"
        "| 5    | Anthropic | 1.48%  | 70× |\n\n"
        "**Why it matters:** no one has published a labor-demand vendor-share "
        "table for dev tools. Copilot leads on raw rate; Claude has the "
        "steepest growth and is closing fast. ChatGPT as a *brand* is "
        "plateauing/declining (0.82% → 0.70%) while specific vendors keep "
        "climbing — employers are specializing their AI vocabulary."
    ))
    cells.append(code("fig = viz_vendor_leaderboard(); plt.show()"))

    cells.append(md(
        "## 5. Big Tech: more posting volume AND more AI density\n\n"
        "Big Tech (Google/Meta/Amazon/Apple/Microsoft/Oracle/Anthropic/OpenAI/…) "
        "shows two simultaneous patterns:\n\n"
        "- **Posting share ROSE** from 2.4% (2024-01) to 7.0% (2026-04). "
        "Surprising direction relative to the public layoff narrative.\n"
        "- **AI density 17pp HIGHER** than the rest in 2026 (44% vs 27%), "
        "robust across all four stress-test slices.\n\n"
        "**Why it matters:** the BT vs rest density gap is larger than v8 "
        "published. It's the strongest stratification signal in the EDA and "
        "the natural anchor for a follow-up paper that pairs the AI-density "
        "trajectory with named-firm layoff timelines (Oracle, Block, "
        "Amazon, Meta) and 10-Q filings."
    ))
    cells.append(code("fig = viz_bigtech_density(); plt.show()"))

    cells.append(md(
        "---\n\n"
        "# What got falsified (the prior narratives we tested and rejected)\n\n"
        "These were genuine starting hypotheses — from the user's gut sense, "
        "from *The Economist*, and from the classic scope-inflation framing in "
        "labor-market literature. The data does not support them."
    ))

    cells.append(md(
        "## Disproven 1 — H1 AI-washing (content level)\n\n"
        "*Hypothesis:* tech firms attribute layoffs to AI as a narrative cover, "
        "not because AI is actually replacing the work. (Backed by *The "
        "Economist* 'Code red' Apr 2026 + Bank of England survey finding "
        "'essentially zero' AI employment impact.)\n\n"
        "*Falsified at the content level:* if H1 were the operative story, "
        "SWE and control occupations would show similar AI-language adoption "
        "(both being subjected to the same narrative). They don't. SWE rose "
        "23× faster than control. The rewrite is real, not narrative-only.\n\n"
        "*Caveat:* H1 might still be true at the **volume/timing** layer (do "
        "specific firms invoke AI as the *public reason* for cuts?) — that's "
        "an interview question for RQ4, not testable from posting content."
    ))
    cells.append(code("fig = viz_disproven_aiwashing(); plt.show()"))

    cells.append(md(
        "## Disproven 2 — H4 industry spread to non-tech (on LinkedIn)\n\n"
        "*Hypothesis (from The Economist):* SWE jobs are spreading from pure-"
        "tech firms into retail (+12% 2022→2025), property (+75%), "
        "construction (+100%). Therefore non-tech share of SWE postings should "
        "be larger in 2026 than 2024.\n\n"
        "*Falsified on LinkedIn posting share:* non-tech industries already "
        "held ~55% of LinkedIn SWE postings in 2024 (arshkon baseline) and "
        "remain at ~55% in 2026 (scraped). No shift in posting composition.\n\n"
        "*Caveat:* The Economist cited BLS *occupational* data (employed "
        "people) — that can be true even if LinkedIn posting share is flat, "
        "because non-tech firms may recruit through different channels. Both "
        "claims can coexist; ours specifically rules out the LinkedIn version."
    ))
    cells.append(code("fig = viz_disproven_industry_spread(); plt.show()"))

    cells.append(md(
        "## Disproven 3 — H5(a) junior-first automation\n\n"
        "*Hypothesis:* AI automates routine tasks first, so junior SWE "
        "postings should be the first to show AI-vocabulary adoption (and to "
        "lose share / volume).\n\n"
        "*Falsified:* AI-vocab rate is essentially uniform across seniority "
        "in 2026-04 (junior 27%, mid 30%, senior 31%, unknown 25%). Combined "
        "with H8 (junior YOE *fell*), the junior-first narrative does not "
        "survive. The consistent reading is H5(b) **senior-restructuring**: "
        "senior postings shifted in *content* (toward orchestration/review/"
        "AI-leverage language per v8 T21), while junior postings remained "
        "structurally similar."
    ))
    cells.append(code("fig = viz_disproven_juniorfirst(); plt.show()"))

    cells.append(md(
        "---\n\n"
        "# Verdict at a glance — all 13 hypotheses"
    ))
    cells.append(code("fig = viz_verdict_table(); plt.show()"))

    cells.append(md(
        "---\n\n"
        "## Recommended next moves (ranked)\n\n"
        "1. **Within-firm rewrite paper** anchored on the 292-co overlap panel "
        "(Headline 1). Strongest causal-identification surface in the data. "
        "Pair with RQ4 interviews: *who inside your firm rewrote the JDs?*\n"
        "2. **Big-Tech-stratified RQ1 paper** pairing the 17pp BT-vs-rest gap "
        "(Headline 5) with named-firm layoff timelines and 10-Q filings.\n"
        "3. **Vendor-specificity paper** (Headline 4) — entirely novel "
        "labor-demand artifact for dev tools. Sensitivity check on regex "
        "over-matching needed.\n"
        "4. **YOE-floor inversion methodological note** (Headline 3) — "
        "explicitly contrast LLM-YOE vs rule-based YOE so future readers "
        "don't conflate them.\n\n"
        "## Limitations\n\n"
        "- LinkedIn-only by construction on `unified_core.parquet`; no "
        "Indeed sensitivity in this pass.\n"
        "- Date granularity is too coarse to test ChatGPT (Nov 2022) vs "
        "Claude Code (Feb 2025) lead/lag specifically.\n"
        "- Vendor regexes (S13) may over-match common-word vendors "
        "(\"cursor\", \"claude\", \"llama\") — read directionally.\n"
        "- Within-firm panel (S17) requires ≥5 SWE postings in both periods "
        "→ excludes most small employers.\n"
        "- LLM-YOE coverage ~78% of in-frame SWE rows; possible selection "
        "bias on rows where LLM refused to extract.\n"
        "- No significance tests. Descriptive EDA only. The formal "
        "analysis-phase plan is pending per AGENTS.md §3."
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
