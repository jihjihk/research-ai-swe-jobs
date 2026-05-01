#!/usr/bin/env python3
"""Build the Wave 5 presentation and evidence package.

This script generates the MkDocs source tree under exploration/site/,
copies the supporting figures/tables/artifacts into the site assets,
and rewrites the raw report links so the audit trail stays usable once
packaged inside the static site.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1].parent
PKG_ROOT = ROOT / "exploration" / "site"
DOCS_ROOT = PKG_ROOT / "docs"
MARP_ROOT = PKG_ROOT / "marp"
SITE_OUT = PKG_ROOT / "site"

FIG_SRC = ROOT / "exploration" / "figures"
TBL_SRC = ROOT / "exploration" / "tables"
ART_SHARED_SRC = ROOT / "exploration" / "artifacts" / "shared"
ART_T25_SRC = ROOT / "exploration" / "artifacts" / "T25"
REPORTS_SRC = ROOT / "exploration" / "reports"
MEMOS_SRC = ROOT / "exploration" / "memos"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(dedent(content).lstrip("\n"), encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)


def rewrite_raw_markdown(text: str, kind: str) -> str:
    """Rewrite absolute and repo-relative links for packaged audit copies."""

    replacements = [
        ("/home/jihgaboot/gabor/job-research/exploration/figures/", "../../../assets/figures/"),
        ("/home/jihgaboot/gabor/job-research/exploration/tables/", "../../../assets/tables/"),
        ("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/", "../../../assets/artifacts/shared/"),
        ("/home/jihgaboot/gabor/job-research/exploration/artifacts/T25/", "../../../assets/artifacts/T25/"),
        ("../figures/", "../../../assets/figures/"),
        ("../tables/", "../../../assets/tables/"),
        ("../artifacts/shared/", "../../../assets/artifacts/shared/"),
        ("../artifacts/T25/", "../../../assets/artifacts/T25/"),
        ("exploration/figures/", "../../../assets/figures/"),
        ("exploration/tables/", "../../../assets/tables/"),
        ("exploration/artifacts/shared/", "../../../assets/artifacts/shared/"),
        ("exploration/artifacts/T25/", "../../../assets/artifacts/T25/"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    if kind == "report":
        text = text.replace("/home/jihgaboot/gabor/job-research/exploration/reports/", "./")
        text = text.replace("/home/jihgaboot/gabor/job-research/exploration/memos/", "../memos/")
        text = text.replace("exploration/reports/", "./")
        text = text.replace("exploration/memos/", "../memos/")
    else:
        text = text.replace("/home/jihgaboot/gabor/job-research/exploration/reports/", "../reports/")
        text = text.replace("/home/jihgaboot/gabor/job-research/exploration/memos/", "./")
        text = text.replace("exploration/reports/", "../reports/")
        text = text.replace("exploration/memos/", "./")
    return text


def render_finding_page(
    title: str,
    claim: str,
    summary: str,
    evidence_lines: list[str],
    sensitivity_lines: list[str],
    figures: list[str],
    raw_links: list[tuple[str, str]],
    takeaways: list[str] | None = None,
) -> str:
    fig_md = "\n\n".join(f"![{Path(fig).name}](../assets/{fig})" for fig in figures)
    links_md = "\n".join(f"- [{label}]({target})" for label, target in raw_links)
    take_md = "\n".join(f"- {line}" for line in (takeaways or []))
    evidence_md = "\n".join(f"- {line}" for line in evidence_lines)
    sensitivity_md = "\n".join(f"- {line}" for line in sensitivity_lines)
    return f"""
# {title}

## Claim
{claim}

{summary}

## Evidence
{evidence_md}

## Figures
{fig_md}

## Sensitivity and caveats
{sensitivity_md}

## Raw trail
{links_md}

{('## What this means\n' + take_md) if take_md else ''}
"""


def build_pages() -> dict[str, str]:
    pages: dict[str, str] = {}

    pages["index.md"] = """
# SWE labor market restructuring, packaged as a navigable evidence set.

This site is organized around one reading of the exploration: **differential densification under template drift**. The strongest current story is that SWE postings became more AI/LLM-centered, more multi-constraint, and more standardized in form. The simple dramatic stories do not survive the evidence: there is no clean junior collapse, no clean management decline, and no robust employer-worker divergence.

The longer, more sectioned postings are real, but they do not prove AI-authorship. Treat AI-assisted drafting as a hypothesis, not a result.

Use the tabs to move between the story, the findings, the methods, and the raw trail. The slide deck below is the fastest entry point.

<iframe class="presentation-frame" src="presentation.html" title="Exploration presentation" loading="lazy"></iframe>

## Start here

| Layer | What it answers | Best entry page |
|---|---|---|
| Story | What is the one-sentence reading of the project? | [Findings hub](findings/index.md) |
| Evidence | What claims are actually supported? | [Findings hub](findings/index.md) |
| Methods | How should the data be read? | [Preprocessing and schema](methods/preprocessing.md) |
| Audit trail | What exactly was done? | [Raw archive](audit/raw/index.md) |

## Headline claims

| Rank | Claim | Status | Basis |
|---|---|---|---|
| 1 | The market's latent structure is domain-first, and AI/LLM became a coherent skill ecosystem. | Strong | T09, T14, T15 |
| 2 | Requirement stacking rose more clearly than raw tech breadth. | Strong | T11, T13 |
| 3 | Job descriptions got longer and more sectioned, but AI-authorship is only a hypothesis. | Strong | T13, T12 |
| 4 | Returning employers changed their own postings; within-company change is the headline. | Strong, cluster labels provisional | T16, V2 |
| 5 | Senior roles add AI/tool orchestration without a clean management decline. | Useful direction, recomputation pending | T20, T21, V2 |
| 6 | Explicit entry is a conservative lower bound; low-YOE postings remain. | Strong | T03, T08, T20 |
| 7 | AI-tool growth is SWE-specific, while length and scope are broader across roles. | Useful direction, recomputation pending | T18, T19, V2 |
| 8 | Ghost-like language looks aspirational and template-heavy, not fake jobs. | Provisional, recomputation pending | T22, V2 |
| 9 | The divergence claim is benchmark-sensitive, not settled. | Calibration only | T23, V2 |

The strongest paper message is not one dramatic claim but the combination of these nine readings. The weakest versions are the ones that collapse them back into a single junior-collapse story.

## Where to go next

- [Domain recomposition and AI/LLM growth](findings/domain-recomposition.md)
- [Requirement stacking and scope](findings/credential-stacking-scope.md)
- [Posting-template drift](findings/template-drift.md)
- [Within-firm decomposition and company strategy](findings/company-strategies.md)
- [Cross-occupation densification](findings/cross-occupation.md)
- [Seniority and orchestration](findings/seniority.md)
- [Junior measurement asymmetry](findings/junior-measurement.md)
- [Ghost and aspiration](findings/ghost-aspiration.md)
- [Benchmark-sensitive divergence](findings/divergence.md)
"""

    pages["findings/index.md"] = """
# Findings that matter most for the paper.

The strongest narrative is no longer the original junior-collapse story. The evidence now points to posting densification around AI/LLM, scope, and structured requirements, with the observed change split across template drift, company composition, and within-firm strategy.

| Rank | Claim | Status | Support / caveat |
|---|---|---|---|
| 1 | [The market's latent structure is domain-first, and AI/LLM became a coherent skill ecosystem.](domain-recomposition.md) | Strong | T09 is the structural result; T14 gives the ecosystem layer; T15 rules out a junior-seniority convergence story. |
| 2 | [Requirement stacking rose more clearly than raw tech breadth.](credential-stacking-scope.md) | Strong | Breadth, stack depth, scope, and AI all move; tech count is not the lead metric. |
| 3 | [Job descriptions got longer and more sectioned, but AI-authorship is only a hypothesis.](template-drift.md) | Strong | T13 is the validity anchor; text growth is real but form changed too. |
| 4 | [Returning employers changed their own postings; the decomposition is the headline.](company-strategies.md) | Strong, cluster labels provisional | T16 decomposition is verified; the exact cluster typology is not. |
| 5 | [Explicit entry is a conservative lower bound; low-YOE postings remain.](junior-measurement.md) | Strong | The instrument split is real and must travel with any junior claim. |
| 6 | [Senior roles add AI/tool orchestration without a clean management decline.](seniority.md) | Useful direction, recomputation pending | T21 points this way; broad management stays out of the lead. |
| 7 | [AI-tool growth is SWE-specific, while length and scope are broader across roles.](cross-occupation.md) | Useful direction, recomputation pending | T18/T19 are directionally useful but still need analysis-phase rederivation. |
| 8 | [Ghost-like language looks aspirational and template-heavy, not fake jobs.](ghost-aspiration.md) | Provisional, recomputation pending | Use as a mechanism hypothesis, not a headline. |
| 9 | [The divergence claim is benchmark-sensitive, not settled.](divergence.md) | Calibration only | T23 belongs in the validity section unless the benchmark strategy is locked. |

Legend: **Strong** means the current evidence is enough to lead with, **useful direction** means the trend is informative but still needs the analysis-phase recomputation noted in the verification memos, **provisional** means the direction is promising but not yet stable enough for a headline, and **calibration only** means the result should not be stated as a binary claim.
"""

    pages["findings/domain-recomposition.md"] = render_finding_page(
        title="The market's latent structure is domain-first, and AI/LLM became a coherent skill ecosystem.",
        claim="The clearest structural result is that posting bundles line up with tech domains far more than with seniority or period, and the AI/LLM layer shows up as an organized stack rather than scattered keyword growth.",
        summary=(
            "T09 is the best latent-structure result in the exploration. NMF with `k=15` is the first representation that cleanly separates backend, data, infra, embedded, frontend, mobile, AI/LLM workflows, and requirements/compliance. T14 then shows the AI/LLM layer is a coherent ecosystem embedded in the stack, not just a larger count of isolated keywords."
        ),
        evidence_lines=[
            "T09 finds `tech_domain` NMI around 0.115-0.123, while seniority NMI is about 0.003.",
            "The AI/LLM workflows archetype is 91.5% 2026 in the full-corpus labels.",
            "T14 shows the underlying technology network now has a stable cloud/frontend backbone with a much denser AI/LLM layer.",
            "T15 rejects a clean junior-senior convergence story, which makes a seniority-first interpretation even weaker.",
        ],
        sensitivity_lines=[
            "BERTopic is useful as a coarse comparator, but it collapses the structure too aggressively for downstream claims.",
            "The exact tech-domain decimal shifts a little under independent rebuilding, but the ordering does not.",
            "The raw-text sensitivity preserves the domain map, but it makes boilerplate more visible and should not replace the cleaned-text analysis.",
        ],
        figures=[
            "figures/T09/T09_embedding_maps.png",
            "figures/T14/T14_cooccurrence_network.png",
            "figures/T15/T15_similarity_heatmaps.png",
        ],
        raw_links=[
            ("T09 report", "../audit/raw/reports/T09.md"),
            ("T14 report", "../audit/raw/reports/T14.md"),
            ("T15 report", "../audit/raw/reports/T15.md"),
            ("Synthesis", "../audit/raw/reports/SYNTHESIS.md"),
        ],
        takeaways=[
            "Lead the paper with domain recomposition, not a broad seniority axis.",
            "Use NMF `k=15` as the downstream archetype map and keep BERTopic as a coarse check only.",
            "Treat AI/LLM as an ecosystem-level shift, not a keyword trend.",
        ],
    )

    pages["findings/credential-stacking-scope.md"] = render_finding_page(
        title="Requirement stacking rose more clearly than raw tech breadth.",
        claim="The stronger content result is that 2026 postings ask for more categories at once, not simply more technologies.",
        summary=(
            "T11 shows the change is concentrated in requirement breadth, credential stack depth, scope density, and AI mentions. `tech_count` moves only modestly and does not cleanly clear the 2024 calibration baseline, so the paper should not lead with a generic 'more tech' claim."
        ),
        evidence_lines=[
            "On the primary company-capped LLM-text subset, `requirement_breadth` and `credential_stack_depth` both rise from 2024 to 2026.",
            "Scope density and AI mentions are the clearest increases.",
            "The junior-like explicit-entry slice and the YOE proxy both show the same direction, but the YOE slice is broader and more complex.",
            "T13 explains why part of the apparent growth is a form change: the postings are more sectioned in 2026.",
        ],
        sensitivity_lines=[
            "Company capping keeps the direction of breadth, stack, and scope intact.",
            "Raw-text fallback nudges the magnitudes, but it does not reverse the story.",
            "Broad management is noisy and should stay out of the lead result.",
        ],
        figures=[
            "figures/T11/T11_complexity_distributions.png",
            "figures/T11/T11_entry_complexity_comparison.png",
            "figures/T13/T13_section_composition.png",
        ],
        raw_links=[
            ("T11 report", "../audit/raw/reports/T11.md"),
            ("T13 report", "../audit/raw/reports/T13.md"),
            ("T08 report", "../audit/raw/reports/T08.md"),
        ],
        takeaways=[
            "The publishable claim is credential stacking around the SWE core, not raw tech inflation.",
            "AI and scope belong at the center of RQ2, not as side examples.",
        ],
    )

    pages["findings/template-drift.md"] = render_finding_page(
        title="Job descriptions got longer and more sectioned, but AI-authorship is only a hypothesis.",
        claim="Length growth is real, but it is partly a template shift: 2026 ads are far more explicitly sectioned than 2024 ads.",
        summary=(
            "T13 is a validity result as much as a substantive one. Cleaned text length rises from roughly 1,969 chars in arshkon to 2,644 in scraped, but the bigger change is that core sections are nearly absent in 2024 and common in 2026. T12 then shows that cleaned-word shifts are messy unless the section structure is respected, which is why AI-authorship should stay a hypothesis rather than a conclusion."
        ),
        evidence_lines=[
            "Only about 1.6% of arshkon docs and 4.5% of asaniczka docs have detected core sections, versus about 61-62% in scraped 2026.",
            "Requirements, responsibilities, preferred, and other structured sections account for a much larger share of the scraped corpus.",
            "The cleaned corpus is denser than the raw corpus; it is not a simpler one.",
            "The raw-vs-cleaned comparison in T12 is too imbalanced to stand alone, which is exactly why the section artifact matters.",
        ],
        sensitivity_lines=[
            "The section parser misses unmarked historical structure, so 'unclassified' is a real residual category.",
            "Raw text remains acceptable only for recall-sensitive checks, not as the primary text surface.",
            "The historical-vs-scraped gap is larger than the within-2024 difference, so template drift is a first-order confound.",
            "Longer and more sectioned postings are compatible with AI-assisted drafting, but the exploration does not identify who wrote the posting.",
        ],
        figures=[
            "figures/T13/T13_section_composition.png",
            "figures/T12/T12_category_summary.png",
            "figures/T05/description_length_overlap.png",
        ],
        raw_links=[
            ("T13 report", "../audit/raw/reports/T13.md"),
            ("T12 report", "../audit/raw/reports/T12.md"),
            ("T05 report", "../audit/raw/reports/T05.md"),
        ],
        takeaways=[
            "The paper should treat posting structure as part of the measurement problem, not a nuisance detail.",
            "Any later text claim has to condition on sectioning or explicitly say it does not.",
            "AI-authorship belongs in the interview and mechanism discussion, not the headline result.",
        ],
    )

    pages["findings/company-strategies.md"] = render_finding_page(
        title="Returning employers changed their own postings; the decomposition is the headline.",
        claim="The company-level evidence is strongest when you read it as within-firm densification rather than a single market-wide shift.",
        summary=(
            "T16 is the clean decomposition result. Among 237 returning companies with at least three SWE postings in both periods, the overlap panel shows most of the AI, scope, and length movement happening within firms. The pooled-2024 baseline flips the explicit-entry sign, which is why the company and seniority instruments cannot be treated as interchangeable. The exact cluster typology remains provisional."
        ),
        evidence_lines=[
            "The verified overlap counts are 237 companies at `>=3` and 122 at `>=5`.",
            "AI, scope, and length changes are mostly within-company; the within-company components dominate the headline deltas.",
            "Explicit entry falls slightly on the arshkon-only panel, but flips positive under pooled 2024; the YOE proxy stays positive in both cases.",
            "The exact four-cluster typology was not reproduced cleanly in verification, so the decomposition should lead and the cluster labels should stay provisional.",
        ],
        sensitivity_lines=[
            "The no-aggregator and cap-25 variants preserve the decomposition direction.",
            "The exact cluster split is spec-dependent and should not be over-literalized.",
            "Use the decomposition as the core result and the cluster names only as a provisional lens.",
            "The verified headline is within-company change; the cluster labels are a secondary heuristic.",
        ],
        figures=[
            "figures/T16/T16_decomposition_bars.png",
            "figures/T16/T16_cluster_heatmap.png",
            "figures/T16/T16_domain_family_shift.png",
        ],
        raw_links=[
            ("T16 report", "../audit/raw/reports/T16.md"),
            ("V2 verification", "../audit/raw/reports/V2_verification.md"),
            ("Synthesis", "../audit/raw/reports/SYNTHESIS.md"),
        ],
        takeaways=[
            "The story is company heterogeneity and within-firm change, not a single market average.",
            "The typology is useful as a heuristic, but the decomposition is the defensible headline.",
            "Lead with the verified overlap panel before mentioning any cluster labels.",
        ],
    )

    pages["findings/cross-occupation.md"] = render_finding_page(
        title="AI-tool growth is SWE-specific, while length and scope are broader across roles.",
        claim="The best cross-occupation reading is porous boundary, not collapse: SWE changes are real, but some posting-template drift is broader than SWE.",
        summary=(
            "T18 shows that AI-tool language is the cleanest SWE-specific signal relative to adjacent and control occupations. Length and scope, by contrast, move more like field-wide template drift. T19 then shows the time structure is windowed, with sharp jumps between the 2024 baseline and the scraped 2026 window. The direction is useful, but both T18 and T19 still need the analysis-phase recomputation noted in V2."
        ),
        evidence_lines=[
            "AI-tool share rises from 1.98% to 20.38% for SWE, from 2.45% to 18.14% for adjacent roles, and only from 1.25% to 1.41% for controls.",
            "Boundary similarity stays high at about 0.80-0.83, which means the SWE-adjacent boundary is porous but not collapsed.",
            "Requirement breadth rises from 5.94 to 6.98, raw description length from 2,974 to 4,891, and AI-tool share from 2.56% to 20.30% across the source windows.",
            "Posting age clusters at one day in the scraped window, which looks more like scrape cadence than backlog age.",
        ],
        sensitivity_lines=[
            "The control-group result keeps the AI-tool story from becoming a generic macro trend.",
            "Scope and length still rise in controls, so they should not be framed as SWE-only breaks.",
            "Remote status is unusable in the selected metro frame and should not be over-interpreted.",
            "Treat the T18/T19 magnitudes as directionally informative until the recomputation pass locks the exact aggregates.",
        ],
        figures=[
            "figures/T18/T18_parallel_trends.png",
            "figures/T18/T18_boundary_similarity.png",
            "figures/T19/T19_source_window_rates.png",
        ],
        raw_links=[
            ("T18 report", "../audit/raw/reports/T18.md"),
            ("T19 report", "../audit/raw/reports/T19.md"),
            ("V2 verification", "../audit/raw/reports/V2_verification.md"),
        ],
        takeaways=[
            "Keep AI-tool language as the SWE-specific signal, and keep length/scope as a broader posting-surface shift.",
            "Do not model the data as a continuous time series; it is a set of windows and snapshots.",
            "Cross-occupation work is a specificity check, not a new lead claim.",
        ],
    )

    pages["findings/seniority.md"] = render_finding_page(
        title="Senior roles add AI/tool orchestration without a clean management decline.",
        claim="The senior-role story is not a clean management decline. The stronger result is orchestration growth layered on top of a stable people-manager base.",
        summary=(
            "T20 shows the seniority ladder is measurable but uneven: `associate -> mid-senior` is the sharpest boundary, while `entry -> associate` is the weakest. T21 then points toward strict management staying flat or slightly up, while AI/tool orchestration and AI-domain strategy grow sharply. That direction is useful, but the strict-management / orchestration aggregates still need the analysis-phase recomputation noted in V2."
        ),
        evidence_lines=[
            "The best 3-class senior role decomposition shifts from people-manager dominance toward an AI/tool-orchestrator cluster and a smaller AI-domain strategist cluster.",
            "The broad management patterns are too noisy to carry the main claim, which is why strict management must lead if management is mentioned at all.",
            "The director slice is important but small; it looks more like a refined version of the same shift than a wholly separate world.",
            "Cross-seniority comparison suggests the senior shift is not simply a downward migration from entry-level patterns.",
        ],
        sensitivity_lines=[
            "Broad management is below the precision bar and should stay a sensitivity only.",
            "Use the YOE proxy beside `seniority_final` whenever the claim touches juniors.",
            "The strict-vs-broad pattern is not just a wording issue; it changes the result materially.",
            "Until T21 is rederived, use the direction of the orchestration shift rather than the exact cluster proportions.",
        ],
        figures=[
            "figures/T20/T20_boundary_auc_comparison.png",
            "figures/T21/T21_cross_seniority_management_comparison.png",
            "figures/T21/T21_management_by_seniority.png",
        ],
        raw_links=[
            ("T20 report", "../audit/raw/reports/T20.md"),
            ("T21 report", "../audit/raw/reports/T21.md"),
            ("T03 report", "../audit/raw/reports/T03.md"),
        ],
        takeaways=[
            "The paper should talk about senior AI/tool orchestration, not management disappearance.",
            "The seniority boundary work matters mainly because it prevents over-reading the junior story.",
            "Management decline is the wrong headline; orchestration growth is the better one.",
        ],
    )

    pages["findings/junior-measurement.md"] = render_finding_page(
        title="Explicit entry is a conservative lower bound; low-YOE postings remain.",
        claim="The junior story depends on the instrument: explicit labels are conservative, while the YOE proxy captures a broader junior-like pool.",
        summary=(
            "T03, T08, and T20 all point in the same direction. `seniority_final = entry` is tiny, but `yoe_extracted <= 2` is materially broader and often rises when the explicit label does not. That is not a nuisance; it is the central measurement constraint on any junior claim. The right headline is not that juniors vanished; it is that low-YOE postings remain even when explicit entry labels stay scarce."
        ),
        evidence_lines=[
            "`seniority_final = entry` is 3.73% in arshkon and 2.18% in scraped 2026-04.",
            "The YOE proxy is 14.98% in arshkon and 16.97% in scraped 2026-04.",
            "Arshkon native `entry` rows are not a stable junior baseline across snapshots, which is why `seniority_native` is diagnostic only.",
            "T20 shows `associate -> mid-senior` is the sharpest boundary, while `entry -> associate` is the weakest.",
        ],
        sensitivity_lines=[
            "Asaniczka cannot be pooled into a `seniority_native` entry baseline because it has zero native entry labels.",
            "Any junior figure should show explicit-entry and YOE proxy side by side.",
            "Material disagreement between the instruments is itself a finding, not a problem to hide.",
        ],
        figures=[
            "figures/T03/T03_junior_share_comparison.png",
            "figures/T03/T03_native_vs_final_heatmaps.png",
            "figures/T08/T08_junior_share_trends.png",
        ],
        raw_links=[
            ("T03 report", "../audit/raw/reports/T03.md"),
            ("T08 report", "../audit/raw/reports/T08.md"),
            ("T20 report", "../audit/raw/reports/T20.md"),
            ("T02 report", "../audit/raw/reports/T02.md"),
        ],
        takeaways=[
            "Junior collapse is too simple; the more defensible claim is instrument-dependent junior visibility.",
            "The analysis-phase rule should be: show explicit labels and YOE together, or do not make a junior claim.",
            "Explicit entry is a lower bound, not a complete census of junior work.",
        ],
    )

    pages["findings/ghost-aspiration.md"] = render_finding_page(
        title="Ghost-like language looks aspirational and template-heavy, not fake jobs.",
        claim="The ghost story is not that postings are fake. It is that some postings are overloaded, hedged, and templated in ways that make them look more aspirational than screening-tight.",
        summary=(
            "T22 shows AI postings have a higher hedge-to-firm ratio than non-AI postings, and direct employers are slightly more ghost-like overall. The best reading is bundled scope plus aspirational wording, not literal fabricated roles. Use this as a mechanism hypothesis until the T22 recomputation is finished."
        ),
        evidence_lines=[
            "On the section-filtered LLM core, AI postings have a hedge/firm ratio of about 0.73 versus 0.52 for non-AI postings.",
            "The raw-text sensitivity preserves the direction, though the magnitudes change.",
            "Direct employers are slightly more ghost-like overall; aggregators contribute a distinct form of template saturation.",
            "The validated management artifact demotes broad management to sensitivity only, which is a useful reminder that generic wording can be misleading.",
        ],
        sensitivity_lines=[
            "The result is noisy enough that it should stay in the validity section, not the paper's headline.",
            "Broad management is a bad proxy for the same reason ghost language is tricky: generic words absorb too much noise.",
            "This is more a mechanism question than a proof of fake jobs.",
            "The exact hedge ratios still need the analysis-phase recomputation from V2.",
        ],
        figures=[
            "figures/T22/T22_ai_aspiration_ratio.png",
            "figures/T22/T22_ghost_score_by_period_seniority.png",
            "figures/T22/T22_template_saturation_top_companies.png",
        ],
        raw_links=[
            ("T22 report", "../audit/raw/reports/T22.md"),
            ("Validated management patterns", "../assets/artifacts/shared/validated_mgmt_patterns.json"),
        ],
        takeaways=[
            "AI wording looks more aspirational than non-AI wording, but that is not the same as fake jobs.",
            "The right follow-up is qualitative: who writes the posting, how often is it templated, and what changes screening practice?",
            "Ghost language is a mechanism lens, not a headline result.",
        ],
    )

    pages["findings/divergence.md"] = render_finding_page(
        title="The divergence claim is benchmark-sensitive, not settled.",
        claim="Posting-side AI language is real, but the employer-worker comparison cannot be reduced to one number or one benchmark.",
        summary=(
            "T23 is useful because it prevents overclaiming. The posting-side AI-tool rate is elevated, but the sign and size of the divergence depend on which benchmark is used. The right publication move is a benchmark band, not a binary outperformance claim. The T23 aggregates still need the analysis-phase recomputation noted in V2, so treat the direction as informative but not final."
        ),
        evidence_lines=[
            "The primary section-filtered AI-tool rate is about 30.3%, rising to 40.7% under raw-text sensitivity.",
            "The comparison flips depending on whether the benchmark is 32.4%, 51%, 84%, or 99%.",
            "That means the divergence issue is a calibration question, not a settled result.",
            "The stronger claim is that employer-side AI signaling is high; the weaker claim is that it clearly outpaces worker usage.",
        ],
        sensitivity_lines=[
            "Benchmark sensitivity is the main result, not a nuisance note.",
            "The right visualization is a range, not a single bar.",
            "This is the one place where a skeptical reader should stay skeptical until the analysis phase.",
            "Do not use the divergence comparison as a headline unless the benchmark strategy is frozen.",
        ],
        figures=[
            "figures/T23/T23_ai_requirement_vs_benchmarks.png",
        ],
        raw_links=[
            ("T23 report", "../audit/raw/reports/T23.md"),
            ("V2 verification", "../audit/raw/reports/V2_verification.md"),
        ],
        takeaways=[
            "Treat RQ3 as a validity section unless the analysis phase locks a benchmark strategy.",
            "The useful output here is not a single gap, but the range of plausible gaps.",
            "Benchmark mismatch is a feature of the result, not a footnote.",
        ],
    )

    pages["methods/index.md"] = """
# Methods and data.

This section explains how the heterogeneous inputs became a comparable corpus, how to read the resulting columns, and how to interpret the findings without over-reading the measurement surface.

## Pages

| Page | What it covers |
|---|---|
| [Preprocessing and schema](preprocessing.md) | How raw Kaggle and scraped postings become `data/unified.parquet`. |
| [Evidence and sensitivity](evidence.md) | What counts as a robust finding and what remains provisional. |

## The short version

- The preprocessing pipeline creates a single comparable corpus from two Kaggle snapshots and a growing daily scrape.
- `description_core_llm` is the only cleaned text column, and it is coverage-limited.
- `seniority_final` is the primary seniority label, but the YOE proxy must travel with any junior claim.
- `selected_for_llm_frame`, `llm_extraction_coverage`, and `llm_classification_coverage` determine which rows are actually labeled.
- Text claims should be read on the cleaned, labeled subset unless the analysis is explicitly recall-oriented.
"""

    pages["methods/preprocessing.md"] = """
# Preprocessing and schema.

The preprocessing pipeline exists because the raw inputs are heterogeneous: two historical Kaggle snapshots and a growing daily scrape. The pipeline normalizes, deduplicates, classifies, and enriches them into `data/unified.parquet` and `data/unified_observations.parquet`.

## Stage-by-stage summary

| Stage | What it does | Why it matters |
|---|---|---|
| 1 | Ingests the three sources and unifies them into a canonical schema. | Creates a single posting table and daily observation table that downstream tasks can compare. |
| 2 | Flags aggregators and staffing agencies, then derives a real-employer field. | Prevents staffing intermediaries from contaminating employer-level claims. |
| 4 | Canonicalizes company names and deduplicates openings. | Keeps company-level and corpus-level counts from being inflated by duplicates and spelling drift. |
| 5 | Classifies SWE status, seniority, and years of experience. | Produces the core analysis labels, including the conservative `seniority_final` field. |
| 6-8 | Normalizes location, temporal, language, and quality flags. | Makes geography, timing, and data quality comparable across sources. |
| 9 | Selects the LLM core frame and removes boilerplate to create `description_core_llm`. | Produces the only cleaned-text column and makes text-sensitive work coverage-explicit. |
| 10 | Runs LLM classification and merges results back into the full table. | Completes the combined seniority, SWE, ghost, and YOE cross-check layer. |
| final | Writes `data/unified.parquet` and `data/unified_observations.parquet`. | Produces the analysis-ready dataset used in the exploration. |

## Output data structure

One row in `data/unified.parquet` is one unique posting. The main column families are:

- identity and provenance
- raw and cleaned text
- company and aggregator fields
- SWE and seniority classification
- YOE extraction
- temporal, geography, and quality flags
- LLM-derived classification and coverage columns

The coverage columns matter. `selected_for_llm_frame` marks the sticky balanced core only. `llm_extraction_coverage` and `llm_classification_coverage` tell you whether a row was actually labeled. `description_core_llm` is the only cleaned text column, and `seniority_final` is the primary seniority column.

## LLM stages and prompts

<details>
<summary>Stage 9: boilerplate removal and cleaned text</summary>

The LLM is asked to identify boilerplate units in a segmented job description and return the cleaned posting text. The important output is `description_core_llm`.

Condensed prompt:

> Given a job posting split into sentence-like units, remove boilerplate and return the core posting text. Keep role content, requirements, and responsibilities. Drop company marketing, legal, and generic filler.

</details>

<details>
<summary>Stage 10: classification and integration</summary>

The LLM is asked to classify SWE status, resolve seniority when the rule-based layer cannot, assess ghost-job risk, and cross-check YOE. The explicit seniority signals are the only valid basis for the LLM seniority result.

Condensed prompt:

> Classify the posting's SWE status, seniority, ghost-job assessment, and minimum years of experience using the cleaned text and explicit signals only. Do not infer seniority from responsibilities, tech stack complexity, or YOE requirements.

</details>

## Budgeting and coverage caveats

Stages 9 and 10 require an explicit `--llm-budget`. That budget caps new LLM calls, so not every row receives LLM-derived columns. Stage 9 and Stage 10 use separate caches, which is why coverage can differ row by row. Findings built on LLM columns should always report the labeled count alongside the eligible count and should distinguish the sticky core from supplemental cache rows.
"""

    pages["methods/evidence.md"] = """
# Evidence and sensitivity.

The safest way to read the exploration is to treat every finding as a specification plus a sensitivity envelope. A result is only robust if it survives the checks that matter for that unit of analysis.

## Sensitivity framework

| Dimension | Primary reading | Why it matters |
|---|---|---|
| Aggregator exclusion | Include all rows unless the claim is about direct employers only. | Staffing intermediaries have different text and seniority patterns. |
| Company capping | Cap prolific employers for corpus-level text work. | Prevents a handful of employers from dominating frequencies and topics. |
| Seniority operationalization | Use `seniority_final`, then check the YOE proxy. | Explicit entry is conservative and junior claims are otherwise fragile. |
| Description text source | Use `description_core_llm` for text-sensitive work. | Raw `description` is recall-oriented and can be boilerplate-driven. |
| Source restriction | Compare arshkon vs scraped, and use pooled 2024 only as a sensitivity. | The historical snapshots are different instruments. |
| Within-2024 calibration | Compare the 2024 source gap to the 2024-to-2026 change. | If the temporal change is smaller than instrument noise, flag it. |
| SWE classification tier | Exclude `title_lookup_llm` when boundary sensitivity matters. | That tier has a higher false-positive risk. |
| LLM coverage | Report labeled rows and total eligible rows separately. | Coverage is thin in scraped text, so the sample is not the population. |
| Indeed cross-platform check | Use it as a sensitivity, not as the primary platform. | It is useful context, but LinkedIn remains the core analysis surface. |

## How to read the paper

- Strong findings usually survive the main sensitivities and the within-2024 calibration check.
- Spec-dependent findings should be labeled as such and should not be treated as headline proof.
- If a result changes by more than about 30 percent across the core sensitivity dimensions, the mechanism matters as much as the direction.
- Whenever explicit seniority and the YOE proxy disagree, the disagreement itself should be reported.
- The data can identify template drift and AI-like language, but it cannot prove who authored a posting. AI-authorship should stay in the hypothesis bucket unless interviews or other evidence resolve it.

## Limitations and open questions

- The cleaned-text coverage on scraped SWE is still thin.
- The section parser exposes a real template shift, which means later text claims must be careful about form versus content.
- AI-authorship is plausible but unproven; the exploration does not identify authors.
- Company composition matters, so pooled market averages can hide within-firm change.
- The benchmark for employer-worker divergence is not settled.
- The interview phase should focus on the contradictions surfaced by T22 and T23.
"""

    pages["audit/index.md"] = """
# Audit trail.

This tab is the navigation layer for the raw evidence. It exists so a reader can move from the packaged claims back to the underlying reports and gate memos without guessing where the material came from.

## Entry points

- [Raw archive](raw/index.md)
- [Synthesis](raw/reports/SYNTHESIS.md)
- [Gate 1 memo](raw/memos/gate_1.md)
- [Gate 2 memo](raw/memos/gate_2.md)
- [Gate 3 memo](raw/memos/gate_3.md)

## Verification memos

- [Gate 2 verification](raw/reports/V1_verification.md)
- [Gate 3 verification](raw/reports/V2_verification.md)

## What to expect

The raw reports are grouped by wave. They preserve the exploration logic, the caveats, and the task-level outputs. The packaged findings pages are the synthesis layer; the raw reports are the audit layer.
"""

    pages["audit/raw/index.md"] = """
# Raw archive.

This archive is organized by wave. The files are copied into the site so the evidence trail stays local to the packaged artifact.

## Wave 1

- [T01 Data profile](reports/T01.md)
- [T02 Seniority comparability](reports/T02.md)
- [T03 Seniority audit](reports/T03.md)
- [T04 SWE classification audit](reports/T04.md)
- [T05 Dataset comparability](reports/T05.md)
- [T06 Company concentration](reports/T06.md)
- [T07 External benchmarks and power](reports/T07.md)

## Wave 2

- [T08 Distribution profiling](reports/T08.md)
- [T09 Archetype discovery](reports/T09.md)
- [T10 Title taxonomy](reports/T10.md)
- [T11 Requirement complexity](reports/T11.md)
- [T12 Open-ended text evolution](reports/T12.md)
- [T13 Linguistic and structural evolution](reports/T13.md)
- [T14 Technology ecosystem](reports/T14.md)
- [T15 Semantic similarity](reports/T15.md)

## Wave 3

- [T16 Company strategies](reports/T16.md)
- [T17 Geographic structure](reports/T17.md)
- [T18 Cross-occupation boundary](reports/T18.md)
- [T19 Temporal patterns](reports/T19.md)
- [T20 Seniority boundary clarity](reports/T20.md)
- [T21 Senior role evolution](reports/T21.md)
- [T22 Ghost forensics](reports/T22.md)
- [T23 Employer-worker divergence](reports/T23.md)

## Wave 4

- [T24 Hypothesis generation](reports/T24.md)
- [T25 Interview artifacts](../../assets/artifacts/T25/README.md)
- [Synthesis](reports/SYNTHESIS.md)

## Gates and verification

- [Gate 0 pre-exploration](memos/gate_0_pre_exploration.md)
- [Gate 1 memo](memos/gate_1.md)
- [Gate 2 memo](memos/gate_2.md)
- [Gate 3 memo](memos/gate_3.md)
- [Gate 2 verification](reports/V1_verification.md)
- [Gate 3 verification](reports/V2_verification.md)
- [Index](reports/INDEX.md)
"""

    raw_reports = [
        "T01.md",
        "T02.md",
        "T03.md",
        "T04.md",
        "T05.md",
        "T06.md",
        "T07.md",
        "T08.md",
        "T09.md",
        "T10.md",
        "T11.md",
        "T12.md",
        "T13.md",
        "T14.md",
        "T15.md",
        "T16.md",
        "T17.md",
        "T18.md",
        "T19.md",
        "T20.md",
        "T21.md",
        "T22.md",
        "T23.md",
        "T24.md",
        "SYNTHESIS.md",
        "INDEX.md",
        "V1_verification.md",
        "V2_verification.md",
    ]
    raw_memos = ["gate_1.md", "gate_2.md", "gate_3.md"]

    for name in raw_reports:
        text = (REPORTS_SRC / name).read_text(encoding="utf-8")
        pages[f"audit/raw/reports/{name}"] = rewrite_raw_markdown(text, "report")

    for name in raw_memos:
        text = (MEMOS_SRC / name).read_text(encoding="utf-8")
        pages[f"audit/raw/memos/{name}"] = rewrite_raw_markdown(text, "memo")

    return pages


def build_marp_deck() -> str:
    slides = [
        {
            "title": "The best current reading is differential densification under template drift.",
            "figure": "assets/figures/T16/T16_decomposition_bars.png",
            "points": [
                "The market changed, but not as a simple junior-collapse story.",
                "Returning firms drive most of the AI, scope, and length movement within-company.",
            ],
        },
        {
            "title": "Job descriptions got longer and more sectioned, but AI-authorship is only a hypothesis.",
            "figure": "assets/figures/T13/T13_section_composition.png",
            "points": [
                "2026 ads are much more explicitly sectioned than 2024 ads.",
                "Longer text is not the same thing as more demand content.",
            ],
        },
        {
            "title": "Requirement stacking rose more clearly than raw tech breadth.",
            "figure": "assets/figures/T11/T11_complexity_distributions.png",
            "points": [
                "AI, scope, YOE, and soft skills bundle together more often.",
                "Tech count moves, but it is not the clean headline.",
            ],
        },
        {
            "title": "The market's latent structure is domain-first, and AI/LLM became a coherent skill ecosystem.",
            "figure": "assets/figures/T09/T09_embedding_maps.png",
            "points": [
                "NMF `k=15` is the first representation that gives a useful domain map.",
                "Tech-domain alignment is far stronger than seniority alignment.",
            ],
        },
        {
            "title": "Returning employers changed their own postings; the decomposition is the headline.",
            "figure": "assets/figures/T16/T16_decomposition_bars.png",
            "points": [
                "The overlap panel shows that returning employers moved together less than aggregate averages suggest.",
                "The exact cluster labels are provisional, so the decomposition should lead the story.",
            ],
        },
        {
            "title": "Explicit entry is a conservative lower bound; low-YOE postings remain.",
            "figure": "assets/figures/T03/T03_junior_share_comparison.png",
            "points": [
                "The explicit label stays tiny while the YOE proxy remains much broader.",
                "The two instruments do not tell the same junior story.",
            ],
        },
        {
            "title": "Senior roles add AI/tool orchestration without a clean management decline.",
            "figure": "assets/figures/T21/T21_cross_seniority_management_comparison.png",
            "points": [
                "Strict management is stable or slightly up.",
                "The growth cluster is the AI/tool-orchestrator type.",
            ],
        },
        {
            "title": "AI-tool growth is SWE-specific, while length and scope are broader across roles.",
            "figure": "assets/figures/T18/T18_parallel_trends.png",
            "points": [
                "Controls do not show the same AI-tool jump that SWE does.",
                "Length and scope are broader posting-surface changes.",
            ],
        },
        {
            "title": "Ghost-like language looks aspirational and template-heavy, not fake jobs.",
            "figure": "assets/figures/T22/T22_ai_aspiration_ratio.png",
            "points": [
                "AI postings hedge more than non-AI postings.",
                "Direct employers and aggregators contribute different kinds of noise.",
            ],
        },
        {
            "title": "The divergence claim is benchmark-sensitive, not settled.",
            "figure": "assets/figures/T23/T23_ai_requirement_vs_benchmarks.png",
            "points": [
                "Posting-side AI language is real.",
                "Outpacing worker usage is not a single settled number.",
            ],
        },
        {
            "title": "No single dramatic story survives the checks.",
            "figure": "assets/figures/T14/T14_tech_shift_heatmap.png",
            "points": [
                "The simple junior-collapse, management-decline, and outpacing-worker stories do not survive the evidence.",
                "AI/LLM, requirement stacking, and template drift are the stable core.",
            ],
        },
    ]

    out = [
        "---",
        "marp: true",
        "theme: default",
        "paginate: true",
        "size: 16:9",
        "style: |",
        "  section { font-size: 24px; }",
        "  h1 { font-size: 32px; }",
        "  img { border-radius: 8px; }",
        "  strong { font-weight: 700; }",
        "---",
        "",
    ]
    for slide in slides:
        out.append(f"# {slide['title']}")
        out.append("")
        out.append(f"![w:1100]({slide['figure']})")
        out.append("")
        for point in slide["points"]:
            out.append(f"- {point}")
        out.append("")
        out.append("---")
        out.append("")
    return "\n".join(out[:-2]) + "\n"


def build_mkdocs_yml() -> str:
    return """
site_name: SWE labor market restructuring
site_description: Exploration findings package and audit trail
docs_dir: docs
site_dir: site
theme:
  name: material
  palette:
    scheme: default
    primary: teal
    accent: green
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - toc.follow
    - content.code.copy
nav:
  - Story: index.md
  - Findings:
      - Findings hub: findings/index.md
      - Domain recomposition and AI/LLM ecosystem: findings/domain-recomposition.md
      - Requirement stacking and scope: findings/credential-stacking-scope.md
      - Posting-template drift and AI-authorship hypothesis: findings/template-drift.md
      - Returning employer recomposition: findings/company-strategies.md
      - Cross-occupation densification: findings/cross-occupation.md
      - Seniority and orchestration: findings/seniority.md
      - Junior measurement asymmetry: findings/junior-measurement.md
      - Ghost and aspiration: findings/ghost-aspiration.md
      - Benchmark-sensitive divergence: findings/divergence.md
  - Methods:
      - Methods hub: methods/index.md
      - Preprocessing and schema: methods/preprocessing.md
      - Evidence and sensitivity: methods/evidence.md
  - Audit trail:
      - Audit hub: audit/index.md
      - Raw archive: audit/raw/index.md
markdown_extensions:
  - admonition
  - attr_list
  - def_list
  - footnotes
  - tables
  - toc:
      permalink: true
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
extra_css:
  - stylesheets/site.css
"""


def build_css() -> str:
    return """
iframe.presentation-frame {
  width: 100%;
  min-height: 72vh;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 8px;
  background: #fff;
}

img {
  border-radius: 8px;
}

table {
  font-size: 0.92rem;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-copy", action="store_true", help="Do not copy assets/raw sources")
    args = parser.parse_args()

    ensure_dir(PKG_ROOT)
    ensure_dir(DOCS_ROOT)
    ensure_dir(MARP_ROOT)

    if not args.skip_copy:
        copy_tree(FIG_SRC, DOCS_ROOT / "assets" / "figures")
        copy_tree(TBL_SRC, DOCS_ROOT / "assets" / "tables")
        copy_tree(ART_SHARED_SRC, DOCS_ROOT / "assets" / "artifacts" / "shared")
        copy_tree(ART_T25_SRC, DOCS_ROOT / "assets" / "artifacts" / "T25")

        ensure_dir(DOCS_ROOT / "audit" / "raw" / "reports")
        ensure_dir(DOCS_ROOT / "audit" / "raw" / "memos")
        for src in REPORTS_SRC.glob("*.md"):
            text = rewrite_raw_markdown(src.read_text(encoding="utf-8"), "report")
            write_text(DOCS_ROOT / "audit" / "raw" / "reports" / src.name, text)
        for src in MEMOS_SRC.glob("*.md"):
            text = rewrite_raw_markdown(src.read_text(encoding="utf-8"), "memo")
            write_text(DOCS_ROOT / "audit" / "raw" / "memos" / src.name, text)

    pages = build_pages()
    for rel_path, content in pages.items():
        write_text(DOCS_ROOT / rel_path, content)

    write_text(PKG_ROOT / "mkdocs.yml", build_mkdocs_yml())
    write_text(DOCS_ROOT / "stylesheets" / "site.css", build_css())
    write_text(MARP_ROOT / "presentation.md", build_marp_deck())


if __name__ == "__main__":
    main()
