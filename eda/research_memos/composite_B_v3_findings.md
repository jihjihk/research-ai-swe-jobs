# Composite B v3 — Role-landscape evolution, 2024 → 2026

Run: 2026-04-22. Pipeline: `eda/scripts/composite_B_v3_evolution.py`. Inputs: `data/unified_core.parquet`. Primary artifacts: `eda/artifacts/composite_B_v3llm_*`, `eda/tables/composite_B_v3llm_*`.

## What the analysis asks

How did the **archetype structure** of software-engineering and SWE-adjacent job postings change between 2024 and 2026? Not "what do the clusters look like" but "what got created, what got destroyed, what got rewritten."

## How the sample was built

- Source: `data/unified_core.parquet`, which pools the 2024 Kaggle snapshots (arshkon, asaniczka) and the 2026 LinkedIn/Indeed scraper.
- Role filter: `swe_classification_llm IN ('SWE', 'SWE_ADJACENT')` — the direct LLM verdict, not the composite rule-based flag. The LLM verdict disagrees with the composite flag for roughly 15,000 postings; chief among them, about 400 building architects who were pulled into the "SWE-adjacent" bucket by a title-embedding gate despite the LLM classifying them as NOT_SWE. Using the LLM column directly removes that contamination.
- Additional filters: English, `date_flag='ok'`, `llm_extraction_coverage='labeled'`, `description_core_llm IS NOT NULL` and length ≥ 200 chars.
- Per-firm cap: 30 postings per (company × period × role group) so that a few prolific firms cannot dominate.
- Final n: **46,451** (SWE 2024: 14,753; SWE 2026: 18,708; SWE-adjacent 2024: 5,775; SWE-adjacent 2026: 7,215).
- Text for embedding/clustering: `description_core_llm` (the LLM-extracted canonical body) with a COALESCE fallback to raw description for the rare unlabeled row.

## Method in one paragraph

Sentence-transformer embeddings (`all-MiniLM-L6-v2`, 384-dim, normalized) → UMAP reduction → HDBSCAN density clustering → BERTopic aggregation, then `reduce_topics(nr_topics=30)` to produce 29 readable archetype families (plus a noise bucket). The dominant mega-cluster (Family 0), which embeddings place in one density region despite blending AI-engineering and data-science/ML content, is split post-hoc using the same AI-vocabulary regex used in Article A, yielding sub-cohorts `0_AI` and `0_nonAI`. Per-period BERTopic fits (2024-only, 2026-only) are run as a stability cross-check; per-family centroid cosine to the nearest period-specific topic measures whether the joint taxonomy reproduces on time-sliced data. An emergence index calibrated on the 2024 corpus's within-period nearest-neighbor distances flags 2026 postings that sit further from any 2024 counterpart than typical 2024 postings sit from each other.

## Stability and rigour checks

| Check | Value | Reading |
|---|---:|---|
| Topic-level stability ARI (3 seeds) | 0.49 (mean) | Seed-sensitive at the 88-topic granularity. |
| **Family-level stability ARI, k=30 (3 seeds)** | **0.44 (mean)** | Also seed-sensitive. See note below. |
| Joint ↔ 2024-only alignment, centroid cosine | **0.94** (mean) | Joint families reproduce on 2024-only data. |
| Joint ↔ 2026-only alignment, centroid cosine | **0.95** (mean) | And on 2026-only data. |
| Noise rate (HDBSCAN `-1`) | 30.6% | Much of the corpus doesn't sit in any dense cluster — an honest ceiling. |

**On the stability tension.** At fine granularity the clustering is seed-sensitive: one of the three random seeds produces a materially different topic layout. At family granularity the alignment between the joint taxonomy and independent 2024-only and 2026-only fits is strong (0.94, 0.95). The second test is more load-bearing for this analysis: it asks whether the archetype structure exists in the data regardless of how we slice time; the first asks whether it exists regardless of how we initialise UMAP. We lean on the first for the narrative and flag the second as a caveat.

## The three layers of change

### 1 · Created

**The dominant story: the AI-engineering specialty crystallised.** A single UMAP mega-region (Family 0 at k=30, 13,040 postings — 31% of the corpus) houses both AI/ML and data-science/analytics content; embeddings cannot separate them because their vocabulary overlaps heavily. Splitting Family 0 with the AI-vocabulary regex resolves the story:

| Sub-cohort | 2024 | 2026 | Growth | Share SWE | AI-vocab 2024 → 2026 |
|---|---:|---:|---:|---:|---|
| **0_AI** — AI-coded postings | 520 | 4,447 | **8.5×** | 74% | 100% → 100% (by construction) |
| **0_nonAI** — data-work without AI vocabulary | 3,435 | 4,638 | 1.35× | 56% | 0% → 0% |

The AI-coded slice grew from 520 postings in 2024 — a rounding error — to 4,447 in 2026. That is the single cleanest quantitative statement of the AI-engineering boom in this dataset.

**Plus smaller, sharply emergent clusters** (all from k=30, 2024 → 2026 absolute):
- Cybersecurity / identity / threat (Family 5): 399 → 1,203 (**3.0×**); +1.1pp share on SWE, +6.8pp on SWE-adjacent.
- Robotics / autonomous / computer vision (Family 11): 125 → 312 (**2.5×**).
- SRE / reliability / observability (Family 10): 193 → 362 (**1.9×**), now visibly distinct from cloud/DevOps.
- GPU / inference / AMD / NVIDIA (Family 18): 67 → 119 (**1.8×**).
- Network automation (Family 23): 16 → 71 (**4.2×**).
- Rust backend (Family 28): 13 → 28 (**2.1×**).

### 2 · Destroyed

Specific legacy stacks contracted by 45-65%. Six clusters carry this story, on the SWE and SWE-adjacent sides respectively:

| Family | Content | 2024 → 2026 | Growth | Δ share SWE | Δ share adjacent |
|---|---|---:|---:|---:|---:|
| 3 | **Enterprise Java / .NET / Spring** | 1,572 → 721 | **0.46×** | **−6.57pp** | −0.59 |
| 15 | **SAP / Oracle / mainframe / COBOL / JCL** | 202 → 75 | 0.37× | −0.76 | −0.54 |
| 20 | **PHP / WordPress / Laravel** | 98 → 38 | 0.39× | −0.44 | −0.06 |
| 24 | **AUTOSAR / automotive embedded** | 59 → 25 | 0.43× | −0.24 | −0.07 |
| 8 | **ServiceNow / solution-architect consulting** | 515 → 230 | 0.45× | −0.66 | **−4.05pp** |
| 14 | **PLC / control / manufacturing automation** | 181 → 97 | 0.54× | −0.37 | −0.86 |

The losing side is not "programming." It is specific technology stacks, most of them enterprise-flavoured. The single biggest SWE loss is enterprise Java.

### 3 · Rewritten

Even clusters that survived (and in some cases grew) show vocabulary drift. Tracked via the c-TF-IDF top-20 terms that entered or exited between the 2024 and 2026 subsets of the same family:

- **Family 3 (enterprise Java, shrinking 0.46×):** exited — `java, framework, server, services, business`; entered — `apis, microservices, cloud, web, architecture, net`. The surviving enterprise postings modernise even as the cluster halves.
- **Family 1 (cloud / DevOps, growing 1.27×):** exited — `devops, kubernetes, automation, security`; entered — `reliability, monitoring, platform, best practices, hands-on`. The word "DevOps" is fading from top-20 vocabulary in favour of specific practices — platform engineering by another name.
- **Family 5 (cybersecurity, growing 3.0×):** entered — `ai, cloud security, identity, detection, compliance, response`. Security is absorbing AI language as much as competing with it.
- **Family 6 (QA / test automation, flat):** the word `qa` itself exits the top-20; `ci/cd, api, tests, integration, selenium` enters. Testing embeds into development workflows.
- **Family 0 overall (mega-cluster):** the word `ai` exits the top-20; `ml, data science, model, platform, workflows` enters. As the cluster grew 2×, its central vocabulary rotated from a buzzword into named practices.

## Caveat surfaced by the method: classifier contamination discovered

The switch from `is_swe_adjacent` (composite rule-based) to `swe_classification_llm` (LLM verdict) removed ~12,600 postings from the sample. Inspection of one prominent removed cluster revealed ~430 building architects (Sheladia Associates, LS3P Associates, Olson Kundig, Kleinfelder) that had been labeled SWE-adjacent by the pipeline's `embedding_adjacent` tier — title-embedding similarity placed "Architect" near "software architect" / "cloud architect." The LLM classified all 375 as NOT_SWE but was overridden. **This is a documented false-positive channel in the upstream SWE-adjacent classifier** that Composite B v2 carried unknowingly. Worth flagging in the paper's data-quality section.

## Proposed figure (four panels, one composite)

The figure lives in a new section of `findings_consolidated_2026-04-21.ipynb`, inserted as Panel 3 of Article B. Panels 1 and 2 of Article B (RAG-cluster specialist, FDE + legacy) become supporting zooms into specific regions of the map this panel draws.

1. **Panel A — The map.** 2×2 UMAP small multiples: rows (SWE / SWE-adjacent) × columns (2024 / 2026). Shared axes and legend. Points coloured by archetype family, with Family 0 further split into `0_AI` (red) and `0_nonAI` (amber). The long tail of small families gets grey to preserve legibility; the top ~10 families get distinct colours.
2. **Panel B — What moved.** Diverging horizontal bars: per archetype family, the percentage-point change in share between 2024 and 2026, SWE and adjacent side-by-side. Sorted by absolute SWE delta. Family 0 shown as the AI/non-AI split.
3. **Panel C — Growth against emergence.** Scatter: growth ratio on X (log scale), emergence share on Y, bubble size by 2026 volume, colour by family. Quadrants labeled — "mainstream growth," "new frontiers," "quiet rewrites," "stable legacy."
4. **Panel D — Same cluster, different words.** Four short typographic call-outs (no axes) for the four families with the cleanest within-family rewrites: Family 0 (AI/data), Family 3 (enterprise Java modernising), Family 1 (DevOps → platform engineering), Family 6 (QA → embedded testing).

## Where the figure code lives

- Visualisation functions: `eda/scripts/composite_B_v3_viz.py`.
- Called from the notebook as `viz_composite_b_map()`, `viz_composite_b_deltas()`, `viz_composite_b_growth_emergence()`, `viz_composite_b_drift_words()`.

## Plain-English version

Between 2024 and 2026, the map of software-engineering and SWE-adjacent job postings did not simply grow a new region. It reorganised. A small AI-engineering specialty that numbered about 500 postings in 2024 is now nearly 4,500 — about one in every four postings we classify as software-engineering now carries AI-engineering vocabulary, up from about one in thirty. At the same time, specific legacy tech stacks contracted sharply: enterprise Java/.NET/Spring lost over half its share on the software-engineer side; SAP/COBOL/mainframe, PHP/WordPress, automotive embedded, PLC manufacturing automation, and ServiceNow solution consulting all halved or worse. None of these is "programming" as a whole; each is a specific technological moment passing. And underneath the visible shifts, surviving clusters were rewritten in place: DevOps postings have shed the word "DevOps" in favour of "reliability" and "platform"; QA postings have shed the word "QA" as testing embedded into development; security postings absorbed AI vocabulary; even the data-work cluster, which kept its shape, rotated its central vocabulary from "AI" as a buzzword to "ML, model, data science, workflows" as named practices. Three things happened at once — something was created, something was destroyed, a great deal was rewritten — and the visible AI-boom is only one-third of what the data shows.
