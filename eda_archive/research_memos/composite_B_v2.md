# Composite B v2 — Role Landscape: BERTopic rerun + article-worthiness verdict

**Author:** Claude (Opus 4.7, follow-up dispatch)
**Date:** 2026-04-21
**New code:** `eda/scripts/S27_v2_bertopic.py` (BERTopic rerun) and `eda/scripts/S27_v2_spotchecks.py`
**New tables:** `eda/tables/S27_v2_*.csv` (8 files)
**New figures:** `eda/figures/S27_v2_*.png` (4 panels)
**New artifact:** `eda/artifacts/composite_B_archetype_labels.parquet` (37,003 rows)
**Predecessor memo:** `eda/research_memos/composite_B_role_landscape.md`

---

## tl;dr

The BERTopic rerun on a 37,003-row company-capped SWE sample (`description_core_llm` text, 2024 + 2026) produced a sharper taxonomy than the v9 T09 8k-row pilot: 95 topics, 36% noise, stability ARI = 0.64 across three seeds, agreement vs v9 NMI = 0.32 (the dominant AI cluster maps cleanly: 76% of v9's `models/systems/llm` lands in our Topic 1). NMF on the same texts converges on the same growth lanes (RAG/AI 12.4×, devops/IaC 2.5×, distributed-backend 3.0×) and the same shrinking lanes (.NET 0.33×, VMware/AD 0.46×, Salesforce 0.56×) — independent method confirmation.

The single sharpest finding from the rerun: **a content cluster that did not exist as a recognisable archetype in 2024 — RAG/AI agents/agentic LLM systems — accounts for 12.7% of 2026 SWE postings, up from 2.5% in 2024 (5.2× share rise on a vocabulary-honest definition).** That is the article's central numerical anchor.

Verdicts on the four threads:
- **Thread 1 Applied-AI: REVISE** — keep, but anchor on cluster-share-of-corpus (2.5% → 12.7%), not on title regex (which gives 5.2×) or v9 T34 cluster (15.6× was inflated by 2026-only n-grams).
- **Thread 2 FDE: KEEP** — clean 41-posting / 31-firm / 1.95× AI density story survives every robustness check, but headline is "a function being adopted", not "a quantified labour-market effect" — frame as colour, not headline number.
- **Thread 3 Emerging clusters: KEEP** — strongly upgraded by the rerun. The two-method agreement (BERTopic + NMF, both ranking the AI cluster as the top grower with very similar AI density) is the article's strongest methodological asset.
- **Thread 4 Legacy substitution: KEEP** — robust direction, robust to vocabulary, structurally complementary to Thread 3. The under-reported half of the role-landscape story.

The composite article holds together as a four-panel piece. Recommended structure unchanged from v1, but with revised headline numbers.

---

## Work stream 1 — BERTopic rerun

### Sample

37,003 SWE rows after filtering (`is_swe=True`, `is_english=True`, `date_flag='ok'`, `llm_extraction_coverage='labeled'`, `len(description_core_llm) ≥ 200`) and capping at 30 postings per `(company_name_canonical, period_bucket)`. Period split: 17,784 / 19,219 (2024 / 2026). Code: `eda/scripts/S27_v2_bertopic.py:65-105`.

### Methods

Embeddings: `all-MiniLM-L6-v2` batch 256, ~12 min. BERTopic primary: UMAP (n_neighbors=15, n_components=5, cosine) + HDBSCAN (min_cluster_size=35, min_samples=10) + CountVectorizer (ngram 1-2, min_df=10, max_df=0.4). NMF comparison: TF-IDF + NMF k=20. Stability: three BERTopic fits at seeds (42, 1337, 2026), pairwise ARI. v9 comparison: ARI + NMI on 6,277 overlapping uids.

### Key numbers

| Metric | Value |
|---|---|
| n sample | 37,003 |
| n topics (primary, seed 42) | 95 |
| noise share | 36.1% |
| stability ARI mean (3 seeds) | 0.643 |
| stability ARI range | 0.620 – 0.673 |
| ARI vs v9 T09 (n=6,277 overlap) | 0.076 |
| NMI vs v9 T09 | 0.320 |
| ARI BERTopic vs NMF | 0.118 |
| NMI BERTopic vs NMF | 0.396 |

The low BERTopic-vs-v9 ARI is mostly granularity: our 95-topic taxonomy splits v9's 30 macro-archetypes. v9 `models/systems/llm` maps 76% (220/291) to our Topic 1; v9 `kubernetes/terraform/cicd` maps 50%+ to our Topic 2. NMI of 0.32 is the more honest scalar — moderate structural agreement.

### Top growers (n_2026 / n_2024) — excluding noise

| topic_id | name | n_2024 | n_2026 | growth | AI rate |
|---|---|---|---|---|---|
| 1 | rag / ai_solutions / ai_systems | 436 | 2,449 | **5.6×** | **79%** |
| 9 | sre / site_reliability / reliability_engineer | 172 | 300 | 1.7× | 13% |
| 18 | robotic / robot / perception | 102 | 165 | 1.6× | 24% |
| 21 | swiftui / ios_development / xcode | 83 | 145 | 1.7× | 14% |
| 2 | devops_engineer / cloudformation / iac | 663 | 1,040 | 1.6× | 8% |
| 28 | perception / deep_learning / autonomous_driving | 38 | 149 | 3.8× | 24% |
| 50 | patient / deep_learning / imaging | 17 | 70 | 3.9× | 77% |
| 80 | founding / founders / founding_engineer | 10 | 33 | 3.1× | 37% |
| 77 | threat / adversarial / attack (security ML) | 3 | 41 | 10.5× | 80% |

### Top shrinkers (smallest growth ratio, n_2024 ≥ 30)

| topic_id | name | n_2024 | n_2026 | growth | AI rate |
|---|---|---|---|---|---|
| 64 | laravel / php | 53 | 6 | 0.13× | 2% |
| 35 | requirements_management / V&V / subsystem | 133 | 23 | 0.18× | 0% |
| 17 | plc / automation_engineer | 233 | 46 | 0.20× | 1% |
| 37 | mainframe / cobol / jcl | 109 | 25 | 0.24× | 1% |
| 49 | engineering_management / technical_integrity | 71 | 17 | 0.25× | 0% |
| 71 | mbse / sysml / model_based_systems_engineering | 40 | 9 | 0.24× | 0% |
| 5 | vmware / active_directory | 538 | 153 | 0.29× | 1% |
| 3 | net_core / net_developer / entity_framework | 871 | 297 | 0.34× | 4% |

The shrinking list spans three families: (a) legacy stack (.NET, PHP, COBOL, VMware), (b) traditional systems-engineering (V&V, MBSE, requirements management), and (c) people-management-flavoured engineering (Topic 49). Topic 49's collapse from 71→17 is small but suggestive — the `senior archetype shift` paper claim has independent support from this clustering.

### NMF triangulation

NMF k=20 on the same texts ranks the AI cluster as the largest single grower:

| NMF topic | top terms | n_2024 → n_2026 | growth | AI rate |
|---|---|---|---|---|
| 1 | ai, agentic, agents, rag, agent, llm, workflows, generative_ai | 143 → 1,785 | **12.4×** | **87%** |
| 0 | ll, product, building, backend, platform, scale, distributed | 539 → 1,611 | 3.0× | 31% |
| 3 | infrastructure, devops, automation, ci/cd, kubernetes, pipelines | 742 → 1,871 | 2.5× | 17% |
| 4 | learning, machine_learning, ml, models, deep_learning, pytorch | 502 → 978 | 1.9× | 47% |
| 8 | pipelines, analytics, data_engineering, snowflake, etl | 870 → 1,442 | 1.7× | 13% |
| 14 | react, frontend, javascript, node, typescript | 1,703 → 2,178 | 1.3× | 19% |
| 9 | net, sql, asp, asp_net, sql_server | 950 → 311 | **0.33×** | 5% |
| 18 | network, server, vmware, windows, microsoft | 1,296 → 602 | 0.46× | 3% |
| 12 | salesforce, apex, lightning | 246 → 138 | 0.56× | 9% |

The two methods place AI/RAG as the top-growing cluster, with the BERTopic version (5.6×) more conservative than the NMF version (12.4×) because BERTopic's HDBSCAN is more aggressive about pushing borderline AI-flavoured backend postings into adjacent clusters or into noise. The 5.6× number is the more defensible headline; 12.4× is the upper bound. **Cluster-share-of-corpus (2.5% → 12.7% in 2026, a 5.2× rise within balanced periods) is the cleanest single number** — vocabulary-honest, ratio-based, computed within the cap-balanced sample.

### Stability

Pairwise ARI between three BERTopic fits with different seeds: 0.635, 0.673, 0.620 — mean 0.643. The dominant clusters (Topic 0/1/2/3, the largest in size) are essentially identical across seeds; volatility lives in the small clusters (n < 100), which is expected for HDBSCAN at this min-cluster-size setting.

### Plot diagnostics

UMAP-by-archetype, by-period and by-seniority figures, plus the growers/shrinkers bar chart, are saved under `eda/figures/S27_v2_*.png`. The by-period plot shows the AI region is overwhelmingly 2026; the by-seniority plot shows clusters do *not* align with seniority — domain trumps seniority as the dominant axis of posting variation, consistent with v9 T09.

### What the rerun changes vs the v9 8k pilot

Sample 4.6× larger, AI cluster sharper (n=2,885 vs v9's 389), founding-engineer archetype now visible (Topic 80, n=43, 37% AI), engineering-management archetype now visible as shrinker (Topic 49, 71→17 — independent textual support for senior-archetype-shift), stack-modernisation second growth lane confirmed.

---

## Work stream 2 — article-worthiness evaluation

For each thread: numerical strength, narrative strength, overlap with `findings_consolidated_2026-04-21.ipynb`, and a 10-posting spot-check (`eda/tables/S27_v2_spotchecks.csv`).

### Thread 1 — Applied-AI

**Headline.** Two defensible numbers, both 5.2×: title regex (366 → 1,896 postings) and BERTopic AI-cluster share (2.5% → 12.7% of corpus). Use cluster-share — it measures "postings that read like AI engineering work, regardless of title", which is the substantive thing. **The v9 T34 15.6× number cannot survive scrutiny** — that cluster was defined by 2026-only n-grams, so its 2024 base of 144 is a structural artefact. BERTopic Topic 1 has a real 2024 base of 436 and produces 5.6× on the same balanced corpus.

**Narrative.** Strong. Industry rhetoric frames AI engineering as "anyone who can prompt" — contradicted by the experience profile. Senior Applied-AI mean YOE 5.90 vs senior baseline 6.36 (slightly *lower* but not democratised); director density 2.12% vs 1.70% (1.25× lift). Frame: "the new specialist still requires deep experience; what's new is the title vocabulary".

**Overlap with consolidated notebook.** The "Other 2026 observations" section ("AI-focused senior roles ask for more experience", median YOE 6) is the v9 T34 framing — *partially contradicted* by the re-derivation here. Composite B should restate carefully.

**Spot-check.** All 10 senior-Applied-AI title-regex postings face-valid: Staff ML Engineer (Oscar), Senior ML Engineer Credit Karma (Intuit), Senior Generative AI Engineer (Hudson), Staff ML Engineer On-Device (Qualcomm), AI Engineer (People In AI), Principal ML Engineer (hackajob), ML Engineer Senior (EY), Senior AI Engineer Agent Workflows (Govini), etc.

**Verdict: REVISE.** Keep as Panel 1, anchor on cluster share 2.5% → 12.7%, footnote title-regex 5.2×. Drop 15.6× T34 claim. Keep senior-Applied-AI experience profile as colour.

### Thread 2 — Forward-Deployed Engineer

**Headline.** 41 postings / 31 firms in 2026 vs 3 postings in 2024 (aggregator-stripped, title-only) — 14× share rise from near-zero base. AI density 1.95× general SWE (53.8% vs 27.6%).

**Narrative.** Strong but small-base. Function-naming, not labour-market scale: Palantir-coined title now adopted by OpenAI, Adobe, AMD, PhysicsX, TRM Labs, Saronic, Foxglove, PwC, Ramp. Firm-list and density premium are the evidence; count is too small to lead.

**Overlap with consolidated notebook.** Not present — FDE is genuinely novel.

**Spot-check.** Real postings at Palantir (incl. Poland internship), OpenAI, Adobe (Senior Manager FDE AI Engineer), AMD (Lead FDE), PhysicsX (Lead FDE), Invisible Technologies, TRM Labs, eranova, PwC (Foundry Senior Manager / Director). Title means what we think.

**Verdict: KEEP.** Panel 2. Frame as function-emergence; lead with firm list, not count.

### Thread 3 — Emerging skill clusters (now BERTopic-backed)

**Headline.** RAG/AI cluster grew 2.5% → 12.7% of corpus (5.2× share, 79% AI density inside cluster). NMF k=20 on same texts independently confirms (Topic 1: 87% AI, 12.4× volume rise — NMF aggregates more borderline backend roles into AI cluster, hence higher multiplier). Second growth lane: stack-modernisation (DevOps/IaC/SRE) 1.3-1.7× across three BERTopic clusters, with steady AI-density drift.

**Narrative.** The article's analytical backbone. Two-method agreement (BERTopic + NMF independently rank the same growth cluster top, with similar AI density) separates a discoverable structural shift from an analyst's clustering choice.

**Overlap with consolidated notebook.** Headline 4 (vendor leaderboard: Copilot, Claude, Cursor) is *vocabulary*-level; this thread is *role*-level. They complement each other.

**Spot-check.** Topic 1 exemplars: AI Engineer (Trinity Tech), Data/ML Engineer (TrueSkilla), Senior ML Engineer AI Automation (Unity), Senior ML Engineer Vector AI (Unity). Topic 80 (founding engineer): Founding Engineer (Pretorin, Autopilot, HeyMilo AI, Healink). Topic 49 (engineering management) cleanly captures the senior-management shrinkage.

**Verdict: KEEP, upgraded.** Panel 3 lead with cluster share 2.5% → 12.7% + two-method confirmation. Sub-stories: stack modernisation (1.3-1.7×) and legacy clusters (.NET 0.33×, VMware 0.29×, mainframe 0.24×) — segue into Thread 4.

### Thread 4 — Legacy substitution

**Headline.** Disappearing-2024 titles (Java architect, Drupal developer, devops architect, scala developer) re-route in 2026 to nearest-neighbour titles (Java developer, web developer, DevOps engineer) whose AI mention rate is **11.9% vs the 27.6% market average** — substitution flows toward stack modernisation, not AI-ification, by a 2.3× margin. v9 T36 strict-vocabulary version (3.6% / 14.4%) gives a 3.9× margin; direction is robust to vocabulary, only magnitude shifts.

**Narrative.** The under-told half of the role-landscape story. Mainframe developer 66 → 2 in two years; C#/.NET developer 120 → 0; Salesforce developer 45 → 5. These are real labour-demand displacements that aren't registered as "AI replacing jobs" because the substitute roles are not AI-flavoured.

**Overlap with consolidated notebook.** "Other 2026 observations" cites this at 3.6% / 14.4%; Composite B refines under wider pattern (11.9% / 27.6%) and supplies the title-frequency-delta backup.

**Spot-check.** Legacy-neighbour postings: Full Stack Java Developer (MPower Plus), DevOps Engineer (Jobs via Dice, CodeplixAI, Shield Consulting, Haystack ×2, American Board of Anesthesiology), Web Developer (Grover Gaming, Jobs via Dice), Java Developer (Capco). Real engineering roles, mostly without AI vocabulary.

**Verdict: KEEP.** Panel 4. Lead with magnitude texture (mainframe 66→2, .NET 120→0, salesforce 45→5) and the 12% / 28% AI-rate gap.

---

## Recommended composite article structure

**Working title:** *How software-engineering roles are being created, destroyed, and rewritten.*

**Lede.** Anchor on the rewrite-not-replacement frame using the consolidated finding (SWE AI-vocab 3% → 28% vs control 0.3% → 1.4%, a 23-to-1 ratio; same companies, same titles). Then set up the role-landscape question: under that vocabulary-rewrite, where are *roles* — distinct content clusters — being created, where are they shrinking, where are they being re-routed?

**Panel 1 — The new specialist (Applied-AI).** BERTopic AI cluster (RAG / agents / GenAI / MLOps) is **12.7% of 2026 SWE postings vs 2.5% in 2024** — a 5.2× share rise. Title-regex anchor: AI/ML/Applied-AI Engineer titles rose 366 → 1,896 (5.2× volume). Senior-Applied-AI experience profile contradicts democratisation rhetoric: mean YOE 5.90 (senior baseline 6.36) and director density 1.25× elevated.

**Panel 2 — The new function (Forward-Deployed).** 41 postings at 31 firms in 2026 vs 3 in 2024. Palantir-coined title now adopted by OpenAI, Adobe, AMD, PhysicsX, TRM Labs, Saronic, Foxglove, PwC, Ramp. AI density 1.95× general SWE. Frame as function-emergence; firm-list is the news.

**Panel 3 — The emerging skill stack.** BERTopic AI cluster 5.6× growth, 79% AI density; stack-modernisation second growth lane (1.3-1.7×). NMF on same texts independently confirms (Topic 1: 12.4× growth, 87% AI). Two-method agreement is the article's methodological asset. Shrinking lanes (.NET 0.33×, VMware 0.29×, mainframe 0.24×, salesforce 0.56×) segue into Panel 4.

**Panel 4 — The quiet substitution (legacy).** Mainframe developer 66 → 2, C#/.NET developer 120 → 0, Salesforce developer 45 → 5. Neighbour titles (Java developer, DevOps engineer, web developer) carry AI vocabulary at 11.9% vs 27.6% market — substitution is stack-modernisation, not AI-ification, by a 2.3× margin. Engineering-management cluster shrinking 71→17 is the labour-side echo.

**Closer.** Three axes: a small visible peak where credentials still bind (Applied-AI, FDE), a dominant content rewrite (RAG/AI cluster + stack modernisation), a quieter floor where legacy roles are absorbed into modernised neighbours mostly without AI rebranding. Same companies write all three changes (consolidated notebook: 292 firms, +19.4 pp within-firm AI rewrite). Industry rhetoric describes the peak; the body of the change is in the rewrite and the floor.

---

## Per-thread verdict summary

| Thread | Verdict | Headline number | Risk |
|---|---|---|---|
| 1. Applied-AI | REVISE | Cluster share 2.5% → 12.7% (5.2× rise) | Drop the 15.6× T34 number entirely; foreground vocabulary-honest cluster definition |
| 2. FDE | KEEP | 41 postings / 31 firms in 2026 vs 3 in 2024; 1.95× AI density | Small base; lead with firm names, not count |
| 3. Emerging clusters | KEEP (upgraded) | BERTopic AI cluster 5.6× growth, 79% AI rate, NMF confirmation 12.4× | Methodologically strongest panel; let two-method agreement do the work |
| 4. Legacy substitution | KEEP | Neighbour AI rate 11.9% vs market 27.6% (2.3× gap) | Vocabulary-pattern sensitivity; cite both wider and strict numbers |

---

## Caveats

Sample restricted to `llm_extraction_coverage='labeled'` (~99% of in-frame SWE; minor selection). Company-capping at 30/period suppresses aggregator dominance but means firm counts in characterisation tables are post-cap. HDBSCAN noise 36% is normal at min_cluster_size=35; dominant clusters are stable across seeds (ARI 0.64) — micro-topics churn. ARI vs v9 of 0.076 is mostly granularity mismatch (95 vs 30 topics); v9 `models/systems/llm` maps 76% into our Topic 1, confirming the structural backbone is shared. No causal claims in any panel — labour-demand-signalling only.

---

## Files

Code: `eda/scripts/S27_v2_bertopic.py`, `eda/scripts/S27_v2_spotchecks.py`. Tables: `eda/tables/S27_v2_*.csv` (8 files: bertopic_topics, bertopic_exemplars, nmf_topics, stability_ari, v9_comparison, v9_crosstab, method_alignment, spotchecks). Figures: `eda/figures/S27_v2_*.png` (4 panels: UMAP-by-archetype, by-period, by-seniority, growers/shrinkers). Artifact: `eda/artifacts/composite_B_archetype_labels.parquet` (37,003 rows; uid, archetype_id, archetype_name, prob) plus `composite_B_bertopic_summary.json`.
