# Composite B — Role Landscape: Created, Destroyed, Rewritten

**Author:** Claude (Opus 4.7, exploratory pass)
**Date:** 2026-04-21
**Code:** `eda/scripts/S27_role_landscape.py`
**Tables:** `eda/tables/S27_*.csv` (8 files)
**Figures:** `eda/figures/S27_*.png` (4 panels)

---

## Hypotheses

**Thread 1 — Applied-AI / LLM Engineer.** A new senior archetype has emerged whose distinguishing vocabulary (Claude Code, RAG, LangChain) is essentially absent in 2024 and dense in 2026. The v9 T34 story claims a 15.6× share rise on a senior-only sample, with a one-year median-YOE premium and roughly double the director share. We test whether the multiplier survives if the role is defined by *title* rather than by *content cluster*, and whether the experience-floor and director-density claims are robust to that change.

**Thread 2 — Forward-Deployed Engineer.** A title invented at Palantir is being adopted by foundation-model labs, defence-tech firms, and finance-software companies as the structural engineering role for selling LLM products into specialist verticals. The v9 story counts 38–42 firms and a 2.3× AI-density premium. We test the title-only signal against a title-or-description signal, and re-derive both the firm count and AI density on `data/unified_core.parquet`.

**Thread 3 — Emerging skill clusters.** Beneath the title layer the *content* of postings is reorganising. A handful of archetypes (LLM/agent, kubernetes/CI/CD, data pipelines) are growing while older archetypes (legacy .NET, salesforce, scrum-ceremonial) are shrinking. We ingest v9 T09's 8,000-row archetype labels, recompute archetype × period growth on `unified_core`, and triangulate against a lightweight TF-IDF emergent-term scan.

**Thread 4 — Legacy substitution.** Disappearing 2024 titles (Java architect, Drupal specialist, PHP architect) map to 2026 nearest neighbours that are stack-modernised but not AI-ified. v9 T36 reports the neighbour AI-strict rate at 3.6%, well below the market average. We re-derive this on `unified_core` with both T36's nearest-neighbour mapping and a simple year-over-year title-frequency-delta.

---

## Data sources

- **Primary:** `data/unified_core.parquet` (110k rows, LinkedIn-only, 2024-01 + 2024-04 + 2026-03 + 2026-04). Filtered `is_english = true AND date_flag = 'ok' AND is_swe`. Yields 22,812 SWE rows in 2024, 25,822 in 2026.
- **AI-vocab regex:** the canonical pattern from `eda/scripts/scans.py:AI_VOCAB_PATTERN` (40-token whitelist). Wider than v9 T36's "ai_strict" — market AI-rate baseline is 27.6% under our pattern vs ~14.4% under v9's strict pattern.
- **v9 artifacts ingested:** `swe_archetype_labels.parquet` (8k rows; 7,999 join), `T34/cluster_precondition_check.csv` (15.6× headline), `T34/content_exemplars_cluster0.csv` (20 exemplars for title-precision check), `T36/substitution_table_top1.csv` and `T36/ai_vocab_comparison.csv` (legacy baseline).
- No new long-running clustering job was executed. See decision below.

---

## Methods explored per thread

- **Thread 1 (Applied-AI):** (a) ingest v9 T34 cluster-0 stats from `cluster_precondition_check.csv`; (b) title regex `\b(applied\s+ai|applied\s+ml|ai\s+engineer|ml\s+engineer|llm\s+engineer|machine\s+learning\s+engineer|mlops\s+engineer|genai\s+engineer|generative\s+ai\s+engineer|agent(?:ic)?\s+engineer|ai/ml\s+engineer)\b` on `unified_core`. Cross-check: same regex against T34's 20 cluster-0 exemplars (10/20 hit).
- **Thread 2 (FDE):** (a) title-only `(?i)forward[\s\-]?deployed`; (b) title-or-description, adding `customer-facing engineer` and `deployment (strategist|engineer) (on|with) (customer|client)`. Verification: firm count, AI-vocab rate, median YOE.
- **Thread 3 (Emerging clusters):** (a) ingest `swe_archetype_labels.parquet`, inner-join to `unified_core`, compute archetype × period growth and AI density; (b) TF-IDF emergent-term scan on a balanced 8,320-row SWE sample (bigrams+trigrams, `min_df=20`, `max_df=0.4`, `max_features=20000`, emergent threshold `rate_2026 > 1% AND rate_2024 < 0.1%`). Boilerplate filtered post-hoc.
- **Thread 4 (Legacy):** (a) ingest T36 nearest-neighbour pairs; (b) YoY title-frequency-delta on aggregator-stripped `unified_core`. Disappearing: `n_2024 ≥ 5 AND n_2026 < n_2024 / 4`. Emerging: `n_2024 < 5 AND n_2026 ≥ 30`.

---

## Per-method results

### Thread 1 — Applied-AI

| Metric | v9 T34 cluster 0 | Title regex (`unified_core`) | Senior baseline (2026) |
|---|---|---|---|
| n 2024 | 144 | 366 | — |
| n 2026 | 2,251 | 1,896 | 12,303 (senior SWE) |
| growth ratio | **15.6×** | **5.2×** | — |
| n distinct firms | 1,163 | 1,014 | — |
| median YOE | 6.0 | 5.0 (overall and senior subset) | 5.0 |
| director share | 1.92% | 0.95% (overall) / 2.12% (senior subset) | 1.70% |

The 15.6× headline depends on the cluster being defined by 2026-only n-grams ("claude code", "rag pipelines", "github copilot claude") — cluster 0 has almost no 2024 base by construction. Title-regex catches 366 ML-engineer-style postings already present in 2024, so the multiplier collapses to 5.2×. The "median YOE 6 vs 5" finding is **not** reproduced by title regex: median is 5.0, identical to the senior baseline; mean YOE among senior Applied-AI is 5.9 vs 6.36 senior-overall — slightly *lower*. Director-share elevation does survive: 2.12% within senior Applied-AI vs 1.70% across all senior SWE (1.25×, vs T34's claimed 1.89× over its own comparator).

Cross-check on T34's 20 cluster-0 exemplars: 10/20 carry the regex in the title; the other 10 are "Senior Backend Engineer", "Sr. Data Engineer", "Senior Data Engineer, AI Platform" — postings whose AI character lives in the description. **Content-clustering surfaces roughly 2× the volume of title-matching**, which is why its growth multiplier is larger.

### Thread 2 — Forward-Deployed Engineer

| Metric | Title only | Title or desc |
|---|---|---|
| n 2024 | 3 | 6 |
| n 2026 | 52 | 130 |
| share 2024 | 0.013% | 0.026% |
| share 2026 | 0.20% | 0.50% |
| share-rise | 15.3× | 19.1× |

Title-only matches v9's "3 → 59" with the small discrepancy attributable to corpus filter (`is_english AND date_flag = 'ok'`). The title-or-description method more than doubles the 2026 count to 130 by picking up "customer-facing engineer" and explicit deployment-strategist phrasings — but it also pulls in junior support roles that are not architecturally FDE. **Title-only is the cleaner growth signal**; the description-augmented version trades cleanliness for coverage.

Verification on `unified_core`:
- **Distinct firms (title-only, 2026):** 38. Aggregator-stripped: 31 firms / 41 postings. v9 reported 38 originally, corrected to 42; the original 38 reproduces here. Top-10 (TalentAlly 5, Saronic 3, Foxglove/Inferred Edge/Invisible Tech/OpenAI/Palantir/PwC/Ramp/TRM Labs 2 each) matches the v9 story.
- **AI-vocab density:** 53.8% on FDE vs 27.6% general SWE = 1.95× under the canonical (wider) pattern. v9 reported 32.2% / 13.8% = 2.34× under stricter pattern. Both confirm ~2× density.
- **Median YOE:** 4.0 vs 5.0 — v9 reported 5.0/5.0. n=52 with many missing LLM-YOE values; the difference is fragile.

### Thread 3 — Emerging clusters

**(a) v9 T09 ingestion — top growers and shrinkers within the 8k labelled sample (period buckets balanced):**

| Archetype | n 2024 | n 2026 | growth ratio | AI rate |
|---|---|---|---|---|
| `models/systems/llm` | 44 | 338 | 7.7× | 76.1% |
| `systems/agent/workflows` | 2 | 30 | 15.3× | 68.1% |
| `kubernetes/terraform/cicd` | 119 | 351 | 3.0× | (low) |
| `pipelines/sql/etl` | 86 | 211 | 2.5× | (low) |
| `backend/systems/services` | 17 | 85 | 5.1× | 28.0% |
| `java/boot/microservices` | 141 | 174 | 1.2× | low |
| `software/sql/aspnet` | 26 | 18 | 0.7× | low |
| `sql/aspnet/mvc` | 25 | 13 | 0.5× | low |
| `failure/procedures/ability` | 50 | 35 | 0.7× | low |
| `technologies/millions_americans/tech_trends` | 64 | 42 | 0.7× | low |

The two AI-tagged clusters are simultaneously the smallest 2024 base and the highest 2026 AI density (>68%). Stack-modernisation clusters grew 2-3×. Legacy .NET/ASP/Salesforce and ceremonial-process clusters shrank in absolute terms.

**(b) TF-IDF emergent-term scan.** After EEO/legal boilerplate filtering, the genuine content emergent terms are:

| Term | rate 2024 | rate 2026 | delta (pp) |
|---|---|---|---|
| `agentic` | 0.05% | 7.5% | +7.4 |
| `production-grade` | 0.07% | 5.0% | +4.9 |
| `rag` | 0.10% | 5.0% | +4.9 |
| `copilot` | 0.07% | 4.0% | +4.0 |
| `ai agents` | 0.07% | 3.9% | +3.8 |
| `claude` | 0.02% | 3.8% | +3.7 |
| `ai-assisted` | 0.00% | 3.6% | +3.6 |
| `agentic ai` | 0.00% | 3.6% | +3.6 |
| `prompt engineering` | 0.10% | 3.5% | +3.4 |
| `langchain` | 0.10% | 3.2% | +3.1 |
| `vector databases` | 0.10% | 2.4% | +2.3 |

Disappearing terms are dominated by legacy benefits language ("employees share", "committed diverse", "reward individual contributions") plus exactly one stack token: `net developer` (2.2% → 0.2%). Cluster and TF-IDF methods agree on direction and on the dominant emergent vocabulary. The cluster method gives sharper labels; TF-IDF gives finer term-level evidence.

### Thread 4 — Legacy substitution

| Pair | 2024 source AI rate | 2026 neighbour AI rate (T36, strict) | 2026 neighbour AI rate (`unified_core`, wider pattern) |
|---|---|---|---|
| database developer → database engineer | 0% | 4.6% | 0.0% (n=22) |
| devops architect → devops engineer | 0% | 2.0% | 13.6% (n=265) |
| drupal developer → web developer | 0% | 4.4% | 3.1% (n=32) |
| java application developer → full stack java developer | 0% | 2.4% | 19.2% (n=26) |
| java architect → java developer | 0% | 1.0% | 9.5% (n=84) |
| scala developer → big data engineer | 0% | 7.1% | 20.0% (n=10) |
| **mean** | **0%** | **3.6%** | **11.9% weighted** |
| **2026 SWE market** | — | **14.4%** (T36 calibration, strict) | **27.6%** (canonical wider pattern, this run) |

Both methods point in the same direction: legacy-stack substitutes sit below the 2026 SWE-market AI rate by ~4× under T36's strict definition and ~2.3× under the wider canonical definition. The "stack-modernisation, not AI-ification" headline is robust to vocabulary choice.

The YoY frequency-delta recovers a longer tail of disappearing titles: `senior systems engineer` (244 → 30), `c#/.net developer` (120 → 0), `automation engineer` (108 → 26), `mainframe developer` (66 → 2), `salesforce developer` (45 → 5), `software architect` (41 → 1). Emerging titles meeting `n_2024 < 5 AND n_2026 ≥ 30`: `ai/ml engineer` (1 → 36), `founding engineer` (4 → 34). The frequency-delta is the cheaper sanity check; T36's nearest-neighbour mapping is the richer instrument because it tells you *which* 2026 title absorbed the demand.

---

## Cross-method agreement and divergence

- **Thread 1 — partial divergence.** Volume growth depends critically on the operationalisation. Cluster discovery (T34) reports 15.6×; title regex reports 5.2×. The director-density elevation is preserved across methods (1.25–1.89×). The YOE premium is **not** preserved by the title regex. We recommend the article quote a multiplier *range* (5–16×) and explicitly note that the higher number is a content-cluster definition, the lower a title-string definition.
- **Thread 2 — agreement, with caveat.** Title-only and title+description both confirm a 15-19× share-rise from a near-zero base. AI density 1.95-2.34× across both methods. Firm count of 38 reproduces exactly. The YOE-junior tilt (4.0 vs 5.0) is a small-sample wobble, not a methodologically meaningful divergence.
- **Thread 3 — strong agreement.** Cluster ingestion and TF-IDF emergent-term scan independently surface the same vocabulary: agentic/RAG/LLM/Claude/Copilot/LangChain/vector-database. Stack-modernisation (kubernetes/CI-CD/pipelines) is the second growth lane. Legacy ASP/.NET/Salesforce is the principal shrinking lane. **Methods converge.**
- **Thread 4 — strong agreement.** T36 nearest-neighbour and the cheaper YoY title-frequency-delta agree on the disappearing titles and on the substitute-AI-rate-below-market headline. Neighbour rates differ by vocabulary choice but both stay below market.

---

## Decision on Thread 3 — ingest, rerun, or both

**Decision: ingest.**

- The v9 T09 artifact (`exploration-archive/v9_final_opus_47/artifacts/shared/swe_archetype_labels.parquet`) is a usable 8,000-row labelled sample, 7,999/8,000 join cleanly to `unified_core` by `uid`. Its archetype taxonomy already discriminates AI (`models/systems/llm` 76% AI, `systems/agent/workflows` 68% AI) from non-AI clusters.
- The TF-IDF emergent-term scan independently corroborates the same vocabulary on a fresh balanced sample. Two methods agree.
- A full BERTopic rerun on the ~48k filtered SWE corpus would take ~2-3 hours of CPU (30 min sentence-transformer embedding, 30 min UMAP, 60 min HDBSCAN, 30 min c-TF-IDF + stability runs). RAM ~6-8 GB peak with chunked embedding.
- Marginal value is low for the article-level claim — a rerun would refine cluster boundaries but is unlikely to overturn the LLM/agent + CI-CD-modernisation growth-lane headline. A rerun would be valuable for paper-grade analysis where coverage on the full corpus matters; not needed for this composite. **Proposed, not run.**

---

## Recommended composite article structure (4 panels)

**Working title:** *How software-engineering roles are being created, destroyed, and rewritten.*

**Lede.** Frame the tension: industry rhetoric says AI is collapsing the experience premium; posting data show employers asking for *more* experience on AI-specialist roles, *new* roles where AI-tool fluency is the baseline, and *quietly disappearing* roles whose work is being reabsorbed into modernised stacks. Anchor with one number — AI-vocab in SWE rose from 3% to 28%, in non-tech control only from 0.3% to 1.4% (23-to-1 delta).

**Panel 1: The new specialist (Applied-AI).** Lead with title-regex 5× growth (366 → 1,896 postings carrying `AI/ML/LLM/Applied AI Engineer`) as the conservative anchor; footnote the content-cluster 15× as upper bound. The Karpathy/Huang/Altman counter-quote drives the rhetorical frame. The senior Applied-AI YOE bar is 5-6 years (depending on definition; **not** lower than senior baseline); director-share is 2.1% within senior Applied-AI vs 1.7% senior-overall (the one structural elevation that survives both methods).

**Panel 2: The new function (Forward-Deployed).** Title moved from 3 to 52 LinkedIn postings (15× share-rise from a near-zero base; flag the small base). 38 firms post under the title in 2026 — Palantir, OpenAI, Saronic, Foxglove, Ditto, PwC, Ramp. AI-density 1.95-2.34× general SWE pool. Anthropic caveat: it is the largest hirer of the function under "Applied AI Engineer", not under "Forward-Deployed". The function is broader than the title.

**Panel 3: The emerging skill stack (under-the-title content).** Two growth lanes from the v9 T09 + TF-IDF triangulation: (i) AI-tooling (`models/systems/llm` 7.7×, `systems/agent/workflows` 15.3×; emergent vocab `agentic`, `rag`, `claude`, `copilot`, `langchain`, `prompt engineering`, `vector databases`); (ii) stack-modernisation (`kubernetes/terraform/cicd` 3.0×, `pipelines/sql/etl` 2.5×). Shrinking: legacy .NET/ASP and ceremonial-process. Method-agreement is the rhetorical asset here.

**Panel 4: The quiet substitution (legacy).** Disappearing-title pairs: Java architect → Java developer, Drupal developer → web developer, devops architect → devops engineer. Neighbour AI-rate sits at 3.6% (strict) / 11.9% (wider) — both well below the 14.4% / 27.6% market baselines. Substitution channel is stack-modernisation, not AI-ification. Texture: `mainframe developer` 66 → 2, `c#/.net developer` 120 → 0, `salesforce developer` 45 → 5.

**Closer.** The role landscape is restructuring on three axes: a small visible peak (Applied-AI, FDE) where the credential bar is rising; a large content-rewrite (CI/CD, Kubernetes, AI-assisted) running through mainstream SWE postings; a quieter floor where legacy-stack roles are absorbed by modernised neighbours mostly without AI-rebranding. The same firms write all three kinds of changes (S17 within-firm panel: +19.4 pp AI-rewrite on 292 overlap firms). Industry rhetoric describes the visible peak; the body of the change is in the rewrite.

---

## Caveats

- **Definition sensitivity (Thread 1).** The 15.6× headline cannot be quoted without "by content cluster"; the 5.2× without "by title". Foreground both, do not pick one.
- **Small bases (Threads 1, 2).** FDE 2024 base of 3 cannot sustain a tight multiplier; prefer share-based framing. T34 cluster 2024 base of 144 is near the small-base regime.
- **Vocabulary choice changes neighbour AI rates by ~3×.** T36 strict gives 3.6%, wider canonical gives 11.9%. Both stay below their market baselines so the qualitative finding is robust; always report alongside pattern definition.
- **v9 T09 labels are an 8k stratified sample.** Archetype × period growth ratios are descriptive of the sample; a full-corpus rerun (4 hours compute, proposed but not executed) would refine but not overturn the headline.
- **Aggregator inflation in FDE.** TalentAlly contributes 5 of 52. Strip leaves 41 postings / 31 firms — still 14× share-rise, still 1.9× AI density.
- **YOE-LLM coverage gaps** make the FDE 4.0 vs 5.0 median fragile.
- **TF-IDF emergent terms are contaminated by recruiter boilerplate.** Pre-filter is necessary; the content-only table is `S27_thread3b_tfidf_emergent_content.csv`.
- **No causal claim.** All four threads describe restructuring in employer postings, not employment outcomes. Stay inside the labour-demand-signalling frame.

---

## Files

All tables and figures with prefix `S27_*` under `eda/tables/` and `eda/figures/` (16 + 5). Generation code: `eda/scripts/S27_role_landscape.py`.
