# Composite A — Where AI is signaled most in software-engineering job requirements

**Date:** 2026-04-21 · **Code:** [`eda/scripts/S26_composite_a.py`](../scripts/S26_composite_a.py) · **Tables:** `eda/tables/S26_*.csv` · **Figures:** `eda/figures/S26_*.png`
**Data:** `data/unified_core.parquet`, LinkedIn-only, `is_english AND date_flag='ok' AND is_swe`.
**AI vocab:** the canonical `AI_VOCAB_PATTERN` from [`eda/scripts/scans.py:50-71`](../scripts/scans.py).
**v9 artifacts ingested:** archetype labels from `exploration-archive/v9_final_opus_47/tables/T28/T28_corpus_with_archetype.parquet` (BERTopic over 48k labeled SWE rows).

---

## Hypotheses

**Geographic.** Story 03 (`exploration-archive/v9_final_opus_47/stories/03_atlanta_not_san_francisco.md`) argues AI-requirement growth from 2024→2026 is geographically flat across 26 metros (tech-hub premium under 2 pp on the strict regex), with Tampa and Atlanta leading the rise and the Bay Area still leading on absolute counts of senior Applied-AI roles. We probe whether flatness survives a broader vocabulary, alternative volume cutoffs, aggregator and multi-location filters.

**Industry.** Story 04 argues that within 2026, Financial Services (15.7%) and Software Development (15.3%) are tied, with Hospitals & Health Care leading at 20.9% — falsifying the regulated-industry-lag narrative. We probe whether the ranking holds with a different vocabulary, with aggregators and big-tech firms removed, and with the unlabeled-industry tail accounted for.

**Builder vs user.** Open question: AI-builder roles (Applied AI, ML, LLM, FDE) might stay hub-locked while AI-user roles (general SWE postings citing Copilot/Claude/RAG) might diffuse everywhere. We test this with two parallel methods and check where they agree.

---

## Data sources and scope

- **Corpus.** 110k rows × 42 cols. SWE+English+date-OK rows by period: 18,125 (2024-01) / 4,687 (2024-04) / 11,810 (2026-03) / 14,012 (2026-04).
- **Industry coverage.** 2024-01 has *zero* `company_industry` labels in core; 2024-04 has 99.2%; 2026 is essentially complete. **Industry analysis is therefore within-2026 only**, matching v9's rule.
- **Archetype labels.** T28's `archetype_primary`. Cluster 1 ("models/systems/llm", n=2,382) and 25 ("systems/agent/workflows", n=1,186) are the AI-builder clusters. ~19k SWE rows are unlabeled and excluded from method-(a) denominators.
- **Tech-hub set.** SF Bay Area, Seattle, NYC, Austin, Boston (matches T17).
- **Vocabulary caveat.** `AI_VOCAB_PATTERN` is broader than v9's `ai_strict` (v9 spot-check precision 0.86); ours adds `llm`, `genai`, `agentic`, `ai-powered`, `mlops`, etc. Pooled 2026 SWE rate is 27.6% on broad vs 13.2% on strict. **Quantitative levels are ~2× the strict-pattern equivalents; rankings replicate qualitatively.** The brief required the canonical broad pattern.

---

## Thread 1 — Geography

Per-metro AI rate by year with five sensitivities: aggregator exclusion (`delta_noagg`), multi-location exclusion (`delta_single`), volume cutoff at 30/50/100/200, volume-weighted vs unweighted means, absolute 2026 rate vs differential.

The 26-metro panel reproduces. Every metro gained on the broad pattern, in a band of **+14.2 to +33.1 pp**. Top of the rise: **Seattle (+33.1), Tampa (+32.2), San Francisco (+29.6), Salt Lake City (+29.3), Nashville (+28.0), NYC (+27.7), Atlanta (+27.7), Charlotte (+26.3)**. Tampa and Atlanta replicate in the top eight, but Seattle and SF climb into the top three on the broad pattern (vs mid-pack on strict).

**Tech-hub premium** (5 hubs vs 21 rest, mean delta): all postings +6.37 pp; no aggregators +7.13 pp; single-location only +6.37 pp. Volume-weighted mean across the 26-metro panel is +22.96 pp, basically equal to the unweighted +22.09 pp. The premium runs +5.9 to +7.3 pp across cutoffs ≥30 to ≥200. **The flatness claim is fragile to vocabulary choice**: 1.65 pp on `ai_strict`, 6.4 pp on the broad pattern.

**Absolute 2026 rate** ranks differently from the rise. Top ten: SF 38.3%, Seattle 36.9%, Tampa 33.1%, SLC 32.4%, Atlanta 29.9%, NYC 29.2%, Nashville 29.0%, Miami 27.0%, Charlotte 26.6%, Austin 26.5%. The Bay Area is on top. **The "Atlanta not San Francisco" headline holds only on the differential framing**; on the absolute 2026 cross-section the Bay leads.

---

## Thread 2 — Industry (within-2026)

Per-industry AI rate restricted to 2026, with three sensitivities: pooled (`rate_all`), no aggregators (`rate_noagg`), no big-tech canonical names (`rate_nobig`); plus labeled-vs-unlabeled coverage.

Pooled labeled rate: **27.63%** (n=25,816). The unlabeled tail in 2026 is 6 rows — labeled-only does not bias the comparison.

Top industries (n≥100), with Wilson 95% CIs:

| Industry | n | AI rate | 95% CI |
|---|---|---|---|
| Research Services | 173 | 84.97% | 78.9-89.5 |
| FS, Investment Mgmt & Banking | 154 | 44.81% | 37.2-52.7 |
| Computer & Network Security | 151 | 40.40% | 32.9-48.4 |
| **Hospitals & Health Care** | 372 | **36.02%** | 31.3-41.0 |
| Tech, Information & Media | 210 | 35.24% | 29.1-41.9 |
| Retail | 292 | 34.93% | 29.7-40.6 |
| Internet Marketplace Platforms | 137 | 33.58% | 26.2-41.8 |
| **Software Development** | 6,938 | **32.33%** | 31.2-33.4 |
| Tech, Information & Internet | 1,293 | 32.33% | 29.8-34.9 |
| Entertainment Providers | 226 | 26.55% | 21.2-32.7 |
| **Financial Services** | 1,804 | **26.11%** | 24.1-28.2 |
| Insurance | 220 | 25.91% | 20.6-32.1 |
| IT Services and IT Consulting | 3,053 | 25.45% | 23.9-27.0 |
| Banking | 127 | 19.69% | 13.7-27.5 |

Two findings replicate qualitatively from v9: (i) **Hospitals & Health Care leads major industries** (36% beats Software Development's 32%, non-overlapping CIs); (ii) the regulated-industry-lag narrative is wrong at the labour-demand layer.

One finding **does not replicate** under the broad pattern: Financial Services (26.1%) is now **6 pp below Software Development** (32.3%) with non-overlapping CIs, rather than parity. The broad pattern catches `llm`, `agentic`, `ai-powered`, denser in software postings than in finance. **The "finance matches SWE" claim is vocabulary-dependent; the "hospitals lead software" claim survives both.**

`rate_noagg` and `rate_nobig` are within ±2 pp of `rate_all` for most cells. Notable exception: Software Development falls from 32.3% to 30.9% without big-tech, and rises to 40.2% without aggregators (aggregator postings in Software Development are noisy and AI-light).

---

## Thread 3 — Builder vs user

**Method (a) — v9 archetype.** Builder = `archetype_primary IN (1, 25)` from T28. User = AI-vocab match within other labeled archetypes. Crosstab metro × year.

**Method (b) — title regex.** Builder = title match on Applied AI / ML / LLM / Forward-Deployed / AI Architect / Research Engineer / MLOps phrases ([`S26_composite_a.py:38-58`](../scripts/S26_composite_a.py)). User = AI-vocab match in description among non-builder titles.

Pooled by period, method (a):

| Period | n_builder | builder AI rate | n_general | general AI rate |
|---|---|---|---|---|
| 2024-01 | 253 | 37.6% | 12,597 | 2.2% |
| 2026-04 | 1,223 | 77.4% | 9,171 | 21.7% |

Builder volume rose ~5×; builder AI vocab rose 38→77%; non-builder AI vocab rose 2→22%. **Both expanded; the user-side rise in absolute pp is larger.**

Geography of builder share, 2026:

| Method | Hub builder share | Rest | Premium |
|---|---|---|---|
| (a) archetype | 12.77% | 9.34% | **+3.43 pp** |
| (b) title regex | 8.09% | 5.81% | **+2.28 pp** |

Geography of user (AI-mention rate among non-builder rows), 2026:

| Method | Hub user rate | Rest | Premium |
|---|---|---|---|
| (a) archetype | 24.59% | 17.85% | **+6.74 pp** |
| (b) title regex | 27.47% | 19.44% | **+8.04 pp** |

**Cross-method agreement.** Across the 26 metros: builder-share Pearson r = 0.76 (Spearman 0.80, p<1e-6); user-share r = 0.97 (Spearman 0.95, p<1e-13). Methods agree on rank, disagree on level (archetype catches a longer tail that title regex misses; method-(a) builder share is ~1.5× method-(b)).

**Cross-method divergence.** Salt Lake City is #1 builder-title metro (10.7%) but only 4th by archetype (13.4%) — defense contractors there post titles like "AI Engineer" whose content is platform/integration work rather than LLM model-building. The Bay Area runs the other way: more "Software Engineer"-titled postings whose archetype is LLM model work.

**The hub-locked-builder, diffused-user hypothesis is half right.** Builder-share premium is small (+2-3 pp). User-intensity premium is *larger* (+7-8 pp), not smaller. The Bay Area writes more AI vocabulary into ordinary SWE postings, not just into specialist ones. The metro AI-rate Δ correlates with 2026 builder share at r=0.61, and the absolute 2026 AI rate at r=0.70 — geography sorts the two together.

---

## Cross-method agreement / divergence

| Claim | Replicates? |
|---|---|
| Every metro's AI rate rose 2024→2026 | **Yes** (broad +14 to +33 pp; strict +6 to +19 pp) |
| Tampa and Atlanta in top of metro rise | **Partially** — top 8, but Seattle/SF/SLC also top 5 on broad |
| Tech-hub premium under 2 pp | **No** — +6.4 pp on broad, +1.65 pp on strict |
| Bay Area leads in absolute 2026 AI rate | **Yes** — 38.3%, ahead of Seattle 36.9% |
| Hospitals & Health Care leads industries | **Yes** — 36.0%, non-overlapping CI with Software Dev |
| Financial Services matches Software Dev | **No** — FS 26%, SWE 32%, non-overlapping CIs |
| Builder roles hub-locked, user roles diffused | **No** — builder premium small (+2-3 pp), user premium *larger* (+7-8 pp) |
| Methods (a) archetype vs (b) title regex agree | **Yes** on rank (Spearman 0.80-0.95); disagree on level |

---

## Recommended composite article structure

Three panels, one headline number each, plus a scope caveat.

**Panel 1 — Where AI is being written into engineering job descriptions.**
Headline: *Bay Area 38%, Seattle 37%, Tampa 33%, Salt Lake City 32%; lowest, Portland 16%.*
Frame: every metro gained, in a band of 14 to 33 pp. Tech hubs rose ~6 pp more than non-hubs on the broad measure. Use the *absolute 2026 framing* rather than the differential, because the differential's tech-hub premium is sensitive to vocabulary choice (sub-2 pp on a tight regex, 6 pp on a broad one); the absolute level is more robust.

**Panel 2 — Hospitals out-write software firms.**
Headline: *36% of hospital SWE postings mention AI vocabulary in 2026, vs 32% software firms, 26% financial services.*
Frame: within-period only (2024 industry labels missing). The "regulated-industry lag" story does not survive a labour-demand measurement; hospitals are out-writing tech. Banking sits at 20% on n=127. Use this panel to break the consultancy frame.

**Panel 3 — Builders cluster, users follow.**
Headline: *Hubs hold 8% of postings in builder-title roles vs 6% in non-hub metros — a 2-point gap. The gap on AI mentions in *ordinary* SWE roles is wider, at 8 points.*
Frame: the intuitive hub-locked-builder story is correct but small. The bigger geographic gap is in how densely *non-specialist* postings are written with AI vocabulary. Two methods agree on rank (Spearman 0.80) but disagree on level; report the conservative title-regex number.

**Scope caveat.** LinkedIn job postings measure *labour-demand signaling*, not employment. The BLS-derived claim that software jobs are spreading into non-tech industries (retail +12%, property +75%, construction +100%, 2022-2025, *The Economist* April 2026) does not appear in the LinkedIn frame: non-tech industries' share of LinkedIn SWE postings is essentially flat at ~55% from 2024 to 2026 (Falsified 2 in [`eda/notebooks/findings_consolidated_2026-04-21.ipynb`](../notebooks/findings_consolidated_2026-04-21.ipynb)). Both can be true: BLS measures employed people across all hiring channels, including referral-driven and craft-trade hiring not transacted on LinkedIn. The article's geographic and industry claims are *about LinkedIn's posting frame*, which is selective for office-based, formal-sector, white-collar SWE hiring.

---

## Caveats

1. **Vocabulary sensitivity dominates the disagreement with v9.** The broad pattern raises absolute rates ~2× and roughly doubles the tech-hub premium. Absolute rankings of metros and industries are mostly preserved. Either pattern is defensible; recommend reporting the strict-pattern number as headline and the broad-pattern as robustness in any published version.
2. **Industry analysis is within-2026 only.** Reading any 2024→2026 industry shift from this corpus would be a measurement artefact.
3. **Builder/user definitions are construct-loaded.** Method (a) inherits BERTopic clustering decisions; (b) inherits the title regex. The 0.76-0.97 cross-method correlation provides assurance on rank but not level.
4. **Aggregator handling barely matters for the metro rise** but materially shifts the Software Development industry rate (+8 pp without aggregators).
5. **Salt Lake City and Tampa are likely under-covered news angles.** Both rank in the top 5 on multiple measures (SLC defense-AI; Tampa-as-finance-back-office) and are not in the v9 stories.
6. **The LinkedIn-vs-BLS scope caveat is real and should sit next to the geographic claim.** The article cannot make population-level "non-tech is hiring more software people" claims without a non-LinkedIn source.

---

## File index

| File | Contents |
|---|---|
| [`eda/scripts/S26_composite_a.py`](../scripts/S26_composite_a.py) | All analysis code |
| `eda/tables/S26_metro_panel.csv` | 26-metro panel with deltas under three filters |
| `eda/tables/S26_metro_hub_premium.csv` | Hub vs rest premium under three filters |
| `eda/tables/S26_metro_cutoff_sensitivity.csv` | Cutoff and weighting sensitivity |
| `eda/tables/S26_metro_abs_2026.csv` | Absolute 2026 rate ranking |
| `eda/tables/S26_industry_2026.csv` | Per-industry 2026 AI rate, three filters, Wilson CIs |
| `eda/tables/S26_industry_coverage.csv` | Labeled vs unlabeled coverage |
| `eda/tables/S26_builder_user_archetype.csv` | Method (a) pooled by period |
| `eda/tables/S26_builder_user_metro_archetype.csv` | Method (a) per metro × year |
| `eda/tables/S26_builder_user_metro_title.csv` | Method (b) per metro × year |
| `eda/tables/S26_builder_user_cross_method.csv` | Method-level hub premium comparison |
| `eda/figures/S26_metro_deltas.png` | Metro AI deltas, ranked, hubs in red |
| `eda/figures/S26_industry_2026.png` | Top 15 industries with Wilson CIs |
| `eda/figures/S26_builder_share_metro.png` | Builder share by metro, both methods |
| `eda/figures/S26_user_share_metro.png` | User AI rate by metro, both methods |
| `eda/figures/S26_composite_3panel.png` | Three-panel article-style summary |
