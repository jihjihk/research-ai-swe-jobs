# Gate 1 Research Memo

## What We Learned

1. **The binding constraints are measurement and comparability, not raw row count.** The default LinkedIn/English/date-ok frame has 63,701 SWE rows across the three LinkedIn sources, but seniority, text, company, and geography each have source-specific limits. The sharpest text constraint is scraped LinkedIn cleaned-text coverage: only 30.7% of scraped LinkedIn SWE rows have `llm_extraction_coverage = 'labeled'`, versus 99.9% for arshkon and 94.0% for asaniczka.

2. **`seniority_final` is defensible but conservative.** It produces YOE-coherent assigned labels, and known-only native agreement is acceptable. The main defect is under-labeling into `unknown`, not noisy over-assignment. Scraped SWE unknown seniority is about 38-41% by source-period in T03 and 53.4% in T05's source-level known/unknown table, so seniority-known subsets must never be treated as representative by default.

3. **Asaniczka native `associate` is not a junior proxy.** T02's decision rule fails: asaniczka `associate` matches arshkon `entry` on only 1 of 4 signals. It has a 30.7% senior-title cue rate compared with 2.7% for arshkon native `entry`, and its `seniority_final` distribution is closer to arshkon `associate` than to arshkon `entry`.

4. **The initial junior-decline story is not supported as a simple aggregate fact.** T30 finds J1-J4 all rise from pooled 2024 to scraped 2026. But T06's matched-company decomposition splits: label-based J1/J2 decline slightly, while YOE-based J3/J4 rise, mostly through between-company reweighting. The right next question is not "did junior share decline?" but "why do employer labels and YOE proxies diverge, and how much is company/domain composition?"

5. **Company composition is a first-order confound.** Company overlap is low, new entrants are 44.8% of scraped LinkedIn SWE postings, and entry posting is a specialized employer activity. Among scraped LinkedIn companies with at least 5 SWE postings, 79.4% have zero J1 entry-labeled rows. Any corpus-level text, topic, or network result must be company-capped and checked against entry-specialist employers.

## What Surprised Us

- **Arshkon native `entry` is YOE-noisy.** Among arshkon native-entry rows with known YOE, 42.6% require 5+ years. By contrast, `seniority_final = entry` has a median YOE of 2 and only about 6% of YOE-known rows at 5+ years in both arshkon and scraped. Native labels are not a clean ground truth.

- **Junior definitions agree in the aggregate but not in matched-company decomposition.** Aggregate J1-J4 all move up; common-company J1/J2 move slightly down while J3/J4 move up. This disagreement is probably more informative than either estimate alone.

- **The senior side is not one story.** S1/S3/S4 decline, but S2 director-only rises. A defensible claim would distinguish broad senior roles from director/top-ladder roles rather than reporting one "senior share" trend.

- **Known-seniority distribution is more similar between arshkon and scraped than between arshkon and asaniczka.** For known seniority, arshkon-vs-scraped Cramer's V is 0.035, below the within-2024 arshkon-vs-asaniczka value of 0.140. This helps arshkon as a baseline but also shows how instrument-specific asaniczka is.

- **Geographic face validity is strong despite instrument concerns.** T07's state-level correlations with OEWS employment are about r = 0.98, which supports the geographic frame. This does not validate all vacancy levels or content claims, but it reduces concern that the sample is geographically arbitrary.

## Evidence Assessment

| Finding | Strength | Sample / n | Calibration and sensitivity | Main confounds | Gate 1 interpretation |
|---|---|---:|---|---|---|
| Data volume is adequate for all-SWE analyses | Strong | 4,691 arshkon SWE; 18,129 asaniczka SWE; 40,881 scraped LinkedIn SWE | T07 all-SWE MDE is 1.2-2.2 pp for binary outcomes | Source instruments differ | All-SWE Wave 2 analyses should proceed. |
| Cleaned text is coverage-limited in scraped | Strong | 12,534 labeled scraped LinkedIn SWE out of 40,881 | Not a sensitivity; it is a coverage constraint | LLM selection may not be random | Text-dependent claims must report labeled/eligible counts and avoid silent raw fallback. |
| `seniority_final` is defensible but conservative | Moderate-strong | 63,701 SWE; known-only native comparison n = 1,667 arshkon and 15,982 scraped | YOE profiles validate final labels; unknown pool remains large | Unknown seniority composition; level-code under-labeling | Use `seniority_final`, but every claim needs T30 panel and unknown reporting. |
| Asaniczka `associate` is not a junior proxy | Strong | 1,820 asaniczka native associate; comparison groups n = 738, 263, 2,231 | Fails 3-of-4 signal rule | Native labels are source-specific | Do not use asaniczka native `associate` as a junior baseline. |
| Aggregate junior share rises across J1-J4 | Moderate | Pooled 2024 vs scraped n ranges from 379/1,275 (J1) to 4,035/9,307 (J4) | Direction unanimous, but signal/noise < 2 against within-2024 calibration | Company composition; unknown seniority; J3/J4 are YOE constructs | Not a lead claim yet. Use as a puzzle for Wave 2. |
| Common-company junior decomposition splits | Moderate | 125 common companies for J1/J2; 121 for YOE variants | J1/J2 down, J3/J4 up | Different estimands; between-company reweighting | This is a key mechanism to investigate, not a nuisance. |
| Broad senior/YOE senior declines while director-only rises | Moderate | S1 n = 13,595/17,713 pooled; S4 n = 10,279/15,938; S2 only 105/295 pooled | 3-of-4 senior panel down, S2 up | Director labels are sparse and may reflect title inflation | Phrase as split senior-ladder pattern, not generic senior decline. |
| Company concentration can distort corpus analyses | Strong | 7,940 scraped LinkedIn companies; top 20 = 18.1% of known-company SWE postings | Aggregator exclusion changes magnitudes; entry-specialist artifact has 308 flagged rows | Canonical company matching; small denominators for YOE specialist flags | Company capping/exclusion is mandatory for Wave 2+ corpus analyses. |
| Geographic representation is plausible | Strong for geography | 51 states in OEWS comparison; r ≈ 0.98 | External benchmark passed; metro null exclusions large | Postings vs employment stock; 26-metro scrape design | Geography can be used with explicit location-known frame. |
| Industry benchmarking is incomplete | Weak | arshkon/scraped internal industry only; asaniczka missing | BLS API threshold blocked industry retrieval | Missing asaniczka industry; platform labels differ | Industry is a source-specific descriptor, not a pooled control yet. |

## Seniority Panel

### Aggregate Junior Share: Pooled 2024 To Scraped 2026

This is the main aggregate junior-share headline from T30. The directional verdict is unanimous, but the cross-period signal is not cleanly above within-2024 source variability.

| Variant | Definition | Effect | Direction | n 2024 / n 2026 | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | `seniority_final = 'entry'` | +3.99 pp | Up | 379 / 1,275 | Unanimous direction; thin J1 power; calibration-sensitive |
| J2 | `seniority_final IN ('entry','associate')` | +3.99 pp | Up | 418 / 1,327 | Unanimous direction; only modest power gain over J1 |
| J3 | `yoe_extracted <= 2` | +5.98 pp | Up | 1,766 / 4,742 | Unanimous direction; powered low-YOE validator |
| J4 | `yoe_extracted <= 3` | +8.12 pp | Up | 4,035 / 9,307 | Unanimous direction; broadest low-YOE construct |

Gate 1 reading: this cannot support the original "junior share declined" narrative. It supports a more cautious puzzle: employer-posted entry labels and low-YOE requirements both become more common in aggregate, but the effect is not clean relative to within-2024 instrument variation.

### Matched-Company Junior Decomposition

This is the mechanism warning from T06. It uses companies with at least 5 SWE postings in both arshkon and scraped LinkedIn.

| Variant | Definition | Full change | Common-panel change | n / panel | Agreement verdict |
|---|---|---:|---:|---:|---|
| J1 | `seniority_final = 'entry'` | -0.5 pp | -0.3 pp | 125 companies | Split against J3/J4 |
| J2 | `seniority_final IN ('entry','associate')` | -0.7 pp | -0.4 pp | 125 companies | Split against J3/J4 |
| J3 | `yoe_extracted <= 2` | +2.2 pp | +7.3 pp | 121 companies | Split against J1/J2 |
| J4 | `yoe_extracted <= 3` | +2.7 pp | +12.1 pp | 121 companies | Split against J1/J2 |

Gate 1 reading: the disagreement is mechanistic. Employer labels and YOE floors are moving differently in the overlap panel, and most J3/J4 common-panel growth is between-company reweighting. Wave 2 should investigate whether this is relabeling, entry-specialist composition, source measurement, or a real shift toward low-YOE non-entry postings.

### Senior Share: Pooled 2024 To Scraped 2026

| Variant | Definition | Effect | Direction | n 2024 / n 2026 | Agreement verdict |
|---|---|---:|---|---:|---|
| S1 | `seniority_final IN ('mid-senior','director')` | -3.99 pp | Down | 13,595 / 17,713 | 3-of-4 senior panel down |
| S2 | `seniority_final = 'director'` | +0.80 pp | Up | 105 / 295 | Contradictory; descriptive only due to power |
| S3 | normalized-title senior regex | -1.59 pp | Down | 446 / 150 | 3-of-4 down, but artifact-prone |
| S4 | `yoe_extracted >= 5` | -7.50 pp | Down | 10,279 / 15,938 | 3-of-4 down; powered YOE validator |

Gate 1 reading: only a qualified claim is admissible. Broad senior and YOE-senior shares decline, but director-only labels rise and S3 is affected by title normalization. Senior role evolution should be investigated as ladder reshaping, not as one monotonic senior trend.

## Narrative Evaluation

- **RQ1: Employer-side restructuring. Needs reframing.** The initial "junior share decline" component is weakened or contradicted at the aggregate level. Junior scope inflation has not been tested yet. Senior restructuring remains plausible but should be framed as a split between broad senior roles and director/top-ladder labels. The current RQ1 should become: *How did the distribution and labeling of SWE seniority change across employer labels, YOE requirements, and company composition from 2024 to 2026?*

- **RQ2: Task and requirement migration. Still open.** Wave 1 does not test requirement migration directly. It does show that any migration claim must separate label-based junior roles from low-YOE roles and must be company-capped, aggregator-tested, and calibrated against 2024 source differences.

- **RQ3: Employer-requirement / worker-usage divergence. Still open, with feasibility caveat.** Wave 1 confirms that all-SWE and broad senior posting analyses are feasible, but the divergence story will depend on benchmark quality. T07's successful FRED/OEWS access is useful context, not a worker-usage benchmark.

- **RQ4: Mechanisms. Strengthened.** The measurement puzzles surfaced in Wave 1 make interviews more important, not less. We need hiring-side accounts of whether low-YOE, entry labels, senior titles, and AI requirements reflect real screening, template choices, or anticipatory signaling.

Initial RQ1-RQ4 narrative status: **weakened as originally stated, not abandoned.** The "junior ladder narrowed" claim is not currently the best lead. A stronger emerging frame is that the SWE posting market may be reorganizing through employer mix, low-YOE labeling, and senior-ladder reshaping, with source instruments and templates shaping what we can observe.

## Emerging Narrative

The data is telling a more complex story than "AI eliminated junior postings." In the aggregate, junior definitions rise; within matched companies, entry labels decline slightly while low-YOE requirements rise; entry posting is concentrated among specialist employers; and 2026 includes many new companies. This points toward **boundary and composition restructuring** rather than a simple junior-volume collapse.

The senior side also looks like boundary restructuring. Broad senior and YOE-senior measures decline, while director-only labels rise. That may reflect top-ladder title inflation, a shift toward director-level accountability, or sparse-label noise. Wave 2 and Wave 3 need to determine whether senior content changes align with this label pattern.

## Research Question Evolution

Proposed current RQ set after Gate 1:

- **RQ1 revised:** How did SWE posting seniority structure change across employer labels, YOE requirements, and company composition between 2024 and 2026?
- **RQ2 revised:** Which requirement bundles changed within and across seniority definitions, and do changes reflect downward migration, domain recomposition, or broad text/template expansion?
- **RQ3 retained:** Do employer-side AI requirements outpace external worker-usage benchmarks, and is any divergence specific to SWE?
- **RQ4 retained and sharpened:** How do engineers and hiring-side actors explain discrepancies between labels, stated YOE, requirements, and actual screening?
- **New candidate RQ5:** To what extent are observed posting changes driven by market recomposition across companies, entry-specialist employers, aggregators, and technology domains rather than within-firm restructuring?

RQ5 may become central if Wave 2 shows that domain/company structure explains more variance than seniority or period.

## Gaps And Weaknesses

- Cleaned-text coverage in scraped LinkedIn is low enough that Wave 2 text findings may apply to a labeled subset unless more LLM budget is allocated later. No Wave 2 agent should run stages 9/10 without explicit user budget approval.
- Unknown seniority is large and structured. We need unknown-pool profiling in Wave 2, especially by company, title family, YOE, and text availability.
- T30's S3 normalized-title senior proxy is not reliable because title normalization strips seniority indicators. Wave 2 should use raw-title senior patterns only as a carefully validated diagnostic, not the current S3 artifact as a lead measure.
- Industry remains weak for cross-source analysis because asaniczka has no industry and OEWS industry benchmarking did not complete.
- T06's entry-specialist artifact is intentionally broad and includes possible small-denominator YOE flags. Downstream tasks should use stricter variants as sensitivities.
- The OOM incident shows that future agents need explicit memory constraints. Shared preprocessing, embeddings, topic modeling, and co-occurrence networks are the highest-risk next steps.

## Direction For Next Wave

First run Wave 1.5 shared preprocessing, but with stricter memory discipline than the original task reference:

- Build shared artifacts sequentially, not in parallel with other local heavy jobs.
- Use DuckDB memory limits and pyarrow chunking.
- For embeddings, batch conservatively and save checkpoints. Avoid holding all texts plus embeddings plus metadata as large pandas objects.
- Given scraped cleaned-text coverage, the shared text artifact must record `text_source` and coverage by source prominently.
- The tech-matrix sanity check is mandatory because escaped C++/C#/.NET style terms can distort period comparisons.

Wave 2 prompt modifications:

- **Primary seniority guidance:** For employer-label junior claims, use J1 as the high-label-quality primary. For content/scope analyses that need power, report J3 as the powered low-YOE primary and J1 as the label-based estimand; J4 is a broad sensitivity. Do not use J2 as a fix for power because it adds little.
- **Senior guidance:** Use S1 as primary and S4 as the powered YOE validator. Treat S2 as descriptive top-ladder evidence and S3 as artifact-prone unless rebuilt from raw titles.
- **Company discipline:** Every aggregate text/topic/network task must use company caps and must test exclusion of `entry_specialist_employers.csv`.
- **Unknown seniority:** Wave 2 distribution profiling should explicitly profile unknowns; dropping unknowns is not acceptable.
- **Narrative discipline:** Agents should treat junior-share direction as unresolved and mechanistic, not as a pre-existing decline.

## Current Paper Positioning

If stopped here, the strongest paper would be a **dataset and measurement paper**, not yet an empirical restructuring paper. The contribution would be a disciplined measurement framework showing how seniority labels, YOE requirements, company composition, and source instruments can produce contradictory narratives about SWE labor-market change.

To become a stronger empirical restructuring paper, Wave 2 must deliver at least one robust, novel content finding that survives: within-2024 calibration, company capping, aggregator exclusion, T30 seniority panel, semantic keyword validation, and text-source sensitivity. The most promising lead candidates after Gate 1 are:

- boundary/composition restructuring rather than junior elimination;
- senior-ladder split between broad senior roles and director/top-ladder labels;
- AI/content changes that persist within overlapping companies and after company caps;
- domain recomposition if unsupervised archetypes explain more structure than seniority.

