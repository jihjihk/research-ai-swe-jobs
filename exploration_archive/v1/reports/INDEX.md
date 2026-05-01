# Exploration Task Index

| Task | Agent | Wave | Status | One-line finding |
|------|-------|------|--------|------------------|
| T01 | A | 1 | done | Core columns 100% covered; seniority 45-71% known (SWE); company_industry/size/skills unusable cross-period |
| T02 | B | 1 | done | **BLOCKER:** native_backfill bug — seniority variants disagree on RQ1 direction (rule-based: entry increasing; native: decreasing) |
| T03 | A | 1 | done | Entry-level SWE: 89 arshkon / 124 asaniczka (imputed) / 132 scraped; associate/entry overlap across sources |
| T04 | B | 1 | done | SWE FP ~5.5% overall; embedding_llm tier 26% FP; FN ~12%; adequate with sensitivity checks |
| T05 | C | 1 | done | Arshkon-scraped seniority distributions indistinguishable (p=0.096); description length +50%; industry labels incomparable |
| T06 | C | 1 | done | Asaniczka concentrated (Capital One+Recruiting from Scratch=15.3%); company-capped sensitivity robust (<1.5pp shift) |
| T07 | D | 1 | done | Geographic r=0.978 (all >0.93); all periods in same labor market regime (Indeed SWE index ~70) |
| T08 | E | 2 | done | Description length +64% (median 3124->5110); entry share 21.0%->13.9% (patched, arshkon->scraped); remote 0%->24.7%; aggregator 28%->12-14% |
| T09 | E | 2 | done | **Direction split:** native-label variants show entry declining 7pp (-33%), title-only variants show +1.5pp (+30%); 3 of 5 variants agree on decline; seniority_llm needed to resolve |
| T10 | F | 2 | done | AI is #1 temporal distinguisher (z=-26.8 junior, z=-43.8 senior); 14-19 AI terms significant per comparison; ownership language shifted to 2026; C5 shows junior 2026 surpasses senior 2024 on AI but trails on experience/management |
| T11 | F | 2 | done | JSD overall 0.038 (entry 0.059 -- 52% higher); AI prevalence 19%->55% (broad), 3.6%->24% (narrow); entry-level YOE dropped 3.7->2.0 years (p<0.001); emerging terms dominated by AI tooling (agentic, claude, cursor, mcp, rag) |
| T12 | G | 2 | done | Scope inflation in organizational language (cross-functional +10pp, ownership +8pp, collaboration +24pp) but not formal credentials (YOE decreased, MS/PhD flat); AI keywords 15%->44% at entry; field-wide shift, not junior-specific |
| T13 | G | 2 | done | 379 overlap companies; within-company entry decline -9.3pp (stronger than overall -3.8pp); AI surge +33pp within-company (composition ~0); large firms (10K+) drive entry squeeze (-13pp) |
| T14 | H | 2 | done | AI requirements tripled 18%->55% (2024->2026); AI-tool mentions (LLM/GPT/Copilot) 3%->26%; employer-worker gap closed from ~40pp to ~0pp; convergence consistent with rapid institutionalization |
| T15 | H | 2 | done | Ghost risk flag too conservative (354/1.16M non-low, all entry-level by construction); YOE contradictions 912 total (22 SWE); description quality >99.8% ok; no analytical threat; needs LLM ghost_assessment for utility |
| T16 | H | 2 | done | **SWE junior decline is occupation-specific:** SWE junior share fell 21%->14% while control INCREASED 23%->44%; AI-tool requirements concentrated in technical roles (SWE 26% vs control 1.3%); findings not confounded by macro trends |
| T17 | I | 3 | done | 5 interview artifacts produced: inflated junior JDs, paired JDs (2024 vs 2026), junior-share trend + AI timeline, senior archetype chart, employer-worker divergence chart |
| T18 | I | 3 | done | Synthesis document written: data quality verdicts, recommended samples, seniority recommendation, confounders, preliminary findings, tensions to resolve, sensitivity requirements, pipeline fixes |

## Gate 1 Notes

### Blocker

**T02: Seniority variant direction disagreement.** A native_backfill bug in Stage 5 (`resolve_seniority_final()`, line 624) causes `seniority_final` to ignore 9,490 SWE rows with usable `seniority_native` labels. Only 1 of 30,283 SWE rows gets backfilled. As a result:
- Rule-based variants (seniority_final, high-confidence) show entry share **increasing** (0.8% -> 4.4% -> 5.7%)
- Native-label variants show entry share **decreasing** (22.9% -> 15.5%)
- The RQ1 conclusion flips depending on which operationalization is used.

**Decision required:** Fix the bug and re-run preprocessing before Wave 2, or proceed with Wave 2 noting the constraint?

### Warnings

1. **Seniority unknown rate 29-55% for SWE** — all seniority-stratified analyses use <71% of data (T01/T03)
2. **Entry-level samples thin** — 89 (arshkon) / 132 (scraped) for seniority_final (T03)
3. **Asaniczka "associate" overlaps with arshkon "entry"** — 60 shared titles; cross-source seniority unreliable without re-labeling (T03)
4. **embedding_llm tier 26% FP rate** — 14% of SWE rows; recommend regex-only sensitivity check (T04)
5. **Description length grew 50%** — keyword counts must be normalized by length (T05)
6. **Industry labels incomparable** — scraped compound labels vs arshkon single labels (T05)
7. **Asaniczka company concentration** — Capital One + Recruiting from Scratch = 15.3% of SWE postings (T06)
8. **"Jobs via Dice" possible aggregator leakage** — 130 scraped postings (3.1%) (T06)

### Clean

- Geographic representativeness exceeds target: all r > 0.93, combined r = 0.978 (T07)
- All collection periods in same labor market regime: Indeed SWE index 69.9-71.9 (T07)
- Company-capped sensitivity robust: seniority shifts <1.5pp after capping at 10/company (T06)
- SWE classification adequate: overall FP ~5.5%, below 10% threshold (T04)

**Seniority column recommendation:** Use `seniority_patched` from the SQL workaround in `exploration/seniority_workaround.sql`. This replicates what the fixed pipeline will produce by applying: strong title signals > native backfill > weak signals > unknown. Results:

| Source | seniority_final unknown % | seniority_patched unknown % | Old entry n | New entry n |
|--------|--------------------------|----------------------------|-------------|-------------|
| arshkon | 54.6% | 19.9% | 89 | 745 |
| asaniczka | 29.3% | 0.0% | 124 | 121 |
| scraped | 44.6% | 6.4% | 132 | 549 |

Wave 2 agents: include the `swe_patched` CTE from `exploration/seniority_workaround.sql` in all seniority-stratified queries. Use `seniority_patched` instead of `seniority_final`, and `seniority_3level_patched` instead of `seniority_3level`. Also report results using the original `seniority_final` as a sensitivity variant.

**Column exclusions for cross-period analysis:** `company_industry` (asaniczka=0%), `company_size` (arshkon-only), `skills_raw` (asaniczka-only), `work_type` (asaniczka=0%). All requirement/skill analysis must use description-text NLP.

## Gate 2 Notes

### No blockers

All Wave 2 tasks completed successfully. No findings invalidate the study design or prevent synthesis.

### Gate 2 checklist

- [x] T08: Distributions show expected patterns. Entry share 21.0%->13.9% (patched). Description length +64%. YOE coverage dropped to 16% in 2026 (warning).
- [x] T09: 3 of 5 variants agree on entry decline (-7pp). 2 title-only variants disagree (+1.5pp). Native-label cluster is better-powered (545-745 vs 87-132). Not a new blocker.
- [x] T10: AI is the dominant distinguisher. Ownership/autonomy language shifted to 2026. Findings strongly align with RQ1/RQ2.
- [x] T11: Entry JSD 52% higher than overall — entry-level postings changed disproportionately. YOE *decreased* (3.7->2.0). AI prevalence 19%->55%.
- [x] T12: Scope inflation confirmed for organizational language but NOT formal credentials. Field-wide, not junior-specific.
- [x] T13: Within-company entry decline -9.3pp (stronger than overall -3.8pp). AI surge is within-company, not compositional. Large firms drive the squeeze.
- [x] T14: Employer-worker AI gap closed from ~40pp to ~0pp (2024->2026). AI-tool category (3%->26%) is cleanest RQ3 variable.
- [x] T15: No systematic data quality issues. Ghost risk flag too conservative for use; needs LLM assessment.
- [x] T16: **Strongest finding** — SWE junior decline is occupation-specific. Control group shows *opposite* trend (+21pp). Rules out macro confounding.

### Warnings

1. **T09: Seniority direction not fully resolved** — 3/5 variants agree on decline, but seniority_llm still needed for definitive answer
2. **T08: YOE extraction coverage dropped** from 34-40% to 16.4% in 2026 — YOE findings are directionally useful but selection-biased
3. **T12: Scope inflation is field-wide** — not disproportionately junior; complicates the "junior squeeze" narrative
4. **T13: Aggregator contamination in overlap set** — TEKsystems, Apex Systems show erratic seniority labels; sensitivity check needed excluding is_aggregator
5. **T12: Collaboration keyword inflation** may partly reflect longer descriptions having more boilerplate

### Findings to reconcile in synthesis

- **YOE paradox:** Entry-level YOE *decreased* (T11, T12) while organizational scope *increased*. Not credential inflation but complexity inflation.
- **Field-wide vs junior-specific:** Scope inflation exists at all levels (T12), but entry decline is SWE-specific (T16) and within-company (T13).
- **Within-company vs composition:** Entry decline -9.3pp within-company, partially masked by +5.6pp composition effect from new companies (T13).
- **AI convergence:** Employer AI requirements caught up with worker usage (T14), consistent with rapid institutionalization narrative.

## Wave 3 Completion Notes

All exploration tasks (T01-T18) are complete.

- [x] T17: 5 interview elicitation artifacts produced in `exploration/artifacts/`
- [x] T18: Synthesis document at `exploration/reports/SYNTHESIS.md` -- the one document the analysis agent reads first

**Handoff to analysis phase:** Read `exploration/reports/SYNTHESIS.md`. It covers data quality verdicts, recommended samples, seniority column guidance, known confounders, preliminary findings, key tensions, sensitivity requirements, and pipeline fixes needed.
