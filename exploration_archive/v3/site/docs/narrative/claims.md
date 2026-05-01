# What the Paper Can and Cannot Claim

This page enumerates the evidence-supported claims and their boundaries, derived from 26 exploration tasks and 4 gate reviews.

## What the paper CAN claim

### Core findings (strong evidence, SWE-specific)

1. **AI competency requirements surged and are SWE-specific.**
   DiD = +24.4pp (SWE +26.7pp vs Control +2.4pp). Validated as genuine hiring requirements, not aspirational language (hedge fraction reversed: AI less hedged than traditional in 2026). Specific tools still modest (LLM 10%, Copilot 4%).

2. **Entry-level SWE posting share declined while control junior share increased.**
   DiD = -24.9pp. Within-company decline (-11.8pp) exceeds aggregate, meaning composition dampens the signal. This is robust across 4 seniority operationalizations (with one caveat noted below).

3. **The SWE domain composition shifted dramatically.**
   ML/AI Engineering: 4% to 27% (+22pp). Frontend/Web: 41% to 24% (-17pp). Method-robust (ARI >= 0.996). Domain is 10x more structurally important than seniority for posting content (NMI: 0.175 vs 0.018).

4. **AI requirements are additive, not substitutive.**
   AI-mentioning postings require 11.4 technologies vs 7.3 for non-AI. Stack diversity increased from 6.2 to 8.3 mean techs/posting. A new 25-technology AI/ML ecosystem emerged. DiD for stack diversity: +0.78.

5. **YOE slots purified.**
   5+ YOE "entry-level" postings: 22.8% to 2.4%. Median entry YOE: 3.0 to 2.0. Confirmed across 5 independent decompositions.

6. **AI and entry-level changes are orthogonal at the firm and metro level.**
   Firm-level r = -0.07 (p = 0.138). Metro-level r = -0.04 (p = 0.850). Companies that adopted AI the most did not systematically cut junior roles.

7. **57% of aggregate change is compositional.**
   Shift-share decomposition on 451-company panel. New entrants arrived AI-forward (24.3% AI rate). Only 43% is within-firm behavioral change.

### Supporting findings (moderate evidence)

8. **Senior orchestration surged.** +16% mid-senior, +46% director. AI-mentioning senior postings have higher orchestration than non-AI senior postings.

9. **GenAI adoption accelerated 8.3x** between within-2024 and cross-period rates, consistent with model release cascade.

10. **Associate level is collapsing toward entry** (relative feature-space position: 0.30 to 0.16). De facto 3-tier seniority system emerging.

11. **All 26 metros show the same directional changes.** No counter-trend metro. Tech hubs vs non-hubs show no significant differences.

## What the paper CANNOT claim

1. **That AI caused junior elimination within firms.** The orthogonality finding (r ~ 0) directly contradicts this. These are parallel market trends.

2. **That management scope inflation is dramatic or SWE-specific.** The +31pp was measurement error (corrected to +4-10pp). Management expansion is field-wide (DiD ~ 0), not SWE restructuring.

3. **That junior and senior postings are semantically converging.** Within-2024 calibration shift exceeds cross-period change. The convergence hypothesis fails the noise floor test.

4. **That posting AI requirements outpace developer usage.** Requirements LAG usage (~41% vs ~75%). The gap narrowed from -45pp to -34pp, but employers are catching up, not inflating beyond reality.

5. **That soft skills expansion is SWE-specific.** SWE grew LESS than control (DiD = -5.1pp).

6. **That management migrated from senior to entry.** It expanded at ALL levels. This is universal template expansion, not seniority-specific migration.

7. **That the aggregate trend reflects within-firm behavior.** 57% is compositional. The "restructuring" is partly about which companies are posting.

## What requires caveats

1. **Entry-level trend direction depends on seniority operationalization.** seniority_native shows decline; seniority_3level in overlap panel appeared to show increase. Planned seniority_llm will resolve.

2. **Text quality asymmetry inflates text-based indicators.** 2024 has LLM-cleaned text; 2026 has 0%. Rule-based cleaning retains more boilerplate in 2026.

3. **Thin 2024 entry-level baseline.** 769 native entry-level SWE from arshkon is our only historical baseline.

4. **Two-point comparison.** Cannot distinguish gradual trend from abrupt shift. The within-2024 sub-period comparison (January vs April) provides a partial third point.

5. **Company overlap panel biased toward large firms.** The 451-company panel is robust but may not generalize to smaller companies (only 18% of companies overlap).
