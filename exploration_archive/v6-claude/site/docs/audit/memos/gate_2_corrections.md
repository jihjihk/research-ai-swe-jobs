# Gate 2 Corrections (post-V1 verification)

Date: 2026-04-15
Source: `exploration/reports/V1_verification.md`

V1 independently re-derived all 5 headline numbers within 1% of Gate 2's reported values. The computational foundation of Gate 2 is solid. However, V1 caught **four citation / validation errors** in Gate 2 that must be applied before Wave 3 dispatch and before any paper draft.

## Correction 1 — Split the "SNR 925" citation (**blocker for paper draft**)

**Error:** Gate 2 reports `"aggregate AI prevalence 5.15% → 28.63% (+23.5 pp, SNR 925)"` as a single cell. This combines two different metrics from two different tasks:

- **5.15% → 28.63%** is from T14's broad 24-tech-term union. Its true SNR is **13.3**, not 925. Still well above the SNR ≥ 2 threshold, but not 925.
- **SNR 925** is from T05's narrow single-pattern LIKE query on the string "ai". The 2024 → 2026 rates on that pattern are **2.81% → 18.78%**.

Both findings are real; both are above threshold; **but they cannot be cited together in the same cell.** Any paper draft or presentation that quotes "5.15 → 28.63 at SNR 925" is making a false precision claim.

**Fix for downstream:**
- "Narrow AI keyword rate (LIKE '%ai%'-style matches, T05): 2.81% → 18.78%, SNR ~925."
- "Broad AI prevalence (24-tech-term union, T14): 5.15% → 28.63%, SNR 13.3."
- Everywhere Gate 2 or Wave 2 reports cite the combined figure, replace with whichever of the two is appropriate to the context. The AI explosion is still the lead finding; this is a bookkeeping correction, not a substantive weakening.

## Correction 2 — Downgrade the management finding

**Error:** Gate 2 reports T11's strict management pattern as "cleanly validated" and cites "`mentor` +55%, `manage` −47%" as a defensible narrow finding.

V1's 50-row stratified precision audit on the strict pattern: **38% precision (people-management only), 50% precision (including task-management).** T11's claimed "100% precision" was tautological — it checked that strict-pattern rows contained strict keywords, not whether the semantics were actually management.

Breakdown of V1's finding:
- **`mentor` sub-pattern: clean** (~90%+ precision). The +55% rise (3.8% → 5.9%) is real.
- **`manage` sub-pattern: task management dominates**, not people management. Matches "manage your time", "manage the codebase", "manage deployments". The "−47%" claim (3.8% → 2.0%) is measuring a category that is not what a reader would assume "management" means.
- **`hire` / `hiring`: HR-boilerplate contaminated.** Matches "we are hiring", "our hiring process", "equal opportunity hiring".

**Fix for downstream:**
- **Elevate `mentor` +55% from Moderate to Strong** as a narrow mentoring-language finding. This is the only clean component.
- **Drop the `manage` −47% claim** as written. The direction may be real but the semantic category is contaminated.
- Drop `hire/hiring` from any management pattern.
- For Wave 3 T21, Agent L must **rebuild the management pattern from scratch**: require object-noun-phrase after `manage` (manage a team, manage people, manage direct reports), remove `hire/hiring`, validate on 50-row stratified samples.
- The overall "senior archetype shift" narrow claim survives as: `mentor` rose, general management language is contaminated and should not be cited.

## Correction 3 — Tighten the scope_density finding

**Error:** Gate 2 cites `scope_density` +85% per 1K chars (T11) as a robust, SNR-strong finding subject only to Wave 3 keyword validation. V1 ran the keyword validation now.

**Scope pattern precision (50-row stratified):**
- `end-to-end`: **96% clean** ✓
- `cross-functional`: **100% clean** ✓
- `ownership`: **58% (36% in 2024, 80% in 2026)** — FAILS
  - 2024 contamination: "employee-owned", "total cost of ownership", "ownership group", "on your own terms"
  - 2026 mostly clean: "end-to-end ownership", "ownership of the service"

The 2024 `ownership` baseline is contaminated with corporate boilerplate. The apparent `ownership` rise is partly a decrease in 2024 boilerplate, not purely a rise in real scope language.

**Fix for downstream:**
- **Direction of `scope_density` rise is real; magnitude is overstated by ~15-30%.**
- Wave 3 T22 must rebuild the scope pattern excluding bare `ownership` (or require adjacent qualifier like "ownership of", "end-to-end ownership", "ownership mindset").
- Keep `end-to-end` and `cross-functional` as clean components.
- Re-report scope_density after the pattern rebuild. Expect +60-70% instead of +85%.

## Correction 4 — Reframe T15 convergence as null, not diverging

**Error:** Gate 2 reports T15 as "junior↔senior postings are slightly DIVERGING, fails SNR by a hair." V1 correctly notes that SNR 1.90-1.94 is **below the threshold of 2**, which means the test cannot reject the null — it cannot distinguish a mild divergence from no change from a mild convergence with our sample size.

**Fix for downstream:**
- **Junior↔senior convergence: null result.** The sample cannot distinguish convergence, divergence, or no change. Both embedding and TF-IDF representations point weakly toward divergence, but neither passes the SNR threshold.
- This is still a useful finding: **the "seniority levels are converging" claim from RQ1's boundary-blurring thread is not supported by the data**, even in the direction of its hypothesis. But we should not say "diverging" — we should say "no evidence of convergence, with a weak (below-threshold) divergence point estimate."

## Correction 5 — "Every archetype rose" is literally false

**Error:** Gate 2 says "within-archetype junior share rose uniformly across every archetype." V1 checked: **GPU/CUDA fell (4.7 → 3.4).** The claim is nearly-but-not-universally true.

**Fix for downstream:**
- Rephrase as: "Junior share rose in every large archetype (n_known ≥ 100)."
- Note the exception: GPU/CUDA (small archetype, possibly noisy).
- Does not materially change the Alt A reading — 9 of 10 large archetypes rose uniformly, which is still a strong signature of measurement change.

## Correction 6 — agents_framework 2024 precision

**Error:** Gate 2 cited T14's "~30% FP in 2024" on `agents_framework`. V1 found 2024 precision is **~10%**, not 30%.

**Fix for downstream:**
- True 2024 baseline is even noisier than Gate 2 reported.
- True cross-period effect is close to +11.4 pp (matching T14's conservative +10-11 pp estimate).
- **No change to direction or significance**, just a bookkeeping correction on the caveat precision.

---

## Gate 2 clearance verdict (updated)

Per V1's re-derivation:

- All 5 headline numbers reproduce within 1% ✓
- Alternative explanations for all 5 headlines ruled out ✓
- Tech network modularity rise holds under cap = 20 and cap = 50 ✓
- T13 section classifier is not systematically mis-assigning requirements to responsibilities ✓
- Asaniczka archetype-template enrichment is negligible (1.11×) ✓
- Arshkon-only entry-share flip is NOT driven by denominator drift ✓ (pct_unknown 51.7% vs 53.4%, stable)

**Gate 2 clears** with the 6 corrections above applied, **conditional on Wave 3 T21 and T22 rebuilding the management and scope patterns from scratch with precision validation before reporting any numbers.**

Wave 3 can proceed. Agent L (T21) and Agent M (T22) must internalize corrections 2 and 3. Agent N (T24-T26 synthesis) must internalize corrections 1, 4, and 5 when writing SYNTHESIS.md. The paper draft must not cite the SNR 925 with the broad 5.15→28.63 rates.
