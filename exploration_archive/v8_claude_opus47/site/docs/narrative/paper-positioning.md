# Paper positioning

**Recommendation: hybrid dataset + substantive empirical labor paper, with "SWE-specific AI rewriting" as the substantive lead.**

## Three contributions stacked

### 1. A longitudinal SWE posting dataset

Harmonizes three sources:

- Kaggle arshkon (2024-04, 4,691 SWE rows with entry labels)
- Kaggle asaniczka (2024-01, 18,129 SWE rows)
- Scraped EC2 (2026-03 onward, 40,881 SWE rows growing)

Plus **155,745 control-occupation rows** for DiD. 34,102 rows carry LLM-cleaned text via Stage 9.

The preprocessing pipeline is documented reproducibly (see [preprocessing](../methodology/preprocessing.md)); all code and schema are checked into the repo.

### 2. A measurement framework for the LLM era

Six elements:

- **T30 seniority panel** — 4 junior × 4 senior definitions × 2 baselines, reported on every seniority-stratified headline.
- **Semantic-precision protocol** — 50-sample, 80% floor, drop failing sub-terms.
- **DiD vs control occupations** — for every "SWE is doing X" claim, test against adjacent and control with bootstrap CI.
- **Length residualization** — for any composite with component r > 0.3 against length.
- **Composite-score correlation check** — prevents length from masquerading as content.
- **LLM-authorship mediation test** (T29) — measures how much apparent change is recruiter-LLM-drafted text.

Usable by any future researcher working on LLM-era posting data.

### 3. Three substantive findings

Ordered by evidence strength:

1. **SWE-specific AI-vocabulary rewriting** (99% DiD) — cross-seniority, cross-archetype (20/22 positive strict), within-company (102%), geographically uniform (26/26 metros).
2. **RQ3 inversion: employers under-specify AI by 15-30 pp vs workers.**
3. **Senior-disproportionate role shift toward mentoring + orchestration + AI-deployment**, with an emergent mgmt+orch+strat+AI sub-archetype.

## Venue recommendations

| Venue | Fit | Why |
|---|---|---|
| **ICWSM / WWW / CSCW** | Primary recommended | Methodological + hybrid dataset positioning lands; social computation audience |
| **Labor economics (ILR Review, JOLE, BE Journal)** | Strong | RQ3 inversion is publishable standalone |
| **CHI / Management Science** | Secondary | The emergent senior role type (H_C, T21 cluster 2) could anchor a specific paper on tech-lead work |
| **Nature Human Behaviour / PNAS** | Aspirational | Cross-occupation extension (H_A) could elevate the paper if implemented |

**Recommended lead submission:** ICWSM dataset & methods track, with a separate labor-economics follow-up. The RQ3 inversion can anchor either a short companion paper or a dedicated section of the main manuscript.

## What the paper must NOT claim

- "Junior share rose" as an unqualified headline — direction is baseline-contingent (SNR < 1).
- "Juniors now look like 2024 seniors" — T12 relabeling diagnostic rejected this; T20 sharpened not blurred boundaries.
- "Employers are hiring more X" — 2026 is a JOLTS Info-sector hiring low; all claims share-of-SWE, not volume.
- "The requirements section grew" — it shrank. Narrative sections grew.
- "AI is a junior-filter" — seniors more AI-specified than juniors.
- "180× period vs seniority in embeddings" — corrected by V1 to ~1.2× centroid-pairwise.

## What the paper SHOULD do

- **Lead with SWE-specificity.** DiD 99%. Not "everyone is writing like this now."
- **Stratify by domain before seniority.** NMI 8.6× domain/period. ML/AI is the fastest-growing archetype; seniority effects are conditional on archetype.
- **Report arshkon-only baseline as primary; pooled as sensitivity.** Asaniczka's 0 entry labels make pooled-2024 structurally broken for senior-side claims.
- **Cite the T29 mediation explicitly.** "15-30% of the rewrite is recruiter-LLM-mediated" is a methodological contribution, not a reframe.
- **Address sampling-frame artifact head-on.** 74.5% new entrants in scraped — every longitudinal claim needs returning-cos-only sensitivity.

## Lead sentence (to open the paper)

> "Between 2024 and 2026, US LinkedIn software-engineering job postings added AI-tool vocabulary at a rate 99 percent specific to SWE versus control occupations (DiD), 102 percent attributable to the same 240 companies rewriting their own postings, while employers simultaneously under-specified AI by 15 to 30 percentage points relative to the most conservative developer-usage benchmark — a cross-seniority, cross-archetype, geographically uniform rewrite that coexists with a senior-disproportionate role-content shift toward mentoring and orchestration and sharpening rather than blurring of seniority boundaries."

## Closing framing recommendation

The paper's intellectual contribution is the intersection of three observations that only an employer-side posting dataset can reveal together:

1. **Employers rewrote postings, but trailed workers.** (RQ3 inversion)
2. **The rewrite was not junior-scope-inflation but senior-role-transformation.** (T21)
3. **Inside this rewrite, seniority became easier to tell apart, not harder.** (T20)

These three together reject the dominant narrative of AI "hollowing out the rung" and replace it with a more interesting story: **AI coincided with seniors becoming explicit team-multipliers, while juniors remained recognizably junior but with the tools now named.**
</content>
</invoke>