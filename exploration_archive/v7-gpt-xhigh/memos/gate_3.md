# Gate 3 Research Memo

## What We Learned

1. **AI/tool expansion is mostly within returning companies.** In 240 arshkon-to-scraped overlap companies, broad-AI prevalence rose from 3.74% to 23.23% and AI-tool-specific prevalence from 2.23% to 19.80%. T16 decomposes 91-92% of the absolute AI/tool change as within-company rather than between-company reweighting.

2. **The AI/tool shift is geographically broad.** T17 finds broad-AI and AI-tool prevalence increased in all 26 eligible metros. Tech hubs have somewhat larger changes, but every eligible metro moved upward and metro AI growth correlates with requirement-breadth growth rather than entry/low-YOE share change.

3. **The shift is SWE-amplified, not SWE-exclusive.** T18 shows SWE and SWE-adjacent roles move in parallel on AI/tool, bounded tech count, and requirement breadth, while controls rise only modestly. This weakens a paper framed as uniquely SWE restructuring, but strengthens a broader technical-work restructuring frame.

4. **Junior-share evidence remains construct-dependent.** T16 common-company arshkon-only panel shows J1/J2 explicit entry labels down, while J3/J4 low-YOE shares rise. T17 pooled geography and T18 cross-occupation panels show J1-J4 up, but arshkon-only geography flips J1/J2 negative. The strongest seniority result is still label-vs-YOE disagreement, not junior disappearance.

5. **Several critical validity checks remain unrun.** T19, T20/T21, and T22/T23 failed after retries. We do not yet have temporal-rate stability, seniority-boundary modeling, ghost/aspirational forensics, or worker-usage divergence evidence. Those omissions constrain what the paper can claim now.

## What Surprised Us

- **Common firms changed their AI/tool language.** The AI/tool increase is not mainly new AI-native employers entering the scraped sample. New 2026 entrants are somewhat more AI-heavy, but returning firms also moved sharply.
- **SWE-adjacent roles moved almost as much as SWE.** T18 finds adjacent roles rose from 2.1% to 21.2% AI-tool prevalence, close to SWE's 2.2% to 24.0%. Data scientist/ML and data engineering are especially close to SWE on the measured requirement surface.
- **Geography is not the mechanism.** The diffusion is too broad to narrate as a Bay Area/Seattle/NYC phenomenon. Tampa Bay and Salt Lake City rank among the largest increases.
- **Remote work fields are unusable.** T17 reports `is_remote_inferred` is zero in scraped 2026 as well as 2024, so remote work cannot be analyzed without a scraper/location-stage audit.
- **The failed tasks are substantively costly.** Without T22/T23, we cannot yet say whether AI/platform/workflow requirements are screened, aspirational, ghost/template language, or divergent from worker usage.

## Evidence Assessment

| Finding | Strength | Sample / n | Calibration / sensitivity | Main confounds | Interpretation |
|---|---|---:|---|---|---|
| Within-company AI/tool expansion | Strong | 240 arshkon-scraped overlap companies; 40,881 scraped SWE rows | Aggregator exclusion stable; uses V1/T14 validated tech indicators | Company canonicalization; raw binary AI can include context not requirements | Core mechanism finding |
| Within-company requirement breadth expansion | Moderate-strong | 223 overlap companies with LLM text | Direction stable; LLM subset only | Scraped LLM text coverage about 31%; regex breadth | Strong support, but label as LLM-cleaned subset |
| Geographic AI/tool diffusion | Strong | 26 metros; 13,454 pooled-2024 and 30,560 scraped-2026 metro rows | Positive in all metros; aggregator/company cap stable | Metro coverage/source composition; multi-location exclusions | Good supporting figure |
| Metro requirement breadth diffusion | Moderate | 12,799 2024 and 9,198 2026 LLM-text metro rows | Positive in all metros | LLM subset and metro composition | Supports breadth/AI co-movement |
| SWE-control AI/tool DiD | Strong | 242,854 cross-occupation rows; SWE n 63,701 | Aggregator, SWE-tier, company-cap sensitivities stable; within-2024 calibration favorable | Occupation classification; raw AI mentions not force-coded | Reframes as technical-work expansion |
| SWE-adjacent parallel movement | Strong | Adjacent 11,816 2024 and 11,592 2026 rows | Same direction as SWE; controls much smaller | Adjacent category heterogeneous | Important narrative correction |
| Generic SWE-adjacent convergence | Weak/rejected | 800-row TF-IDF sample | Centroid similarity falls, pairwise rises slightly | Sample-based, text subset | Do not claim occupation collapse |
| Junior-share decline | Weak/mixed | T16/T17/T18 panels disagree by construct/source | Fails T30 unanimity and source stability | J1/J2 labels vs J3/J4 YOE constructs | Use as boundary ambiguity, not lead |
| Ghost/aspirational validity | Missing | T22 failed | Not assessed | Unknown requirement force | Major gap before causal hiring-bar claims |
| Employer/worker usage divergence | Missing | T23 failed | Not assessed | Benchmark quality unknown | RQ3 cannot be lead yet |

## Seniority Panel

### Common-Company Junior Share, Arshkon To Scraped

T16 is the cleanest within-company seniority test. It is **split by construct**, so it cannot support a lead claim that junior roles rose or fell.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | `seniority_final = 'entry'` | -0.44 pp | Down | 240 companies | Split |
| J2 | entry/associate | -0.53 pp | Down | 240 companies | Split |
| J3 | `yoe_extracted <= 2` | +6.63 pp | Up | 222 companies with YOE-known rows | Split |
| J4 | `yoe_extracted <= 3` | +10.32 pp | Up | 222 companies with YOE-known rows | Split |

Interpretation: explicit entry labels and low-YOE floors are measuring different market phenomena. This is a finding, not a nuisance.

### Metro-Level Junior Change, Pooled 2024 To Scraped

T17 pooled geography shows mostly upward junior/low-YOE shares, but arshkon-only source restriction turns J1/J2 negative. Cite only with source caveats.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | entry label | +1.12 pp mean metro change | Up | 26 metros | 4-of-4 pooled, source-sensitive |
| J2 | entry/associate | +1.06 pp | Up | 26 metros | 4-of-4 pooled, source-sensitive |
| J3 | YOE <= 2 | +4.95 pp | Up | YOE-known metro rows | 4-of-4 pooled |
| J4 | YOE <= 3 | +6.47 pp | Up | YOE-known metro rows | 4-of-4 pooled |

Interpretation: pooled geography reinforces that junior disappearance is not robust. The arshkon-only reversal reinforces source/instrument caution.

### Cross-Occupation SWE Junior And Senior Shares

T18 finds SWE junior variants all rise, while senior variants are mixed.

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| J1 | entry label | +1.46 pp | Up | SWE default frame | Unanimous junior up |
| J2 | entry/associate | +1.41 pp | Up | SWE default frame | Unanimous junior up |
| J3 | YOE <= 2 | +5.98 pp | Up | YOE-known denominator | Unanimous junior up |
| J4 | YOE <= 3 | +8.12 pp | Up | YOE-known denominator | Unanimous junior up |

| Variant | Definition | Effect | Direction | n basis | Agreement verdict |
|---|---|---:|---|---:|---|
| S1 | mid-senior/director | -16.25 pp | Down | SWE default frame | Mixed |
| S2 | director | +0.26 pp | Up | sparse | Mixed |
| S3 | senior title regex | -14.42 pp | Down | diagnostic | Mixed |
| S4 | YOE >= 5 | -7.50 pp | Down | YOE-known denominator | 3-of-4 down, director exception |

Interpretation: cross-occupation evidence again rejects simple junior decline. Senior results are not clean enough to lead without T20/T21 mechanism work.

## Narrative Evaluation

- **RQ1: Employer-side restructuring. Reframed and partly strengthened.** The strongest employer-side change is within-company AI/tool and skill-breadth expansion, not junior-volume decline. Junior share remains mixed and construct-dependent.
- **RQ2: Task and requirement migration. Strengthened but broadened.** T16/T17/T18 support AI/tool/platform/workflow expansion across companies, metros, and technical occupations. This is not simple downward migration from senior to junior roles.
- **RQ3: Employer-requirement / worker-usage divergence. Still untested.** T23 failed. Posting-side AI requirements are strong, but the divergence claim needs external usage benchmarks and force-of-requirement parsing.
- **RQ4: Mechanisms. More important than before, but quantitatively unresolved.** T22 failed, so interviews and future analysis must distinguish screened requirements from aspirational/template/ghost language.

Original narrative status:

- Junior scope inflation: **supported for breadth**, but ghost/force validity unresolved.
- Junior share decline: **contradicted or weakened**; do not lead with it.
- Senior archetype shift: **insufficient Wave 3 evidence**; Wave 2 suggests complexity/mentorship rather than management decline, but T21 failed.
- Employer-usage divergence: **pending/missing**.

Alternative framings considered:

1. **SWE-specific restructuring.** Attractive because the dataset is SWE-centered, but T18 shows adjacent roles move almost as much as SWE. This framing overclaims uniqueness.
2. **Technical-work skill-surface expansion.** Best supported: SWE and adjacent technical roles absorb AI/tool/platform requirements much faster than controls; within-company and geographic evidence shows diffusion rather than isolated composition.
3. **Posting-platform/template inflation.** Still plausible because T22 failed and raw/LLM coverage differs, but T13/V1/T16 argue the signal is not only boilerplate or new-company mix. Treat as a validity threat, not the lead narrative.
4. **Junior-labor collapse.** Not supported. Direction flips by seniority construct and source.

Preferred current framing: **AI-era skill-surface expansion across software-producing technical work, with seniority labels becoming less reliable measurement anchors.**

## Emerging Narrative

The paper is becoming less about "AI eliminated junior SWE roles" and more about **how AI-era requirements diffused across the technical labor market**. SWE postings changed, but SWE-adjacent roles changed too. The novel contribution is measuring where the expansion appears: within returning firms, across nearly all metros, across SWE and adjacent technical occupations, and across seniority definitions for requirement breadth.

Seniority still matters, but as a measurement problem and boundary phenomenon. Explicit entry labels, associate labels, YOE floors, and semantic role content do not agree reliably. The paper should foreground that ambiguity instead of forcing a single junior-share estimate.

## Research Question Evolution

Recommended RQs after Gate 3:

- **RQ1 revised:** How much did AI/tool/platform and requirement-breadth language expand in SWE postings from 2024 to 2026, and is the expansion within-company, geographic, and cross-occupation robust?
- **RQ2 revised:** Is this expansion uniquely SWE-specific or shared across software-adjacent technical occupations relative to controls?
- **RQ3 revised:** How do seniority labels, YOE floors, and requirement profiles diverge, and what does that imply for measuring junior/senior labor-market restructuring?
- **RQ4 retained but demoted until evidence exists:** Are AI/tool requirements screened requirements, aspirational/ghost/template language, or divergent from worker usage?
- **RQ5 qualitative:** How do recruiters, hiring managers, and engineers explain broadened AI/platform/workflow requirements?

Changes since Gate 2: cross-occupation specificity becomes central, while employer/worker usage divergence and senior-role archetype claims should be demoted until T23/T21-style evidence exists.

## Gaps And Weaknesses

- **T19 missing:** no temporal-rate, scraped-window stability, posting-age, or backlog-vs-flow evidence. We should not make precise annualized rate claims.
- **T20/T21 missing:** no boundary classifier, associate-distance, or senior-role deep dive. Use Wave 2/T16 seniority evidence cautiously.
- **T22 missing:** no ghost/aspirational analysis. Requirement breadth and AI/tool prevalence remain posting-language claims, not proven hiring-bar claims.
- **T23 missing:** no external usage benchmark comparison. RQ3 remains a hypothesis.
- **T28/T29 not run:** no domain-stratified decomposition beyond T16/T17's limited archetype checks and no LLM-authorship artifact test.
- **LLM-cleaned subset coverage:** scraped LLM-text coverage remains about 31%, so breadth/semantic/archetype claims need careful denominator language.
- **Agent reliability:** after OOM recovery, several Wave 3 agents stalled without compute. Future runs should use smaller single-task prompts, explicit wall-clock limits, and maybe prewritten scripts rather than broad multi-task delegation.

## Direction For Next Wave

Before Wave 4 synthesis, the highest-value recovery work is not another broad wave. It is targeted validation:

1. **Recover T22 first** with a hand-scoped script or a single-task agent. This is the most important missing validity check.
2. **Run T28 domain decomposition** if compute permits. It directly sharpens the current lead story by separating within-domain expansion from domain recomposition.
3. **Run a lightweight T19** only for scraped-window stability and source date ranges, not elaborate rate modeling.
4. **Demote T21/T23** unless a smaller prompt can finish quickly; neither should block synthesis.

For Wave 4 Agent N, provide this memo and instruct it to treat T22/T23/T20/T19 as explicit gaps, not as findings to infer.

## Current Paper Positioning

If stopped here, the strongest paper is an **empirical posting-content restructuring paper about AI-era skill-surface expansion in software-producing technical work**. It has a measurement contribution around seniority operationalization and source calibration.

Provisional core argument:

> Between 2024 and 2026, SWE and software-adjacent postings did not show a robust collapse of junior demand. Instead, postings broadened around AI/tool/platform requirements across firms, metros, and technical occupations, while seniority labels and YOE floors diverged as measurement signals.

Ranked findings by strength, novelty, and narrative value:

1. **Within-company AI/tool expansion**: strong evidence, high narrative value, likely lead figure.
2. **SWE-amplified but adjacent-parallel expansion**: strong evidence, high reframing value.
3. **Geographically broad diffusion**: strong supporting evidence, useful figure.
4. **Requirement breadth/tech breadth expansion across seniority**: strong from Wave 2 and reinforced by T16/T17, but LLM-subset caveat.
5. **Seniority label-vs-YOE divergence**: moderate evidence, high conceptual value, needs T20-style mechanism work.

Five likely paper figures/tables:

1. Within-company decomposition of AI-tool, tech breadth, requirement breadth, and J1/J3 shares.
2. Cross-occupation DiD table or slope plot for SWE, adjacent, control.
3. Metro heatmap showing all-metro AI/tool increases.
4. T30 seniority panel showing J1/J2 vs J3/J4 divergence.
5. Requirement/tech breadth expansion with LLM coverage and source calibration caveats.

