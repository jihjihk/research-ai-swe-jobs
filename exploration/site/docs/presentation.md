---
marp: true
theme: default
paginate: true
header: 'SWE Labor Market Exploration'
footer: 'April 2026'
style: |
  section { font-size: 24px; }
  h1 { font-size: 36px; }
---

# The SWE Labor Market Restructured Between 2024 and 2026 -- Through Domain Recomposition, Not Seniority Migration

**Exploration findings from 52K LinkedIn postings**

April 2026

---

# The question: How did AI coding tools change what employers ask of software engineers?

Between 2024 and 2026, AI coding tools went from novelty to near-universal developer adoption (~75% usage). We analyzed ~52K SWE job postings and ~142K control-occupation postings from LinkedIn to measure what changed on the employer side.

Four research questions:
- **RQ1:** Which SWE labor demand changes are SWE-specific vs field-wide?
- **RQ2:** Are AI skill requirements additive or substitutive?
- **RQ3:** Do employer AI requirements lag or lead developer usage?
- **RQ4:** How do practitioners explain the restructuring?

---

# Why it matters: The dominant narrative ("AI replaces junior devs") has no within-firm support

Public discourse assumes AI coding tools directly cause junior developer elimination. Our data tells a more nuanced story: AI adoption and junior hiring changes are **parallel market trends, not causally linked within organizations**. Getting this right matters for workforce policy, CS education, and hiring strategy.

---

# Data: Three LinkedIn snapshots spanning the AI adoption window

| Source | Period | SWE rows | Entry-level | Key strength |
|--------|--------|----------|-------------|--------------|
| Kaggle arshkon | Apr 2024 | 5,019 | 769 | Native entry labels |
| Kaggle asaniczka | Jan 2024 | 23,213 | -- | Volume (no entry labels) |
| Scraped | Mar 2026 | 24,095 | 3,255 | Fresh, search metadata |

- **Control sample:** ~142K non-SWE postings for difference-in-differences
- **Company overlap panel:** 451 firms with 3+ SWE postings in both periods
- All LinkedIn platform, English language, validated date ranges

---

# Method: Cross-occupation DiD with within-firm decomposition and self-correction

- **Primary design:** Difference-in-differences (SWE vs control occupations)
- **Within-firm check:** 451-company overlap panel with shift-share decomposition
- **Validation:** Within-2024 calibration (arshkon vs asaniczka as noise floor)
- **Self-correction:** Wave 3 overturned our own Wave 2 headline (management +31pp became +4-10pp)
- **26 exploration tasks** across 4 waves, with gate memos tracking narrative evolution

---

# AI competency requirements surged +24pp (DiD), validated as genuine hiring requirements

- SWE AI mention rate: 8% (2024) to 33% (2026); control: 1% to 3%
- **DiD = +24.4pp** -- massively SWE-specific
- Entry-level AI requirements: 3.9% to 27.5%
- AI requirements are **less hedged** than traditional requirements in 2026 (20% vs 30% hedge fraction) -- reversed from 2024
- Specific tools: LLM 10%, Copilot 4%, Claude 4%

[Deep dive: AI Requirements Surge](./findings/ai-requirements.md)

---

# Entry-level SWE share declined while control occupations hired more juniors (DiD = -25pp)

- SWE entry share: 22.3% to 14.0% (seniority_native, arshkon-only baseline)
- Control junior share **increased** -- opposite direction
- **DiD = -24.9pp**
- Within-company decline (-11.8pp) exceeds aggregate -- composition dampens the signal
- YOE slot purification: 5+ YOE "entry-level" postings nearly vanished (22.8% to 2.4%)

[Deep dive: Junior Hiring Decline](./findings/junior-decline.md)

---

# The SWE domain landscape recomposed: ML/AI grew from 4% to 27%, Frontend contracted from 41% to 24%

- 14 posting archetypes identified via BERTopic (method-robust: ARI >= 0.996)
- ML/AI Engineering: 4% to 27% (+22pp) -- the largest structural shift
- Frontend/Web: 41% to 24% (-17pp)
- Data/Analytics: stable at ~21%
- **Domain NMI is 10x seniority NMI** -- technology domain structures the market more than seniority level

[Deep dive: Domain Recomposition](./findings/domain-recomposition.md)

---

# AI is additive to technology stacks: AI-mentioning postings require 56% more technologies

- AI-mentioning postings: 11.4 technologies vs 7.3 for non-AI postings
- Overall stack diversity: 6.2 to 8.3 mean techs/posting
- A new 25-technology AI/ML community emerged (LangChain, RAG, vector DBs, agent frameworks)
- **61 technologies rising, 12 declining, 73 stable** (FDR-corrected)
- Python became majority: 34.6% to 50.1%

[Deep dive: Technology Ecosystem](./findings/technology-expansion.md)

---

# YOE requirements purified: 5+ YOE "entry-level" postings dropped from 22.8% to 2.4%

- Median entry-level YOE: 3.0 (2024) to 2.0 (2026)
- Confirmed across 5 independent decompositions (title, company, aggregator, domain, geography)
- Fewer junior slots, but the surviving ones are more genuinely junior
- "Years experience" bigram dropped from 5.6 to 2.1 per 1K characters
- Entry-level 2026 postings are MORE readable (FK 15.3 vs 16.5)

[Deep dive: Junior Hiring Decline](./findings/junior-decline.md)

---

# AI adoption and junior decline are orthogonal: r = -0.07 (firm), r = -0.04 (metro)

- Companies that went AI-forward did NOT systematically cut junior roles
- All 26 metros show same direction on all five metrics -- no counter-trend metro
- **The paper cannot claim AI caused junior elimination within firms**
- These are parallel market trends operating through separate mechanisms
- The orthogonality puzzle is the most important open question for interviews

[Deep dive: The Orthogonality Puzzle](./findings/orthogonality.md)

---

# 57% of aggregate change is compositional -- different companies posting, not same companies changing

- T16 shift-share decomposition on 451-company overlap panel
- New market entrants: 24.3% AI rate (vs 2.5% for companies that exited)
- Only 18% of companies overlap between periods
- Four company clusters: Stable Traditional (55%), AI Transformers (11%), New AI-Native (22%), Declining Legacy (12%)
- **Within-firm changes are real but account for only 43% of what we observe**

[Deep dive: The Orthogonality Puzzle](./findings/orthogonality.md)

---

# Senior roles shifted toward technical orchestration (+46% at director level)

- Management language expanded at ALL levels -- not migration from senior to entry
- The distinctive senior change: **orchestration surged** (+16% mid-senior, +46% director)
- AI-mentioning senior postings have higher orchestration and lower management density than non-AI senior postings
- Director/mid-senior boundary blurred (AUC 0.75 to 0.64)
- Associate collapsing toward entry (relative position 0.30 to 0.16) -- de facto 3-tier system emerging

---

# What we corrected: The +31pp management indicator was measurement error, corrected to +4-10pp

- `\bleading\b`: 33% match rate, but 99.4% were adjective usage ("a leading company")
- `\bcross-functional\b`: 19% match rate, 84% were collaboration not management
- `\bleadership\b`: 12% match rate, 77% were company boilerplate
- **Corrected entry-level management increase: +4-10pp** (validated pattern set)
- Management expansion is also **field-wide** (DiD ~ 0), not SWE-specific

[Deep dive: Corrections & Revisions](./findings/corrections.md)

---

# What we corrected: Soft skills expansion and semantic convergence also failed scrutiny

- **Soft skills:** SWE grew LESS than control occupations (DiD = -5.1pp) -- not SWE-specific
- **Junior-senior convergence:** Within-2024 calibration shift exceeds cross-period change. Fails noise floor test.
- **Requirements vs usage direction:** Requirements LAG usage (~41% vs ~75%), not the other way around. Gap narrowing from -45pp to -34pp.
- **Management migration:** Did not migrate from senior to entry. Expanded at ALL levels.

[Deep dive: Corrections & Revisions](./findings/corrections.md)

---

# What the paper can claim: Three SWE-specific restructuring mechanisms operating in parallel

1. **AI competency requirements surged** (+24pp DiD), validated as genuine
2. **Entry-level SWE share declined** while control junior share increased (DiD = -25pp), but orthogonal to AI at firm level
3. **Domain landscape recomposed**: ML/AI 4% to 27%, Frontend 41% to 24%

Supporting: AI is additive to stacks, YOE purification, senior orchestration shift, GenAI acceleration 8.3x

---

# What the paper cannot claim

- That AI caused junior elimination within firms (r ~ 0 at firm and metro level)
- That management scope inflation is dramatic or SWE-specific (+4-10pp, field-wide)
- That junior and senior postings are semantically converging (fails calibration)
- That posting requirements outpace developer usage (they lag by ~34pp)
- That soft skills expansion is SWE-specific (SWE grew less than control)
- That the aggregate trend reflects within-firm behavior (57% is compositional)

---

# Alternative narratives the data cannot rule out

1. **HR template modernization:** Management/leadership language expansion may reflect changes in how postings are written, not changes in actual roles
2. **Compositional story:** The "restructuring" is partly about different companies entering the market (AI-native startups) and others exiting
3. **Cyclical hiring:** Junior share decline could reflect macroeconomic hiring slowdown, not AI-driven restructuring
4. **Time-lag causation:** Two snapshots cannot distinguish gradual trends from abrupt shifts; the AI-entry link might exist with a lag our data cannot capture

---

# Risk 1: Seniority operationalization may reverse the entry-level trend direction

- `seniority_native` (arshkon only): 22.3% to 14.0% = **decline**
- `seniority_3level` in overlap panel: appeared to show **increase** (3.4% to 13.5%)
- Resolution: `seniority_llm` (explicit-signal-only LLM classification) will be the canonical column
- **This is the binding constraint on RQ1**

---

# Risk 2: Text quality asymmetry and thin baselines

- **Text asymmetry:** 2024 Kaggle data has LLM-cleaned text; 2026 scraped has 0% LLM-cleaned. Rule-based cleaning inflates apparent text-based changes.
- **Thin arshkon baseline:** 769 native entry-level SWE (our only 2024 entry baseline)
- **Description length growth (57-67%):** The single largest effect (Cohen's d = 0.77). All text metrics must be length-normalized.
- **Two-point comparison:** Cannot distinguish gradual trend from abrupt shift with only 2024 and 2026 snapshots.

---

# Open question 1: What explains the orthogonality puzzle?

AI adoption surged and junior hiring declined simultaneously at the market level, but these trends are completely uncorrelated within individual companies.

Possible explanations:
- **(a)** Separate mechanisms operating in parallel
- **(b)** Market-level equilibrium effects invisible within firms
- **(c)** Time-lag causation our snapshots cannot capture
- **(d)** The trends are genuinely independent

**This is the most important question for practitioner interviews (RQ4).**

---

# Open question 2: Does domain recomposition drive the junior decline?

ML/AI engineering has a lower entry-level share AND grew from 4% to 27%. The compositional shift alone could account for much of the aggregate junior decline.

**H1 (most testable hypothesis):** Within-domain entry share decomposition would show whether junior roles declined within each domain or only in aggregate due to the domain shift.

**This is the first analysis the formal phase should run.**

---

# Next steps: From exploration to formal analysis

1. **Run domain-stratified entry share decomposition** (H1 -- may reframe entire junior decline story)
2. **Reproduce DiD with corrected management indicators** (strict T22 pattern set)
3. **Build formal robustness tables** for each core finding (5 sensitivity dimensions)
4. **Within-firm panel analysis** with company fixed effects (451 companies)
5. **Await seniority_llm** (resolves operationalization discrepancy -- highest priority data improvement)
6. **Conduct practitioner interviews** around the orthogonality puzzle, AI requirement reality, and domain recomposition experience

---

# Appendix: Exploration task map

| Wave | Tasks | Key contribution |
|------|-------|-----------------|
| Wave 1 (Foundation) | T01-T07 | Data quality, sample sizes, within-company decline, power analysis |
| Wave 2 (Discovery) | T08-T15 | YOE purification, domain recomposition, AI additive, scope inflation (later corrected) |
| Wave 3 (Dynamics) | T16-T23 | SWE-specificity confirmed, orthogonality discovered, management corrected, requirements lag usage |
| Wave 4 (Synthesis) | T24-T26 | New hypotheses, interview artifacts, this synthesis |

26 tasks, 4 gate memos, 95 figures, 1 self-correction. Full evidence base: [mkdocs site](./index.md)
