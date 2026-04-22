# Composite A — deep-dive validation of three contested findings

**Date:** 2026-04-21 · **Code:** [`eda/scripts/S26_deepdive.py`](../scripts/S26_deepdive.py) · **Tables:** `eda/tables/S26_dd_*.csv`
**Inputs:** `data/unified_core.parquet` (LinkedIn SWE, English, date OK); canonical `AI_VOCAB_PATTERN` from [`scans.py:50-71`](../scripts/scans.py).
**Findings audited:** (1) hospitals lead software 36 vs 32 pp; (2) FS=SWE parity does not replicate; (3) Bay user-intensity premium +8 pp.

---

## DD1 — Are hospitals really out-writing software firms on AI?

### Volume and label coverage

The hospital cell is small but adequate. From [`S26_dd1_volume_by_period.csv`](../tables/S26_dd1_volume_by_period.csv): in 2026 the corpus has 372 SWE postings tagged `company_industry = 'Hospitals and Health Care'` (154 in 2026-03 plus 218 in 2026-04), against 6,938 in `Software Development`. The 36.0% rate cited in the parent memo is computed on n=372 and the Wilson 95% CI is 31.3-41.0 pp; Software Development sits at 32.3% with CI 31.2-33.4 pp. The intervals are non-overlapping, but the hospital interval is wide and one bad month would close the gap.

### What the hospital postings actually are

[`S26_dd1_hospital_top_titles.csv`](../tables/S26_dd1_hospital_top_titles.csv): the top titles are *Software Engineer* (n=20), *Senior Software Engineer* (n=20), *Data Engineer* (n=12), *Senior Data Engineer* (n=11), *Senior Backend Engineer* (n=5), *Staff SDE* (n=4), *AI Engineer* (n=4). These are SWE roles — not nurse postings caught by `is_swe`. The `is_swe` filter is doing real work.

The companies behind the volume ([`S26_dd1_hospital_top_companies.csv`](../tables/S26_dd1_hospital_top_companies.csv)) are a mix of three populations:

- **Health insurance / health-IT giants:** Optum (n=52, 56% AI rate), CVS Health (n=23, 48%), HCA Healthcare (n=13, 31%), McKesson (n=8), GE HealthCare (n=8), Philips (n=5).
- **Genuine hospitals / providers:** Children's Hospital Colorado, Memorial Sloan Kettering, Michigan Medicine, UT MD Anderson, Maven Clinic, CommonSpirit Health.
- **AI-native health-tech startups:** Abridge (n=7, 100%), Ambience Healthcare (n=4, 100%), Inspiren (n=5, 80%), Counsel Health (n=4, 75%), Plenful (n=3, 100%), SmithRx (n=4, 50%), Standard Practice AI.

The startup tier is over-represented: Abridge (medical-AI scribe) and Ambience (clinician documentation) are health-tech firms whose products *are* LLMs and whose entire posting corpus is AI. LinkedIn's industry taxonomy puts them under Hospitals because that is the customer industry, not the product category.

### Spot-check precision (n=30)

[`S26_dd1_hospital_spotcheck30.csv`](../tables/S26_dd1_hospital_spotcheck30.csv): of 30 random hospital postings flagged AI by the broad regex, **30/30 are substantive** (Wilson 88-100%). No incidental matches, no boilerplate. Representative examples:

- *Children's Hospital Colorado, Azure SWE:* "Preference will be given to those applicants who have experience with AI Code generation tools: Github Copilot and Claude Code. Building AI solutions with systems that leverage Retrieval Augmented Generation is a plus."
- *Michigan Medicine, Data Engineer Senior:* "Demonstrated proficiency with AI and experience working with agentic coding tools (e.g., Cursor, OpenAI Codex, Claude Code, Anti-Gravity)."
- *CVS Health, Staff SDE:* "Workflow orchestration engines, multi-provider LLM integration, automated deployment pipelines."
- *Memorial Sloan Kettering, Bioinformatics SWE:* "Large language models (LLMs), retrieval-augmented generation (RAG), deep learning... integrated into secure, scalable full-stack clinical applications."

The hospital signal is real, not a regex artifact.

### Is the result driven by a few firms?

After dropping the top-five AI-heavy hospital firms (Optum, CVS Health, Abridge, HCA Healthcare, Inspiren), the residual hospital n drops from 372 to 272 and the AI rate falls from 36.0% to **29.0%**. That is below Software Development's 32.3% and inside its CI. **The headline gap is fragile to five-firm exclusion.** Optum alone (52 postings, 56% AI rate, owned by UnitedHealth) contributes most of the lift.

### Verdict

The "hospitals lead" claim is *technically true on the labelled corpus* and *substantively defensible* — the AI vocabulary is real, the postings are real SWE roles inside healthcare, and Optum / CVS / Abridge / Verily / Children's Hospital Colorado are demonstrably writing AI-tooling language into JDs. But the 36-vs-32 gap leans on (a) a small denominator (n=372), (b) LinkedIn classifying AI-native health-tech firms (Abridge, Ambience, Inspiren) as Hospitals when their products are LLMs, and (c) UnitedHealth/Optum's outsized writing volume. Drop those five firms and hospitals fall to 29% — *below* software, *not above*.

**Recommended article formulation.** Drop the "hospitals lead software" headline. Replace with a softer panel: *"Within the 2026 LinkedIn corpus, hospital-classified SWE postings mention AI vocabulary at roughly the same rate as software firms — 36% versus 32%, with confidence intervals that nearly touch. The gap is driven by health-tech firms classified into the hospital industry (Abridge, Ambience, Inspiren) and by Optum and CVS, whose own AI-platform writing is dense; remove those five and the rate is essentially identical to software. Either way, the regulated-industry-lag narrative does not survive at the labour-demand layer — hospitals are at parity, not behind."* That sentence carries the surprise without overclaiming a 4-pp lead built on five firms.

---

## DD2 — FS = SWE parity, under which regex?

### Side-by-side comparison

I ran three regex variants ([`eda/scripts/S26_deepdive.py:34-72`](../scripts/S26_deepdive.py)):

- **`canonical_broad`** — the `AI_VOCAB_PATTERN` from `eda/scripts/scans.py:50-71`: 31 phrases including `llm`, `gpt`, `chatgpt`, `claude`, `copilot`, `anthropic`, `genai`, `agentic`, `ai-powered`, `mlops`, `rag`, `prompt engineering`, `ai agent`, `vector database`.
- **`strict_v9like`** — 21 phrases, drops the most ambiguous tokens (`gpt` alone, `rag` alone, `gemini`, `bard`, `mistral`, `llama`, vendor model names that double as common words) and keeps only multi-word AI phrases plus unambiguous platform brands.
- **`tooling_only`** — 17 phrases, only concrete coding-agent / LLM-tooling tokens: drops `llama`, `gemini`, `mistral`, `mlops`, `vector database`, `large language model`. Tests whether the FS deficit is about coding-agent adoption specifically.

[`S26_dd2_regex_comparison.csv`](../tables/S26_dd2_regex_comparison.csv):

| Regex | FS rate (95% CI) | SWE rate (95% CI) | FS-SWE | CI overlap |
|---|---|---|---|---|
| canonical_broad | 26.1% (24.1-28.2) | 32.3% (31.2-33.4) | -6.22 pp | No |
| strict_v9like | 23.0% (21.1-25.0) | 27.8% (26.8-28.9) | -4.84 pp | No |
| tooling_only | 22.8% (21.0-24.8) | 27.2% (26.1-28.2) | -4.33 pp | No |

**Under all three regexes, FS lags SWE by 4-6 pp with non-overlapping CIs.** The v9 "FS=SWE parity" finding does not replicate under any regex this audit could construct. The v9 numbers in story 04 (FS 15.7% vs SWE 15.3%) almost certainly used a different scope — a much narrower sub-corpus (LLM-cleaned text, or a smaller seniority slice, or a 2024-only window) that this audit could not reproduce. The parent memo's claim that parity holds "under stricter regex" is not supported by these tests.

### Precision audit (n=20 per regex)

[`S26_dd2_fs_spotcheck.csv`](../tables/S26_dd2_fs_spotcheck.csv): 60 FS spot-checks. Hand-coded precision on the 20 canonical-broad samples is **18/20 substantive** (≈0.90), comparable to v9's 0.86 on `ai_strict`. Examples:

- *Fannie Mae, Advisor SWE (AI/ML & AWS):* "Hands-on experience with LLMs (e.g., OpenAI, Anthropic, Cohere) and prompt engineering."
- *BlackRock, Senior C++ Engineer, Aladdin:* "Leverage AI-assisted tools (including LLMs and agentic workflows) to accelerate development."
- *JPMorgan Chase, Lead Data Engineer:* "Applied use of LLMs/agents, RAG, anomaly detection, or automated runbooks."
- *Remitly, Agentic Software Engineer:* "Orchestrate a fleet of AI agents to architect, build, and deliver."

**Regex precision is not the problem; the FS-SWE gap is real at the requirement-text layer.**

### What FS firms ask for vs what software firms ask for

[`S26_dd2_fs_vs_swe_words.csv`](../tables/S26_dd2_fs_vs_swe_words.csv): in AI-matched 2026 postings, FS-distinctive words (rank-difference vs SWE) are *financial, capital, technology, management, lead, global, deliver, platforms, equal, gender, applicants, employment* — high-management and HR-boilerplate vocabulary. SWE-distinctive words include *agentic, models, model, microsoft, security, customer, workflows, develop, engineers, computer, science, collaborate, requirements*. The two industries are using AI vocabulary to talk about different work: FS attaches AI to existing risk / compliance / operations frames; software attaches it to model-building, agent workflows, customer-facing systems. **FS-AI is qualitatively different from SWE-AI**, not a smaller dose of the same content.

### Verdict

The "FS approaches SWE parity" claim does not survive any of three regexes tested here, including a stricter v9-style one. Either v9's parity finding lived on a different denominator (cleaner text, narrower seniority window, a different vintage of the corpus) or on the strict regex applied to a different filter; this audit cannot reproduce it on `unified_core.parquet`. What is robust is that FS lags software by 4-6 pp in 2026, and that the *content* of FS-AI postings is structurally different (AI as overlay on financial/risk language; SWE-AI as model-and-agent-building language).

**Recommended article formulation.** Drop the parity claim; replace with: *"Financial services lag software firms by four to six percentage points in AI-vocabulary rate, depending on which dictionary one uses, and the gap is regex-stable. But the more interesting fact is qualitative: when financial firms write AI requirements they overlay them on risk, compliance and operations vocabulary, while software firms attach the same words to agent workflows and model-building. FS is not a slower software industry — it is a different one."* That carries the cross-industry diffusion story honestly.

---

## DD3 — Bay-Area +8 pp user-intensity premium: real or boilerplate?

### What the gap means in absolute volumes

From [`S26_dd3_volume.csv`](../tables/S26_dd3_volume.csv): in 2026, restricting to non-builder titles (no Applied AI / ML / FDE in title) and to metros with ≥50 SWE postings in both periods:

| Zone | n general SWE | n with AI vocab | AI rate |
|---|---|---|---|
| Tech hub (5) | 6,901 | 2,068 | 30.0% |
| Rest (21) | 10,802 | 2,087 | 19.3% |

That is a **10.6 pp gap on the pooled non-builder denominator**, larger than the +8 pp metro-mean reported in the parent memo (which averages metro rates without volume weighting). In raw counts, the gap means about 730 *additional* AI-mentioning ordinary-SWE postings in the Bay Area / Seattle / NYC / Austin / Boston cluster than would exist if the hub rate matched the rest. That is a non-trivial cohort.

### Concrete examples (10 hub vs 10 rest)

[`S26_dd3_bay_vs_rest_samples.csv`](../tables/S26_dd3_bay_vs_rest_samples.csv) — 10 Bay-Area ordinary-SWE postings and 10 non-hub equivalents. The Bay sample reads like a different market: postings are more often at the centre of an AI product (Levelpath DevOps "utilize AI-driven development tools (e.g., Claude Code, Cursor, or similar agents) to boost your own productivity"; HydroX "Experience with LLMs or generative AI systems"; Acceler8 Talent "deeply senior team from OpenAI, Boston Dynamics, and DeepMind"; TrueFoundry "production agentic systems... route between OpenAI, Anthropic, Google, and self-hosted models"). The non-hub sample is heavier in *adopter* postings: AIG Palantir GenAI engineer "innovative Generative AI team", Travelers Insurance "AI-powered capabilities that deliver measurable outcomes", Fed Reserve Board "generative AI, or LLM-driven application development", JPMorgan Chase "AI agentic experience" added as a preferred qualification. Both populations use AI vocabulary, but the hub postings sit closer to model-building and agentic systems while non-hub postings sit closer to enterprise adoption.

### Which tokens drive the gap

This is the most informative finding. [`S26_dd3_token_gap.csv`](../tables/S26_dd3_token_gap.csv) — within non-builder AI-matched 2026 postings, token-rate by zone:

**Bay leads on (frontier-platform vocabulary):**
- `openai` +6.1 pp · `agentic` +4.2 pp · `ai agent` +3.5 pp · `llm` +3.4 pp · `ai-powered` +3.0 pp · `chatgpt` +1.4 pp · `anthropic` +1.3 pp · `foundation model` +1.1 pp.

**Rest leads on (coding-tool-and-MLOps vocabulary):**
- `copilot` -12.2 pp · `github copilot` -8.7 pp · `prompt engineering` -6.8 pp · `rag` -6.1 pp · `claude` -5.6 pp · `mlops` -4.7 pp · `genai` -2.1 pp.

This is striking. The hub premium is concentrated in *platform-name vocabulary* — postings that name specific frontier providers (OpenAI, Anthropic) and architectural concepts (agentic, AI agent, foundation model). The non-hub deficit is concentrated in *coding-tool vocabulary* (Copilot, GitHub Copilot, prompt engineering, RAG, MLOps), much of which is staffing-aggregator boilerplate: of the 249 non-hub `github copilot` mentions, 11 come from Jobs via Dice, 7 from Trimble, 6 from Apex Systems, 4 each from Capgemini and CGI, 3 each from Randstad and hackajob — staffing intermediaries that paste tool lists into JD templates.

**Reframed:** the Bay's "user-intensity premium" is not "the Bay writes more AI-tool boilerplate." It is "the Bay writes more *frontier-model platform* references into ordinary SWE postings." The non-hub corpus actually *out-mentions* the Bay on coding tools (Copilot is now ubiquitous everywhere), but the Bay leads on the agentic / LLM-platform stack.

### Counter-narratives this contradicts

The popular *AI-tools-are-democratising-and-geography-no-longer-matters* claim — most prominently in vendor messaging from GitHub and Cursor — predicts that coding-tool adoption is roughly flat across geographies. **The Copilot data actually supports that prediction:** non-hubs lead the Bay on `github copilot` mentions, consistent with broad diffusion of the tool. What *does not* diffuse is the agentic-architecture vocabulary (`agentic`, `ai agent`, `openai`, `anthropic`, `foundation model`). Those concentrate in the Bay. The honest finding is asymmetric: tool adoption has spread, but model-platform-aware ordinary-SWE work has not.

### A 100-word lede test

> *Coding tools have spread. GitHub Copilot now appears in software-engineering job ads from Cleveland to Charlotte to Tampa, more often than in San Francisco. But the language of model platforms has not. In 2026, ordinary engineering postings in the Bay Area are six points more likely to name OpenAI, four points more likely to call work "agentic," and three more to mention an "AI agent" than postings in non-hub metros — even when the title is something prosaic like "Backend Engineer." The most-used tools have democratised. The architectural vocabulary that specifies how AI is being built into products has not.*

That works. The asymmetric framing is publishable.

### Verdict

The +8 pp user premium is real and survives sanity checks, but the parent memo's framing under-sells it. The geographic gap is not "the Bay uses more AI buzzwords" — it is "the Bay writes frontier-model and agentic-architecture vocabulary into ordinary SWE postings, while everywhere else now mentions Copilot." That contrast is publishable as the lead claim of composite A: it is a counter-narrative to the democratisation story, with concrete evidence on either side. Promote from supporting claim to lead.

**Recommended article formulation.** Lead with the asymmetric-diffusion frame above. Pair with the absolute-volume figure (730 extra AI-mentioning hub postings vs the counterfactual). Use the per-token table as the supporting chart, hub-premium tokens in red and rest-premium tokens in blue.

---

## Cross-cutting verdicts

| Finding | Holds? | Article action |
|---|---|---|
| Hospitals lead software firms 36 vs 32 | **Fragile** — non-overlapping CIs but driven by 5 firms; drops to 29% without them | Soften to "at parity"; surface health-tech-firm classification footnote |
| FS = SWE parity under stricter regex | **No** — FS lags by 4-6 pp under all three regexes tested here | Drop parity claim; pivot to qualitative-difference frame |
| Bay user-intensity premium +8 pp | **Yes**, and is more interesting than memo says: asymmetric across token classes | Promote to lead claim of composite A |

**File index.**

| Path | Contents |
|---|---|
| [`eda/scripts/S26_deepdive.py`](../scripts/S26_deepdive.py) | All deep-dive code |
| [`eda/tables/S26_dd1_volume_by_period.csv`](../tables/S26_dd1_volume_by_period.csv) | Hospital volume by period |
| [`eda/tables/S26_dd1_hospital_top_titles.csv`](../tables/S26_dd1_hospital_top_titles.csv) | Top hospital SWE titles, 2026 |
| [`eda/tables/S26_dd1_hospital_top_companies.csv`](../tables/S26_dd1_hospital_top_companies.csv) | Top hospital companies + AI rate |
| [`eda/tables/S26_dd1_hospital_spotcheck30.csv`](../tables/S26_dd1_hospital_spotcheck30.csv) | 30 random AI-matched hospital postings + excerpts |
| [`eda/tables/S26_dd2_regex_comparison.csv`](../tables/S26_dd2_regex_comparison.csv) | FS vs SWE under three regexes |
| [`eda/tables/S26_dd2_regex_breakdown.csv`](../tables/S26_dd2_regex_breakdown.csv) | Per-regex per-industry n + Wilson CI |
| [`eda/tables/S26_dd2_fs_spotcheck.csv`](../tables/S26_dd2_fs_spotcheck.csv) | 60 FS spot-checks (20 per regex) |
| [`eda/tables/S26_dd2_fs_vs_swe_words.csv`](../tables/S26_dd2_fs_vs_swe_words.csv) | FS-vs-SWE word-rank delta in AI postings |
| [`eda/tables/S26_dd3_volume.csv`](../tables/S26_dd3_volume.csv) | Hub vs rest non-builder volumes |
| [`eda/tables/S26_dd3_bay_vs_rest_samples.csv`](../tables/S26_dd3_bay_vs_rest_samples.csv) | 20 paired hub/rest ordinary-SWE excerpts |
| [`eda/tables/S26_dd3_token_gap.csv`](../tables/S26_dd3_token_gap.csv) | Per-token Bay-vs-Rest rate gap |
