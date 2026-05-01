# Artifact 2 — Wells Fargo & Capital One same-title AI drift exemplars

**Source:** T31 pair-level drift analysis. Top-20 AI drift pairs.

## Exemplar 1: Wells Fargo "Lead Software Engineer"

**Pair statistics:**
- 2024 n: 4 postings | 2026 n: 17 postings (4.25× volume growth at same title)
- **Δ ai_strict_v1: +52.9 pp** (0% 2024 → 52.9% 2026) — LARGEST single-pair AI drift in the top-20
- Δ requirement_breadth_resid: +5.71 (scope broadened dramatically within-title)
- Δ desc_length: +1,520 chars (descriptions substantially longer)
- is_aggregator: False

The Wells Fargo "Lead Software Engineer" 2026 descriptions explicitly reference:
- Generative AI systems integration at bank-scale
- LLM-powered customer experience features
- AI-enabled fraud detection
- RAG architectures for internal knowledge systems

Of 17 Wells Fargo "Lead SWE" postings in 2026, **9 mention AI/LLM/RAG explicitly** — vs 0 of 4 in 2024.

## Exemplar 2: Wells Fargo "Senior Software Engineer"

**Pair statistics:**
- 2024 n: 5 postings | 2026 n: 20 postings (4× volume growth)
- **Δ ai_strict_v1: +35.0 pp**
- Δ requirement_breadth_resid: −0.76 (slight scope narrowing within-title — opposite of Lead SWE)
- Δ desc_length: +450 chars

At the "senior" (mid-senior ladder rung) level, Wells Fargo added AI content WITHOUT broadening scope — the AI language displaces rather than augments existing requirements. At the "lead" level (staff/senior-lead ladder rung), AI and scope both expanded together.

## Exemplar 3: Capital One "Senior Lead Software Engineer, Full Stack"

**Pair statistics:**
- 2024 n: 13 postings | 2026 n: 4 postings (3.25× volume CONTRACTION)
- Δ ai_strict_v1: +25.0 pp
- Δ requirement_breadth_resid: +1.79 (scope broadened)

Capital One's "Senior Lead SWE, Full Stack" 2026 postings feature:
- "Build with modern AI/ML patterns including RAG, vector databases, LLM integration"
- "Partner with AI platform teams on secure financial-services AI deployments"
- "Model risk management compliance integration"

Despite CONTRACTING posting volume at this title (13 → 4), Capital One's remaining postings are more AI-forward and broader in scope — consistent with T38's finding that hiring-volume-down is NOT linked to scope-broadening at the firm level, but individual firms may concentrate their reduced posting into higher-bar roles.

## Exemplar 4: Capital One "Lead Software Engineer"

**Pair statistics:**
- 2024 n: 6 postings | 2026 n: 5 postings (flat volume)
- Δ ai_strict_v1: +20.0 pp
- Δ desc_length: −124 chars (descriptions SHORTER)

Capital One "Lead SWE" postings become more AI-mentioning while getting slightly shorter — AI-adoption language is displacing older content, not accumulating on top.

---

## Cross-exemplar pattern

Wells Fargo and Capital One exemplars jointly demonstrate:

1. **Financial services leads the AI-rewriting signal at same-title pairs.** 4 of the top-10 AI-drift pairs are Wells Fargo + Capital One. T11 top-1% AI-saturated outliers (JPM, Citi, Wells, Visa, GEICO, Solera) confirm financial-services concentration.

2. **"Senior Lead" and "Lead" rungs are the most AI-rewritten.** Both firms show larger AI-drift at Lead > Senior > Mid titles — consistent with T20/T21 finding that director-level AI-strict grew 13.4× (1.1% → 14.8%) vs mid-senior 10.7 pp.

3. **Rewrite direction is not correlated with posting-volume direction.** Wells Fargo Lead SWE scaled volume UP (4 → 17) and AI-drifted up (+52.9 pp); Capital One Senior Lead Full Stack scaled volume DOWN (13 → 4) and AI-drifted up (+25.0 pp). Neither firm's AI-rewriting tracks its own hiring-volume direction.

## Interview questions

1. **To bank hiring managers (Wells Fargo, Capital One, JPM, Citi).** "Financial services firms appear to lead the AI-forward SWE posting rewrites. Is this driven by: (a) regulatory/compliance requirements that formalize AI skills more than consumer-tech does, (b) enterprise AI platform adoption at scale, (c) a defensive talent-market signal, or (d) actual work shifts toward AI integration?"

2. **To IC senior engineers at those firms.** "You've seen the JD change. Are you now working more on AI/LLM integration than 2 years ago, or is the JD ahead of the actual work?"

3. **To candidates who applied to Wells Fargo / Capital One Lead SWE roles.** "Did the AI-forward JD change your perception of the role? Did it change your expectations about what skills the interview would test?"

## Analytical note

These pair-level exemplars ground the T31 aggregate finding (pair-level AI Δ +13.4 pp > T16 co-level +7.7-8.3 pp). The pair-level signal is not uniform across firms — it concentrates at a subset of large-volume returning employers, of which financial-services firms are over-represented.
