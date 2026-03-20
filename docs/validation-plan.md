# Validation Plan: What to Look For & ML/AI Approaches

## Guiding Principle

Every hypothesis implies a **measurable signal** in job posting text or metadata. This plan maps each RQ to concrete observables, the simplest credible test, and a more powerful ML approach where warranted.

---

## RQ1: How are SWE postings restructuring across seniority levels?

**What to look for in the data:**
- Volume: monthly count of postings tagged junior/entry-level.
- Composition: junior share of total SWE postings.
- Content drift: are "junior" postings in 2025-26 textually closer to what "mid-level" or "senior" postings looked like in 2023-24?
- Senior-role drift: are senior postings using less management language and more orchestration / review / architecture language?

**Simple test:** Time-series plot of junior share plus a senior archetype shift index.

**ML approach — Seniority Classifier:**
Train a text classifier (fine-tuned BERT or SetFit with few-shot) on 2023-2024 benchmark postings where seniority labels are relatively clean. Features: full posting text. Labels: junior / mid / senior. Then run inference on 2025-2026 postings. The key metric is **predicted-seniority drift** — what fraction of title-tagged "junior" postings the model classifies as mid or senior? Rising misclassification = redefinition, not disappearance.

Why this works: it separates title inflation (cosmetic) from genuine content convergence (structural). If the classifier says a 2026 "Junior SWE" posting reads like a 2023 "Mid-Level SWE" posting, that's direct evidence of redefinition.

---

## RQ2: Which requirements migrated, and in what order?

**What to look for:**
- Per-skill prevalence curves in junior postings over time (monthly resolution).
- Temporal ordering: which skills crossed 10% prevalence first?

**Simple test:** Keyword-dictionary counts on posting text. Plot prevalence by month for target skills and requirement bundles (system design, CI/CD, cross-functional leadership, prompt engineering, ownership language, mentorship language, etc.). Identify crossing points.

**ML approach — Dynamic Topic Modeling + Skill Embedding Trajectories:**

1. **BERTopic with temporal binning.** Cluster posting text by quarter. Track which topic clusters appear in junior postings over time. Advantage over keyword counting: captures emergent skills not in your predefined dictionary (you don't know what you don't know).

2. **Skill embedding trajectories.** Embed each posting with a sentence transformer (e.g., `all-MiniLM-L6-v2`). For each skill term, compute its average context embedding in junior vs. senior postings per quarter. Measure when the junior-context embedding for "system design" converges with the senior-context embedding — this captures not just mention frequency but whether the skill is used in the same *way* (ownership vs. exposure).

---

## RQ3: Do posting-side changes show breaks or sharp accelerations around the broader AI release era?

**What to look for:**
- A discrete level shift and/or slope change in junior posting share, scope inflation, senior archetype shift, or embedding similarity.
- Breaks around a release era rather than one assumed universal date.

**Simple test:** Bai-Perron endogenous breakpoint detection on monthly series. Annotate candidate release windows rather than imposing one `Post-agent` date. Use placebo tests at arbitrary dates to confirm specificity.

**ML approach — Change Point Detection via Bayesian Online Methods:**
Use `ruptures` (Python) or a Bayesian online changepoint detector (BOCPD) on multivariate features simultaneously — posting volume, skill breadth, embedding centroid shift, AI-keyword prevalence. Multivariate detection has higher power than univariate Bai-Perron because it pools signals. If multiple feature streams independently flag the same quarter, that's stronger evidence than any single series.

Sensitivity check: run BOCPD on a rolling window excluding the most recent 1, 2, 3 months to test whether the detected break is stable or an artifact of endpoint effects.

---

## RQ4: Do posting-side AI requirements outpace observed workplace AI usage, and what explains the gap?

**What to look for:**
- Whether posting-side AI mentions and AI-related requirements rise faster than observed occupation-level AI usage benchmarks.
- Whether the gap differs by seniority, source, or metro.
- Whether interviews suggest the gap reflects real workflow change, template inflation, or anticipatory beliefs.

**Simple test:** Build a posting-usage divergence index by comparing posting AI mention rates against external occupation-level usage benchmarks. Report this descriptively by seniority and over time.

**ML / integration approach:**
Use interview coding plus external usage benchmarks to adjudicate the mechanism behind divergence. If needed, add a secondary comparison to low-exposure non-SWE postings later, but do not make that the core design.

Robustness: compare divergence patterns across LinkedIn-only, LinkedIn + Indeed pooled, and junior vs. senior subsamples.

---

## RQ5: What follows for training and apprenticeship?

**What to look for:**
- The empirical outputs from RQ1–4 directly parameterize recommendations. No separate ML needed.
- Cross-validate against profession parallels: did radiology residencies restructure when AI diagnostic tools deployed? Timeline + structure comparison.

**Approach:** Qualitative comparative analysis informed by the task migration map from RQ2. The prescriptive framework (Appendix B) should map 1:1 onto empirical findings — each recommendation traceable to a specific result.

---

## Recommended ML/AI Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Text embeddings | `sentence-transformers` (MiniLM or MPNet) | Posting similarity, drift measurement |
| Seniority classifier | SetFit or fine-tuned BERT | Detect redefinition vs. disappearance |
| Topic modeling | BERTopic | Emergent skill discovery |
| Changepoint detection | `ruptures` + BOCPD | Multivariate structural break |
| Benchmark integration | External usage benchmarks + interview coding | Quantify and explain posting-usage divergence |
| Skill extraction | Curated dictionaries + zero-shot NLI | Catch skills outside the predefined schema |

## Pipeline Order

```
1. Ingest & clean (benchmark + scraped postings → standardize fields, deduplicate)
2. Embed all postings (batch inference, cache embeddings)
3. Train seniority classifier on 2023-2024 benchmark data
4. Run classifier on full corpus → redefinition metric (RQ1)
5. Compute per-skill prevalence curves + BERTopic (RQ2)
6. Time-series assembly → Bai-Perron + BOCPD (RQ3)
7. Compute posting-usage divergence + integrate interview evidence (RQ4)
8. Synthesize into training framework (RQ5)
```

## Key Judgment Calls

**Where ML adds value over simple methods:** RQ1 (classifier detects redefinition that keyword counts miss), RQ2 (BERTopic finds emergent skills), RQ3 (multivariate changepoint has more power), RQ4 (zero-shot or embedding support can help classify AI-requirement language at scale).

**Where simple methods suffice:** Posting volume trends, individual requirement prevalence from curated dictionaries, and RQ5 (qualitative synthesis).

**What to watch out for:** The post-break window is still the binding constraint on RQ3. Be transparent about statistical power. The seniority classifier must be validated on held-out benchmark data before being applied forward — otherwise you're measuring model drift, not labor market drift.
