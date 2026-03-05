# Validation Plan: What to Look For & ML/AI Approaches

## Guiding Principle

Every hypothesis implies a **measurable signal** in job posting text or metadata. This plan maps each RQ to concrete observables, the simplest credible test, and a more powerful ML approach where warranted.

---

## RQ1: Are junior roles disappearing or being redefined?

**What to look for in the data:**
- Volume: monthly count of postings tagged junior/entry-level (by title keywords + experience ≤ 2 YoE).
- Content drift: are "junior" postings in 2025–26 textually closer to what "mid-level" postings looked like in 2022–23?

**Simple test:** Time-series plot of junior share (junior posts / total SWE posts). Two-sided: check both numerator decline and denominator growth.

**ML approach — Seniority Classifier:**
Train a text classifier (fine-tuned BERT or SetFit with few-shot) on 2020–2022 postings where seniority labels are clean. Features: full posting text. Labels: junior / mid / senior. Then run inference on 2023–2026 postings. The key metric is **predicted-seniority drift** — what fraction of title-tagged "junior" postings the model classifies as mid or senior? Rising misclassification = redefinition, not disappearance.

Why this works: it separates title inflation (cosmetic) from genuine content convergence (structural). If the classifier says a 2026 "Junior SWE" posting reads like a 2021 "Mid-Level SWE" posting, that's direct evidence of redefinition.

---

## RQ2: Which competencies migrated, and in what order?

**What to look for:**
- Per-skill prevalence curves in junior postings over time (monthly resolution).
- Temporal ordering: which skills crossed 10% prevalence first?

**Simple test:** Keyword/taxonomy counts from Lightcast's parsed skill fields. Plot prevalence by month for ~15 target skills (system design, CI/CD, cross-functional leadership, prompt engineering, etc.). Identify crossing points.

**ML approach — Dynamic Topic Modeling + Skill Embedding Trajectories:**

1. **BERTopic with temporal binning.** Cluster posting text by quarter. Track which topic clusters appear in junior postings over time. Advantage over keyword counting: captures emergent skills not in your predefined dictionary (you don't know what you don't know).

2. **Skill embedding trajectories.** Embed each posting with a sentence transformer (e.g., `all-MiniLM-L6-v2`). For each skill term, compute its average context embedding in junior vs. senior postings per quarter. Measure when the junior-context embedding for "system design" converges with the senior-context embedding — this captures not just mention frequency but whether the skill is used in the same *way* (ownership vs. exposure).

---

## RQ3: Structural break in late 2025?

**What to look for:**
- A discrete level shift and/or slope change in junior posting share, skill breadth, or embedding similarity around Dec 2025.

**Simple test:** Bai-Perron endogenous breakpoint detection on the monthly time series. Supplement with Chow test at the hypothesized break date. Placebo tests at 6 arbitrary dates to confirm specificity.

**ML approach — Change Point Detection via Bayesian Online Methods:**
Use `ruptures` (Python) or a Bayesian online changepoint detector (BOCPD) on multivariate features simultaneously — posting volume, skill breadth, embedding centroid shift, AI-keyword prevalence. Multivariate detection has higher power than univariate Bai-Perron because it pools signals. If multiple feature streams independently flag the same quarter, that's stronger evidence than any single series.

Sensitivity check: run BOCPD on a rolling window excluding the most recent 1, 2, 3 months to test whether the detected break is stable or an artifact of endpoint effects (critical given your ~3-month post-break window).

---

## RQ4: SWE-specific or broader trend?

**What to look for:**
- Whether the same signals (volume drop, skill breadth increase, embedding drift) appear in control occupations (civil eng, nursing, mech eng).

**Simple test:** Difference-in-differences. Treatment = SWE postings. Control = low-AI-exposure occupations (selected via Felten et al. 2023 AI exposure scores). Estimate the interaction term (SWE × PostAgent).

**ML approach — Synthetic Control Method:**
Rather than hand-picking 3–4 control occupations, construct a **synthetic counterfactual** from a weighted combination of all non-AI-exposed occupations that best matches pre-treatment SWE trends. This is more defensible than arbitrary control selection and provides a single, optimized counterfactual. Implemented in Python via `SparseSC` or `SyntheticControlMethods`. The gap between actual SWE trajectory and synthetic control *is* the treatment effect.

Robustness: run placebo synthetic controls for each donor occupation. If many placebos show gaps as large as SWE's, the finding is not significant.

---

## RQ5: Training implications

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
| Causal inference | Synthetic control (`SparseSC`) | Isolate SWE-specific effect |
| Skill extraction | Lightcast taxonomy + zero-shot NLI | Catch skills outside standard taxonomies |

## Pipeline Order

```
1. Ingest & clean (Lightcast dump → standardize fields, deduplicate)
2. Embed all postings (batch inference, cache embeddings)
3. Train seniority classifier on 2020–22 labeled data
4. Run classifier on full corpus → redefinition metric (RQ1)
5. Compute per-skill prevalence curves + BERTopic (RQ2)
6. Time-series assembly → Bai-Perron + BOCPD (RQ3)
7. Merge control occupations → DiD + synthetic control (RQ4)
8. Synthesize into training framework (RQ5)
```

## Key Judgment Calls

**Where ML adds value over simple methods:** RQ1 (classifier detects redefinition that keyword counts miss), RQ2 (BERTopic finds emergent skills), RQ3 (multivariate changepoint has more power), RQ4 (synthetic control beats hand-picked controls).

**Where simple methods suffice:** Posting volume trends (just counts), individual skill prevalence (keyword matching on Lightcast taxonomy), and RQ5 (qualitative synthesis).

**What to watch out for:** The 3-month post-break window is the binding constraint on RQ3. Be transparent about statistical power. The seniority classifier must be validated on held-out 2020–22 data *before* being applied forward — otherwise you're measuring model drift, not labor market drift.
