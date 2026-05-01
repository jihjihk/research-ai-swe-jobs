# Artifact 6 — Seniority-boundary sharpening: unsupervised + supervised convergence

**Source:** T15 unsupervised TF-IDF cosine + T20 supervised logistic-regression AUC.
**Figures:** `exploration/figures/T20/auc_by_boundary.png`, `exploration/figures/T15/similarity_heatmap_tfidf.png`, `exploration/figures/T20/gap_evolution.png`.

This artifact communicates the Tier A finding that seniority boundaries SHARPENED between 2024 and 2026 — contradicting a pre-exploration prior that AI might blur or compress the ladder.

---

## Headline numbers

### T15 unsupervised (TF-IDF cosine between junior and senior centroids)

| Period | Junior↔Senior cosine | Δ |
|---|---|---|
| 2024 | 0.946 |  |
| 2026 | **0.863** | **−0.083 (diverging)** |

Lower cosine = MORE distinct. Junior and senior SWE postings became MORE distinguishable between 2024 and 2026.

Within-2024 arshkon-vs-asaniczka same-seniority cosine: 0.946 junior, 0.933 senior. Cross-period Δ (−0.083) is **2.5× the within-2024 noise floor**. Real signal, not instrument artifact.

### T20 supervised (logistic regression AUC on 9 structural features)

| Boundary | 2024 AUC | 2026 AUC | Δ |
|---|---|---|---|
| entry ↔ associate | 0.708 | 0.801 | **+0.093** (sharpened) |
| associate ↔ mid-senior | 0.743 | 0.893 | **+0.150** (sharpened; cleanest signal) |
| mid-senior ↔ director | 0.723 | 0.701 | −0.022 (softened slightly) |
| junior ↔ senior (3-level) | 0.825 | 0.836 | +0.010 |

The middle of the ladder sharpened the most. The director boundary slightly softened (see T34 cluster 0 with 2× director share — AI-oriented directors look like mid-senior orchestrators on structured features).

### T20 continuous YOE × period interaction (OLS)

| Outcome | YOE × period coef | p |
|---|---|---|
| requirement_breadth | +0.273 | 4.98e-44 |
| tech_count | +0.177 | 5.49e-24 |
| credential_stack_depth | +0.017 | 3.09e-07 |
| ai_binary | +0.0004 | 0.655 (null) |

5 of 6 outcomes have significant positive YOE × period interactions: **YOE sorts roles MORE strongly on structured features in 2026 than 2024**. AI-binary is the exception — AI mentions rose uniformly across YOE levels.

---

## Counter-intuitive findings to probe

### 1. AI is at all YOE levels, not senior-gated

T11 + T20 agree: the AI-binary × YOE × period interaction is null. AI-mention rose by the same amount across all YOE levels. GitHub Copilot rate at junior (J3 4.6%) is HIGHER than at senior (S4 4.1%). AI is not a senior-specific requirement.

### 2. 2026 junior postings look like 2024 junior postings, NOT like 2024 senior postings

T15 nearest-neighbor retrieval: 2026 junior postings retrieve 2024-junior neighbors at 2.79× base rate (21.9% vs 7.9% base), while 2024-senior neighbors appear at EXACTLY the base rate (45.7% actual vs 45.6% base).

Interpretation: the "junior is being rewritten as senior" hypothesis is **falsified**. Junior roles still look most like junior roles — they haven't absorbed senior-role content.

### 3. Associate tier didn't disappear — it became more distinct

T20 centroid-distance analysis on z-standardized features:
- associate ↔ entry distance: 0.71 → 0.82 (+0.11)
- associate ↔ mid-senior distance: 1.21 → 1.44 (+0.23)
- associate ↔ director distance: 2.08 → 2.22 (+0.14)

All three distances GREW. Associate as a level is becoming MORE differentiated, not absorbing into neighbors. (Note: associate n=121 in 2026 is thin; the direction is robust but magnitudes are fragile.)

---

## Interview questions

### To hiring managers

1. **"The data shows that SWE seniority boundaries sharpened — entry vs associate, associate vs mid-senior became more distinguishable by 2026. Does this match your hiring experience? Did you tighten seniority-level distinctions in your JDs?"**

2. **"Was there a discussion at your firm about seniority-level ambiguity between 2024 and 2026? Did you explicitly clarify entry vs. associate vs. mid-senior expectations in postings?"**

3. **"AI adoption is often framed as 'democratizing' or 'leveling' the field. Our data shows the opposite — AI-oriented senior roles ask for MORE experience (median YOE 6) than traditional senior roles (5). Why might AI be sharpening rather than flattening the seniority ladder?"**

### To senior IC engineers

4. **"Do you feel that the distinction between 'senior' and 'mid-senior' has become clearer or murkier in your firm over the past 2 years? The data says clearer."**

5. **"The AI-oriented senior archetype (T34 cluster 0) has 2× the director-share of the traditional senior cluster. Is 'AI-focused director' a real role-ladder at your firm, or is director-title being used as a compensation / signaling lever on AI roles?"**

### To junior / early-career engineers

6. **"The data shows 2026 entry-level SWE postings are more distinctly 'entry-level' in content than 2024 entry postings — not more senior. Does this match what you see in job listings? Do entry-level JDs look more cleanly entry-level now?"**

7. **"An alternative reading: entry JDs ask for MORE specialized AI skills in 2026 (LLM exposure, RAG familiarity, etc.) but ALSO more clearly entry-level. Do you perceive this as a raised bar, a lowered bar, or just a different bar?"**

### To candidates across the ladder

8. **"If the seniority ladder is sharpening (not blurring), how do you position yourself when applying for a role that sits between two levels? Is there a 'between-level' gray zone, or do you pick a target level and commit?"**

## Mechanistic readings to probe

- **(a) LLM-authored JDs are more template-consistent within-level.** Recruiter LLM tooling may produce more uniform junior JDs and more uniform senior JDs, mechanically sharpening boundaries even if underlying work didn't change.
- **(b) Employers are more cautious about seniority-level specification.** In a tight labor market (JOLTS 2026 Info 0.71× 2023), employers may be more explicit about experience asks to manage applicant flow.
- **(c) Real content shifts underlying the ladder.** Senior-specific content (orchestration, AI-systems, platform integration) may be genuinely distinct from junior content (individual-contribution features, bounded systems).

T37's robustness finding (13/15 headlines robust on returning cohort; J3/S4 directions INTENSIFY) suggests sharpening is NOT a sampling-frame artifact.

## Counter-findings interviewees may surface

Interviewees may report that their firm's hiring is LESS distinguishable across levels in practice (e.g., "everyone does everything"), even if JDs are more distinguishable. The T15/T20 finding is about POSTING LANGUAGE; the interview can probe whether posting-language sharpening reflects or precedes actual work-allocation sharpening.
