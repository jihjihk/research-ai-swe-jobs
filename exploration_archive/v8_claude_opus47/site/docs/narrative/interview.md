# Interview elicitation

Six artifacts support RQ4 qualitative interviews. The interviews should adjudicate five questions that the quantitative data cannot resolve on its own.

## Five adjudication questions

### 1. Scope realism

**Question.** Would a 2024 or 2026 entry-level candidate realistically meet the stated scope in these postings?

**Probe.** Hiring-manager intent vs recruiter boilerplate. Does the hiring manager read the JD the same way a candidate does?

**Artifact.** [inflated_junior_jds.md](../interview/inflated_junior_jds.md) — four ghost-flagged entry-labeled JDs ranked high on (tech_count × org_scope) × LLM ghost rubric. Diversified across 2024/2026, AI vs traditional, aggregator vs direct.

### 2. Content change vs stylistic change

**Question.** What in the 2026 version of each paired JD reflects real hiring-requirement change vs LLM-augmented template?

**Probe.** Which 2026 deltas would the hiring manager actually screen for? Which are recruiter-tool output?

**Artifact.** [paired_jds_over_time.md](../interview/paired_jds_over_time.md) — four same-company 2024/2026 mid-senior JD pairs from the 240-company overlap panel. Ranked by company-level increase in AI-strict + mentor binary rates.

### 3. Employer-usage gap mechanism

**Question.** Why do SWE employers mention AI tools in 46.8% of JDs when developers use them at 62-76% rates?

**Probe.** Four candidate mechanisms:

1. **JD template lag** — hiring managers are slower to update than developers are to adopt.
2. **Implicit assumption** — "AI tools are table-stakes; we don't list them" (parallel to git).
3. **AI as coordination signal, not skill demand** — postings communicate firm-level AI adoption to investors/candidates, not a skill requirement.
4. **Internal filter, not public requirement** — firms screen on AI tool use in interviews but omit from postings.

**Artifact.** [employer_usage_divergence.png](../figures/employer_usage_divergence.png) — overlays SWE employer rates against Stack Overflow 2024, Octoverse 2024, Anthropic 2025 usage benchmarks.

![Employer-worker divergence](../figures/employer_usage_divergence.png)

### 4. Senior role redefinition

**Question.** Is the mgmt + orch + strat + AI senior profile a new role or a relabeling of existing seniors?

**Probe.** Use `systems_engineering` as natural-experiment control — AI-strict +0.16 pp (essentially zero); senior mentor Δ +7.6 pp (part of corpus-wide mentor rise). If the sub-archetype is real, its members will describe their work differently from a systems-engineering senior.

**Artifact.** [senior_archetype_chart.png](../figures/senior_archetype_chart.png) — grouped bars for S1 binary-presence share of management (V1-refined strict), orchestration, strategic scope, and AI (V1-refined strict). 2024 vs 2026 with Δ labels.

![Senior archetype shift](../figures/senior_archetype_chart.png)

### 5. LLM-authorship

**Question.** When drafting a JD today vs three years ago, what is your pipeline? What percentage of JD language came from an LLM draft vs a copy-edit of an existing template?

**Probe.** T29's authorship-score shift is +1.14 standard deviations. Quantitative estimate of recruiter-LLM mediation: 15-30% of the apparent rewrite. This probes that estimate qualitatively.

**Artifact.** [junior_share_trend.png](../figures/junior_share_trend.png) — J1/J2/J3/J4 trajectories with dashed markers at GPT-4, Claude 3, GPT-4o, Claude 3.5 Sonnet, o1, Claude 4, Gemini 2.5 Pro.

![Junior share trend with LLM releases](../figures/junior_share_trend.png)

## Extra reference figures

### AI gradient — SWE vs adjacent vs control

![AI gradient interview chart](../figures/ai_gradient_chart.png)

From T18 (primary) with V2 cross-check. Frames the SWE-specificity of the rewrite for the interview opening.

## Sample sizes and stratification

Target sample (from the T25 handoff):

- **10-15 hiring managers** — stratified by archetype (AI/ML, generic SWE, systems-engineering control) and firm size (entry-specialist vs direct employer).
- **5 recruiters** — focused on JD-drafting pipeline and HR-tool defaults.
- **3 HR-tooling operators** — to validate the authorship-score mediation quantitatively.

## Reproduction

All six artifacts are deterministic outputs of:

```
/home/jihgaboot/gabor/job-research/.venv/bin/python \
  /home/jihgaboot/gabor/job-research/exploration/scripts/T25_interview_artifacts.py
```

Full README: [exploration/artifacts/T25_interview/README.md](../raw/wave-4/). Original paired-JD markdowns are copied to this site's `interview/` folder.
</content>
</invoke>