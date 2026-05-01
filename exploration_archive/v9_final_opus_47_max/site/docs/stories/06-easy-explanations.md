# The rejection of the easy explanation

### *Five plausible-sounding causes for the 2024-to-2026 rise in artificial-intelligence requirements in software-engineering postings. Four have been formally rejected; one has been partly reduced. The rise remains.*

The cleanest way to discredit a surprising labour-market finding is to replace it with a more prosaic explanation. In the twenty-four months since AI-requirement language began to appear in software-engineering job advertisements in volume, five such explanations have attracted public support. Each has been tested against 68,000 postings. One survives as a partial mechanism; four do not.

**The recruiter was using ChatGPT.** This is the loudest of the five. *Fortune* reported in September 2025 that "artificial intelligence has been training on crappy job descriptions"; Josh Bersin's recruiting newsletter celebrates AI-automated requisition workflows as a 2025 trend; SHRM says 66% of recruiters using AI already use it for job descriptions. The inference is that the rise in AI-related language in 2026 postings is the rise of language-model-mediated authorship: recruiters asking GPT to write a modern, impressive-sounding requisition, GPT inserting "RAG" and "LangChain" the way it inserts "innovative" and "dynamic". A validated signature-vocabulary score was computed across the 68,000 postings. The 2026 density of language-model stylistic tells rose (Cohen's d = 0.22, above within-2024 noise). Content was then re-tested on the bottom quartile by that score, the postings least stylistically machine-authored. AI-related content preserved 80-to-130% of its full-corpus effect size. Length preserved 52%. Length growth is, on this evidence, *about half* machine-authorship-mediated. AI-content rise is not. The authorship mechanism explains the prose style, not the new requirements.

**The bar was lowered.** The sister explanation, popular in human-resources press: firms shortened their requirements sections because a slack labour market allowed them to be less demanding of juniors. HR Dive reports 25% of employers dropping degree requirements; LinkedIn's *Future of Recruiting 2025* leans in the same direction. In the posting data, requirements-section length did shrink among junior roles. But across 243 panel firms the correlation between requirements-section contraction and any measure of hiring-bar lowering is small: absolute Pearson's ρ at or below 0.09 under one classifier, at or below 0.28 with sign-flipping across classifiers. A hand-read sample of fifty postings that saw the largest requirements-section contraction contained zero postings with explicit "we are lowering the bar" language; 56% showed technical requirements migrating from the requirements section to the responsibilities section; 30% showed benefits and culture expanding. Text was not removed. It was relocated.

**The hiring market was tight, and hiring firms were selective.** JOLTS data for the information sector in 2026 sits at 0.71× the 2023 average. Multiple authorities (Indeed's Hiring Lab, CNN, KPMG, McKinsey's 2025 *State of AI*) have suggested that firms respond to soft hiring by raising what they ask. If that were the mechanism, volume-down firms would have written denser AI requirements than volume-up firms. The opposite is what the data show. The only significant correlation in the selectivity test set runs the wrong way: volume-up firms write *longer* job descriptions, not shorter (Pearson r = +0.20, p = 0.0015, on 243 panel firms). Across the panel, every content shift tested is uncorrelated with a firm's 2024-to-2026 change in posting volume. The hiring trough is real. It is not the mechanism.

**The sample was different.** A subtle methodological critique: the 2024 corpus comes from two sources, the 2026 corpus from a scraped collection, and the three samples do not cover identical firms. If the 2026 sample over-represents AI-forward employers, the apparent 2024-to-2026 AI rise would be artefactual. The critique can be tested: restrict both periods to the 2,109 firms that posted in both. Fourteen of fifteen headline findings retain at least 80% of their full-sample magnitude (the self-audit report-text gives 13/15; the saved output table gives 14/15 at that ratio). The junior-share rise and senior-share fall both intensify on the restricted sample. Within-company AI rewriting on this restricted sample lands within 0.45 percentage points of the full-sample estimate. Sampling-frame change is not the mechanism.

**Legacy roles were relabelled.** A last defence: the Java architect, the .NET specialist, the Drupal developer are not vanishing; they are being rebranded as AI-enabled roles. Text-based neighbour mapping can test this on a thin sample. Of eleven 2024 titles that shrink most markedly, six could be matched with confidence to their closest 2026 titles by description similarity. The six 2026 neighbour roles carry an average AI-related mention rate of 3.6%, materially below the market's 14.4%. The substitution happens; it is substitution into modern-stack generalist roles (Postgres, Terraform, microservices, continuous integration), not into AI-enabled ones. The architect is absorbed by the backend engineer, not by the prompt engineer. The small sample (six matched titles) means the effect size is well-established but the breadth of legacy coverage is narrower than the full picture.

What remains, after these explanations are set aside or partly reduced, is the finding that was presumed to need explaining. Software-engineering postings contain more AI-requirement language in 2026 than in 2024. LLM-mediated authorship explains some of the length but not the content. Hiring-bar relaxation does not correlate with the change. Selectivity-response does not correlate with the change. Sampling-frame change cannot account for the change. Legacy-role substitution carries AI language at a rate below the market average. The positive finding, that employers have chosen to write more AI-requirement language into their advertisements, is not proved by the rejections; it is what remains after the alternatives have failed their own tests.

That is not always how investigations end. But it is how this one ended.

---

??? note "Evidence and sources"

    **Headline numbers**

    - **LLM-authorship partially rejected (T29):** content effects preserved at 80-130% on low-LLM-quartile subset; length growth only 52% preserved (half LLM-mediated). Cohen's d 0.22 and SNR 4.8 on signature-vocabulary density. Fact-check 06 verified.
    - **Hiring-bar lowering rejected (T33):** |ρ| ≤ 0.09 on hiring-bar proxies under T13 classifier; |ρ| ≤ 0.28 with sign-flipping across classifiers; 0 of 50 largest-contraction postings contain explicit loosening language; 56% show technical requirements migrating to responsibilities, 30% culture/benefits expanding.
    - **Hiring-selectivity rejected (T38):** |r| < 0.11 across breadth, AI-strict, scope, mentor, YOE metrics on n=243 firms; desc-length r = +0.20, p = 0.0015 (positive, opposite of selectivity).
    - **Sampling-frame artefact rejected (T37):** 14 of 15 headlines robust at ratio ≥ 0.80 on 2,109-firm returning cohort (saved-table count; report-text miscounted 13/15); J3 +5.05 pp → +6.17 pp; S4 −7.62 pp → −8.29 pp on cohort; within-co AI +7.91 pp matches full-panel +7.65-8.34 pp within 0.45 pp.
    - **Legacy-to-AI substitution rejected (T36):** 11 disappearing 2024 titles attempted; 6 matched; 2026 neighbour AI-strict rate 3.6% (below 14.36% market); substitutions go to Postgres / CI-CD / microservices / Terraform stacks.

    **Prior-art sources (ranked by loudness)**

    *Fortune* 2025 LLM-JD training piece; Josh Bersin recruiting-newsletter; SHRM 66% of AI-using recruiters; HR Dive degree-drop piece; LinkedIn *Future of Recruiting 2025*; Stanford Brynjolfsson "Canaries" paper; Indeed Hiring Lab JOLTS commentary; KPMG / McKinsey 2025 *State of AI*; GitHub / Stack Overflow blog posts on role rebranding; Choi & Marinescu SSRN methods note on sampling.

    **Sensitivity verdict**

    All five rejections survive their own sensitivity checks. LLM-authorship is partial rather than complete — length growth is half-authorship-mediated. Hiring-bar rejection is classifier-robust in direction but the precise correlation band widens from 0.09 to 0.28 depending on classifier. Legacy-substitution is small-n (six matched titles). The collective rejection is stronger than any single rejection.

---

## Related in Findings

- [What we rejected](../findings/rejections.md) — the academic summary of the same five rejections.
