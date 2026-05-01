# Style brief — V9 rewrite

Rewriting `findings/`, `stories/`, `methodology/`. Gate memos stay as-is. The rewritten site should read like the three passages below: numbers carry the prose, caveats sit next to the claims they caveat, the writer commits where evidence is strong and hedges precisely where it's weak, sentence length varies. Keep `_glossary.md` open for canonical phrasings.

## Exemplars

### 1. Defining a term — OWID, "Life expectancy"

<example>
The term 'life expectancy' refers to the number of years a person can expect to live. By definition, life expectancy is based on an estimate of the average age that members of a particular population group will be when they die.

In societies with high infant mortality rates many people die in the first few years of life; but once they survive childhood, people often live much longer.

Indeed, this is a common source of confusion in the interpretation of life expectancy figures: It is perfectly possible that a given population has a low life expectancy at birth, and yet has a large proportion of old people.
</example>

Source: <https://ourworldindata.org/life-expectancy-how-is-it-calculated-and-how-should-it-be-interpreted>

Define, name the naive misreading, land the counterintuitive fact as the climax. No jargon, but real statistical work (distinguishing a distribution from a central tendency).

### 2. Numbers in prose, caveat in body — Construction Physics

<example>
Allmon et al (2000) looked at productivity changes for 20 different construction tasks from 1974 through 1996 using RS Means estimating guide data, and found that labor productivity increased for seven tasks, decreased for two tasks, and was unchanged for 11 tasks. Goodrum et al (2002) looked at productivity changes between 1976 and 1998 for 200 different construction tasks using data from several different estimating guides. They found that labor productivity declined for 30 tasks, was unchanged for 64 tasks, and improved for 107 tasks, with an average growth rate in labor productivity ranging from 0.8% to 1.8% depending on the estimating guide.

Between 2019 and 2024, revisions to the UK KLEMS data resulted in a swing from showing positive construction productivity growth between 1996 and 2016 to those same years showing negative productivity growth. Swedish data revisions showed the opposite, going from negative productivity growth using 2019 data to flat productivity growth using 2024 data.
</example>

Source: <https://www.construction-physics.com/p/trends-in-us-construction-productivity>

Numbers shaped as readable prose ("7 / 2 / 11"), not scanned off a table. Data-revision sensitivity surfaced as a structural claim in the body, not a footnote.

### 3. Argument against conventional wisdom — Noahpinion

<example>
Notice that the workers in their 30s, 40s, and 50s who are judged to be most heavily exposed to AI have seen robust employment growth since late 2022.

How can we square this fact with a story about AI destroying jobs? Sure, maybe companies are reluctant to fire their long-standing workers, so that when AI causes them to need less labor, they respond by hiring less instead of by conducting mass firings. But that can't possibly explain why companies would be rushing to hire new 40-year-old workers.

Unless we can come up with a compelling story for why AI should only replace the young, this finding smacks a bit of 'specification search'.
</example>

Source: <https://www.noahpinion.blog/p/ai-and-jobs-again>

Voice the steelman counter, then name what it fails to explain. "Can't possibly" (hard commit) next to "smacks a bit of" (precise hedge) — two epistemic registers in the same paragraph.

## Mechanics

**Inline definitions.** First use of a statistical term or code on a page → explain it. Short gloss as parenthetical; longer gloss as its own sentence. Not a footnote or admonition. Canonical phrasings in `_glossary.md`. Duplication across pages is intentional — each page stands alone.

**Codes.**
- Strip from prose: T-codes, V1/V2/Phase, Wave, Gate, H-codes, schema column names (`ai_strict`, `yoe_min_years_llm`, etc.).
- Expand on first use per page: J\*/S\* seniority codes ("J3 (postings asking for two or fewer years of experience)"), then bare "J3" is fine.
- Rename: "Sample N" → content name ("the returning-firms cohort", "the same-title pair panel").
- A1–A6: never as headline; acceptable as cross-reference label.
- Keep stripped codes in a "Source tasks" footer per page if the link is useful.

**Caveats.** Scope limits live in the same paragraph as the claim — or the one immediately after — not in a collapsed Evidence block.

**Preserve exactly.** Every number. Every citation. Every scope caveat (content — location may move). Every claim and its direction. Figure references.

**Don't add.** New claims, numbers, citations, opinion, or speculation that wasn't in the original.

## Before/afters

### Finding claim — `findings/a1-cross-occupation-divergence.md`

**Before.** **A1 — Cross-occupation AI divergence is universal, SWE-specific in magnitude.** … SWE DiD magnitude is +14.02 pp [+13.67, +14.37]; control drift is +0.17 pp. Spearman(worker-mid, employer-2026) = +0.92 — employers rank occupations identically to workers, at 10 to 30% of worker levels.

**After.** **Employers describe AI work at a fraction of the rate workers report using it.** Across 16 occupation subgroups (software engineering, adjacent technical roles, and a panel of non-tech controls), postings mention AI at universally lower rates than workers say they actually use it on the job. For software engineering, the gap grew by 14 percentage points between 2024 and 2026 (precisely +14.02 pp, 95% CI [+13.67, +14.37]); for control occupations it barely moved (+0.17 points). The ranking is stable: a Spearman rank-correlation of +0.92 shows employers order occupations the same way workers do, at 10 to 30% of the worker-reported intensity.

### Story paragraph — `stories/01-narrowing-middle.md`

**Before.** Among 68,000 software-engineering advertisements scraped from LinkedIn across 2024 and 2026, the junior share — the share of labelled postings calling for two or fewer years of experience — rose from 9.2% to 14.2%. That is five percentage points of increase on a base of nine points: a relative rise of roughly 55%. Under the more conservative arshkon-only baseline (the dataset's within-2024 calibration floor) the junior rise is smaller, 1.2 points, closer to the noise floor; under the 2,109-firm subset that posted in both 2024 and 2026 it intensifies to 6.2 points.

**After.** The study draws on roughly 68,000 software-engineering LinkedIn postings from 2024 and 2026. Among postings that explicitly named a seniority or a years-of-experience floor, the junior share (postings asking for two or fewer years of experience) rose from 9.2% to 14.2%. Five points on a base of nine is a 55% proportional rise, a number that deserves a caveat. On the pooled 2024 sample, the within-2024 noise (measured by comparing the two 2024 sub-sources against each other) is almost as large as the 2024-to-2026 move, so the pooled rise reads as directional, not precisely calibrated. On the 2,109 firms that posted in both periods, the junior share rose by 6.2 points, above the noise.

### Findings index — `findings/index.md`

**Before.** Every claim on this site traces to a task report (T01 through T38) or a verification report (V1, V2). The six Tier A findings below are the paper's lead candidates.

**After.** Six headline findings from this study. Each links to the analysis that produced it and to the robustness checks that tested it.

## Hard floor (greppable; the verifier will flag any match)

- No em dashes as connectors between clauses. (A true mid-sentence interruption inside a quotation is fine.)
- No `Moreover`, `Furthermore`, `In addition` as sentence openers.
- No `leverage` as a verb; no `delve` / `dive into` / `deep dive`.
- No `fundamentally`, `in essence`, `at its core`, `it's worth noting`.
- No `T\d+`, `H_[A-Z]`, `V[12] Phase`, `Wave \d`, `Gate \d`, `Tier A` in body prose (tables and footers exempt).
- Every statistical term glossed, and every J\*/S\* code expanded, on first use per page.

## Self-check

1. Sentence-length variation visible — at least one sentence over 25 words, at least one under 10.
2. At least one moment of visible judgment — a hard commit where evidence is strong, a precise hedge where it's weak — not uniform flat confidence.
3. Every number from the original present and correct; every scope caveat surfaced.
4. All insider codes stripped or expanded per the mechanics above.

If a passage still reads AI-generated and none of these caught it, reread the three exemplars and rewrite that passage from scratch.
