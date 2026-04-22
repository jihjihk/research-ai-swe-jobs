# Composite C — Article-Worthiness Evaluation

Author: evaluation dispatch. Date: 2026-04-21.
Inputs: `eda/research_memos/composite_C_junior_label.md`,
`eda/scripts/composite_C_junior_label.py`, `eda/tables/S28_*.csv`.

---

## 1. The actual story, in one sentence

Best one-line framing: **"The vocabulary employers use to mark entry-level
software jobs has bifurcated — defense contractors and staffing agencies still
say 'junior'; everyone else has quietly moved to numeric level codes — but the
share of true-entry postings carrying *any* explicit marker doubled while the
'junior' word itself stayed flat at under 1% of all SWE postings."**

This is a hybrid of (a) and (c) from the prompt, with (b) as the colourful
sub-story. The single most defensible factual claim — a doubling of explicit
markers driven entirely by level codes (`S28_06`: 11.7% → 23.8% any-explicit
among yoe ≤ 2; `S28_01`: junior/jr title rate 0.50% → 0.79%) — is genuinely a
"vocabulary shift, not a content shift" finding.

## 2. Why an Economist reader would care

Honestly: **not very much, on its own.** The reader already suspects
- that defense contractors use weird grade codes (priced in),
- that big tech uses level numbers (priced in),
- that "junior" sounds dated (folk knowledge).

The non-obvious payoff is narrow: a reader who had been planning to track
"junior posting share" as a labor-market indicator now learns that it would
have *missed* the doubling of explicit early-career markers. That is a
methodological warning to the labor-economics community, not a finding about
the labor market itself. The "Engineer III in defense = entry-level" wrinkle
(`S28_08_substitute_ngram_n2.csv`: 92 of 1,184 unlabelled-entry titles in
2026-04) is a good cocktail-party fact but it doesn't change anyone's model
of how AI is reshaping SWE work.

Crucially, **this finding does not advance RQ1-RQ4** as defined in
`docs/1-research-design.md`. RQ1 is about junior scope inflation and senior
archetype shift; the sharper supporting fact from this memo (`S28_05`:
junior-titled postings carry the *highest* tech count, 7.29 vs 6.10 for
unlabelled entry) is already a Composite-A/B finding in spirit and should
live there.

## 3. Comparable published findings

The labelling-conventions question is **already worked over** in adjacent
literature, and our finding partially *contradicts* the published trend:

- **Datapeople (2019-2021 tech postings)** report a **20-40% decrease** in
  numerical-suffix titles ("Engineer I/II/III/IV") and a 100% rise in
  "Lead/Principal" titles. Our 2024 → 2026 SWE finding runs the *opposite*
  direction: numerical-suffix and level-code share is rising. That is
  publishable as a reversal — IF the comparison is methodologically clean
  (Datapeople used different scopes and platforms).
- **Indeed Hiring Lab (Sept 2025)** reports junior postings down 7% YoY and
  notes "fewer than 2% of postings in tech and engineering are for junior
  roles." We sit in the same ballpark (0.79% junior-titled, 3.8% any-junior-
  family among true-entry) but with a finer breakdown.
- **Maasoum & Lichtinger (SSRN 2025), "Generative AI as Seniority-Biased
  Technological Change"** — a much heavier paper that uses résumé + posting
  data to show junior employment falling at AI-adopting firms. They are
  measuring the underlying labor demand, not the lexical surface.
- **Hampole, Datapeople, Robert Walters, ClearanceJobs** all touch title
  inflation but none publish the level-code-versus-junior-word decomposition
  for a recent SWE-only window.

**Novelty assessment:** the lexical decomposition itself (level-code share
rising while junior-word share stays flat) is mildly novel because the
existing pieces are either coarser (Indeed Hiring Lab) or older (Datapeople,
on a window that ends before the agentic-AI inflection). The defense-
contractor "Engineer III" wrinkle appears to be genuinely unwritten. But
none of this is a *labor-market* finding the way Maasoum & Lichtinger is —
it is a labelling finding.

## 4. Article lede draft, 200 words, Economist style

> Defense contractors still call them juniors. Almost no-one else does.
> In April 2026, just 0.79% of American software-engineering postings on
> LinkedIn carried the word "junior" or "jr." in their title — a share
> indistinguishable from January 2024. That word has retreated to a corner
> of the labour market: nearly all of the firms that use it most heavily are
> defense primes (Peraton, Leidos, SAIC, CACI, General Dynamics) or IT
> staffing agencies (Insight Global, Brooksource, Cynet, Yoh, Experis).
> Among employers posting at least 100 software roles, the rate falls to
> 0.50%. Google, Microsoft, AWS and NVIDIA used the word zero times.
>
> Yet the share of genuinely entry-level postings — those asking for two
> years of experience or fewer — that carry *some* explicit early-career
> marker has roughly doubled, from 11.7% in early 2024 to 23.8% in
> April 2026. Every percentage point of that growth comes from numeric
> grade codes such as "Engineer I", "Engineer II" and, more confusingly,
> "Engineer III", a defense-contractor pay-band that to most readers
> announces seniority but, in context, signals its opposite. The vocabulary
> changed; the rung did not move.

Verdict on the lede: **competent but thin.** It would run as a 600-word
column on a slow news day. It would not run as a feature.

## 5. Alternative framings worth considering

a. **"The rise of the level code"** as a measurement piece. Anchor: Datapeople's
   2019-2021 finding ran the opposite way; our 2024-2026 SWE-only window
   reverses it. This would matter to anyone building seniority dashboards
   from posting text. Strongest standalone framing IF a quick robustness
   pass against the broader Indeed/Datapeople scope confirms the reversal.

b. **"What defense contractors call entry-level"** as a sector vignette.
   Strong colour, weak generalization. Best as a sidebar inside a piece
   that is already about SWE labelling or defense-sector hiring.

c. **"Engineer III as misleading credential"** as a methodological aside.
   This is the most useful thing in the memo for anyone *else* doing
   labor-text research and the natural home for it is a methods footnote
   in the dataset paper, not a standalone piece.

d. **Folded into Composite A (junior scope inflation):** the `S28_05` fact
   that junior-titled postings carry +1.51 tech-count residual versus −0.36
   for the unlabelled-entry pool is a clean piece of evidence *for* the
   junior-scope-inflation thesis. That fact wants to live in the junior-
   scope-inflation chapter, not in a labelling chapter.

## 6. Tier placement

Composite C does not survive as a standalone Tier 1 piece. It is a labelling
study where the underlying labour-market finding (junior scope inflation)
already belongs to Composite A, and the lexical finding has been partly
preempted by Datapeople (with opposite sign, which is a story but a small one).

**Recommended placement:**

- Move the **scope/breadth contrast** (`S28_05`: junior-titled = 7.29 tech
  count, unlabelled entry = 6.10) into **Composite A**, where it sharpens
  the junior-scope-inflation claim by showing that the postings most
  literally labelled "junior" are *also* the most demanding entry-tier
  postings — H1 supported on breadth.
- Move the **level-code displacement** finding (`S28_06`: 11.7% → 23.8%
  any-explicit-marker rate) into a **methodological sidebar** inside
  Composite A or the dataset paper. Frame as "If you want to track
  entry-level posting share over time, you must use a multi-pattern
  detector; the word 'junior' alone misses the doubling of marker
  prevalence." Cite Datapeople 2019-2021 as the comparable; flag the
  reversal.
- Keep the **defense-contractor concentration** (`S28_07_who_labels_top_
  companies_2026_04.csv`, top 15 firms) as a **two-paragraph colour box**
  inside that same sidebar. The "Engineer III in defense" wrinkle (`S28_08`)
  belongs as a footnote, not a story.

## VERDICT

**DEMOTE to sidebar.** Composite C does not stand on its own as a Tier 1
article. The lexical finding is real but small; the labour-market finding
underneath it (junior scope inflation) belongs to Composite A, not here.

Concretely:
- **Fold into Composite A** — the breadth-contrast finding (junior-titled
  postings = highest tech count among entry candidates) becomes a paragraph
  in the junior-scope-inflation section.
- **Add a methods sidebar** — "How to count entry-level postings when the
  vocabulary is shifting" — built on the level-code-displacement finding
  (`S28_06`) and the precision-table apparatus from the original memo.
  This is a 400-600 word sidebar, not a feature.
- **Defense-contractor and 'Engineer III' material** — colour box and
  footnote inside that sidebar.

If a future reviewer pushes for a standalone version, the strongest framing
to revive would be **"the rise of the level code"** (5a above), pitched
explicitly as a reversal of Datapeople's 2019-2021 finding. That requires
one additional analysis the original memo does not contain: a back-of-
envelope replication of Datapeople's measurement on our 2024 frame to
verify the comparison is apples-to-apples. Until that check is done, the
reversal claim is provisional.

No new analysis was run for this evaluation; all numbers cited come from
`eda/tables/S28_01`, `S28_05`, `S28_06`, `S28_07_who_labels_top_companies_
2026_04.csv`, and `S28_08_substitute_ngram_n2.csv`.
