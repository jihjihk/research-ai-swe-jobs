# Stage 3 synthesis — template

Per design.md §13.5 the orchestrator produces a single prose document
under 2,000 words at `figures/bertopic/memos/synthesis.md`. The user
reads and acts on it; if more material than fits the budget exists,
that is a signal to cut harder, not pad more.

The synthesis is **prose, not bullet lists where prose works better**.
Tonal anchors per `AGENTS.md` writing style: Anthropic Economic Index,
Indeed Hiring Lab, Derek Thompson — data-driven journalism, claims
next to caveats, hard commit beside precise hedge.

## Structure

### 1. Executive summary (≈ 60 words, three sentences)

What survived the three-gate cull, what didn't, what the paper narrative
should commit to. State the survivor count plainly. Name the largest
single result (positive or negative).

### 2. Survivors (priority order)

Per finding:
- **Claim it supports.** C1 / C2 / C3 / C4 / T1–T4 / §1.4.3 / §1.4.4.
- **Headline number.** The cleanest single quantitative statement plus
  the figure / table path that backs it.
- **Three-gate evaluation.** One short paragraph: which gates passed
  and how comfortably; any caveats the prose must carry.
- **Proposed prose** for the paper, one paragraph in The Economist
  register. Editor-ready.

Findings that fail one gate but have strong robustness elsewhere appear
as "exploratory" with an explicit hedge ("the data are consistent with X
but the leave-one-out spread leaves room for Y").

### 3. Cuts

Per finding dropped:
- Which gate failed and by how much.
- Whether the cut is a "redesign and retry" or "the underlying signal
  is not there."
- Brief evidence trail (memo path, headline number, threshold).

Reviewers may ask why a particular ablation was not in the body of the
paper; the cut log is what we point them to.

### 4. Lessons

Observations the human authors should carry forward: mega-cluster
behavior, embedding-model surprises, anchor sensitivity, design-doc
inconsistencies that surfaced in execution, items the §5.2 labelling
protocol should specifically address. Limit to genuinely new
information — do not restate what the design doc already said.

### 5. Recommendations for the paper

Per surviving finding: headline-worthy / appendix-worthy / drop.
Headline = body figure or table at a primary claim's first mention.
Appendix = robustness or supporting evidence. Drop = does not enter
the paper at all.

For findings that depend on §7.7 L1 / L2 crosstab (T-l1l2 queued
because columns absent), state explicitly that the crosstab will run
in a later session and the paper either notes the absence or runs the
crosstab at revision time.

### 6. Open questions for human work

What the authors must do next:
- §5.2 cluster-naming protocol decisions and when frozen labels are
  committed.
- Anchor-set domain validation (any §6.1 axis whose held-out hit rate
  surfaced an issue).
- T-l1l2 dispatch when L1 / L2 columns land in `unified_core.parquet`.
- Stage 4 reproducible notebook authorship (a separate human session
  per design doc §13.6).

## What the synthesis is not

- Not a results dump. Tables and figures live in the artifacts; the
  synthesis points at them and tells the story.
- Not advocacy. If a finding survived the gates by a margin of 0.001,
  say so.
- Not under-hedged: numbers need their robustness band attached. Not
  over-hedged: a finding that survived three of three gates does not
  need a "more research is needed" qualifier.
