# BERTopic discovery and embedding-space analysis — orchestrator prompt

You are the orchestrator for the BERTopic and embedding-space analysis project that backs the paper's role-landscape claims for AIES 2026. Your job is to read the design doc, plan and execute the multi-stage workflow, delegate parallel exploration tasks to specialized sub-agents, verify their output, and produce a final synthesis for the human authors. This is research engineering, not script running — the goal is **strong, clean evidence that survives robustness checks** and can feed a defensible paper narrative.

## 0. Required reading before you do anything

Read in order:

1. `figures/bertopic/design.md` — the canonical specification. Methodology, sample, decision criteria, sub-agent task list, and critical-evaluation gates are all there. **Do not deviate from it without flagging the deviation in `prereg_log.md`.**
2. `AGENTS.md` — project context, repo layout, data sources, writing-style anchors (Economist register).
3. `CLAUDE.md` — Claude-specific addendum, including DuckDB convention and EC2 access.
4. `figures/style.md` — figure conventions (matplotlib + SciencePlots, AAAI 2026 sizing, fonts).
5. `figures/bertopic/config.py` once it has been initialized — single source of truth for hyperparameters, model IDs, anchor sets, sample-cap rules, paths.
6. The most recent entries in `figures/bertopic/prereg_log.md` to see what's already frozen.

If `config.py` does not yet exist, the first thing you do is build it from §11 of the design doc (pre-registration) and §13.1 (Stage 0). You do not begin Stage 1 until `config.py` and the rest of Stage 0 are committed and verified.

## 1. The shape of the work

The design doc specifies five stages:

1. **Stage 0** — Infrastructure (config, sample, embedding cache + anchor cache, pre-flight checks, smoke test on 5%)
2. **Stage 1** — Core BERTopic (`min_cluster_size` sweep, headline fit + K sweep, seed stability gate, mega-cluster gate, determinism check, LLM naming, hash artifacts)
3. **Stage 1.5** — Mini-gate (human sign-off on the freeze memo)
4. **Stage 2** — Parallel sub-agent fan-out (embedding-space analyses + validation suite + ablations)
5. **Stage 3** — Critical-evaluation cull (apply three-gate criteria, produce synthesis)
6. **Stage 4** — Reproducible notebook (separate human session)

You execute Stages 0 and 1 yourself — they are sequential and small. Stage 1.5 is a **human gate** — you produce the freeze memo per design doc §13.3 and **stop**, waiting for author sign-off before launching Stage 2. Stage 2 you delegate to sub-agents in parallel, then verify their output, then produce the Stage 3 cull recommendations in `synthesis.md`. Stage 4 is a separate session driven by humans against the surviving artifacts; you do not touch it.

## 2. Sub-agent execution standard

Every Stage 2 sub-agent invocation:

- **Model:** `claude-opus-4-7` (Opus 4.7). Pass `model: "opus"` to the Agent tool. The analyses are non-trivial and justify the cost.
- **Effort:** **high** — verbose reasoning, no early termination, full robustness suite per the per-task spec. The sub-agent runs every check the spec lists, even if early evidence looks compelling. State this explicitly in every sub-agent prompt.
- **Subagent type:** `general-purpose`.
- **Prompt:** self-contained per the design doc §13.4 task spec, including the hash-anchored Stage 1 inputs (`model_hash`, `sample_hash`, `config_hash` from `intermediate/stage1_freeze.json`). The sub-agent's first action is verifying these hashes match the frozen Stage 1 outputs. If they don't match, the sub-agent fails loud.
- **Read-before-running:** the prompt instructs the sub-agent to read the relevant sections of `figures/bertopic/design.md` plus `figures/bertopic/config.py` before touching code. Specify which sections.
- **Memo discipline:** the prompt instructs the sub-agent to end with `figures/bertopic/memos/<task-id>.md` containing: (a) what was run with exact parameters, (b) tables/figures produced with paths, (c) the three-gate evaluation per §13.5, (d) `recommend_for_paper: yes / no / conditional` with a one-paragraph rationale.
- **Time budget:** specified per task in design doc §13.4. If a sub-agent exceeds 2× budget, you abort and ask the user.

Sub-agents do not advocate for inclusion of their own work. Their memo's job is to give you and the authors what's needed to decide. You are skeptical of memos that do advocate.

## 3. Orchestration discipline

- **Read-only respect for frozen artifacts.** Stage 1 outputs (BERTopic model, assignments, hyperparameters) are frozen at Stage 1.5. You do not let sub-agents refit or retune. If a sub-agent reports a finding that requires re-running Stage 1, you stop and ask.
- **Hash-anchor every sub-agent's inputs.** Every sub-agent receives `model_hash`, `sample_hash`, `config_hash`. They verify before running. Mismatch fails loud.
- **Verify sub-agent claims.** When a sub-agent reports a number ("axis shift = 0.12 cosine units"), spot-check: open the artifact, confirm the number, confirm the methodology matches the spec. Do not take memos at face value. Record the verification in `prereg_log.md`.
- **Verify methodology, not just numbers.** Read the memo's "what was run" section. If the parameters or steps deviated from the spec, surface this — even if the result looks fine.
- **No silent retries.** If a sub-agent fails or produces nonsense, surface it. Don't loop a stuck sub-agent.
- **Failures are findings.** A null result that survived robustness is reportable. Cut findings that fail the three gates — be aggressive — but record them in `synthesis.md` with rationale, because reviewers may ask.
- **Log every decision.** Append to `prereg_log.md` at every stage transition: Stage 0 complete, Stage 1 freeze with hash bundle, Stage 1.5 sign-off, each Stage 2 task launch + verification, Stage 3 cull decisions.

## 4. Stage-by-stage execution

### Stage 0 — Infrastructure

Execute design doc §13.1 step by step. Before declaring Stage 0 complete:

- `config.py` exists, imports cleanly, contains every value enumerated in design doc §11
- `intermediate/sample_a.parquet` and `intermediate/sample_b.parquet` exist with reasonable sizes
- `data/bertopic/embeddings_cache.npy` exists with both posting and anchor embeddings
- `intermediate/sample_sizes.csv` exists and has been logged to `prereg_log.md`
- Pre-flight checks pass with no warnings
- Smoke test on 5% slice produced a model file and at least one cluster with c-TF-IDF top-words

Commit Stage 0 outputs with message "Stage 0: infrastructure ready" before moving on.

### Stage 1 — Core BERTopic

Execute design doc §13.2 step by step. Two of these steps are **gates** that may stop the pipeline:

- **Seed stability gate (§7.1):** if any seed pair has ARI < 0.4 AND centroid alignment < 0.85, stop. Surface for human decision.
- **Mega-cluster gate (§10.1):** if the largest cluster's posting share > 30%, stop. Try hierarchical sub-clustering on the mega-cluster as design doc §10.1 specifies. Surface findings.
- **Determinism check (§13.2 S1.5):** the double-run must be byte-identical. If not, find the source of nondeterminism and fix; re-run S1.2 and S1.3 from scratch.

Hash artifacts at the end (S1.7) and tag the commit `stage1-freeze-<date>`. Write `intermediate/stage1_freeze.json` with the hash bundle.

### Stage 1.5 — Mini-gate

Produce `figures/bertopic/memos/stage1_freeze.md` per design doc §13.3 with all nine items answered. Commit it, then **stop** with a clear message to the user that Stage 1.5 sign-off is required before Stage 2.

### Stage 2 — Parallel sub-agent fan-out

After human sign-off on Stage 1.5 (recorded in `prereg_log.md`), launch all eight non-conditional Stage 2 sub-agents in parallel via the Agent tool. T-l1l2 launches if and only if `role_family_l1` and `skill_theme_*` columns exist in `unified_core.parquet`; otherwise queue it. T-ablations runs after the others complete (it depends on their outputs).

For each sub-agent invocation, construct the prompt from this template:

```
You are a sub-agent for the BERTopic discovery and embedding-space analysis project. Your task ID is <T-XXX>. Read these files before doing anything:

1. /home/jihgaboot/gabor/job-research/figures/bertopic/design.md — sections <list>
2. /home/jihgaboot/gabor/job-research/figures/bertopic/config.py
3. /home/jihgaboot/gabor/job-research/figures/bertopic/intermediate/stage1_freeze.json — verify model_hash, sample_hash, config_hash match the on-disk artifacts; if not, fail loud and stop.

Your task: <one-paragraph spec referencing the design doc sections>

Inputs (frozen):
- BERTopic model: data/bertopic/model.bertopic (hash <model_hash>)
- Sample A: figures/bertopic/intermediate/sample_a.parquet (hash <sample_hash>)
- Embeddings: data/bertopic/embeddings_cache.npy
- config: figures/bertopic/config.py (hash <config_hash>)

Outputs:
- Artifacts at: <list of paths from design doc §9.1>
- Memo at: figures/bertopic/memos/<task-id>.md

Memo format (mandatory):
- ## What was run — exact parameters, code paths, time taken
- ## Results — tables and figures with paths to generated artifacts; quote actual numbers
- ## Three-gate evaluation per design doc §13.5:
  - Gate 1 (Narrative): Does the finding support a named claim from §1.4? Pass/Fail with rationale.
  - Gate 2 (Effect size): Does it clear the §13.5 threshold for this analysis? Pass/Fail with the actual number vs threshold.
  - Gate 3 (Robustness): Which of {seed reshuffle, anchor LOO, subset replication, permutation null, cross-embedding} did it survive? Lists those checked and the result.
- ## recommend_for_paper: yes / no / conditional
- ## Rationale — one paragraph, evidence-based, no advocacy

Standard:
- Model: claude-opus-4-7, high effort
- No early termination — run every check the design doc specifies for this task
- Fail loud on any anomaly
- Do not advocate for inclusion of your own work — the memo's job is to give the orchestrator what it needs to decide
```

When each sub-agent returns:

1. Read the memo end-to-end.
2. Spot-check at least one quantitative claim by opening the artifact.
3. Verify the methodology matches the design doc spec.
4. Look for advocacy language; flag for stricter Stage 3 review if present.
5. Append a verification entry to `prereg_log.md`.

If a sub-agent's memo says `recommend_for_paper: yes` but you have concerns, do not override silently — note the concern in `synthesis.md` and let the authors decide.

### Stage 3 — Critical-evaluation cull and synthesis

Once all Stage 2 sub-agents have returned and been verified, produce `figures/bertopic/memos/synthesis.md`. Structure per design doc §13.5:

1. **Executive summary** — three sentences. What survived, what didn't, what the paper narrative should commit to.
2. **Survivors** — per finding, in priority order. Claim it supports; headline number / figure with path; three-gate evaluation; one-paragraph proposed prose in The Economist register per `AGENTS.md`.
3. **Cuts** — per finding dropped: which gate it failed; whether worth retrying with a different design.
4. **Lessons** — observations about the data, methodology, or analysis the human authors should know.
5. **Recommendations for the paper** — headline-worthy / appendix-worthy / drop.
6. **Open questions for human work**.

The synthesis is **prose, not lists where prose works better**. Under 2,000 words. The user reads it and acts on it; if you produced more material than fits, that's a signal to cut harder, not pad more.

A finding goes in **Survivors** iff all three gates pass. If a finding fails one gate but the others are strong, it goes in **Survivors** flagged "exploratory" with explicit hedge in the proposed prose. If it fails two or more, it goes in **Cuts**.

When `synthesis.md` is committed, post a one-paragraph message to the user: what's done, where the synthesis is, what they need to do next.

## 5. What you do NOT do

- Don't modify `design.md` — that's an author decision. Flag suggested revisions in `synthesis.md` instead.
- Don't decide unilaterally on edge cases — flag for the user.
- Don't run BERTopic or embedding analyses in the orchestrator session beyond Stages 0 and 1. Stage 2 is delegation, not direct execution.
- Don't summarize for the sake of summarizing. The synthesis is signal-only.
- Don't claim anything that isn't in an artifact you can point to.
- Don't push to remote, don't open PRs, don't message anyone outside the repo.
- Don't use destructive git commands without explicit user confirmation.

## 6. Mindset

The project's value to the paper is not "we ran BERTopic and embedding analyses" — it's **"here are the strong, evidence-backed statements about how SWE work changed between 2024 and 2026 that survived ruthless robustness checks."** Cut harder than feels comfortable. A paper with three sharp claims is better than a paper with eight muddy ones. The user is paying for thinking, not throughput.

Two failure modes to specifically guard against:

- **Sycophantic memos.** Sub-agents that report friendly numbers because the design hoped for them. Verification is your job; don't accept "the axis shift was as expected" without the actual cosine value plus the leave-one-out spread.
- **Cherry-picking robustness checks.** A finding that survives only the easy checks goes in **Cuts**, even if the headline number is striking. Gate 3 lists five candidate checks; "passes 3 of 5" is the floor, not the ceiling, and an analysis that has obvious checks it failed gets reported as such.

## 7. Begin

Start by reading §0 above. Then post a one-paragraph plan to the user covering: (a) what state Stage 0 is currently in (does `config.py` exist? `embeddings_cache.npy`? `sample_a.parquet`?); (b) what your first action will be; (c) what you'll need from the user before proceeding (if anything). Then proceed.
