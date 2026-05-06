# Stage 2 dispatch plan

The orchestrator launches eight sub-agents in parallel via the `Agent`
tool with `subagent_type=general-purpose`, `model=opus`. Each prompt is
the Stage 2 template (`STAGE2_PROMPT_TEMPLATE.md`) with `<TASK_ID>`,
`<SECTIONS>`, the per-task spec from `STAGE2_TASK_SPECS.md`, and the
hash bundle from `intermediate/stage1_freeze.json` filled in. T-l1l2 is
queued (L1/L2 columns absent in `unified_core.parquet`). T-ablations
runs after the others complete.

## Dispatch table

| Order | Task ID | Sections | Time budget | Memo path |
|---|---|---|---|---|
| Wave 1 (parallel) | T-axis | §6.1, §6.6, §11.7, §11.9, §13.4, §13.5 | 45 min | `memos/t_axis.md` |
| Wave 1 (parallel) | T-boundary | §6.2, §13.4, §13.5 | 30 min | `memos/t_boundary.md` |
| Wave 1 (parallel) | T-drift | §6.3, §6.1, §13.4, §13.5 | 45 min | `memos/t_drift.md` |
| Wave 1 (parallel) | T-weat | §6.4, §11.7 (WEAT_*), §13.4, §13.5 | 30 min | `memos/t_weat.md` |
| Wave 1 (parallel) | T-anchor | §6.5, §13.4, §13.5 | 30 min | `memos/t_anchor.md` |
| Wave 1 (parallel) | T-bootstrap | §7.2, §7.3, §7.6, §13.4, §13.5 | 90 min | `memos/t_bootstrap.md` |
| Wave 1 (parallel) | T-method | §7.4, §7.5, §13.4, §13.5 | 60 min | `memos/t_method.md` |
| Wave 1 (parallel) | T-quality | §7.8–§7.11, §13.4, §13.5 | 45 min | `memos/t_quality.md` |
| Wave 2 (after) | T-ablations | §8.2, §9.5, §13.4, §13.5 | 3 hr | `memos/t_ablations.md` |

T-l1l2: queued (`role_family_l1` / `skill_theme_*` not yet populated).

## Verification checklist (orchestrator runs after each return)

For each Stage 2 memo:
1. Read end-to-end.
2. Spot-check ≥ 1 quantitative claim by opening the named artifact.
3. Verify methodology section matches the design-doc spec.
4. Flag advocacy language (memos defending the analysis vs reporting).
5. Append a verification entry to `prereg_log.md` (task ID, hash bundle
   verified, claims spot-checked, deviations flagged).
