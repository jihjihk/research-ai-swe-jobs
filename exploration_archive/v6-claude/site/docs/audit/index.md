# Audit trail

This tab is the raw evidence. Every task report, every gate memo, both verification passes, and the full task inventory.

## Structure

- **[Task inventory](reports/INDEX.md)** — the full state table for all 26 tasks, organized by wave, with headline summaries.
- **Wave 1 reports** — T01 through T07. Data foundation.
- **Wave 2 reports** — T08 through T15. Open structural discovery.
- **Wave 3 reports** — T16 through T23, T28, T29. Market dynamics.
- **Wave 4 reports** — T24, T25. Hypothesis generation + interview artifacts. T26 is the SYNTHESIS.md document in the Narrative tab.
- **Gate memos** — gate_0 through gate_3_corrections.
- **Verifications** — V1 (Gate 2) and V2 (Gate 3) verification passes.

## Use this tab when

- You want to check a specific number cited in a finding.
- You are a reviewer and want to see the original reasoning, not just the synthesis.
- You want to understand a sensitivity check that a finding page mentions.
- You want to see what was dropped and why (gate corrections).

## Figure references

Where task reports cite figure paths like `exploration/figures/T21/density_profile_shift.png`, the corresponding files are available at `../../assets/figures/T21/density_profile_shift.png` within this site, and have been preserved at their original paths in the repository. The most important figures are embedded directly in the Findings pages.

## One-line rule: SYNTHESIS is the primary, reports are the audit

If a claim appears in SYNTHESIS.md, it has been through V1 and V2 verification. If a claim appears only in a task report, it may have been overturned later — check the gate memos and the verifications before citing.
