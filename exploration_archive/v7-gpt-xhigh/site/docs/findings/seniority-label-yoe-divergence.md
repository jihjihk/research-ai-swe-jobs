# Entry labels and low-YOE floors identify different postings.

<p class="lead">The exploration should not report one junior-share number. Employer labels, low-YOE floors, and semantic neighborhoods point to different populations.</p>

## The junior-collapse claim weakened

T30 finds J1-J4 all rise from pooled 2024 to scraped 2026, but the signal is not cleanly above within-2024 source variability. T16 then shows a split in the cleaner arshkon-to-scraped common-company panel: explicit entry labels decline slightly while low-YOE shares rise.

| Panel | J1 entry label | J3 YOE <= 2 | Interpretation |
|---|---:|---:|---|
| T30 pooled 2024 -> scraped 2026 | +3.99 pp known-denominator effect | +5.98 pp known-denominator effect | Directionally up, source-calibration constrained |
| T16 common companies, arshkon -> scraped | -0.44 pp | +6.63 pp | Label and YOE constructs split |
| T17 metros, pooled 2024 -> scraped | +1.12 pp mean metro change | +4.95 pp mean metro change | Mostly up, but arshkon-only flips label direction |
| T18 occupation panel | +1.46 pp | +5.98 pp | SWE junior definitions up |

Sources: [T30](../raw/reports/T30.md), [T16](../raw/reports/T16.md), [T17](../raw/reports/T17.md), [T18](../raw/reports/T18.md).

<div class="figure-frame">
  <img src="../assets/figures/T08/seniority_panel_changes.png" alt="Seniority panel changes from T08">
  <div class="figure-caption">T08 shows why junior-share direction should be treated as a measurement puzzle, not a lead collapse claim.</div>
</div>

## Low-YOE rows are usually not entry-labeled

In scraped 2026, T08 reports that 72.4% of rows with `yoe_extracted <= 2` are `seniority_final = unknown`, 21.5% are `mid-senior`, and only 5.5% are J1 entry.

T15 adds semantic evidence: label-based J1 rows remain junior-neighbor enriched, while low-YOE J3/J4 rows are more senior-neighbor-heavy under embeddings. That supports boundary ambiguity rather than generic junior-senior convergence.

<div class="figure-frame">
  <img src="../assets/figures/T30/junior_overlap_heatmap.png" alt="Junior definition overlap heatmap from T30">
  <div class="figure-caption">T30's overlap panel shows that label-based and YOE-based junior definitions are related but not interchangeable.</div>
</div>

## Seniority implications

The senior side is also split. T30 and T08 find broad senior and YOE-senior shares down, while director-only labels rise. T11 finds senior requirement breadth and mentorship/coordination language rising, so the defensible senior claim is broadening and ladder ambiguity, not a decline in management work.

## Boundaries

T20 failed, so there is no recovered seniority-boundary classifier or formal low-YOE unknown-pool model. T21 failed, so senior-role redefinition should not be led beyond T11-supported breadth, mentorship, and coordination evidence.
