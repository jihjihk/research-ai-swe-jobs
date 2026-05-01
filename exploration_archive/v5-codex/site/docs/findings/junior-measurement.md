# Explicit entry is a conservative lower bound; low-YOE postings remain.

## Claim
The junior story depends on the instrument: explicit labels are conservative, while the YOE proxy captures a broader junior-like pool.

T03, T08, and T20 all point in the same direction. `seniority_final = entry` is tiny, but `yoe_extracted <= 2` is materially broader and often rises when the explicit label does not. That is not a nuisance; it is the central measurement constraint on any junior claim. The right headline is not that juniors vanished; it is that low-YOE postings remain even when explicit entry labels stay scarce.

## Evidence
- `seniority_final = entry` is 3.73% in arshkon and 2.18% in scraped 2026-04.
- The YOE proxy is 14.98% in arshkon and 16.97% in scraped 2026-04.
- Arshkon native `entry` rows are not a stable junior baseline across snapshots, which is why `seniority_native` is diagnostic only.
- T20 shows `associate -> mid-senior` is the sharpest boundary, while `entry -> associate` is the weakest.

## Figures
![T03_junior_share_comparison.png](../assets/figures/T03/T03_junior_share_comparison.png)

![T03_native_vs_final_heatmaps.png](../assets/figures/T03/T03_native_vs_final_heatmaps.png)

![T08_junior_share_trends.png](../assets/figures/T08/T08_junior_share_trends.png)

## Sensitivity and caveats
- Asaniczka cannot be pooled into a `seniority_native` entry baseline because it has zero native entry labels.
- Any junior figure should show explicit-entry and YOE proxy side by side.
- Material disagreement between the instruments is itself a finding, not a problem to hide.

## Raw trail
- [T03 report](../audit/raw/reports/T03.md)
- [T08 report](../audit/raw/reports/T08.md)
- [T20 report](../audit/raw/reports/T20.md)
- [T02 report](../audit/raw/reports/T02.md)

## What this means
- Junior collapse is too simple; the more defensible claim is instrument-dependent junior visibility.
- The analysis-phase rule should be: show explicit labels and YOE together, or do not make a junior claim.
- Explicit entry is a lower bound, not a complete census of junior work.
