# AI/tool expansion diffused across metros rather than staying inside a few hubs.

<p class="lead">Geography is supporting evidence for broad diffusion. T17 finds positive AI/tool movement in every eligible metro, with tech hubs somewhat larger but not uniquely responsible.</p>

## Evidence

Across 26 metros with at least 50 SWE LinkedIn postings in both pooled 2024 and scraped 2026:

| Metric | Mean change | Median change | Minimum | Maximum | Positive metros |
|---|---:|---:|---:|---:|---:|
| Broad AI prevalence | +17.6 pp | +17.8 pp | +8.7 pp | +27.4 pp | 26 / 26 |
| AI-tool strict prevalence | +16.1 pp | +16.4 pp | +6.7 pp | +24.1 pp | 26 / 26 |
| Requirement breadth, LLM text | +2.14 | +2.20 | +0.53 | +3.48 | 26 / 26 |
| Mean tech count | +2.00 | +2.02 | +0.31 | +3.21 | 26 / 26 |

Source: [T17](../raw/reports/T17.md).

<div class="figure-frame">
  <img src="../assets/figures/T17/metro_metric_delta_heatmap.png" alt="Metro metric delta heatmap from T17">
  <div class="figure-caption">The metro heatmap is useful because the all-positive direction is the point. It is not a claim about precise local vacancy rates.</div>
</div>

## Mechanism clues

Broad-AI change is correlated with requirement-breadth change (r = 0.44, p = 0.026), but not with J1 entry-share change, J3 low-YOE share change, or 2024 posting volume. T17 therefore supports diffusion of posting language more than a metro-specific junior-labor explanation.

## Boundaries

- Metro rollups exclude multi-location rows and unresolved locations.
- The pooled-2024 geography result is source-sensitive for label-based junior shares.
- Remote work is not usable: T17 found the 2026 remote field all-zero in this frame, requiring a scraper/location-stage audit.
- V2 did not independently verify T17.
