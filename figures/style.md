# Figure style — AAAI 2026

**Read this before writing any figure script.** Every figure that ends up in the paper must go through `figures/style.py` so the manuscript has one consistent visual language.

## Library

matplotlib + [SciencePlots](https://github.com/garrettj403/SciencePlots). Install once: `pip install scienceplots`. Do not introduce seaborn, plotnine, plotly, altair, or bokeh — one library, one style.

## Required idiom

```python
from figures.style import setup, save, FIGSIZE_SINGLE, FIGSIZE_DOUBLE
import matplotlib.pyplot as plt

setup()
fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)  # FIGSIZE_DOUBLE for full-width spans
# ... plot ...
save(fig, "fig_skill_trends")
```

`save()` writes `figures/output/fig_skill_trends.pdf` (canonical asset for `\includegraphics`) and a `.png` preview alongside.

## Sizing

AAAI 2026 is two-column: textwidth 7.0 in, column 3.3125 in. Use `FIGSIZE_SINGLE` for single-column figures, `FIGSIZE_DOUBLE` for figures spanning both columns (the latter goes inside `figure*`, not `figure`).

In the .tex, include with `\includegraphics[width=\columnwidth]{name}` (single) or `\includegraphics[width=\textwidth]{name}` (double). Figures are authored at the final size, so this is a near-no-op scale. If LaTeX raises an overfull warning, fall back to `0.95\columnwidth`.

## What the style does for you

- **Times serif, 9 pt** — matches AAAI body type at native scale.
- **Off-black `#1f1f1f`** instead of pure black — softer in print and on screen.
- **No top/right spines, faint horizontal grid behind data** — Tufte-ish framing; keeps attention on the data.
- **Tol bright palette** — colorblind-safe, distinguishable in grayscale.
- **TrueType (`pdf.fonttype = 42`)** — AAAI forbids Type 3 fonts; matplotlib's default would violate this and is the most common reason a submission gets bounced.
- **Tight bbox, 300 DPI raster.**

## Don'ts

- Don't override `rcParams` inside a figure script. If the style needs to change, edit `figures/style.py` once so every figure inherits the change.
- Don't `plt.title()` for the paper title — use the LaTeX caption.
- Don't ship `.png` as the primary asset. `.pdf` is what `\includegraphics` should reference.
