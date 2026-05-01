"""T18 Step 9 — DiD visualization: ranked DiD with CIs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TAB = ROOT / "exploration" / "tables" / "T18"
FIG = ROOT / "exploration" / "figures" / "T18"

did = pd.read_csv(TAB / "T18_did_table.csv")
base = did[~did["exclude_aggregator"]].copy()
base["abs_did_share"] = base["did_share_of_swe_change"].abs()

# Normalize each metric into a DiD share comparable across scales.
# Show did_share_of_swe_change and flag swe_specific vs mostly_macro.
plot_df = base.sort_values("did_share_of_swe_change")

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = plot_df["flag"].map({
    "swe_specific": "#d62728",
    "mostly_macro": "#1f77b4",
    "no_swe_change": "#bbbbbb",
})
ax.barh(plot_df["metric"], plot_df["did_share_of_swe_change"], color=colors)
ax.axvline(0, color="black", linewidth=1)
ax.axvline(0.5, color="gray", linestyle="--", label="50% threshold")
ax.set_xlabel("DiD / SWE-only change (share)")
ax.set_title("T18 DiD — fraction of SWE-only change that is SWE-specific vs macro")
ax.grid(True, axis="x", alpha=0.3)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "T18_did_share.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote", FIG / "T18_did_share.png")
