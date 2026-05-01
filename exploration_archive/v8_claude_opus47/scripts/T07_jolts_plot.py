"""Plot JOLTS Information-sector monthly openings with our observation windows highlighted."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
OUT_FIGS = ROOT / "exploration" / "figures" / "T07"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "exploration" / "tables" / "T07" / "jolts_info_monthly.csv",
                 parse_dates=["date"])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["date"], df["value_thousands"], marker="o", markersize=4,
        linewidth=1.5, color="#1f77b4", label="JOLTS Info sector openings")
# Our windows
ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31"),
           color="orange", alpha=0.25, label="Asaniczka (2024-01)")
ax.axvspan(pd.Timestamp("2024-04-01"), pd.Timestamp("2024-04-30"),
           color="green", alpha=0.25, label="Arshkon (2024-04)")
ax.axvspan(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-30"),
           color="red", alpha=0.25, label="Scraped (2026-03/04)")

# Annotate values
for _, r in df.iterrows():
    if r["date"] in [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-04-01"),
        pd.Timestamp("2026-02-01"),  # most recent available
    ]:
        ax.annotate(f"{r['value_thousands']}K",
                    (r["date"], r["value_thousands"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("Job openings (thousands)")
ax.set_xlabel("Month")
ax.set_title("JOLTS Information Sector Job Openings (FRED: JTS510000000000000JOL)\n"
             "Note: 2026-April JOLTS not yet published; window positioned at most-recent Feb-2026 value")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "jolts_info_openings.png", dpi=150)
plt.close()
print(f"Saved: {OUT_FIGS / 'jolts_info_openings.png'}")

# Contextual ratios
feb_2026 = df[df["date"] == pd.Timestamp("2026-02-01")]["value_thousands"].iloc[0]
avg_2023 = df[df["year"] == 2023]["value_thousands"].mean()
avg_2024 = df[df["year"] == 2024]["value_thousands"].mean()
avg_2025 = df[df["year"] == 2025]["value_thousands"].mean()
print(f"\n2023 avg: {avg_2023:.0f}K")
print(f"2024 avg: {avg_2024:.0f}K")
print(f"2025 avg: {avg_2025:.0f}K")
print(f"2026 Feb: {feb_2026}K")
print(f"2026 vs 2023: {feb_2026 / avg_2023:.2f}x")
print(f"2026 vs 2024: {feb_2026 / avg_2024:.2f}x")
