"""T07 — JOLTS Information sector context plot.

Reads the cached FRED CSV at /tmp/fred_JTU5100JOL.csv (pre-downloaded by the
BLS fetch step) and copies it into exploration/tables/T07/, then produces a
contextual plot showing our two sampling windows (2024-01 / 2024-04 / 2026-03
/ 2026-04) overlaid on JOLTS information-sector job openings since 2023.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "exploration" / "tables" / "T07"
FIG = REPO / "exploration" / "figures" / "T07"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SRC = Path("/tmp/fred_JTU5100JOL.csv")
if not SRC.exists():
    raise SystemExit(f"missing source: {SRC} — run the bls fetch script first")

DEST = OUT / "jolts_info_openings.csv"
shutil.copy(SRC, DEST)
print(f"Wrote {DEST}")

with SRC.open() as f:
    rows = list(csv.DictReader(f))

dates = [datetime.strptime(r["observation_date"], "%Y-%m-%d") for r in rows]
values = [float(r["JTU5100JOL"]) if r["JTU5100JOL"] else None for r in rows]

# Filter last 5 years for context.
cutoff = datetime(2021, 1, 1)
pairs = [(d, v) for d, v in zip(dates, values) if d >= cutoff and v is not None]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot([p[0] for p in pairs], [p[1] for p in pairs], color="steelblue", linewidth=1.5)
ax.set_ylabel("JOLTS Information sector\njob openings (thousands)")
ax.set_xlabel("Month")
ax.set_title(
    "JOLTS Information sector openings vs our sampling windows\n"
    "(source: FRED JTU5100JOL, not seasonally adjusted)"
)
# Shade our sampling windows
windows = [
    (datetime(2024, 1, 1), datetime(2024, 2, 1), "arshkon\n2024-04", "#e8f1f8"),
    (datetime(2024, 4, 1), datetime(2024, 5, 1), "asaniczka\n2024-01", "#fde9d9"),
    (datetime(2026, 3, 1), datetime(2026, 5, 1), "scraped\n2026-03/04", "#e2f0d9"),
]
for start, end, label, color in windows:
    ax.axvspan(start, end, color=color, alpha=0.9, zorder=0)

# Annotations
# Find y positions
def val_near(target):
    for d, v in pairs:
        if d >= target:
            return v
    return None

for start, end, label, color in windows:
    mid = start + (end - start) / 2
    y = val_near(mid)
    if y:
        ax.annotate(label, xy=(mid, y), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_png = FIG / "jolts_info_context.png"
plt.savefig(out_png, dpi=150)
print(f"Wrote {out_png}")

# Summary stats for the windows
print("\nJOLTS info openings summary (thousands):")
def avg(start, end):
    vs = [v for d, v in zip(dates, values) if v is not None and start <= d < end]
    return sum(vs) / len(vs) if vs else None

def fmt(x):
    return f"{x:.0f}" if x is not None else "n/a"

print(f"  2024-01 to 2024-05 (our 2024 windows):     avg={fmt(avg(datetime(2024,1,1), datetime(2024,5,1)))}")
print(f"  2026-01 to 2026-03 (closest to scraped):   avg={fmt(avg(datetime(2026,1,1), datetime(2026,3,1)))}")
print(f"  2025 avg:                                  avg={fmt(avg(datetime(2025,1,1), datetime(2026,1,1)))}")
print(f"  2023 avg:                                  avg={fmt(avg(datetime(2023,1,1), datetime(2024,1,1)))}")
print(f"  2021-2022 avg (post-pandemic peak):        avg={fmt(avg(datetime(2021,1,1), datetime(2023,1,1)))}")
print("\nNote: FRED JTU5100JOL last observation is 2026-02; the scraped window "
      "extends beyond the last released JOLTS point.")
