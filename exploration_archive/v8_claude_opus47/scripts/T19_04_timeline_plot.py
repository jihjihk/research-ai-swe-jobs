"""T19 Step 6 — Timeline contextualization.

Plot AI-mention (strict) over time with AI tool releases annotated.
Qualitative overlay linking our 3 snapshots to model-release chronology.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
ART_T18 = ROOT / "exploration" / "artifacts" / "T18"
TAB = ROOT / "exploration" / "tables" / "T19"
FIG = ROOT / "exploration" / "figures" / "T19"
FIG.mkdir(parents=True, exist_ok=True)

SNAPSHOTS = [
    ("Asaniczka 2024-01", datetime(2024, 1, 15), None, "kaggle_asaniczka"),
    ("Arshkon 2024-04", datetime(2024, 4, 12), None, "kaggle_arshkon"),
    ("Scraped 2026-03", datetime(2026, 3, 25), datetime(2026, 3, 30), "scraped_march"),
    ("Scraped 2026-04", datetime(2026, 4, 7), datetime(2026, 4, 14), "scraped_april"),
]

RELEASES = [
    ("GPT-4", datetime(2023, 3, 14)),
    ("Claude 3", datetime(2024, 3, 4)),
    ("GPT-4o", datetime(2024, 5, 13)),
    ("Claude 3.5 Sonnet", datetime(2024, 6, 20)),
    ("o1", datetime(2024, 9, 12)),
    ("Claude 4", datetime(2025, 9, 1)),
    ("Gemini 2.5 Pro", datetime(2026, 3, 1)),
]


def main():
    feat = pd.read_parquet(ART_T18 / "T18_posting_features.parquet")
    swe = feat[feat["group"] == "SWE"].copy()
    snap_map = {
        "kaggle_asaniczka": datetime(2024, 1, 15),
        "kaggle_arshkon": datetime(2024, 4, 12),
    }
    swe["dt"] = swe.apply(
        lambda r: snap_map.get(r["source"]) if r["source"] != "scraped" else pd.to_datetime(r["scrape_date"]),
        axis=1,
    )
    swe = swe.dropna(subset=["dt"])
    # Bucket into days
    swe["ds"] = pd.to_datetime(swe["dt"]).dt.date

    agg = (
        swe.groupby("ds")
        .agg(
            n=("uid", "count"),
            ai_strict=("ai_strict_binary", "mean"),
            ai_broad=("ai_broad_binary", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(TAB / "T19_timeline_daily_ai.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(pd.to_datetime(agg["ds"]), agg["ai_strict"], "o-", label="AI-mention (strict)", color="#d62728")
    ax.plot(pd.to_datetime(agg["ds"]), agg["ai_broad"], "o-", label="AI-mention (broad)", color="#ff7f0e", alpha=0.6)

    # Annotate snapshots
    ymax = agg["ai_broad"].max() * 1.1
    for label, dt, _, _ in SNAPSHOTS:
        ax.axvspan(dt - pd.Timedelta(days=1), dt + pd.Timedelta(days=1), color="#2ca02c", alpha=0.15)

    # Annotate releases (only within plot range)
    dates = pd.to_datetime(agg["ds"])
    xmin, xmax = dates.min() - pd.Timedelta(days=30), dates.max() + pd.Timedelta(days=30)
    for label, dt in RELEASES:
        if xmin <= dt <= xmax:
            ax.axvline(dt, color="gray", linestyle=":", alpha=0.6)
            ax.text(dt, ymax * 0.95, label, rotation=90, fontsize=8, va="top", ha="right")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("AI-mention prevalence (SWE)")
    ax.set_title("T19 Timeline — AI-mention prevalence with release annotations")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "T19_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {FIG / 'T19_timeline.png'}")


if __name__ == "__main__":
    main()
