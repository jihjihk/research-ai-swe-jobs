"""T19 plots: timeline, scraped daily trends, within-arshkon, March vs April."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import datetime as dt

TAB = "exploration/tables/T19"
OUT = "exploration/figures/T19"
os.makedirs(OUT, exist_ok=True)

# -------- 1. Scraped March+April daily (AI rate, entry_native, median_len) --------
m = pd.read_csv(os.path.join(TAB, "scraped_march_daily.csv"))
a = pd.read_csv(os.path.join(TAB, "scraped_april_daily.csv"))
daily = pd.concat([m, a], ignore_index=True)
daily["scrape_date"] = pd.to_datetime(daily["scrape_date"])
daily = daily.sort_values("scrape_date")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].bar(daily.scrape_date, daily.n, width=0.7, color="#888")
axes[0].set_ylabel("n SWE postings")
axes[0].set_title("Scraped daily SWE posting volume (2026-03 + 2026-04)")

axes[1].plot(daily.scrape_date, daily.ai_rate * 100, "-o", label="AI rate", color="#d62728")
axes[1].plot(daily.scrape_date, daily.agentic_rate * 100, "-s", label="agentic", color="#8b4513")
axes[1].set_ylabel("% postings")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title("AI keyword prevalence (daily)")

axes[2].plot(daily.scrape_date, daily.median_len, "-o", color="#2ca02c", label="median_len")
axes[2].set_ylabel("chars")
ax3 = axes[2].twinx()
ax3.plot(daily.scrape_date, daily.entry_native_rate * 100, "--x", color="#1f77b4", label="entry_native")
ax3.set_ylabel("entry_native %")
axes[2].set_title("Median description length and entry_native share")
axes[2].grid(alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "scraped_daily.png"), dpi=140, bbox_inches="tight")
plt.close()

# -------- 2. Rate-of-change comparison --------
rates = pd.read_csv(os.path.join(TAB, "rates_annualized.csv"))
fig, ax = plt.subplots(figsize=(11, 5))
rates_plot = rates[rates.metric.isin(["entry_best", "yoe_le2", "ai_rate", "agentic_rate", "scope_rate"])].copy()
x = np.arange(len(rates_plot))
w = 0.35
ax.bar(x - w / 2, rates_plot.within_2024_annualized * 100, w, label="within-2024 annualized", color="#1f77b4")
ax.bar(x + w / 2, rates_plot.cross_period_annualized * 100, w, label="cross-period annualized", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(rates_plot.metric, rotation=25, ha="right")
ax.set_ylabel("pp/year")
ax.set_title("Annualized rate of change: within-2024 vs cross-period")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "rates_annualized.png"), dpi=140, bbox_inches="tight")
plt.close()

# -------- 3. Timeline with AI tool releases and our snapshots --------
releases = [
    ("GPT-4", "2023-03-14"),
    ("Claude 3", "2024-03-04"),
    ("GPT-4o", "2024-05-13"),
    ("Claude 3.5 Sonnet", "2024-06-20"),
    ("o1", "2024-09-12"),
    ("DeepSeek V3", "2024-12-26"),
    ("GPT-4.5", "2025-02-27"),
    ("Claude 3.6 Sonnet", "2025-04-01"),
    ("Claude 4 Opus", "2025-09-01"),
    ("Gemini 2.5 Pro", "2026-03-01"),
]
snapshots = [
    ("asaniczka\nJan 2024", "2024-01-14", "#1f77b4"),
    ("arshkon\nApr 2024", "2024-04-13", "#ff7f0e"),
    ("scraped\nMar 2026", "2026-03-24", "#d62728"),
    ("scraped\nApr 2026", "2026-04-03", "#8b0000"),
]
fig, ax = plt.subplots(figsize=(13, 4.5))
for lbl, d in releases:
    dte = pd.to_datetime(d)
    ax.axvline(dte, color="#999", linestyle="--", linewidth=0.8)
    ax.text(dte, 0.95, lbl, rotation=55, va="top", ha="right", fontsize=8, color="#555")
for lbl, d, c in snapshots:
    dte = pd.to_datetime(d)
    ax.axvline(dte, color=c, linewidth=2.5, alpha=0.9)
    ax.text(dte, 0.3, lbl, rotation=0, va="center", ha="center", fontsize=9,
            fontweight="bold", color=c, bbox=dict(facecolor="white", edgecolor=c, pad=3))
ax.set_xlim(pd.to_datetime("2023-01-01"), pd.to_datetime("2026-06-01"))
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_title("AI tool releases and data snapshots (qualitative context)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "timeline.png"), dpi=140, bbox_inches="tight")
plt.close()

# -------- 4. March vs April snapshot comparison --------
cmp = pd.read_csv(os.path.join(TAB, "march_vs_april_2026.csv"))
metrics = ["ai_rate", "agentic_rate", "entry_best", "yoe_le2", "entry_native"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metrics))
w = 0.35
vals_m = cmp[cmp.period == "2026-03"][metrics].iloc[0] * 100
vals_a = cmp[cmp.period == "2026-04"][metrics].iloc[0] * 100
ax.bar(x - w / 2, vals_m, w, label="2026-03", color="#d62728")
ax.bar(x + w / 2, vals_a, w, label="2026-04", color="#8b0000")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=20, ha="right")
ax.set_ylabel("%")
ax.set_title("2026-03 vs 2026-04 stability check (SWE)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "march_vs_april_2026.png"), dpi=140, bbox_inches="tight")
plt.close()

print("Plots saved to", OUT)
