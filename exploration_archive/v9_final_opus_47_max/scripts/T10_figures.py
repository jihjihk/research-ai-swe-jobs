"""T10 figures — title evolution charts."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "exploration" / "tables" / "T10"
FIG = ROOT / "exploration" / "figures" / "T10"
FIG.mkdir(parents=True, exist_ok=True)


def fig_concentration():
    df = pd.read_csv(TAB / "step2_concentration_full.csv")
    cap20 = pd.read_csv(TAB / "step2_concentration_cap20.csv")
    cap50 = pd.read_csv(TAB / "step2_concentration_cap50.csv")
    order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    labels = order
    y_full = df.set_index("group").loc[order, "unique_per_1k"]
    y_cap20 = cap20.set_index("group").loc[order, "unique_per_1k"]
    y_cap50 = cap50.set_index("group").loc[order, "unique_per_1k"]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, y_full, width=w, label="full", color="steelblue")
    ax.bar(x, y_cap50, width=w, label="cap=50", color="orange")
    ax.bar(x + w, y_cap20, width=w, label="cap=20", color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Unique titles per 1K postings")
    ax.set_title("Title vocabulary richness (unique-per-1K)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    y_top10_full = df.set_index("group").loc[order, "top10_share"]
    y_top50_full = df.set_index("group").loc[order, "top50_share"]
    ax2.bar(x - 0.2, y_top10_full, width=0.4, label="top-10 share", color="teal")
    ax2.bar(x + 0.2, y_top50_full, width=0.4, label="top-50 share", color="salmon")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("Title concentration (head of distribution)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "concentration.png", dpi=150)
    plt.close(fig)


def fig_ai_terms():
    df = pd.read_csv(TAB / "step3_ai_compound_titles_full.csv")
    order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    df = df.set_index("group").loc[order]
    terms = ["ai", "ml", "data", "llm", "agent", "genai"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    w = 0.12
    for i, t in enumerate(terms):
        ax.bar(x + (i - len(terms) / 2) * w + w / 2, df[f"share_{t}"] * 100, width=w, label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Share of SWE titles containing term (%)")
    ax.set_title("Compound / hybrid title tokens (AI, ML, data, LLM, agent, genai)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ai_terms_share.png", dpi=150)
    plt.close(fig)


def fig_seniority_markers():
    df = pd.read_csv(TAB / "step5_title_inflation_full.csv")
    order = ["arshkon_2024", "asaniczka_2024", "pooled_2024", "scraped_2026"]
    df = df.set_index("group").loc[order]
    markers = ["junior", "senior", "lead", "principal", "staff", "director", "manager"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    w = 0.12
    for i, m in enumerate(markers):
        ax.bar(x + (i - len(markers) / 2) * w + w / 2, df[f"share_{m}"] * 100, width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Share of SWE titles with seniority marker (%)")
    ax.set_title("Title-embedded seniority markers 2024 → 2026")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "seniority_markers.png", dpi=150)
    plt.close(fig)


def fig_categories():
    df = pd.read_csv(TAB / "step6_emerging_categories_full.csv")
    order = ["pooled_2024", "scraped_2026"]
    df = df.set_index("group").loc[order]
    cats = [c.replace("share_", "") for c in df.columns if c.startswith("share_")]
    # compute delta and sort
    deltas = (df.loc["scraped_2026", [f"share_{c}" for c in cats]].values
              - df.loc["pooled_2024", [f"share_{c}" for c in cats]].values) * 100
    idx = np.argsort(deltas)
    cats_sorted = [cats[i] for i in idx]
    delta_sorted = deltas[idx]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cats_sorted, delta_sorted, color=["firebrick" if d < 0 else "steelblue" for d in delta_sorted])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Share change (pp): pooled 2024 → scraped 2026")
    ax.set_title("Emerging vs declining title-category shares")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "category_deltas.png", dpi=150)
    plt.close(fig)


def fig_content_alignment():
    df = pd.read_csv(TAB / "step4_title_content_alignment_full.csv")
    cal = pd.read_csv(TAB / "step4_within2024_cosine_calibration_full.csv")
    df = df.merge(cal[["title", "within_2024_cosine"]], on="title")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    w = 0.28
    ax.bar(x - w, df["within_2024_cosine"], width=w, label="within-2024 (arshkon vs asaniczka)", color="green")
    ax.bar(x, df["cos_arshkon_vs_scraped"], width=w, label="arshkon 2024 vs scraped 2026", color="steelblue")
    ax.bar(x + w, df["cos_pooled2024_vs_scraped"], width=w, label="pooled 2024 vs scraped 2026", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(df["title"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TF-IDF cosine similarity (mean-vector)")
    ax.set_title("Same-title content drift 2024 → 2026 (lower = more drift; within-2024 = null floor)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "content_alignment.png", dpi=150)
    plt.close(fig)


def main():
    fig_concentration()
    fig_ai_terms()
    fig_seniority_markers()
    fig_categories()
    fig_content_alignment()
    print("figs written to", FIG)


if __name__ == "__main__":
    main()
