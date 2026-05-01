"""T29 figures: authorship score distribution, company profile, correlation heatmap, unifying-mechanism re-test."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TBL = ROOT / "exploration" / "tables" / "T29"
FIG = ROOT / "exploration" / "figures" / "T29"
FIG.mkdir(parents=True, exist_ok=True)


def fig1_score_distribution():
    feat = pd.read_parquet(TBL / "authorship_flags.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-2, 3, 60)
    for p, color in [("2024", "#2b8cbe"), ("2026", "#e34a33")]:
        s = feat[feat["period2"] == p]["authorship_score"]
        ax.hist(s, bins=bins, alpha=0.55, label=f"{p} (n={len(s)})", color=color, density=True)
    med_24 = feat[feat["period2"] == "2024"]["authorship_score"].median()
    med_26 = feat[feat["period2"] == "2026"]["authorship_score"].median()
    ax.axvline(med_24, color="#2b8cbe", linestyle="--", label=f"2024 median {med_24:.2f}")
    ax.axvline(med_26, color="#e34a33", linestyle="--", label=f"2026 median {med_26:.2f}")
    ax.set_xlabel("LLM-authorship score (higher → more LLM-styled)")
    ax.set_ylabel("Density")
    ax.set_title("Authorship-score distribution by period — SWE LinkedIn cleaned text")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig1_score_distribution.png", dpi=150)
    plt.close()


def fig2_per_feature_shift():
    feat = pd.read_parquet(TBL / "authorship_flags.parquet")
    cols = [
        ("tell_density", "LLM tell density / 1K chars"),
        ("emdash_density", "Em-dash density / 1K chars"),
        ("sent_mean_words", "Mean sentence length (words)"),
        ("sent_std_words", "Sentence length std (words)"),
        ("type_token_ratio", "Type-token ratio"),
        ("bullet_density", "Bullet density / 1K chars"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (col, label) in zip(axes.ravel(), cols):
        vals = []
        labels = []
        for p, color in [("2024", "#2b8cbe"), ("2026", "#e34a33")]:
            s = feat[feat["period2"] == p][col].dropna()
            # Clip outliers for bullet/tell density
            if col in ("tell_density", "emdash_density", "bullet_density"):
                s = s.clip(upper=np.percentile(s[s > 0], 99) if (s > 0).any() else s.max() + 1)
            vals.append(s.to_numpy())
            labels.append(f"{p}\nmed={s.median():.2f}")
        ax.boxplot(vals, labels=labels, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#cccccc"))
        ax.set_title(label, fontsize=10)
    plt.suptitle("Per-feature distributions by period")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_per_feature_shift.png", dpi=150)
    plt.close()


def fig3_correlation_heatmap():
    corr = pd.read_csv(TBL / "correlation_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Authorship-score correlation with Wave 2 content metrics")
    plt.tight_layout()
    plt.savefig(FIG / "fig3_correlation_heatmap.png", dpi=150)
    plt.close()


def fig4_unifying_mechanism():
    d = pd.read_csv(TBL / "unifying_mechanism_test.csv")
    d = d.rename(columns={"Unnamed: 0": "metric"}) if "Unnamed: 0" in d.columns else d
    if "metric" not in d.columns:
        d.columns = ["metric"] + list(d.columns[1:])
    metrics = [
        "char_len",
        "requirement_breadth",
        "tech_count",
        "credential_stack_depth",
        "any_ai_narrow",
        "any_ai_broad",
        "scope_density",
    ]
    d = d[d["metric"].isin(metrics)].set_index("metric")
    d = d.reindex(metrics)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(d))
    w = 0.4
    full_vals = d["full"].to_numpy()
    low_vals = d["low_llm"].to_numpy()
    # Normalize to proportion change where full is reference
    ax.bar(x - w / 2, full_vals, width=w, label="Full sample delta", color="#2b8cbe")
    ax.bar(x + w / 2, low_vals, width=w, label="Low-LLM subset delta", color="#e34a33")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=25, ha="right", fontsize=9)
    ax.set_title("Gate 2 headline deltas: full vs low-LLM-score subset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig4_unifying_mechanism.png", dpi=150)
    plt.close()


def fig5_company_examples():
    top_2026 = pd.read_csv(TBL / "top_llm_companies_2026.csv").head(15)
    low_2026 = pd.read_csv(TBL / "low_llm_companies_2026.csv").head(15)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, title, color in [
        (axes[0], top_2026, "Highest LLM-score companies 2026", "#e34a33"),
        (axes[1], low_2026, "Lowest LLM-score companies 2026", "#2b8cbe"),
    ]:
        df = df.sort_values("score_mean")
        y = np.arange(len(df))
        ax.barh(y, df["score_mean"].to_numpy(), color=color)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(df["company_name_canonical"], df["n"])], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_company_examples.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig1_score_distribution()
    fig2_per_feature_shift()
    fig3_correlation_heatmap()
    fig4_unifying_mechanism()
    fig5_company_examples()
    print("[T29] figures written to", FIG)
