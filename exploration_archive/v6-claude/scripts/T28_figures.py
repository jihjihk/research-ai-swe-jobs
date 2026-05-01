"""T28 — Figures: domain x seniority decomposition, per-archetype scope, junior/senior comparison, LLM/GenAI."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
TBL = ROOT / "exploration" / "tables" / "T28"
FIG = ROOT / "exploration" / "figures" / "T28"
FIG.mkdir(parents=True, exist_ok=True)


def fig1_decomposition_summary():
    s = pd.read_csv(TBL / "step2_entry_decomposition_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = s["metric"].tolist()
    within = s["within_domain_component"].to_numpy() * 100
    between = s["between_domain_component"].to_numpy() * 100
    inter = s["interaction_component"].to_numpy() * 100
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, within, width=w, label="Within-domain", color="#2b8cbe")
    ax.bar(x, between, width=w, label="Between-domain", color="#e34a33")
    ax.bar(x + w, inter, width=w, label="Interaction", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0)
    ax.set_ylabel("Contribution to 2024→2026 entry share change (pp)")
    ax.set_title("Kitagawa decomposition of entry-share change")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig1_decomposition_summary.png", dpi=150)
    plt.close()


def fig2_uniform_rise():
    d = pd.read_csv(TBL / "step2_uniform_rise_check.csv")
    large = d[(d["n_2024"] >= 100) & (d["n_2026"] >= 100)].copy()
    pivot = large.pivot_table(index="archetype_name", columns="spec", values="delta_pp").fillna(np.nan)
    pivot = pivot[["seniority_final", "yoe_le2_of_all", "yoe_le2_of_known"]]
    pivot = pivot.sort_values("seniority_final")
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(pivot))
    h = 0.27
    colors = {"seniority_final": "#2b8cbe", "yoe_le2_of_all": "#e34a33", "yoe_le2_of_known": "#31a354"}
    for i, col in enumerate(pivot.columns):
        ax.barh(y + (i - 1) * h, pivot[col].to_numpy(), height=h, label=col, color=colors[col])
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Δ entry-share (pp, 2024 → 2026)")
    ax.set_title("Within-archetype entry-share change (large archetypes only)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_uniform_rise.png", dpi=150)
    plt.close()


def fig3_archetype_scope_changes():
    d = pd.read_csv(TBL / "step3_domain_scope_changes.csv")
    wanted = ["requirement_breadth", "any_ai_broad", "credential_stack_depth", "any_mentor"]
    sub = d[d["metric"].isin(wanted)].copy()
    sub = sub[sub["archetype"] != -2]
    # Per-archetype delta per metric
    piv = sub.pivot_table(
        index=["archetype", "archetype_name"], columns="metric", values="delta"
    ).reset_index()
    piv = piv[piv["archetype_name"].notna()]
    # Only keep larger archetypes
    large_sizes = sub.groupby("archetype_name")["n_2024"].first()
    piv = piv[piv["archetype_name"].map(lambda a: large_sizes.get(a, 0) >= 200)]
    piv = piv.sort_values("requirement_breadth")

    fig, axes = plt.subplots(1, 4, figsize=(13, 5), sharey=True)
    y = np.arange(len(piv))
    names = piv["archetype_name"].to_list()
    for ax, col, title, color in zip(
        axes,
        wanted,
        ["Δ requirement breadth", "Δ any-AI (broad) rate", "Δ credential stack depth", "Δ mentor mention rate"],
        ["#2b8cbe", "#e34a33", "#7f3b08", "#31a354"],
    ):
        ax.barh(y, piv[col].to_numpy(), color=color)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n[:35] for n in names], fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_archetype_scope_changes.png", dpi=150)
    plt.close()


def fig4_junior_senior_gap_change():
    d = pd.read_csv(TBL / "step4_junior_senior_gap.csv")
    metrics = ["requirement_breadth", "credential_stack_depth", "any_ai_broad", "any_mentor"]
    sub = d[d["metric"].isin(metrics) & d["archetype"].ne(-2)].copy()
    piv = sub.pivot_table(index="archetype_name", columns="metric", values="gap_change")
    # Only large archetypes
    piv = piv.dropna(how="all")
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=9, rotation=20)
    ax.set_title("Junior↔Senior gap change (2026−2024)\nNegative = convergence")
    plt.colorbar(im, ax=ax, label="Δ gap (mid_senior − entry)")
    plt.tight_layout()
    plt.savefig(FIG / "fig4_junior_senior_gap_change.png", dpi=150)
    plt.close()


def fig5_llm_genai_profile():
    p = pd.read_csv(TBL / "step6_llm_genai_profile.csv")
    metrics = [
        "entry_final_rate",
        "yoe_le2_rate",
        "any_ai_broad_rate",
        "any_mentor_rate",
        "requirement_breadth_mean",
        "tech_count_mean",
    ]
    labels = ["Entry (final)", "YOE≤2", "any-AI broad", "any-mentor", "Req breadth", "Tech count"]
    vals_24 = p[p["period2"].astype(str) == "2024"][metrics].iloc[0].to_numpy()
    vals_26 = p[p["period2"].astype(str) == "2026"][metrics].iloc[0].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, vals_24, width=w, label="2024", color="#2b8cbe")
    ax.bar(x + w / 2, vals_26, width=w, label="2026", color="#e34a33")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_title("LLM/GenAI archetype profile, 2024 vs 2026")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "fig5_llm_genai_profile.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig1_decomposition_summary()
    fig2_uniform_rise()
    fig3_archetype_scope_changes()
    fig4_junior_senior_gap_change()
    fig5_llm_genai_profile()
    print("[T28] figures written to", FIG)
