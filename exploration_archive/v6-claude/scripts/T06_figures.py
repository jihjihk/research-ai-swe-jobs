"""T06 — figures: concentration curves and entry share distribution."""
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/jihgaboot/gabor/job-research")
FIG = ROOT / "exploration" / "figures" / "T06"
FIG.mkdir(parents=True, exist_ok=True)

BASE = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""


def lorenz(values: np.ndarray):
    v = np.sort(values.astype(float))
    cum = np.cumsum(v)
    if cum[-1] == 0:
        return np.array([0, 1]), np.array([0, 1])
    cum = cum / cum[-1]
    x = np.arange(1, len(v) + 1) / len(v)
    return np.concatenate([[0], x]), np.concatenate([[0], cum])


def main():
    c = duckdb.connect()
    c.execute(f"CREATE VIEW d AS SELECT * FROM '{ROOT / 'data' / 'unified.parquet'}'")

    # Lorenz curves (with + without aggregators)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, excl in zip(axes, [False, True]):
        for src, color in [("kaggle_arshkon", "C0"), ("kaggle_asaniczka", "C1"), ("scraped", "C2")]:
            where = f"{BASE} AND source = '{src}'"
            if excl:
                where += " AND is_aggregator = false"
            rows = c.execute(
                f"""
                SELECT COUNT(*) FROM d WHERE {where} AND company_name_canonical IS NOT NULL
                GROUP BY company_name_canonical
                """
            ).fetchall()
            vals = np.array([r[0] for r in rows])
            x, y = lorenz(vals)
            ax.plot(x, y, label=src, color=color, linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect equality")
        ax.set_xlabel("Cumulative share of companies")
        ax.set_title(f"{'Excluding aggregators' if excl else 'All companies'}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cumulative share of postings")
    axes[0].legend(loc="upper left")
    fig.suptitle("Lorenz curves — company concentration of SWE postings", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "lorenz_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Entry-share distribution among companies with >=5 SWE (scraped, excluding aggregators)
    fig, ax = plt.subplots(figsize=(9, 5))
    for src, color in [("kaggle_arshkon", "C0"), ("kaggle_asaniczka", "C1"), ("scraped", "C2")]:
        rows = c.execute(
            f"""
            SELECT
              AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE seniority_final != 'unknown') AS entry_share,
              COUNT(*) AS n
            FROM d WHERE {BASE} AND source = '{src}' AND company_name_canonical IS NOT NULL
            GROUP BY company_name_canonical
            HAVING COUNT(*) >= 5
            """
        ).fetchall()
        shares = np.array([r[0] if r[0] is not None else 0 for r in rows])
        bins = np.linspace(0, 1, 21)
        ax.hist(shares, bins=bins, alpha=0.4, label=f"{src} (n={len(shares)})", color=color)
    ax.set_xlabel("Within-company entry share (seniority_final)")
    ax.set_ylabel("Companies (>=5 SWE postings)")
    ax.set_title("Entry-level posting is concentrated: many companies have 0%, some are specialists")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "entry_share_within_company.png", dpi=150)
    plt.close(fig)

    # Top-10 companies bar: share of entry_final pool in scraped
    rows = c.execute(
        f"""
        SELECT company_name_canonical, COUNT(*) n
        FROM d WHERE {BASE} AND source = 'scraped' AND seniority_final = 'entry'
        GROUP BY 1 ORDER BY n DESC LIMIT 15
        """
    ).fetchall()
    total = c.execute(
        f"SELECT COUNT(*) FROM d WHERE {BASE} AND source = 'scraped' AND seniority_final = 'entry'"
    ).fetchone()[0]
    names = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    shares = [n / total for n in counts]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], shares[::-1])
    ax.set_xlabel("Share of scraped entry_final pool")
    ax.set_title(f"Top 15 employers drive {sum(shares):.1%} of the scraped entry_final pool (n={total})")
    for i, (s, n) in enumerate(zip(shares[::-1], counts[::-1])):
        ax.text(s, i, f" {n}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "top15_entry_final_pool_scraped.png", dpi=150)
    plt.close(fig)

    # Decomposition figure
    import csv
    rows = []
    with (ROOT / "exploration" / "tables" / "T06" / "decomposition_overlap_panel.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = [r["metric"] for r in rows]
    within = [float(r["within_component"]) for r in rows]
    between = [float(r["between_component"]) for r in rows]
    # normalize by arshkon mean for comparability
    norm = [float(r["arshkon_panel_mean"]) if float(r["arshkon_panel_mean"]) != 0 else 1 for r in rows]
    within_pct = [w / n * 100 for w, n in zip(within, norm)]
    between_pct = [b / n * 100 for b, n in zip(between, norm)]

    x = np.arange(len(metrics))
    ax.bar(x - 0.2, within_pct, 0.4, label="within-company (%)", color="steelblue")
    ax.bar(x + 0.2, between_pct, 0.4, label="between-company (%)", color="coral")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Change as % of arshkon baseline")
    ax.set_title("Within- vs between-company decomposition (arshkon↔scraped overlap panel, n=125)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "within_vs_between_decomposition.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
