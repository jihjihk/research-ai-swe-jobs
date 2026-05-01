"""T02 — Asaniczka `associate` as a junior proxy.

Compares asaniczka's `associate`-labeled SWE rows (under `seniority_native`)
against arshkon's `entry`, `associate`, and `mid-senior` SWE rows on four
signals:

  1. Top-title Jaccard: how similar is the top-50 title distribution?
  2. Explicit junior / senior title-cue rates.
  3. `yoe_extracted` distribution (median, mean, p75, known share).
  4. `seniority_final` distribution conditional on native label (are native
     `associate` rows re-classified as entry or as mid-senior by the
     combined rule+LLM column?).

Plus: per-source SWE entry-level effective sample under `seniority_final`,
broken down by `seniority_final_source`.

All queries go through DuckDB; no full materialization into pandas.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/jihgaboot/gabor/job-research")
PARQUET = ROOT / "data" / "unified.parquet"
TABLE_DIR = ROOT / "exploration" / "tables" / "T02"
FIG_DIR = ROOT / "exploration" / "figures" / "T02"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = (
    "source_platform = 'linkedin' AND is_english = true AND date_flag = 'ok'"
)

# ----- Title cue patterns --------------------------------------------------
# Junior cues: explicit entry-level markers. Use word-boundary-safe patterns
# with lowercase matching. "i/II/III" and roman level codes are included.
JUNIOR_CUES = [
    r"\bjunior\b",
    r"\bjr\.?\b",
    r"\bentry[- ]?level\b",
    r"\bassociate\b",
    r"\bgraduate\b",
    r"\bnew[- ]?grad\b",
    r"\bintern\b",
    r"\btrainee\b",
    r"\bapprentice\b",
    r"\bearly[- ]?career\b",
    r"\bi\b(?!\s*/)",  # standalone "I" as level — may be noisy; we'll report
    r"\bii\b",         # level II
    # "Software Engineer I" / "Engineer I" style level codes
]

SENIOR_CUES = [
    r"\bsenior\b",
    r"\bsr\.?\b",
    r"\bstaff\b",
    r"\bprincipal\b",
    r"\blead\b",
    r"\barchitect\b",
    r"\bdistinguished\b",
    r"\bhead\s+of\b",
    r"\bdirector\b",
    r"\bvp\b",
    r"\bvice\s+president\b",
    r"\bchief\b",
    r"\biv\b",
    r"\bv\b(?!\s)",  # standalone V level code
]

# -------------------------------------------------------------------------
# Sanity-check asserts on the regex patterns (TDD discipline per preamble).
# -------------------------------------------------------------------------

def _compile_group(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


JUNIOR_RE = _compile_group(JUNIOR_CUES)
SENIOR_RE = _compile_group(SENIOR_CUES)


def _assert_patterns() -> None:
    # Junior cue hits
    assert JUNIOR_RE.search("Junior Software Engineer")
    assert JUNIOR_RE.search("Software Engineer - Associate")
    assert JUNIOR_RE.search("New Grad SWE")
    assert JUNIOR_RE.search("Software Engineer II")
    assert JUNIOR_RE.search("Entry-Level Backend Developer")
    assert JUNIOR_RE.search("Jr Developer")
    # Junior negatives — should not fire (we want to avoid false positives)
    assert not JUNIOR_RE.search("Senior Software Engineer")  # "senior" contains "ior" not "junior"
    assert not JUNIOR_RE.search("Staff Software Engineer")
    assert not JUNIOR_RE.search("Principal Software Engineer")
    assert not JUNIOR_RE.search("Software Engineer IV")

    # Senior cue hits
    assert SENIOR_RE.search("Senior Software Engineer")
    assert SENIOR_RE.search("Staff Software Engineer")
    assert SENIOR_RE.search("Principal Engineer")
    assert SENIOR_RE.search("Lead Backend Developer")
    assert SENIOR_RE.search("Sr. Software Engineer")
    assert SENIOR_RE.search("Director of Engineering")
    assert SENIOR_RE.search("Head of Platform")
    assert SENIOR_RE.search("Software Engineer IV")
    # Senior negatives
    assert not SENIOR_RE.search("Junior Software Engineer")
    assert not SENIOR_RE.search("Software Engineer II")
    assert not SENIOR_RE.search("New Grad SWE")

    # Known edge case: "lead" in "lead generation" — this is a limitation we
    # accept for T02 (rough directional signal only).


_assert_patterns()


def fetch_titles(con: duckdb.DuckDBPyConnection, source: str, native: str) -> pd.DataFrame:
    """Return DataFrame with title, title_normalized, yoe_extracted, seniority_final, seniority_final_source
    for SWE rows in the given (source, seniority_native) cell."""
    where = (
        f"source = '{source}' AND seniority_native = '{native}' AND is_swe "
        f"AND {DEFAULT_FILTER}"
    )
    sql = f"""
    SELECT title, title_normalized, yoe_extracted, seniority_final,
           seniority_final_source
    FROM '{PARQUET}'
    WHERE {where}
    """
    return con.execute(sql).df()


def top_title_counts(df: pd.DataFrame, k: int = 50) -> pd.Series:
    # Use title_normalized for Jaccard (strips level-indicator suffixes per
    # schema) so that "Software Engineer" and "Software Engineer III" do not
    # collapse together — we want the directional signal.
    s = df["title"].fillna("").str.strip().str.lower()
    return s.value_counts().head(k)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def weighted_jaccard(a: pd.Series, b: pd.Series) -> float:
    """Tanimoto / weighted Jaccard: sum_min / sum_max over union of titles."""
    union = a.index.union(b.index)
    a2 = a.reindex(union, fill_value=0).astype(float)
    b2 = b.reindex(union, fill_value=0).astype(float)
    num = np.minimum(a2, b2).sum()
    den = np.maximum(a2, b2).sum()
    return float(num / den) if den else 0.0


def cue_rates(df: pd.DataFrame) -> tuple[float, float]:
    titles = df["title"].fillna("").astype(str)
    if len(titles) == 0:
        return float("nan"), float("nan")
    jr = titles.str.contains(JUNIOR_RE).mean()
    sr = titles.str.contains(SENIOR_RE).mean()
    return float(jr), float(sr)


def yoe_summary(df: pd.DataFrame) -> dict:
    v = df["yoe_extracted"].dropna()
    known_share = (df["yoe_extracted"].notna().sum() / len(df)) if len(df) else 0.0
    return {
        "rows": int(len(df)),
        "yoe_known_share": float(known_share),
        "yoe_median": float(v.median()) if len(v) else float("nan"),
        "yoe_mean": float(v.mean()) if len(v) else float("nan"),
        "yoe_p25": float(v.quantile(0.25)) if len(v) else float("nan"),
        "yoe_p75": float(v.quantile(0.75)) if len(v) else float("nan"),
        "yoe_p90": float(v.quantile(0.90)) if len(v) else float("nan"),
        "yoe_le2_share_of_known": float((v <= 2).mean()) if len(v) else float("nan"),
        "yoe_ge5_share_of_known": float((v >= 5).mean()) if len(v) else float("nan"),
    }


def seniority_final_conditional(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    counts = df["seniority_final"].value_counts(normalize=True)
    return {
        "final_entry_share": float(counts.get("entry", 0.0)),
        "final_associate_share": float(counts.get("associate", 0.0)),
        "final_mid_senior_share": float(counts.get("mid-senior", 0.0)),
        "final_director_share": float(counts.get("director", 0.0)),
        "final_unknown_share": float(counts.get("unknown", 0.0)),
    }


def analyze(con: duckdb.DuckDBPyConnection) -> dict:
    cells = {
        "arshkon_entry": ("kaggle_arshkon", "entry"),
        "arshkon_associate": ("kaggle_arshkon", "associate"),
        "arshkon_mid_senior": ("kaggle_arshkon", "mid-senior"),
        "asaniczka_associate": ("kaggle_asaniczka", "associate"),
    }

    data = {label: fetch_titles(con, src, native) for label, (src, native) in cells.items()}
    print(
        "[T02] cell rows:",
        {k: len(v) for k, v in data.items()},
        file=sys.stderr,
    )

    # Top-title distributions (k=50)
    top = {label: top_title_counts(df, k=50) for label, df in data.items()}

    # Jaccard matrix (set-level and weighted)
    labels = list(cells.keys())
    jac_set = pd.DataFrame(index=labels, columns=labels, dtype=float)
    jac_w = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            jac_set.at[a, b] = jaccard(set(top[a].index), set(top[b].index))
            jac_w.at[a, b] = weighted_jaccard(top[a], top[b])

    # Cue rates and yoe stats
    metrics_rows = []
    for label, df in data.items():
        jr, sr = cue_rates(df)
        y = yoe_summary(df)
        sc = seniority_final_conditional(df)
        row = {"cell": label, "cell_n": int(len(df)), "junior_cue_rate": jr, "senior_cue_rate": sr}
        row.update(y)
        row.update(sc)
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)

    # seniority_final cross-tab (counts, not shares) for the four cells
    stabs = []
    for label, df in data.items():
        tab = df["seniority_final"].value_counts().to_frame("n")
        tab["share"] = tab["n"] / tab["n"].sum() if tab["n"].sum() else 0.0
        tab["cell"] = label
        tab = tab.reset_index().rename(columns={"index": "seniority_final"})
        stabs.append(tab)
    st_full = pd.concat(stabs, ignore_index=True)

    # seniority_final_source breakdown (how the label was resolved)
    sstabs = []
    for label, df in data.items():
        tab = df.groupby(
            ["seniority_final", "seniority_final_source"]
        ).size().to_frame("n").reset_index()
        tab["cell"] = label
        sstabs.append(tab)
    ss_full = pd.concat(sstabs, ignore_index=True)

    return {
        "data": data,
        "top_titles": top,
        "jaccard_set": jac_set,
        "jaccard_weighted": jac_w,
        "metrics": metrics,
        "seniority_final_tab": st_full,
        "seniority_final_source_tab": ss_full,
    }


def per_source_entry_sizes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = f"""
    SELECT source, seniority_final, seniority_final_source, count(*) n
    FROM '{PARQUET}'
    WHERE is_swe AND {DEFAULT_FILTER}
      AND seniority_final IN ('entry', 'associate')
    GROUP BY source, seniority_final, seniority_final_source
    ORDER BY source, seniority_final, seniority_final_source
    """
    return con.execute(sql).df()


def top_titles_print(top: dict[str, pd.Series], k: int = 20) -> pd.DataFrame:
    rows = []
    for label, s in top.items():
        for rank, (title, n) in enumerate(s.head(k).items(), start=1):
            rows.append({"cell": label, "rank": rank, "title": title, "n": int(n)})
    return pd.DataFrame(rows)


def render_figures(result: dict) -> None:
    # Figure 1: Jaccard heatmap (set-based, top-50 titles)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (jac, label) in zip(axes, [
        (result["jaccard_set"].astype(float), "Set Jaccard (top-50 titles)"),
        (result["jaccard_weighted"].astype(float), "Weighted Jaccard (top-50 title freq)"),
    ]):
        im = ax.imshow(jac.values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(jac)))
        ax.set_yticks(range(len(jac)))
        ax.set_xticklabels(jac.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(jac.index, fontsize=9)
        ax.set_title(label, fontsize=10)
        for i in range(len(jac)):
            for j in range(len(jac)):
                ax.text(j, i, f"{jac.iat[i,j]:.2f}", ha="center", va="center",
                        color="white" if jac.iat[i,j] < 0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("T02 — top-title similarity between cells (SWE, default filter)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "jaccard_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: yoe_extracted distributions (KDE overlay, but using histogram
    # since kde may fail on repeated values). Use shared x-axis.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bins = np.arange(0, 21, 1)
    for label, df in result["data"].items():
        v = df["yoe_extracted"].dropna().clip(upper=20)
        if len(v) == 0:
            continue
        hist, edges = np.histogram(v, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, label=f"{label} (n={len(v)})", linewidth=1.6)
    ax.set_xlabel("yoe_extracted (capped at 20)")
    ax.set_ylabel("density")
    ax.set_title("T02 — YOE distributions by cell (SWE, rows with extracted YOE)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "yoe_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Bar chart of the 4 decision-rule signals
    m = result["metrics"].set_index("cell")
    # Signals: junior cue rate, senior cue rate, yoe_median, final_entry_share
    # Normalize yoe_median by dividing by 10 so it fits on same scale
    panels = [
        ("junior_cue_rate", "Junior title-cue rate"),
        ("senior_cue_rate", "Senior title-cue rate"),
        ("yoe_median", "YOE median (known)"),
        ("final_entry_share", "seniority_final = entry share"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (col, title) in zip(axes, panels):
        vals = m[col]
        bars = ax.bar(range(len(vals)), vals.values, color=[
            "#2ecc71" if "entry" in idx else "#3498db" if "associate" in idx and "asaniczka" not in idx else "#e67e22" if "asaniczka" in idx else "#9b59b6"
            for idx in vals.index
        ])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals.values):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.2f}" if col != "yoe_median" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)
    plt.suptitle("T02 — four decision-rule signals (asaniczka associate vs arshkon cells)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "decision_rule_signals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_tables(result: dict, con: duckdb.DuckDBPyConnection) -> None:
    result["jaccard_set"].to_csv(TABLE_DIR / "jaccard_set.csv")
    result["jaccard_weighted"].to_csv(TABLE_DIR / "jaccard_weighted.csv")
    result["metrics"].to_csv(TABLE_DIR / "cell_metrics.csv", index=False)
    result["seniority_final_tab"].to_csv(TABLE_DIR / "seniority_final_by_cell.csv", index=False)
    result["seniority_final_source_tab"].to_csv(
        TABLE_DIR / "seniority_final_source_by_cell.csv", index=False
    )
    top_titles = top_titles_print(result["top_titles"], k=20)
    top_titles.to_csv(TABLE_DIR / "top20_titles_by_cell.csv", index=False)
    per_src = per_source_entry_sizes(con)
    per_src.to_csv(TABLE_DIR / "entry_level_sample_sizes.csv", index=False)


def main() -> None:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='16GB'")
    con.execute("PRAGMA threads=4")

    result = analyze(con)
    write_tables(result, con)
    render_figures(result)

    print("\n=== CELL METRICS ===")
    print(result["metrics"].to_string(index=False))
    print("\n=== JACCARD (set, top-50 titles) ===")
    print(result["jaccard_set"].astype(float).round(3).to_string())
    print("\n=== JACCARD (weighted, top-50 title freq) ===")
    print(result["jaccard_weighted"].astype(float).round(3).to_string())
    print("\n=== seniority_final by cell ===")
    print(result["seniority_final_tab"].to_string(index=False))
    print("\n=== seniority_final_source by cell ===")
    print(result["seniority_final_source_tab"].to_string(index=False))
    print("\n=== Top-20 titles per cell ===")
    print(top_titles_print(result["top_titles"], k=20).to_string(index=False))
    print("\n=== per-source entry-level sample sizes (seniority_final in entry, associate) ===")
    print(per_source_entry_sizes(con).to_string(index=False))


if __name__ == "__main__":
    main()
