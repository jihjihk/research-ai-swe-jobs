"""T02: Seniority comparability & label quality.

Tests whether asaniczka `associate` (seniority_native) can serve as a junior
proxy by comparing it against arshkon entry/associate/mid-senior on multiple
signals: exact title overlap, explicit junior/senior title cues, yoe_extracted
distributions, and seniority_final conditional on the native label.

Also reports entry-level effective sample sizes per source under seniority_final
broken down by seniority_final_source.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "unified.parquet"
OUT_TABLES = REPO / "exploration" / "tables" / "T02"
OUT_FIGS = REPO / "exploration" / "figures" / "T02"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DEFAULT_FILTER = "source_platform = 'linkedin' AND is_english AND date_flag = 'ok'"
SWE_FILTER = f"{DEFAULT_FILTER} AND is_swe"


def q(con, sql):
    return con.execute(sql).fetchdf()


# ---- inline TDD asserts for the title-cue patterns --------------------------
# We use SQL LIKE/regex patterns on lowercased title_normalized. Validate basic
# edge cases in Python first so the SQL behavior is predictable.
import re

JUNIOR_CUES = re.compile(
    r"\b(junior|jr\.?|jr\b|entry[- ]level|entry\b|new[- ]grad|graduate|intern|associate|apprentice|i\b|ii\b)"
)
SENIOR_CUES = re.compile(
    r"\b(senior|sr\.?|sr\b|staff|principal|lead|director|architect|head of|vp\b|distinguished|fellow|iii\b|iv\b)"
)

assert JUNIOR_CUES.search("junior software engineer")
assert JUNIOR_CUES.search("software engineer i")
assert JUNIOR_CUES.search("software engineer, jr.")
assert JUNIOR_CUES.search("new grad software engineer")
assert not JUNIOR_CUES.search("software engineer")
assert SENIOR_CUES.search("senior software engineer")
assert SENIOR_CUES.search("principal engineer")
assert SENIOR_CUES.search("staff software engineer")
assert SENIOR_CUES.search("software engineer iii")
assert not SENIOR_CUES.search("software engineer")


def main():
    con = duckdb.connect()
    con.execute(f"CREATE VIEW u AS SELECT * FROM '{DATA.as_posix()}'")

    # SQL regex patterns mirrored from the Python asserts
    JR_SQL = (
        r"regexp_matches(lower(title_normalized), "
        r"'\b(junior|jr\.?|entry[- ]level|entry|new[- ]grad|graduate|intern|associate|apprentice|\b[i]\b|\b[i][i]\b)')"
    )
    SR_SQL = (
        r"regexp_matches(lower(title_normalized), "
        r"'\b(senior|sr\.?|staff|principal|lead|director|architect|head of|\bvp\b|distinguished|fellow|\b[i][i][i]\b|\b[i][v]\b)')"
    )

    # --- 1. Label inventory per source ---------------------------------------
    native_inv = q(
        con,
        f"""
        SELECT source, seniority_native, count(*) AS n
        FROM u
        WHERE {SWE_FILTER}
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    native_inv.to_csv(OUT_TABLES / "seniority_native_inventory_swe.csv", index=False)
    print("\nseniority_native inventory (SWE, linkedin filter):\n", native_inv)

    final_inv = q(
        con,
        f"""
        SELECT source, seniority_final, seniority_final_source, count(*) AS n
        FROM u
        WHERE {SWE_FILTER}
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    final_inv.to_csv(OUT_TABLES / "seniority_final_inventory_swe.csv", index=False)

    # --- 2. Core comparability audit: native-labeled subsets -----------------
    # Create a view with title-cue flags
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW swe AS
        SELECT *,
               CASE WHEN {JR_SQL} THEN 1 ELSE 0 END AS jr_cue,
               CASE WHEN {SR_SQL} THEN 1 ELSE 0 END AS sr_cue
        FROM u
        WHERE {SWE_FILTER}
        """
    )

    groups_sql = """
        SELECT
            CASE
              WHEN source='kaggle_arshkon' AND seniority_native='entry' THEN 'arshkon_entry'
              WHEN source='kaggle_arshkon' AND seniority_native='associate' THEN 'arshkon_associate'
              WHEN source='kaggle_arshkon' AND seniority_native='mid-senior' THEN 'arshkon_mid_senior'
              WHEN source='kaggle_asaniczka' AND seniority_native='associate' THEN 'asaniczka_associate'
              WHEN source='kaggle_asaniczka' AND seniority_native='mid-senior' THEN 'asaniczka_mid_senior'
            END AS grp
    """

    group_profile = q(
        con,
        f"""
        SELECT grp,
               count(*) AS n,
               avg(jr_cue) AS jr_cue_rate,
               avg(sr_cue) AS sr_cue_rate,
               avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_present_rate,
               avg(yoe_extracted) AS yoe_mean,
               median(yoe_extracted) AS yoe_median,
               quantile_cont(yoe_extracted, 0.25) AS yoe_p25,
               quantile_cont(yoe_extracted, 0.75) AS yoe_p75,
               avg(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2_rate,
               avg(CASE WHEN yoe_extracted <= 3 THEN 1 ELSE 0 END) AS yoe_le3_rate,
               avg(CASE WHEN yoe_extracted >= 5 THEN 1 ELSE 0 END) AS yoe_ge5_rate
        FROM ({groups_sql}, * FROM swe)
        WHERE grp IS NOT NULL
        GROUP BY grp
        ORDER BY grp
        """,
    )
    group_profile.to_csv(OUT_TABLES / "native_group_profile.csv", index=False)
    print("\nNative-label group profile:\n", group_profile)

    # seniority_final distribution conditional on native
    final_cond = q(
        con,
        f"""
        SELECT grp, seniority_final, count(*) AS n
        FROM ({groups_sql}, * FROM swe)
        WHERE grp IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    final_cond.to_csv(OUT_TABLES / "seniority_final_conditional_on_native.csv", index=False)

    # Pivot for readability
    piv = final_cond.pivot(index="grp", columns="seniority_final", values="n").fillna(0)
    piv["total"] = piv.sum(axis=1)
    for col in piv.columns:
        if col != "total":
            piv[f"{col}_pct"] = piv[col] / piv["total"]
    piv.to_csv(OUT_TABLES / "seniority_final_conditional_pivot.csv")
    print("\nseniority_final conditional on native (pct):\n", piv)

    # --- 3. Exact title_normalized overlap ---------------------------------
    # For each group, top 50 titles, plus computed Jaccard between each asaniczka_associate vs arshkon groups
    top_titles = q(
        con,
        f"""
        WITH g AS ({groups_sql}, * FROM swe WHERE title_normalized IS NOT NULL)
        SELECT grp, title_normalized, count(*) AS n
        FROM g
        WHERE grp IS NOT NULL
        GROUP BY 1,2
        QUALIFY row_number() OVER (PARTITION BY grp ORDER BY n DESC) <= 50
        ORDER BY grp, n DESC
        """,
    )
    top_titles.to_csv(OUT_TABLES / "top_titles_by_group.csv", index=False)

    # Jaccard over title-sets (top-200 titles per group covers the bulk)
    top200 = q(
        con,
        f"""
        WITH g AS ({groups_sql}, * FROM swe WHERE title_normalized IS NOT NULL)
        SELECT grp, title_normalized, count(*) AS n
        FROM g
        WHERE grp IS NOT NULL
        GROUP BY 1,2
        QUALIFY row_number() OVER (PARTITION BY grp ORDER BY n DESC) <= 200
        """,
    )
    sets = {g: set(sub["title_normalized"]) for g, sub in top200.groupby("grp")}
    jac_rows = []
    if "asaniczka_associate" in sets:
        a = sets["asaniczka_associate"]
        for other in ["arshkon_entry", "arshkon_associate", "arshkon_mid_senior", "asaniczka_mid_senior"]:
            if other in sets:
                b = sets[other]
                inter = len(a & b)
                uni = len(a | b)
                jac_rows.append(
                    {
                        "a": "asaniczka_associate",
                        "b": other,
                        "intersection": inter,
                        "union": uni,
                        "jaccard_top200": inter / uni if uni else np.nan,
                        "a_in_b_pct": inter / len(a) if a else np.nan,
                    }
                )
    jac_df = pd.DataFrame(jac_rows)
    jac_df.to_csv(OUT_TABLES / "title_overlap_jaccard.csv", index=False)
    print("\nTitle-set Jaccard overlap (top-200):\n", jac_df)

    # --- 4. YOE distributions for each native group (histogram data) --------
    yoe_hist = q(
        con,
        f"""
        SELECT grp, yoe_extracted, count(*) AS n
        FROM ({groups_sql}, * FROM swe)
        WHERE grp IS NOT NULL AND yoe_extracted IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    yoe_hist.to_csv(OUT_TABLES / "yoe_histogram_by_group.csv", index=False)

    # Plot YOE CDFs
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp, sub in yoe_hist.groupby("grp"):
        sub = sub.sort_values("yoe_extracted")
        sub["cum"] = sub["n"].cumsum() / sub["n"].sum()
        ax.plot(sub["yoe_extracted"], sub["cum"], label=grp, marker="o", markersize=3)
    ax.set_xlabel("yoe_extracted (min years)")
    ax.set_ylabel("cumulative share")
    ax.set_title("YOE distribution by native-label group (SWE, LinkedIn)")
    ax.set_xlim(0, 12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "yoe_cdf_by_group.png", dpi=150)
    plt.close()

    # Bar chart of key signals for comparison
    if not group_profile.empty:
        metrics = ["jr_cue_rate", "sr_cue_rate", "yoe_le2_rate", "yoe_ge5_rate"]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(group_profile))
        w = 0.2
        for i, m in enumerate(metrics):
            ax.bar(x + (i - 1.5) * w, group_profile[m].values, width=w, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(group_profile["grp"].values, rotation=20, ha="right")
        ax.set_ylabel("rate")
        ax.set_title("Comparability signals by native-label group")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "signals_by_group.png", dpi=150)
        plt.close()

    # --- 5. Entry-level effective sample sizes under seniority_final --------
    entry_n = q(
        con,
        f"""
        SELECT source, seniority_final_source, count(*) AS n,
               avg(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_cov,
               avg(CASE WHEN metro_area IS NOT NULL THEN 1 ELSE 0 END) AS metro_cov,
               avg(CASE WHEN description_core_llm IS NOT NULL AND description_core_llm <> '' THEN 1 ELSE 0 END) AS desc_core_llm_cov,
               avg(CASE WHEN company_name_canonical IS NOT NULL THEN 1 ELSE 0 END) AS co_cov
        FROM u
        WHERE {SWE_FILTER} AND seniority_final='entry'
        GROUP BY 1,2
        ORDER BY 1,2
        """,
    )
    entry_n.to_csv(OUT_TABLES / "entry_level_sample_sizes.csv", index=False)
    print("\nEntry-level effective N by source x seniority_final_source:\n", entry_n)

    # Totals per source
    entry_totals = q(
        con,
        f"""
        SELECT source, count(*) AS n_entry,
               sum(CASE WHEN seniority_final_source='title_keyword' THEN 1 ELSE 0 END) AS n_kw,
               sum(CASE WHEN seniority_final_source='title_manager' THEN 1 ELSE 0 END) AS n_mgr,
               sum(CASE WHEN seniority_final_source='llm' THEN 1 ELSE 0 END) AS n_llm,
               sum(CASE WHEN seniority_final_source='unknown' THEN 1 ELSE 0 END) AS n_unknown
        FROM u
        WHERE {SWE_FILTER} AND seniority_final='entry'
        GROUP BY 1
        ORDER BY 1
        """,
    )
    entry_totals.to_csv(OUT_TABLES / "entry_level_totals.csv", index=False)
    print("\nEntry totals by source:\n", entry_totals)

    # seniority_final_source distribution over ALL SWE rows (full composition)
    src_dist = q(
        con,
        f"""
        SELECT source, seniority_final, seniority_final_source, count(*) AS n
        FROM u
        WHERE {SWE_FILTER}
        GROUP BY 1,2,3
        ORDER BY 1,2,3
        """,
    )
    src_dist.to_csv(OUT_TABLES / "seniority_final_source_full.csv", index=False)

    # --- 6. Summary JSON ----------------------------------------------------
    summary = {
        "group_profile": group_profile.to_dict(orient="records"),
        "entry_totals": entry_totals.to_dict(orient="records"),
        "jaccard": jac_df.to_dict(orient="records"),
    }
    with open(OUT_TABLES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nT02 done.")


if __name__ == "__main__":
    main()
