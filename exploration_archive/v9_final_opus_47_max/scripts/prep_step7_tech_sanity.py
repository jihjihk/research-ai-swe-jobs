"""Wave 1.5 Agent Prep - Step 7: Tech matrix sanity check.

Compute per-tech mention rate in arshkon/asaniczka/scraped SWE LinkedIn subsets.
Flag techs where arshkon:scraped rate differs by more than 3x or less than 0.33x.
Also include a diagnostic column showing pre-escape vs post-escape scraped rate
for C++, C#, .NET.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

OUT_DIR = Path("exploration/artifacts/shared")
OUT_PATH = OUT_DIR / "tech_matrix_sanity.csv"
MATRIX_PATH = OUT_DIR / "swe_tech_matrix.parquet"

# pre vs post escape diag numbers collected during build
ESCAPE_DIAG_PATH = OUT_DIR / "tech_escape_diagnostic.txt"


def main() -> None:
    t0 = time.time()
    con = duckdb.connect()

    # Build a join: tech matrix + source labels
    # Read tech matrix columns
    techs_info = con.execute(f"DESCRIBE SELECT * FROM '{MATRIX_PATH}'").df()
    tech_cols = [r for r in techs_info["column_name"].tolist() if r != "uid"]
    print(f"[step7] {len(tech_cols)} technologies")

    # Join with source labels from unified
    src_df = con.execute(
        """
        SELECT uid, source
        FROM 'data/unified.parquet'
        WHERE is_swe AND source_platform='linkedin'
          AND is_english = true AND date_flag='ok'
        """
    ).df()

    # Totals per source
    totals = src_df.groupby("source").size()
    print(f"[step7] source totals:\n{totals}")

    # For each tech, compute share by source via SQL
    sum_cols_sql = ", ".join(
        f"sum(CASE WHEN m.{t} THEN 1 ELSE 0 END) AS {t}" for t in tech_cols
    )
    q = f"""
        SELECT u.source,
               count(*) AS n,
               {sum_cols_sql}
        FROM '{MATRIX_PATH}' m
        JOIN 'data/unified.parquet' u USING (uid)
        WHERE u.is_swe AND u.source_platform='linkedin'
          AND u.is_english = true AND u.date_flag='ok'
        GROUP BY u.source
    """
    agg = con.execute(q).df()
    print("[step7] per-source aggregates computed")

    # Reshape: rate per tech per source
    sources = sorted(agg["source"].tolist())
    print(f"[step7] sources: {sources}")
    rates: dict[str, dict[str, float]] = {}
    for _, row in agg.iterrows():
        src = row["source"]
        n = row["n"]
        rates[src] = {t: row[t] / n for t in tech_cols}

    # Read escape diag
    escape_rates: dict[str, tuple[float, float]] = {}
    if ESCAPE_DIAG_PATH.exists():
        lines = ESCAPE_DIAG_PATH.read_text().splitlines()
        for line in lines[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    escape_rates[parts[0]] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    pass

    # Build output rows
    rows = []
    for t in tech_cols:
        arshkon = rates.get("kaggle_arshkon", {}).get(t, 0.0)
        asaniczka = rates.get("kaggle_asaniczka", {}).get(t, 0.0)
        scraped = rates.get("scraped", {}).get(t, 0.0)

        if scraped > 0 and arshkon > 0:
            ratio = arshkon / scraped
        elif arshkon == 0 and scraped == 0:
            ratio = 1.0  # both zero — not a signal
        elif arshkon == 0:
            ratio = 0.0  # never seen in 2024 but seen in 2026
        else:
            ratio = float("inf")  # seen in 2024 but not 2026

        flag = "ok"
        notes_parts: list[str] = []
        # Threshold logic: only flag when we have meaningful presence (>0.5% on at least one side)
        meaningful = (arshkon > 0.005) or (scraped > 0.005)
        if meaningful:
            if ratio > 3.0 and ratio != float("inf"):
                flag = "over_detected_2024"
                notes_parts.append(f"arshkon {arshkon:.3f} > 3x scraped {scraped:.3f}")
            elif ratio < 0.33 and ratio != 0.0:
                flag = "under_detected_2024"
                notes_parts.append(f"arshkon {arshkon:.3f} < 0.33x scraped {scraped:.3f}")
            elif ratio == 0.0 and scraped > 0.005:
                flag = "new_in_2026"
                notes_parts.append("zero in arshkon; appears only in scraped")
            elif ratio == float("inf") and arshkon > 0.005:
                flag = "vanished_2026"
                notes_parts.append("zero in scraped; present in arshkon")

        pre_post = escape_rates.get(t)
        if pre_post is not None:
            pre, post = pre_post
            notes_parts.append(f"pre_escape_scraped={pre:.4f} post_escape_scraped={post:.4f}")

        rows.append({
            "technology": t,
            "arshkon_rate": round(arshkon, 5),
            "asaniczka_rate": round(asaniczka, 5),
            "scraped_rate": round(scraped, 5),
            "arshkon_vs_scraped_ratio": "inf" if ratio == float("inf") else round(ratio, 3),
            "flag": flag,
            "notes": "; ".join(notes_parts),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"[step7] wrote -> {OUT_PATH}")
    # Summarize
    counts = out["flag"].value_counts()
    print("[step7] flag counts:")
    print(counts)
    # Show suspects
    suspects = out[out["flag"] != "ok"]
    if len(suspects) > 0:
        print("[step7] flagged techs:")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 120):
            print(suspects[["technology", "arshkon_rate", "asaniczka_rate", "scraped_rate", "arshkon_vs_scraped_ratio", "flag", "notes"]])
    elapsed = time.time() - t0
    print(f"[step7] elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
