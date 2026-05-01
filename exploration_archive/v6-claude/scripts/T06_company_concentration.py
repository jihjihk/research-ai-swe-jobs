"""T06 — Company concentration deep investigation.

Covers:
  1. Concentration metrics (HHI / Gini / top-k share) per source, with/without aggregators
  2. Top-20 employer profile per source
  3. Duplicate-template audit (verification)
  4. Entry-poster concentration (CRITICAL)
  5. Within-company vs between-company decomposition on overlap panel
     (entry share under seniority_final AND YOE proxy; AI rate; desc length; tech count)
  6. Aggregator profile
  7. New entrants profile
  8. Per-finding concentration prediction table

Outputs to exploration/tables/T06/, figures to exploration/figures/T06/.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TBL = ROOT / "exploration" / "tables" / "T06"
FIG = ROOT / "exploration" / "figures" / "T06"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BASE = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""

SRCS = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]


def con():
    c = duckdb.connect()
    c.execute(f"CREATE VIEW d AS SELECT * FROM '{DATA}'")
    return c


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    v = np.sort(values.astype(float))
    n = v.size
    if v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / v[-1]) / n if False else (2 * np.sum((np.arange(1, n + 1)) * v) / (n * v.sum()) - (n + 1) / n)


def hhi(shares: np.ndarray) -> float:
    return float(np.sum(shares ** 2) * 10000)


# ---------- step 1: concentration metrics ----------
def concentration_metrics(c):
    rows = []
    for src in SRCS:
        for excl_aggr in [False, True]:
            where = f"{BASE} AND source = '{src}'"
            if excl_aggr:
                where += " AND is_aggregator = false"
            counts = c.execute(
                f"""
                SELECT company_name_canonical, COUNT(*) n
                FROM d WHERE {where} AND company_name_canonical IS NOT NULL
                GROUP BY 1 ORDER BY n DESC
                """
            ).fetchall()
            if not counts:
                continue
            arr = np.array([r[1] for r in counts], dtype=float)
            total = arr.sum()
            shares = arr / total
            row = {
                "source": src,
                "exclude_aggregators": excl_aggr,
                "n_companies": len(counts),
                "n_postings": int(total),
                "hhi": hhi(shares),
                "gini": gini(arr),
                "top1_share": float(arr[:1].sum() / total),
                "top5_share": float(arr[:5].sum() / total),
                "top10_share": float(arr[:10].sum() / total),
                "top20_share": float(arr[:20].sum() / total),
                "top50_share": float(arr[:50].sum() / total),
                "singletons": int((arr == 1).sum()),
                "singleton_share": float((arr == 1).sum() / len(counts)),
            }
            rows.append(row)
    with (TBL / "concentration_metrics.csv").open("w") as f:
        cols = ["source", "exclude_aggregators", "n_companies", "n_postings", "hhi", "gini",
                "top1_share", "top5_share", "top10_share", "top20_share", "top50_share",
                "singletons", "singleton_share"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c_]) if not isinstance(r[c_], float) else f"{r[c_]:.4f}" for c_ in cols) + "\n")
    return rows


# ---------- step 2: top-20 employer profile per source ----------
def top20_profile(c):
    for src in SRCS:
        rows = c.execute(
            f"""
            WITH base AS (
              SELECT * FROM d WHERE {BASE} AND source = '{src}' AND company_name_canonical IS NOT NULL
            ),
            topc AS (
              SELECT company_name_canonical, COUNT(*) n FROM base
              GROUP BY 1 ORDER BY n DESC LIMIT 20
            )
            SELECT
              t.company_name_canonical,
              t.n,
              ANY_VALUE(b.company_industry) AS industry,
              ANY_VALUE(b.is_aggregator) AS is_aggregator,
              AVG(b.description_length) AS mean_desc_len,
              AVG(b.yoe_extracted) FILTER (WHERE b.yoe_extracted IS NOT NULL) AS mean_yoe,
              AVG(CASE WHEN b.seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE b.seniority_final != 'unknown') AS entry_share_final,
              SUM(CASE WHEN b.seniority_final = 'entry' THEN 1 ELSE 0 END) AS entry_final_n,
              SUM(CASE WHEN b.seniority_final != 'unknown' THEN 1 ELSE 0 END) AS seniority_known_n,
              AVG(CASE WHEN b.yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE b.yoe_extracted IS NOT NULL) AS yoe_le2_share,
              SUM(CASE WHEN b.yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2_n,
              SUM(CASE WHEN b.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_n
            FROM topc t JOIN base b USING (company_name_canonical)
            GROUP BY t.company_name_canonical, t.n
            ORDER BY t.n DESC
            """
        ).fetchall()
        # compute source share
        src_total = c.execute(f"SELECT COUNT(*) FROM d WHERE {BASE} AND source = '{src}'").fetchone()[0]
        with (TBL / f"top20_{src}.csv").open("w") as f:
            f.write("company,n,share_of_source,industry,is_aggregator,mean_desc_len,mean_yoe,entry_share_final,entry_final_n,seniority_known_n,yoe_le2_share,yoe_le2_n,yoe_n\n")
            for r in rows:
                comp, n, ind, agg, mdl, myoe, eshare, enn, skn, yshare, yn, ytot = r
                share = n / src_total if src_total else 0
                f.write(
                    f"\"{comp}\",{n},{share:.4f},\"{ind or ''}\",{agg},{(mdl or 0):.0f},{(myoe or 0):.2f},"
                    f"{(eshare or 0):.4f},{enn or 0},{skn or 0},{(yshare or 0):.4f},{yn or 0},{ytot or 0}\n"
                )


# ---------- step 3: duplicate template audit ----------
def duplicate_templates(c):
    for src in SRCS:
        # dup templates within company (postings / distinct desc hashes)
        rows = c.execute(
            f"""
            SELECT company_name_canonical,
                   COUNT(*) AS posts,
                   COUNT(DISTINCT description_hash) AS distinct_descs,
                   CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT description_hash), 0) AS dup_ratio
            FROM d WHERE {BASE} AND source = '{src}' AND description_hash IS NOT NULL
              AND company_name_canonical IS NOT NULL
            GROUP BY 1
            HAVING COUNT(*) >= 10
            ORDER BY dup_ratio DESC, posts DESC
            LIMIT 20
            """
        ).fetchall()
        with (TBL / f"dup_templates_{src}.csv").open("w") as f:
            f.write("company,posts,distinct_descs,dup_ratio\n")
            for r in rows:
                f.write(f"\"{r[0]}\",{r[1]},{r[2]},{(r[3] or 0):.3f}\n")

    # cross-company collision audit: same description_hash used by >1 company
    rows = c.execute(
        f"""
        SELECT description_hash, COUNT(*) posts, COUNT(DISTINCT company_name_canonical) n_companies
        FROM d WHERE {BASE} AND description_hash IS NOT NULL
        GROUP BY description_hash
        HAVING COUNT(DISTINCT company_name_canonical) > 1
        ORDER BY posts DESC
        LIMIT 30
        """
    ).fetchall()
    with (TBL / "cross_company_hash_collisions.csv").open("w") as f:
        f.write("description_hash,posts,n_companies\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")


# ---------- step 4: entry-poster concentration (CRITICAL) ----------
def entry_poster_concentration(c):
    # For each source, for companies with >=5 SWE postings
    per_source = []
    for src in SRCS:
        rows = c.execute(
            f"""
            SELECT
              company_name_canonical,
              COUNT(*) AS n,
              SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS entry_final,
              SUM(CASE WHEN seniority_final != 'unknown' THEN 1 ELSE 0 END) AS seniority_known,
              SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2,
              SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known,
              ANY_VALUE(is_aggregator) AS is_aggregator,
              ANY_VALUE(company_industry) AS industry,
              AVG(description_length) AS mdl
            FROM d WHERE {BASE} AND source = '{src}' AND company_name_canonical IS NOT NULL
            GROUP BY 1
            """
        ).fetchall()

        # full-source: any entry post at all?
        all_entry_posters = sum(1 for r in rows if r[2] > 0)
        all_yoe_posters = sum(1 for r in rows if r[4] > 0)
        total_companies = len(rows)

        ge5 = [r for r in rows if r[1] >= 5]
        zero_entry_ge5 = [r for r in ge5 if r[2] == 0]
        zero_yoe_ge5 = [r for r in ge5 if r[4] == 0]
        specialists_final = [
            r for r in ge5 if (r[2] / r[3] if r[3] else 0) > 0.5
        ]
        # distribution for entry-share among entry-posters (>=5)
        entry_share_ge5_posters = [
            (r[0], r[1], r[2], r[3], (r[2] / r[3] if r[3] else 0), r[7] or "", r[6])
            for r in ge5 if r[2] > 0 and r[3] > 0
        ]

        per_source.append({
            "source": src,
            "n_companies": total_companies,
            "any_entry_final_companies": all_entry_posters,
            "any_entry_final_share": all_entry_posters / total_companies if total_companies else 0,
            "any_yoe_le2_companies": all_yoe_posters,
            "n_companies_ge5": len(ge5),
            "zero_entry_ge5": len(zero_entry_ge5),
            "zero_entry_ge5_share": len(zero_entry_ge5) / len(ge5) if ge5 else 0,
            "zero_yoe_ge5": len(zero_yoe_ge5),
            "zero_yoe_ge5_share": len(zero_yoe_ge5) / len(ge5) if ge5 else 0,
            "specialists_entry_final": len(specialists_final),
            "specialist_share_ge5": len(specialists_final) / len(ge5) if ge5 else 0,
        })

        # save distribution histogram input
        shares = [x[4] for x in entry_share_ge5_posters]
        if shares:
            bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.01]
            hist, _ = np.histogram(shares, bins=bins)
            with (TBL / f"entry_share_distribution_{src}.csv").open("w") as f:
                f.write("lo,hi,n_companies\n")
                for i in range(len(bins) - 1):
                    f.write(f"{bins[i]},{bins[i+1]},{hist[i]}\n")

        # specialist profile
        with (TBL / f"entry_specialists_{src}.csv").open("w") as f:
            f.write("company,n,entry_n,seniority_known,entry_share,industry,is_aggregator\n")
            for (comp, n, en, sk, share, ind, agg) in sorted(entry_share_ge5_posters, key=lambda x: -x[4])[:30]:
                f.write(f"\"{comp}\",{n},{en},{sk},{share:.3f},\"{ind}\",{agg}\n")

    with (TBL / "entry_poster_concentration.csv").open("w") as f:
        cols = ["source", "n_companies", "any_entry_final_companies", "any_entry_final_share",
                "any_yoe_le2_companies", "n_companies_ge5", "zero_entry_ge5", "zero_entry_ge5_share",
                "zero_yoe_ge5", "zero_yoe_ge5_share", "specialists_entry_final", "specialist_share_ge5"]
        f.write(",".join(cols) + "\n")
        for r in per_source:
            f.write(",".join(
                f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in cols
            ) + "\n")
    return per_source


# ---------- step 5: within vs between decomposition ----------
def decomposition(c):
    """Oaxaca-style decomposition on overlap panel: companies with >=5 SWE in both arshkon and scraped."""
    # Build overlap panel
    c.execute("""
    CREATE TEMP TABLE panel AS
    WITH a AS (
      SELECT company_name_canonical, COUNT(*) AS n_ar
      FROM d WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
        AND source = 'kaggle_arshkon' AND company_name_canonical IS NOT NULL
      GROUP BY 1 HAVING COUNT(*) >= 5
    ),
    s AS (
      SELECT company_name_canonical, COUNT(*) AS n_sc
      FROM d WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
        AND source = 'scraped' AND company_name_canonical IS NOT NULL
      GROUP BY 1 HAVING COUNT(*) >= 5
    )
    SELECT a.company_name_canonical, a.n_ar, s.n_sc
    FROM a JOIN s USING (company_name_canonical)
    """)
    panel_size = c.execute("SELECT COUNT(*) FROM panel").fetchone()[0]
    print(f"Panel size (>=5 in both arshkon and scraped): {panel_size}")

    # metric per company per source
    c.execute("""
    CREATE TEMP TABLE panel_metrics AS
    SELECT
      b.source,
      b.company_name_canonical,
      COUNT(*) AS n,
      AVG(CASE WHEN b.seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE b.seniority_final != 'unknown') AS entry_final_rate,
      SUM(CASE WHEN b.seniority_final != 'unknown' THEN 1 ELSE 0 END) AS seniority_known_n,
      AVG(CASE WHEN b.yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE b.yoe_extracted IS NOT NULL) AS yoe_le2_rate,
      SUM(CASE WHEN b.yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_known_n,
      AVG(b.description_length) AS mean_desc_len,
      AVG(CASE WHEN
         lower(b.description) LIKE '%copilot%' OR lower(b.description) LIKE '%cursor%' OR
         lower(b.description) LIKE '%llm%' OR lower(b.description) LIKE '%claude%' OR
         lower(b.description) LIKE '%chatgpt%' OR lower(b.description) LIKE '%gpt-4%'
         THEN 1.0 ELSE 0.0 END) AS ai_rate
    FROM d b JOIN panel p USING (company_name_canonical)
    WHERE b.source IN ('kaggle_arshkon','scraped') AND b.source_platform='linkedin'
      AND b.is_english=true AND b.date_flag='ok' AND b.is_swe=true
    GROUP BY b.source, b.company_name_canonical
    """)

    # pull into python
    df = c.execute("SELECT * FROM panel_metrics").fetchall()
    cols = [d[0] for d in c.description]
    rows_ar = {r[cols.index("company_name_canonical")]: r for r in df if r[cols.index("source")] == "kaggle_arshkon"}
    rows_sc = {r[cols.index("company_name_canonical")]: r for r in df if r[cols.index("source")] == "scraped"}
    common = sorted(set(rows_ar) & set(rows_sc))
    print(f"Common companies with metrics: {len(common)}")

    # Oaxaca: total change = sum over companies (w_sc*m_sc - w_ar*m_ar)
    # Decompose into:
    #   within  = sum_c w_avg * (m_sc - m_ar)            (same composition, metric changes)
    #   between = sum_c m_avg * (w_sc - w_ar)            (composition changes, metric held constant)
    # where w are share weights within the panel.

    def metric_vals(row, key):
        idx = cols.index(key)
        return row[idx]

    n_ar_total = sum(metric_vals(rows_ar[c_], "n") for c_ in common)
    n_sc_total = sum(metric_vals(rows_sc[c_], "n") for c_ in common)

    out_rows = []
    for metric in ["entry_final_rate", "yoe_le2_rate", "mean_desc_len", "ai_rate"]:
        total_ar = 0.0
        total_sc = 0.0
        within = 0.0
        between = 0.0
        n_valid = 0
        for c_ in common:
            ra, rs = rows_ar[c_], rows_sc[c_]
            wa = metric_vals(ra, "n") / n_ar_total
            ws = metric_vals(rs, "n") / n_sc_total
            ma = metric_vals(ra, metric)
            ms = metric_vals(rs, metric)
            if ma is None or ms is None:
                continue
            w_avg = (wa + ws) / 2
            m_avg = (ma + ms) / 2
            within += w_avg * (ms - ma)
            between += m_avg * (ws - wa)
            total_ar += wa * ma
            total_sc += ws * ms
            n_valid += 1
        total_change = total_sc - total_ar
        out_rows.append({
            "metric": metric,
            "n_companies": n_valid,
            "arshkon_panel_mean": total_ar,
            "scraped_panel_mean": total_sc,
            "total_change": total_change,
            "within_component": within,
            "between_component": between,
            "residual": total_change - within - between,
        })

    with (TBL / "decomposition_overlap_panel.csv").open("w") as f:
        cols_out = ["metric", "n_companies", "arshkon_panel_mean", "scraped_panel_mean", "total_change", "within_component", "between_component", "residual"]
        f.write(",".join(cols_out) + "\n")
        for r in out_rows:
            f.write(",".join(
                f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in cols_out
            ) + "\n")

    # panel size metric
    with (TBL / "overlap_panel_info.csv").open("w") as f:
        f.write(f"panel_size_ge5,{panel_size}\n")
        f.write(f"common_companies_used,{len(common)}\n")
        f.write(f"total_arshkon_rows_in_panel,{int(n_ar_total)}\n")
        f.write(f"total_scraped_rows_in_panel,{int(n_sc_total)}\n")
    return out_rows


# ---------- step 6: aggregator profile ----------
def aggregator_profile(c):
    rows = c.execute(
        f"""
        SELECT source, is_aggregator,
          COUNT(*) n,
          AVG(description_length) mdl,
          AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE seniority_final != 'unknown') entry_share,
          AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE yoe_extracted IS NOT NULL) yoe_le2,
          AVG(yoe_extracted) FILTER (WHERE yoe_extracted IS NOT NULL) mean_yoe,
          AVG(CASE WHEN
             lower(description) LIKE '%copilot%' OR lower(description) LIKE '%cursor%' OR
             lower(description) LIKE '%llm%' OR lower(description) LIKE '%claude%' OR
             lower(description) LIKE '%chatgpt%' OR lower(description) LIKE '%gpt-4%'
             THEN 1.0 ELSE 0.0 END) ai_rate
        FROM d WHERE {BASE}
        GROUP BY 1,2 ORDER BY 1,2
        """
    ).fetchall()
    with (TBL / "aggregator_profile.csv").open("w") as f:
        f.write("source,is_aggregator,n,mean_desc_len,entry_share,yoe_le2_share,mean_yoe,ai_rate\n")
        for r in rows:
            f.write(
                f"{r[0]},{r[1]},{r[2]},{(r[3] or 0):.0f},{(r[4] or 0):.4f},{(r[5] or 0):.4f},{(r[6] or 0):.2f},{(r[7] or 0):.4f}\n"
            )


# ---------- step 7: new entrants ----------
def new_entrants(c):
    rows = c.execute(
        f"""
        WITH a24 AS (
          SELECT DISTINCT company_name_canonical FROM d
          WHERE {BASE} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
            AND company_name_canonical IS NOT NULL
        ),
        sc AS (
          SELECT * FROM d
          WHERE {BASE} AND source = 'scraped' AND company_name_canonical IS NOT NULL
        )
        SELECT
          CASE WHEN company_name_canonical IN (SELECT company_name_canonical FROM a24) THEN 'returning' ELSE 'new_2026' END AS bucket,
          COUNT(DISTINCT company_name_canonical) n_companies,
          COUNT(*) n_posts,
          AVG(description_length) mdl,
          AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE seniority_final != 'unknown') entry_share,
          AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE yoe_extracted IS NOT NULL) yoe_le2,
          AVG(CASE WHEN
             lower(description) LIKE '%copilot%' OR lower(description) LIKE '%cursor%' OR
             lower(description) LIKE '%llm%' OR lower(description) LIKE '%claude%' OR
             lower(description) LIKE '%chatgpt%' OR lower(description) LIKE '%gpt-4%'
             THEN 1.0 ELSE 0.0 END) ai_rate,
          AVG(CASE WHEN is_aggregator THEN 1.0 ELSE 0.0 END) aggregator_share
        FROM sc GROUP BY 1
        """
    ).fetchall()
    with (TBL / "new_entrants_profile.csv").open("w") as f:
        f.write("bucket,n_companies,n_posts,mean_desc_len,entry_share,yoe_le2,ai_rate,aggregator_share\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{(r[3] or 0):.0f},{(r[4] or 0):.4f},{(r[5] or 0):.4f},{(r[6] or 0):.4f},{(r[7] or 0):.4f}\n")


def main():
    c = con()
    print("[T06] concentration metrics")
    concentration_metrics(c)
    print("[T06] top-20 profile")
    top20_profile(c)
    print("[T06] duplicate templates")
    duplicate_templates(c)
    print("[T06] entry-poster concentration")
    entry_poster_concentration(c)
    print("[T06] decomposition")
    decomposition(c)
    print("[T06] aggregator profile")
    aggregator_profile(c)
    print("[T06] new entrants")
    new_entrants(c)
    print("[T06] done")


if __name__ == "__main__":
    main()
