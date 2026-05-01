"""T05 — Cross-dataset comparability.

Tests whether dataset differences reflect real labor-market changes vs data-collection artifacts.
Covers:
  1. Description length (KS test + histograms)
  2. Company overlap (Jaccard)
  3. Geographic distribution (chi-squared on state shares)
  4. Seniority distributions (chi-squared)
  5. Title vocabulary overlap
  6. Industry (arshkon vs scraped only)
  7. Artifact diagnostic (qualitative)
  8. Within-2024 calibration (arshkon vs asaniczka)
  9. Platform labeling stability test (top-20 titles, YOE, Indeed cross-check)

SWE, LinkedIn-only, is_english, date_flag='ok' default filter.
Writes tables to exploration/tables/T05/ and figures to exploration/figures/T05/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path("/home/jihgaboot/gabor/job-research")
DATA = ROOT / "data" / "unified.parquet"
TBL_DIR = ROOT / "exploration" / "tables" / "T05"
FIG_DIR = ROOT / "exploration" / "figures" / "T05"
TBL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_FILTER = """
  source_platform = 'linkedin'
  AND is_english = true
  AND date_flag = 'ok'
  AND is_swe = true
"""


def con():
    c = duckdb.connect()
    c.execute(f"CREATE VIEW d AS SELECT * FROM '{DATA}'")
    return c


# ---------- 1. description length ----------
def desc_length_ks(c):
    out = {}
    for src in ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]:
        rows = c.execute(
            f"SELECT description_length FROM d WHERE {BASE_FILTER} AND source = ? AND description_length IS NOT NULL",
            [src],
        ).fetchnumpy()["description_length"]
        out[src] = np.asarray(rows, dtype=float)

    def summary(a):
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "std": float(a.std()),
        }

    stats_rows = []
    for src, a in out.items():
        s = summary(a)
        s["source"] = src
        stats_rows.append(s)
    with (TBL_DIR / "desc_length_summary.csv").open("w") as f:
        f.write("source,n,mean,median,p25,p75,std\n")
        for s in stats_rows:
            f.write(f"{s['source']},{s['n']},{s['mean']:.1f},{s['median']:.1f},{s['p25']:.1f},{s['p75']:.1f},{s['std']:.1f}\n")

    # pairwise KS
    pairs = [
        ("kaggle_arshkon", "scraped"),
        ("kaggle_asaniczka", "scraped"),
        ("kaggle_arshkon", "kaggle_asaniczka"),
    ]
    ks_rows = []
    for a, b in pairs:
        stat, pv = stats.ks_2samp(out[a], out[b])
        ks_rows.append({"a": a, "b": b, "ks_stat": float(stat), "p_value": float(pv)})
    with (TBL_DIR / "desc_length_ks.csv").open("w") as f:
        f.write("source_a,source_b,ks_statistic,p_value\n")
        for r in ks_rows:
            f.write(f"{r['a']},{r['b']},{r['ks_stat']:.4f},{r['p_value']:.3e}\n")

    # histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 8000, 61)
    colors = {"kaggle_arshkon": "C0", "kaggle_asaniczka": "C1", "scraped": "C2"}
    for src, a in out.items():
        clipped = np.clip(a, 0, 8000)
        ax.hist(clipped, bins=bins, alpha=0.45, density=True, label=f"{src} (n={a.size})", color=colors[src])
    ax.set_xlabel("description_length (chars, clipped at 8000)")
    ax.set_ylabel("density")
    ax.set_title("Description length distribution by source — SWE/LinkedIn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "desc_length_hist.png", dpi=150)
    plt.close(fig)

    return stats_rows, ks_rows


# ---------- 2. company overlap ----------
def company_overlap(c):
    srcs = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    sets = {}
    for s in srcs:
        rows = c.execute(
            f"SELECT DISTINCT company_name_canonical FROM d WHERE {BASE_FILTER} AND source = ? AND company_name_canonical IS NOT NULL",
            [s],
        ).fetchall()
        sets[s] = {r[0] for r in rows}

    pairs = [
        ("kaggle_arshkon", "scraped"),
        ("kaggle_asaniczka", "scraped"),
        ("kaggle_arshkon", "kaggle_asaniczka"),
    ]
    rows_out = []
    for a, b in pairs:
        sa, sb = sets[a], sets[b]
        inter = len(sa & sb)
        union = len(sa | sb)
        jac = inter / union if union else 0.0
        rows_out.append(
            {
                "a": a, "b": b, "n_a": len(sa), "n_b": len(sb),
                "intersection": inter, "union": union, "jaccard": jac,
                "overlap_rate_a": inter / len(sa) if sa else 0,
                "overlap_rate_b": inter / len(sb) if sb else 0,
            }
        )
    with (TBL_DIR / "company_overlap.csv").open("w") as f:
        f.write("source_a,source_b,n_a,n_b,intersection,union,jaccard,overlap_rate_a,overlap_rate_b\n")
        for r in rows_out:
            f.write(
                f"{r['a']},{r['b']},{r['n_a']},{r['n_b']},{r['intersection']},{r['union']},{r['jaccard']:.4f},{r['overlap_rate_a']:.3f},{r['overlap_rate_b']:.3f}\n"
            )

    # top 50 by count per source
    for s in srcs:
        rows = c.execute(
            f"""
            SELECT company_name_canonical, COUNT(*) as n
            FROM d WHERE {BASE_FILTER} AND source = ?
              AND company_name_canonical IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT 50
            """,
            [s],
        ).fetchall()
        with (TBL_DIR / f"top50_companies_{s}.csv").open("w") as f:
            f.write("company,posts\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]}\n")

    # top-50 overlap across pairs
    top50 = {}
    for s in srcs:
        rows = c.execute(
            f"""
            SELECT company_name_canonical FROM d
            WHERE {BASE_FILTER} AND source = ? AND company_name_canonical IS NOT NULL
            GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 50
            """,
            [s],
        ).fetchall()
        top50[s] = {r[0] for r in rows}
    with (TBL_DIR / "top50_overlap.csv").open("w") as f:
        f.write("source_a,source_b,overlap_count,jaccard\n")
        for a, b in pairs:
            inter = len(top50[a] & top50[b])
            union = len(top50[a] | top50[b])
            f.write(f"{a},{b},{inter},{inter/union if union else 0:.3f}\n")
    return rows_out


# ---------- 3. geographic ----------
def geographic(c):
    srcs = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    state_counts = {}
    for s in srcs:
        rows = c.execute(
            f"""
            SELECT state_normalized, COUNT(*) as n
            FROM d WHERE {BASE_FILTER} AND source = ?
              AND state_normalized IS NOT NULL AND is_multi_location = false
            GROUP BY 1 ORDER BY n DESC
            """,
            [s],
        ).fetchall()
        state_counts[s] = dict(rows)

    all_states = sorted(set().union(*[set(d.keys()) for d in state_counts.values()]))
    matrix = {s: [state_counts[s].get(st, 0) for st in all_states] for s in srcs}

    with (TBL_DIR / "state_counts.csv").open("w") as f:
        f.write("state," + ",".join(srcs) + "\n")
        for i, st in enumerate(all_states):
            f.write(st + "," + ",".join(str(matrix[s][i]) for s in srcs) + "\n")

    def chi2(a, b):
        # build 2xk matrix for states where both have at least 1
        keys = [k for k in all_states if state_counts[a].get(k, 0) + state_counts[b].get(k, 0) > 10]
        arr = np.array([[state_counts[a].get(k, 0) for k in keys], [state_counts[b].get(k, 0) for k in keys]])
        if arr.shape[1] < 2:
            return None
        chi, pv, _, _ = stats.chi2_contingency(arr)
        return {"chi2": float(chi), "p": float(pv), "dof": arr.shape[1] - 1}

    pairs = [
        ("kaggle_arshkon", "scraped"),
        ("kaggle_asaniczka", "scraped"),
        ("kaggle_arshkon", "kaggle_asaniczka"),
    ]
    rows = []
    for a, b in pairs:
        r = chi2(a, b)
        if r:
            rows.append({"a": a, "b": b, **r})
    with (TBL_DIR / "state_chi2.csv").open("w") as f:
        f.write("source_a,source_b,chi2,p_value,dof\n")
        for r in rows:
            f.write(f"{r['a']},{r['b']},{r['chi2']:.2f},{r['p']:.3e},{r['dof']}\n")

    # top-10 states share per source
    with (TBL_DIR / "top10_states.csv").open("w") as f:
        f.write("source,state,share\n")
        for s in srcs:
            tot = sum(state_counts[s].values())
            top = sorted(state_counts[s].items(), key=lambda x: -x[1])[:10]
            for st, n in top:
                f.write(f"{s},{st},{n/tot:.3f}\n")
    return rows


# ---------- 4. seniority ----------
def seniority_dist(c):
    srcs = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    dist = {}
    for s in srcs:
        rows = c.execute(
            f"""
            SELECT seniority_final, COUNT(*) n
            FROM d WHERE {BASE_FILTER} AND source = ? AND seniority_final != 'unknown'
            GROUP BY 1 ORDER BY 1
            """,
            [s],
        ).fetchall()
        dist[s] = dict(rows)
    levels = sorted(set().union(*[set(d.keys()) for d in dist.values()]))
    with (TBL_DIR / "seniority_final_dist.csv").open("w") as f:
        f.write("level," + ",".join(srcs) + "," + ",".join(f"{s}_share" for s in srcs) + "\n")
        totals = {s: sum(dist[s].values()) for s in srcs}
        for lv in levels:
            row = [lv] + [str(dist[s].get(lv, 0)) for s in srcs] + [f"{dist[s].get(lv, 0)/totals[s]:.3f}" for s in srcs]
            f.write(",".join(row) + "\n")

    # chi-sq pairwise
    pairs = [
        ("kaggle_arshkon", "scraped"),
        ("kaggle_asaniczka", "scraped"),
        ("kaggle_arshkon", "kaggle_asaniczka"),
    ]
    rows = []
    for a, b in pairs:
        arr = np.array([[dist[a].get(lv, 0) for lv in levels], [dist[b].get(lv, 0) for lv in levels]])
        mask = arr.sum(axis=0) > 0
        arr = arr[:, mask]
        chi, pv, dof, _ = stats.chi2_contingency(arr)
        rows.append({"a": a, "b": b, "chi2": float(chi), "p": float(pv), "dof": int(dof)})
    with (TBL_DIR / "seniority_chi2.csv").open("w") as f:
        f.write("source_a,source_b,chi2,p_value,dof\n")
        for r in rows:
            f.write(f"{r['a']},{r['b']},{r['chi2']:.2f},{r['p']:.3e},{r['dof']}\n")
    return dist, rows


# ---------- 5. title vocabulary ----------
def title_vocab(c):
    srcs = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    titles = {}
    for s in srcs:
        rows = c.execute(
            f"""
            SELECT DISTINCT title_normalized FROM d WHERE {BASE_FILTER} AND source = ?
              AND title_normalized IS NOT NULL
            """,
            [s],
        ).fetchall()
        titles[s] = {r[0] for r in rows}

    pairs = [
        ("kaggle_arshkon", "scraped"),
        ("kaggle_asaniczka", "scraped"),
        ("kaggle_arshkon", "kaggle_asaniczka"),
    ]
    rows_out = []
    for a, b in pairs:
        sa, sb = titles[a], titles[b]
        inter = len(sa & sb)
        union = len(sa | sb)
        jac = inter / union if union else 0
        rows_out.append({"a": a, "b": b, "n_a": len(sa), "n_b": len(sb), "intersection": inter, "jaccard": jac})
    with (TBL_DIR / "title_overlap.csv").open("w") as f:
        f.write("source_a,source_b,n_a,n_b,intersection,jaccard\n")
        for r in rows_out:
            f.write(f"{r['a']},{r['b']},{r['n_a']},{r['n_b']},{r['intersection']},{r['jaccard']:.4f}\n")

    # titles unique to one period (scraped vs pooled 2024)
    pooled_2024 = titles["kaggle_arshkon"] | titles["kaggle_asaniczka"]
    only_2026 = titles["scraped"] - pooled_2024
    only_2024 = pooled_2024 - titles["scraped"]
    # find the top unique titles by count in their period
    top_unique_2026 = c.execute(
        f"""
        SELECT title_normalized, COUNT(*) n FROM d
        WHERE {BASE_FILTER} AND source = 'scraped' AND title_normalized IN ({','.join(['?']*len(only_2026))})
        GROUP BY 1 ORDER BY n DESC LIMIT 30
        """,
        list(only_2026),
    ).fetchall() if only_2026 else []
    top_unique_2024 = c.execute(
        f"""
        SELECT title_normalized, COUNT(*) n FROM d
        WHERE {BASE_FILTER} AND source IN ('kaggle_arshkon','kaggle_asaniczka')
          AND title_normalized IN ({','.join(['?']*len(only_2024))})
        GROUP BY 1 ORDER BY n DESC LIMIT 30
        """,
        list(only_2024),
    ).fetchall() if only_2024 else []
    with (TBL_DIR / "titles_unique_2026.csv").open("w") as f:
        f.write("title_normalized,count\n")
        for t, n in top_unique_2026:
            f.write(f"{t},{n}\n")
    with (TBL_DIR / "titles_unique_2024.csv").open("w") as f:
        f.write("title_normalized,count\n")
        for t, n in top_unique_2024:
            f.write(f"{t},{n}\n")
    return rows_out


# ---------- 6. industry ----------
def industry(c):
    # arshkon vs scraped
    rows = c.execute(
        f"""
        SELECT source, company_industry, COUNT(*) n
        FROM d WHERE {BASE_FILTER} AND source IN ('kaggle_arshkon','scraped')
          AND company_industry IS NOT NULL
        GROUP BY source, company_industry
        """
    ).fetchall()
    arsh, scr = {}, {}
    for s, ind, n in rows:
        (arsh if s == "kaggle_arshkon" else scr)[ind] = n
    inds = sorted(set(arsh.keys()) | set(scr.keys()), key=lambda k: -(arsh.get(k, 0) + scr.get(k, 0)))[:30]
    with (TBL_DIR / "industry_dist.csv").open("w") as f:
        ta = sum(arsh.values())
        ts = sum(scr.values())
        f.write("industry,arshkon_n,scraped_n,arshkon_share,scraped_share\n")
        for i in inds:
            f.write(f"{i},{arsh.get(i,0)},{scr.get(i,0)},{arsh.get(i,0)/ta:.3f},{scr.get(i,0)/ts:.3f}\n")


# ---------- 9. platform labeling stability ----------
def platform_stability(c):
    # top-20 SWE titles appearing in BOTH arshkon and scraped (by combined count)
    rows = c.execute(
        f"""
        WITH in_both AS (
          SELECT title_normalized
          FROM d WHERE {BASE_FILTER} AND source IN ('kaggle_arshkon','scraped')
            AND title_normalized IS NOT NULL
          GROUP BY title_normalized
          HAVING COUNT(DISTINCT source) = 2
        )
        SELECT title_normalized, COUNT(*) n
        FROM d
        WHERE {BASE_FILTER} AND source IN ('kaggle_arshkon','scraped')
          AND title_normalized IN (SELECT title_normalized FROM in_both)
        GROUP BY title_normalized
        ORDER BY n DESC LIMIT 20
        """
    ).fetchall()
    top20 = [r[0] for r in rows]

    # For each title: seniority_native distribution by source (arshkon-only valid for asaniczka; here scraped vs arshkon)
    with (TBL_DIR / "top20_title_seniority_native.csv").open("w") as f:
        f.write("title,source,entry,mid,senior,executive,associate,director,total\n")
        for t in top20:
            for src in ["kaggle_arshkon", "scraped"]:
                dist = dict(
                    c.execute(
                        f"""
                        SELECT seniority_native, COUNT(*) FROM d
                        WHERE {BASE_FILTER} AND source = ? AND title_normalized = ?
                          AND seniority_native IS NOT NULL
                        GROUP BY seniority_native
                        """,
                        [src, t],
                    ).fetchall()
                )
                total = sum(dist.values())
                f.write(
                    f"{t},{src},{dist.get('entry',0)},{dist.get('mid_senior',0)+dist.get('mid',0)},{dist.get('senior',0)},{dist.get('executive',0)},{dist.get('associate',0)},{dist.get('director',0)},{total}\n"
                )

    # YOE distribution per title×source
    with (TBL_DIR / "top20_title_yoe.csv").open("w") as f:
        f.write("title,source,n_with_yoe,mean_yoe,median_yoe,share_yoe_le2\n")
        for t in top20:
            for src in ["kaggle_arshkon", "scraped"]:
                r = c.execute(
                    f"""
                    SELECT COUNT(*), AVG(yoe_extracted), MEDIAN(yoe_extracted),
                           AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END)
                    FROM d WHERE {BASE_FILTER} AND source = ? AND title_normalized = ?
                      AND yoe_extracted IS NOT NULL
                    """,
                    [src, t],
                ).fetchone()
                n, mn, md, share = r
                f.write(f"{t},{src},{n or 0},{mn or 0:.2f},{md or 0:.1f},{share or 0:.3f}\n")

    # seniority_final share per title×source
    with (TBL_DIR / "top20_title_seniority_final.csv").open("w") as f:
        f.write("title,source,entry_share,n,total\n")
        for t in top20:
            for src in ["kaggle_arshkon", "scraped"]:
                r = c.execute(
                    f"""
                    SELECT
                      AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END),
                      COUNT(*)
                    FROM d WHERE {BASE_FILTER} AND source = ? AND title_normalized = ?
                      AND seniority_final != 'unknown'
                    """,
                    [src, t],
                ).fetchone()
                share, tot = r
                f.write(f"{t},{src},{share or 0:.3f},{tot or 0},{tot or 0}\n")

    # Indeed cross-validation: entry-share on Indeed scraped rows using seniority_final + YOE proxy
    r = c.execute(
        """
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END) AS entry_n,
          SUM(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END) AS unknown_n,
          SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END) AS yoe_le2_n,
          SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END) AS yoe_n
        FROM d
        WHERE source = 'scraped' AND source_platform = 'indeed'
          AND is_swe = true AND is_english = true AND date_flag = 'ok'
        """
    ).fetchone()
    indeed_stats = {
        "n": r[0], "entry_n": r[1], "unknown_n": r[2], "yoe_le2_n": r[3], "yoe_n": r[4],
        "entry_share_of_known": (r[1] / (r[0] - r[2])) if (r[0] - r[2]) > 0 else None,
        "yoe_le2_share": (r[3] / r[4]) if r[4] else None,
    }

    # also for LinkedIn scraped, the same proxy for reference
    r = c.execute(
        f"""
        SELECT COUNT(*),
          SUM(CASE WHEN seniority_final = 'entry' THEN 1 ELSE 0 END),
          SUM(CASE WHEN seniority_final = 'unknown' THEN 1 ELSE 0 END),
          SUM(CASE WHEN yoe_extracted <= 2 THEN 1 ELSE 0 END),
          SUM(CASE WHEN yoe_extracted IS NOT NULL THEN 1 ELSE 0 END)
        FROM d WHERE {BASE_FILTER} AND source = 'scraped'
        """
    ).fetchone()
    linkedin_stats = {
        "n": r[0], "entry_n": r[1], "unknown_n": r[2], "yoe_le2_n": r[3], "yoe_n": r[4],
        "entry_share_of_known": (r[1] / (r[0] - r[2])) if (r[0] - r[2]) > 0 else None,
        "yoe_le2_share": (r[3] / r[4]) if r[4] else None,
    }
    with (TBL_DIR / "indeed_vs_linkedin_entry.csv").open("w") as f:
        f.write("platform,n,entry_n,unknown_n,yoe_le2_n,yoe_n,entry_share_of_known,yoe_le2_share\n")
        for name, s in [("indeed", indeed_stats), ("linkedin", linkedin_stats)]:
            f.write(
                f"{name},{s['n']},{s['entry_n']},{s['unknown_n']},{s['yoe_le2_n']},{s['yoe_n']},"
                f"{(s['entry_share_of_known'] or 0):.4f},{(s['yoe_le2_share'] or 0):.4f}\n"
            )


# ---------- within-2024 calibration table ----------
def calibration_table(c):
    """Metric by source: mean desc length, entry share (seniority_final), YOE<=2 share, mean YOE, AI mention rate."""
    srcs = ["kaggle_arshkon", "kaggle_asaniczka", "scraped"]
    rows = []
    for s in srcs:
        r = c.execute(
            f"""
            SELECT
              COUNT(*),
              AVG(description_length),
              AVG(CASE WHEN seniority_final = 'entry' THEN 1.0 ELSE 0.0 END) FILTER (WHERE seniority_final != 'unknown'),
              AVG(CASE WHEN yoe_extracted <= 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE yoe_extracted IS NOT NULL),
              AVG(yoe_extracted) FILTER (WHERE yoe_extracted IS NOT NULL),
              AVG(CASE WHEN lower(description) LIKE '%copilot%' OR lower(description) LIKE '%cursor%' OR
                           lower(description) LIKE '%llm%' OR lower(description) LIKE '%claude%' OR
                           lower(description) LIKE '%chatgpt%' OR lower(description) LIKE '%gpt-4%' THEN 1.0 ELSE 0.0 END)
            FROM d WHERE {BASE_FILTER} AND source = ?
            """,
            [s],
        ).fetchone()
        rows.append(
            {
                "source": s, "n": r[0], "mean_desc_len": r[1], "entry_share": r[2],
                "yoe_le2_share": r[3], "mean_yoe": r[4], "ai_mention_rate": r[5],
            }
        )

    with (TBL_DIR / "calibration_table.csv").open("w") as f:
        f.write("source,n,mean_desc_len,entry_share_final,yoe_le2_share,mean_yoe,ai_mention_rate\n")
        for r in rows:
            f.write(
                f"{r['source']},{r['n']},{r['mean_desc_len']:.0f},{(r['entry_share'] or 0):.4f},"
                f"{(r['yoe_le2_share'] or 0):.4f},{(r['mean_yoe'] or 0):.3f},{(r['ai_mention_rate'] or 0):.4f}\n"
            )
    # ratios: cross-period / within-2024 baseline
    ar = next(r for r in rows if r["source"] == "kaggle_arshkon")
    asa = next(r for r in rows if r["source"] == "kaggle_asaniczka")
    sc = next(r for r in rows if r["source"] == "scraped")
    metrics = ["mean_desc_len", "entry_share", "yoe_le2_share", "mean_yoe", "ai_mention_rate"]
    with (TBL_DIR / "signal_to_noise.csv").open("w") as f:
        f.write("metric,arshkon,asaniczka,scraped,within_2024_abs,cross_period_abs,snr\n")
        for m in metrics:
            a = ar[m] or 0
            b = asa[m] or 0
            s = sc[m] or 0
            within = abs(a - b)
            cross = abs(((a + b) / 2) - s)
            snr = cross / within if within > 0 else float("inf")
            f.write(f"{m},{a:.4f},{b:.4f},{s:.4f},{within:.4f},{cross:.4f},{snr:.2f}\n")


def main():
    c = con()
    print("[T05] desc length")
    desc_length_ks(c)
    print("[T05] company overlap")
    company_overlap(c)
    print("[T05] geographic")
    geographic(c)
    print("[T05] seniority")
    seniority_dist(c)
    print("[T05] title vocab")
    title_vocab(c)
    print("[T05] industry")
    industry(c)
    print("[T05] platform stability")
    platform_stability(c)
    print("[T05] calibration")
    calibration_table(c)
    print("[T05] done")


if __name__ == "__main__":
    main()
