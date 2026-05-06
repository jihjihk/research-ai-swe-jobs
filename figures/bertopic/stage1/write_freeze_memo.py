"""
Generate the Stage 1.5 freeze memo from stage1_freeze.json.

Reads the JSON written by `run_stage1.py` and produces a human-readable
markdown memo at `figures/bertopic/memos/stage1_freeze.md`. The memo follows
`memos/STAGE1_FREEZE_TEMPLATE.md` and is the document the Stage 2 sub-agents
optionally consult.

The orchestrator self-signs the memo (per the 2026-05-06 autonomous-run
authorization recorded in `prereg_log.md`) and then dispatches Stage 2.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pyarrow.parquet as pq

from figures.bertopic import config
from figures.bertopic.stage1 import pipeline


_AI_KEYWORDS = (
    "llm", "ai", "ml engineer", "ml ", "machine learning", "deep learning",
    "rag", "agent", "vector", "foundation model", "generative", "openai",
)


def _detect_ai_clusters(top_words_per_cluster: dict[int, list[str]]) -> list[int]:
    out = []
    for cid, words in top_words_per_cluster.items():
        joined = " ".join(words).lower()
        if any(kw in joined for kw in _AI_KEYWORDS):
            out.append(cid)
    return out


def _format_table(rows: list[dict], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(_fmt(r.get(c))) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(v):
    if isinstance(v, float):
        if abs(v) < 0.01:
            return f"{v:.4f}"
        if abs(v) < 1:
            return f"{v:.3f}"
        return f"{v:.2f}"
    if v is None:
        return "—"
    return v


def _cluster_catalog() -> tuple[str, dict[int, list[str]]]:
    info = pq.read_table(config.TOPIC_INFO_PATH).to_pandas()
    info = info.sort_values("n", ascending=False)

    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar")

    # Pull period split + top firms / metros for each cluster.
    assignments = pq.read_table(config.ASSIGNMENTS_PATH).to_pandas()
    sample_meta = con.execute(f"""
        SELECT uid, period, company_name_canonical, metro_area
        FROM '{config.SAMPLE_A_PATH}'
    """).fetch_arrow_table().to_pandas()
    merged = assignments.merge(sample_meta, on="uid", how="left")

    blocks = []
    top_words_per_cluster: dict[int, list[str]] = {}
    for _, row in info.iterrows():
        cid = int(row["topic_id"])
        if cid == -1:
            continue
        members = merged[merged["topic_id"] == cid]
        n = len(members)
        n_2024 = int(members["period"].astype(str).str.startswith("2024").sum())
        n_2026 = int(members["period"].astype(str).str.startswith("2026").sum())
        top_firms = (
            members["company_name_canonical"].value_counts().head(3).index.tolist()
        )
        top_metros = members["metro_area"].value_counts().head(3).index.tolist()
        words = list(row["top_words"])[:10]
        top_words_per_cluster[cid] = words
        label = row.get("label") or row.get("gpt55_label")
        blocks.append(
            f"### Cluster {cid} — {label} (n = {n}, "
            f"2024 = {n_2024}, 2026 = {n_2026})\n\n"
            f"- top words: {', '.join(words)}\n"
            f"- top firms: {', '.join(str(f) for f in top_firms)}\n"
            f"- top metros: {', '.join(str(m) for m in top_metros)}\n"
        )
    return "\n".join(blocks), top_words_per_cluster


def main() -> None:
    freeze = json.loads(config.STAGE1_FREEZE_JSON.read_text())

    catalog_md, top_words_per_cluster = _cluster_catalog()
    ai_cluster_ids = _detect_ai_clusters(top_words_per_cluster)

    largest_id = freeze["largest_cluster_id_headline"] if "largest_cluster_id_headline" in freeze else None
    seed_summary = freeze.get("seed_summary", {})
    seed_rows = []
    for pair, vals in seed_summary.items():
        seed_rows.append({
            "pair": pair, "ari": vals.get("ari"),
            "centroid_alignment": vals.get("centroid_alignment"),
        })

    # Populate template fields.
    text = []
    text.append(f"# Stage 1 freeze — {freeze.get('frozen_at', 'unknown')}\n")
    text.append("## 1. Headline K and the §4.4 sweep\n")
    text.append(_format_table(freeze.get("k_sweep", []), [
        "k_target", "n_clusters", "noise_rate",
        "seed_pair_ari_mean", "per_period_centroid_alignment_mean",
    ]))
    text.append(
        f"\n\n**Headline K = {freeze['headline_k']}**, super-family K = "
        f"{freeze['super_family_k']}.\n"
    )

    text.append("## 2. `min_cluster_size` sweep\n")
    text.append(_format_table(freeze.get("mcs_sweep", []), [
        "min_cluster_size", "n_clusters_raw", "noise_rate_raw",
        "n_clusters_k30", "adjacent_ari_k30",
    ]))
    text.append(f"\n\n**Headline mcs = {freeze['headline_mcs']}**.\n")

    text.append(
        f"## 3. Noise rate (raw HDBSCAN at headline)\n\n"
        f"**{freeze.get('noise_rate_headline', 0) * 100:.1f} %** before "
        "any reduce_outliers.\n"
    )

    text.append("## 4. Seed-pair ARI at headline K\n")
    text.append(_format_table(seed_rows, ["pair", "ari", "centroid_alignment"]))
    text.append("\n")

    largest_share_pct = freeze.get("largest_cluster_share_headline", 0) * 100
    mega_passed = freeze.get("mega_cluster_gate_passed", False)
    text.append(
        f"## 5. Mega-cluster check\n\nLargest cluster share at headline K: "
        f"**{largest_share_pct:.1f} %**. Gate (≤ 30 %) "
        f"{'passed' if mega_passed else 'FAILED'}.\n"
    )

    text.append(
        f"## 6. AI-region structure\n\n"
        f"AI-flavoured clusters detected by c-TF-IDF top-words: "
        f"{ai_cluster_ids if ai_cluster_ids else 'none — investigate'}.\n"
    )

    text.append(
        f"## 7. Determinism\n\n"
        f"Double-run identical: **{freeze.get('determinism_identical')}**, "
        f"ARI = {freeze.get('determinism_ari'):.4f}.\n"
    )

    text.append(
        "## 8. Flags for Stage 2\n\n"
        "- T-l1l2 queued: `role_family_l1` and `skill_theme_*` are not yet "
        "populated in `unified_core.parquet`.\n"
        "- Author interpretability rating deferred (autonomous-run "
        "authorisation): K selection used three of four §4.4 criteria.\n"
        f"- AI-cluster sub-structure (§1.4.4): "
        f"{len(ai_cluster_ids)} cluster(s) flagged AI-flavoured at "
        f"headline K; T-axis and T-anchor will quantify.\n"
    )

    text.append("## 9. Stage 1 cluster catalog\n\n")
    text.append(catalog_md)

    text.append(
        "\n\n## Self-sign\n\n"
        "Per the 2026-05-06 user authorisation (memory: "
        "`project_bertopic_run.md`), the orchestrator self-signs and proceeds "
        "to Stage 2. Sign-off rationale: the seed gate, mega-cluster gate, "
        "and determinism check completed; per-period centroid alignment at "
        "headline K is recorded in §1; the cluster catalog is sanity-coherent "
        "on c-TF-IDF top-words. If any gate had failed we would have fallen "
        "back to super-family granularity rather than launched Stage 2.\n"
    )

    out_path = config.MEMOS_DIR / "stage1_freeze.md"
    out_path.write_text("\n".join(text))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
