"""V1 — Independent re-derivation of Gate 2 top headline numbers.

Target 1: Per-tool AI prevalence SNR (T14 table)
Target 2: Aggregate AI prevalence and SNR (T05/T14)
Target 4: Within-archetype uniform junior-share rise (T09)
Target 5: Arshkon-only `seniority_final` entry share flip (T08)

Target 3 (section anatomy) uses the T13 section classifier — see
V1_re_derive_T13.py.
"""

from __future__ import annotations

import duckdb
import pandas as pd

UNI = "/home/jihgaboot/gabor/job-research/data/unified.parquet"
TECH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_tech_matrix.parquet"
META = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"
ARCH = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_archetype_labels.parquet"
OUT = "/home/jihgaboot/gabor/job-research/exploration/tables/V1"

con = duckdb.connect()


def classify_period(source: str, period: str) -> str:
    if source == "scraped":
        return "2026"
    return "2024"


# -----------------------------------------------------------------------------
# Target 1 + 2: Per-tool AI prevalence
# -----------------------------------------------------------------------------
def target_1_2_tech_snr() -> pd.DataFrame:
    print("=" * 72)
    print("TARGET 1 + 2: Per-tool AI prevalence SNR (independent)")
    print("=" * 72)

    # Load tech matrix joined to metadata (source + period)
    df = con.execute(
        f"""
        SELECT
          m.uid,
          m.source,
          m.period,
          t.claude_tool,
          t.copilot,
          t.langchain,
          t.agents_framework,
          t.embedding,
          -- for aggregate AI-any we build a broad union below
          t.llm,
          t.rag,
          t.openai_api,
          t.claude_api,
          t.prompt_engineering,
          t.fine_tuning,
          t.mcp,
          t.transformer_arch,
          t.machine_learning,
          t.deep_learning,
          t.pytorch,
          t.tensorflow,
          t.cursor_tool,
          t.chatgpt,
          t.gemini_tool,
          t.codex_tool,
          t.huggingface,
          t.nlp,
          t.langgraph
        FROM '{META}' m
        INNER JOIN '{TECH}' t USING (uid)
        """
    ).fetchdf()

    df["period_bucket"] = df.apply(
        lambda r: classify_period(r["source"], r["period"]), axis=1
    )

    # Build AI-any indicator (broad union — matches T14 "ai keyword any / ai tool specific" concept)
    ai_cols = [
        "claude_tool",
        "copilot",
        "langchain",
        "agents_framework",
        "embedding",
        "llm",
        "rag",
        "openai_api",
        "claude_api",
        "prompt_engineering",
        "fine_tuning",
        "mcp",
        "transformer_arch",
        "machine_learning",
        "deep_learning",
        "pytorch",
        "tensorflow",
        "cursor_tool",
        "chatgpt",
        "gemini_tool",
        "codex_tool",
        "huggingface",
        "nlp",
        "langgraph",
    ]
    df["ai_any"] = df[ai_cols].any(axis=1).astype(int)

    focus = ["claude_tool", "copilot", "langchain", "agents_framework", "embedding", "ai_any"]

    results = []
    for tech in focus:
        rates = {}
        for bucket, sub in df.groupby("period_bucket"):
            rates[f"share_{bucket}"] = sub[tech].mean()
        # within-2024: arshkon vs asaniczka effect size (absolute rate diff)
        rows24 = df[df["period_bucket"] == "2024"]
        ars = rows24[rows24["source"] == "kaggle_arshkon"][tech].mean()
        asa = rows24[rows24["source"] == "kaggle_asaniczka"][tech].mean()
        within_2024 = abs(ars - asa)
        cross_period = abs(rates["share_2026"] - rates["share_2024"])
        snr = cross_period / within_2024 if within_2024 > 0 else float("inf")
        results.append(
            {
                "tech": tech,
                "share_2024": rates["share_2024"],
                "share_2026": rates["share_2026"],
                "arshkon_2024": ars,
                "asaniczka_2024": asa,
                "within_2024": within_2024,
                "cross_period": cross_period,
                "snr": snr,
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(f"{OUT}/V1_tech_snr.csv", index=False)
    with pd.option_context("display.width", 180, "display.max_columns", 99):
        print(out.to_string(index=False))
    return out


# -----------------------------------------------------------------------------
# Target 5: Arshkon-only seniority_final entry share flip
# -----------------------------------------------------------------------------
def target_5_arshkon_only_entry() -> None:
    print()
    print("=" * 72)
    print("TARGET 5: Arshkon-only `seniority_final` entry share")
    print("=" * 72)

    q = f"""
    SELECT source, period, seniority_final, COUNT(*) AS n
    FROM '{UNI}'
    WHERE is_swe = true
      AND source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
    GROUP BY source, period, seniority_final
    ORDER BY source, period, seniority_final
    """
    df = con.execute(q).fetchdf()

    # Arshkon-only 2024 baseline
    ars = df[df["source"] == "kaggle_arshkon"]
    scr = df[df["source"] == "scraped"]

    def entry_shares(d: pd.DataFrame, label: str) -> dict:
        total = d["n"].sum()
        entry = d[d["seniority_final"] == "entry"]["n"].sum()
        known = d[~d["seniority_final"].isin(["unknown"])]["n"].sum()
        return {
            "slice": label,
            "n_total": int(total),
            "n_entry": int(entry),
            "n_known": int(known),
            "entry_share_of_all": entry / total,
            "entry_share_of_known": entry / known if known else float("nan"),
        }

    rows = [
        entry_shares(ars, "arshkon_only_2024"),
        entry_shares(scr, "scraped_2026"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/V1_arshkon_only_entry_share.csv", index=False)
    with pd.option_context("display.width", 180, "display.max_columns", 99):
        print(out.to_string(index=False))

    print()
    print(
        "Gate 2 reported: arshkon-only 7.72% -> scraped 6.70% entry-of-known (-1.0 pp)."
    )


# -----------------------------------------------------------------------------
# Target 4: Within-archetype uniform junior-share rise
# -----------------------------------------------------------------------------
def target_4_archetype_junior_share() -> None:
    print()
    print("=" * 72)
    print("TARGET 4: Within-archetype junior share (T09 uniform-rise check)")
    print("=" * 72)

    # Load archetype labels
    q = f"""
    SELECT a.*, m.source, m.period, m.seniority_3level, m.seniority_final
    FROM '{ARCH}' a
    INNER JOIN '{META}' m USING (uid)
    """
    df = con.execute(q).fetchdf()
    print(f"archetype-labeled rows: {len(df):,}")
    print("columns:", df.columns.tolist())

    # Bucket period
    df["period_bucket"] = df.apply(
        lambda r: classify_period(r["source"], r["period"]), axis=1
    )

    # The archetype label column is likely 'archetype_label' or similar
    label_col = None
    for c in df.columns:
        if c not in ("uid", "source", "period", "seniority_3level", "seniority_final", "period_bucket"):
            # prefer human-friendly name
            if "label" in c or "archetype" in c.lower() or "name" in c:
                label_col = c
                break
    if label_col is None:
        # fallback: use first non-key non-meta column
        for c in df.columns:
            if c not in ("uid", "source", "period", "seniority_3level", "seniority_final", "period_bucket"):
                label_col = c
                break
    print(f"Using label column: {label_col}")

    def jr_share(sub: pd.DataFrame) -> float:
        known = sub[sub["seniority_3level"].isin(["junior", "mid", "senior"])]
        if len(known) == 0:
            return float("nan")
        return (known["seniority_3level"] == "junior").sum() / len(known)

    rows = []
    for (arch, p), sub in df.groupby([label_col, "period_bucket"]):
        known = sub[sub["seniority_3level"].isin(["junior", "mid", "senior"])]
        n = len(sub)
        nk = len(known)
        if nk == 0:
            continue
        rows.append(
            {
                "archetype": arch,
                "period": p,
                "n": n,
                "n_known": nk,
                "junior_share": (known["seniority_3level"] == "junior").sum() / nk,
            }
        )
    by = pd.DataFrame(rows)
    # pivot
    pivot = by.pivot(index="archetype", columns="period", values="junior_share").fillna(0)
    counts = by.pivot(index="archetype", columns="period", values="n_known").fillna(0)
    pivot.columns = [f"jr_{c}" for c in pivot.columns]
    counts.columns = [f"n_{c}" for c in counts.columns]
    out = pivot.join(counts).reset_index()
    out = out.sort_values("archetype")
    out.to_csv(f"{OUT}/V1_archetype_junior_share.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 99, "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    target_1_2_tech_snr()
    target_5_arshkon_only_entry()
    target_4_archetype_junior_share()
