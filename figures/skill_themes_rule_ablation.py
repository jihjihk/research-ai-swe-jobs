#!/usr/bin/env python3
"""
Rule-based skill-theme ablation for the Stage 12 LLM classifier.

Applies the curated keyword vocab in `paper/vocab_lists/vocab_lists.json` (one
topic per skill theme, 8 themes total) as a plain regex sweep over
`description_core_llm` for every classified SWE row in `unified_core.parquet`,
producing a `skill_themes_rule` list-column parallel to the LLM's
`skill_themes`. Outputs a row-aligned parquet plus an agreement summary CSV
suitable for inclusion as an ablation table in the paper.

This is intentionally minimal: each topic's keywords are OR'd into a single
case-insensitive `\b(kw1|kw2|...)\b` pattern, with no exclusion guards. The
LLM's contribution shows up as the delta between this naive rule sweep and
the model's labels.

Outputs:
  figures/output/skill_themes_rule.parquet      (uid, skill_themes_rule)
  figures/output/skill_themes_rule_vs_llm.csv   (per-topic agreement)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = PROJECT_ROOT / "paper" / "vocab_lists" / "vocab_lists.json"
CORE_PATH = PROJECT_ROOT / "data" / "unified_core.parquet"
OUT_DIR = PROJECT_ROOT / "figures" / "output"
OUT_PARQUET = OUT_DIR / "skill_themes_rule.parquet"
OUT_CSV = OUT_DIR / "skill_themes_rule_vs_llm.csv"

CHUNK = 10_000


def compile_topic_patterns(vocab_path: Path) -> dict[str, re.Pattern]:
    """Return {topic_name: compiled regex} where the regex matches if any
    keyword in the topic appears (case-insensitive, word-boundary anchored)."""
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    topics = vocab["topics"]
    patterns: dict[str, re.Pattern] = {}
    for topic_name, topic in topics.items():
        keywords: list[str] = []
        for concept in topic.get("core_concepts", []):
            keywords.extend(concept.get("keywords", []))
        # De-duplicate and sort longest-first so multi-word forms win over
        # any prefix overlap.
        keywords = sorted({k.strip() for k in keywords if k.strip()}, key=lambda k: (-len(k), k))
        if not keywords:
            continue
        alternation = "|".join(re.escape(k) for k in keywords)
        patterns[topic_name] = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
    return patterns


def apply_rule(text: str | None, patterns: dict[str, re.Pattern]) -> list[str]:
    """Return the list of topic names whose pattern fires on `text`."""
    if not text:
        return []
    return sorted(name for name, pat in patterns.items() if pat.search(text))


def run(core_path: Path, vocab_path: Path) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = compile_topic_patterns(vocab_path)
    topic_order = sorted(patterns)

    pf = pq.ParquetFile(core_path)
    top_level = set(pf.schema_arrow.names)
    required = {"uid", "description_core_llm", "skill_themes", "is_swe"}
    if not required <= top_level:
        raise RuntimeError(f"{core_path} missing required columns: {sorted(required - top_level)}")

    # Counts per topic: TP/FP/FN/TN against the LLM's `skill_themes`.
    counts = {t: {"rule_only": 0, "llm_only": 0, "both": 0, "neither": 0} for t in topic_order}
    n_swe = 0

    out_uids: list[str] = []
    out_rule: list[list[str] | None] = []

    for batch in pf.iter_batches(batch_size=CHUNK, columns=["uid", "description_core_llm", "skill_themes", "is_swe"]):
        uids = batch.column("uid").to_pylist()
        descs = batch.column("description_core_llm").to_pylist()
        llms = batch.column("skill_themes").to_pylist()
        is_swe = batch.column("is_swe").to_pylist()
        for uid, desc, llm_labels, swe in zip(uids, descs, llms, is_swe):
            if not swe:
                # Match the LLM convention: NULL on non-SWE rows.
                out_uids.append(uid)
                out_rule.append(None)
                continue
            n_swe += 1
            rule_labels = apply_rule(desc, patterns)
            out_uids.append(uid)
            out_rule.append(rule_labels)

            llm_set = set(llm_labels) if llm_labels is not None else set()
            rule_set = set(rule_labels)
            for t in topic_order:
                rule_hit = t in rule_set
                llm_hit = t in llm_set
                if rule_hit and llm_hit:
                    counts[t]["both"] += 1
                elif rule_hit:
                    counts[t]["rule_only"] += 1
                elif llm_hit:
                    counts[t]["llm_only"] += 1
                else:
                    counts[t]["neither"] += 1

    # Write per-row parquet
    table = pa.table({
        "uid": pa.array(out_uids, type=pa.string()),
        "skill_themes_rule": pa.array(out_rule, type=pa.list_(pa.string())),
    })
    pq.write_table(table, OUT_PARQUET, compression="zstd")

    # Build the agreement table
    rows = []
    for t in topic_order:
        c = counts[t]
        n = n_swe
        rule_pct = (c["rule_only"] + c["both"]) / n * 100 if n else 0.0
        llm_pct = (c["llm_only"] + c["both"]) / n * 100 if n else 0.0
        # Treat the LLM as the reference label set, rule as the predictor.
        # Precision = both / (rule_only + both) — when rule fires, how often does LLM agree?
        # Recall    = both / (llm_only + both)  — of LLM-positive rows, how many does rule catch?
        prec = c["both"] / (c["rule_only"] + c["both"]) if (c["rule_only"] + c["both"]) else None
        rec = c["both"] / (c["llm_only"] + c["both"]) if (c["llm_only"] + c["both"]) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        rows.append({
            "topic": t,
            "n_swe": n,
            "rule_pct": round(rule_pct, 2),
            "llm_pct": round(llm_pct, 2),
            "rule_only": c["rule_only"],
            "llm_only": c["llm_only"],
            "both": c["both"],
            "neither": c["neither"],
            "precision_rule_vs_llm": round(prec, 3) if prec is not None else None,
            "recall_rule_vs_llm": round(rec, 3) if rec is not None else None,
            "f1_rule_vs_llm": round(f1, 3) if f1 is not None else None,
        })

    # Write CSV
    field_order = [
        "topic", "n_swe", "rule_pct", "llm_pct",
        "rule_only", "llm_only", "both", "neither",
        "precision_rule_vs_llm", "recall_rule_vs_llm", "f1_rule_vs_llm",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"n_swe": n_swe, "rows": rows, "parquet": OUT_PARQUET, "csv": OUT_CSV}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based skill-theme ablation")
    parser.add_argument("--core", type=Path, default=CORE_PATH)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    args = parser.parse_args()

    summary = run(args.core, args.vocab)
    print(f"Classified SWE rows: {summary['n_swe']:,}")
    print(f"Output parquet:      {summary['parquet']}")
    print(f"Output CSV:          {summary['csv']}")
    print()
    print("--- Rule vs LLM agreement (per topic) ---")
    print(f"{'topic':<25} {'rule%':>6} {'llm%':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for r in summary["rows"]:
        p = f"{r['precision_rule_vs_llm']:.2f}" if r["precision_rule_vs_llm"] is not None else "  -"
        rec = f"{r['recall_rule_vs_llm']:.2f}" if r["recall_rule_vs_llm"] is not None else "  -"
        f1 = f"{r['f1_rule_vs_llm']:.2f}" if r["f1_rule_vs_llm"] is not None else "  -"
        print(f"  {r['topic']:<23} {r['rule_pct']:>6.1f} {r['llm_pct']:>6.1f} {p:>6} {rec:>6} {f1:>6}")


if __name__ == "__main__":
    main()
