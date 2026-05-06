"""Render the 25 pilot postings + per-model labels into a single skim-friendly markdown.

Use: a human reads this for quick sanity check. Not formal validation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
RESULTS = DIR / "pilot_results.jsonl"
SAMPLE = DIR / "pilot_sample.parquet"
OUT = DIR / "spot_check.md"

LABELS = (
    "people_management", "orchestration", "verification", "mentorship",
    "performance", "process_scaffolding", "legacy_stack", "context_infrastructure",
)

DESC_TRUNCATE_CHARS = 1200  # keep spot-check readable


def majority_labels(label_lists):
    if not label_lists:
        return []
    counts = defaultdict(int)
    for lst in label_lists:
        for l in lst:
            counts[l] += 1
    threshold = (len(label_lists) + 1) // 2
    return sorted(l for l, c in counts.items() if c >= threshold)


def fmt_set(s):
    if not s:
        return "_(none)_"
    return ", ".join(f"`{l}`" for l in sorted(s))


def main():
    sample = pd.read_parquet(SAMPLE)
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    models = sorted(df["model"].unique())

    # by (uid, model) -> list of rep labelsets
    grouped = df.groupby(["uid", "model"])["labels"].apply(list).to_dict()

    md = []
    md.append("# LLM classification pilot — spot-check\n")
    md.append(f"_25 postings × {len(models)} models × "
              f"{df.groupby(['uid','model']).size().max()} reps. Read each block, eyeball whether the "
              f"LLM-majority labels make sense given the description._\n")
    md.append("Stratification:")
    md.append("- **A_single** (1-8): one regex topic only — easy single-label cases.")
    md.append("- **B_multi** (9-13): 3+ regex topics — multi-label stress test.")
    md.append("- **C_zero** (14-18): zero regex topics — false-positive/recall risk.")
    md.append("- **D_noise** (19-23): hits persistent-fluff regex concepts — does the LLM resist the fluff?")
    md.append("- **E_random** (24-25): sanity baseline.\n")
    md.append("---\n")

    for i, row in sample.iterrows():
        idx = i + 1
        title = row["title"] or "_(no title)_"
        desc = row["description_core_llm"] or ""
        if len(desc) > DESC_TRUNCATE_CHARS:
            desc = desc[:DESC_TRUNCATE_CHARS].rstrip() + " …"
        regex_topics = sorted(row["regex_topics"]) if row["regex_topics"] is not None else []

        md.append(f"## #{idx}  [{row['stratum']}]  {title}\n")
        md.append(f"**uid:** `{row['uid']}`  · **source:** `{row['source']}/{row['period']}`  "
                  f"· **company:** {row.get('company_name_effective','—') or '—'}\n")
        md.append("**Description (truncated):**")
        md.append(f"> {desc.replace(chr(10), ' ')}\n")
        md.append(f"**Regex topics:** {fmt_set(regex_topics)}\n")

        md.append("**LLM majority labels (per model):**")
        md.append("")
        md.append("| Model | Majority | Per-rep |")
        md.append("|---|---|---|")
        for m in models:
            reps = grouped.get((row["uid"], m), [])
            mj = majority_labels(reps)
            per_rep_str = " · ".join(
                "{" + ",".join(sorted(r)) + "}" if r else "{}" for r in reps
            )
            md.append(f"| `{m}` | {fmt_set(mj)} | {per_rep_str} |")
        md.append("")
        md.append("---")
        md.append("")

    OUT.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
