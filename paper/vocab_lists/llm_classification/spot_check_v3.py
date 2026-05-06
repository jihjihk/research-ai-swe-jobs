"""Render v3 pilot results into a 3-variant side-by-side spot-check markdown."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
RESULTS = DIR / "pilot_v3_results.jsonl"
SAMPLE = DIR / "pilot_sample_v3.parquet"
OUT = DIR / "spot_check_v3.md"

DESC_TRUNCATE_CHARS = 1000


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
    variants = ["skill", "role_family", "combined"]

    # by (uid, variant, model) -> list of rep labelsets per axis
    grouped_skill = (
        df[df["variant"].isin(["skill", "combined"])]
        .groupby(["uid", "variant", "model"])["skill_themes"]
        .apply(list).to_dict()
    )
    grouped_role = (
        df[df["variant"].isin(["role_family", "combined"])]
        .groupby(["uid", "variant", "model"])["role_families"]
        .apply(list).to_dict()
    )

    md = []
    md.append("# V3 pilot — spot-check\n")
    md.append(f"_{len(sample)} postings × 3 variants × {len(models)} models × "
              f"{df.groupby(['uid','variant','model']).size().max()} reps. Read each block; "
              f"compare standalone-variant labels vs combined-variant labels per axis._\n")
    md.append("Strata in this sample:")
    md.append("- **role:\\<family\\>** — 1 posting per role-family by title heuristic")
    md.append("- **skill:multi** — 3+ regex skill themes")
    md.append("- **skill:zero** — 0 regex skill themes")
    md.append("- **era:\\<source\\>** — era-balance fillers\n")
    md.append("---\n")

    for i, row in sample.iterrows():
        idx = i + 1
        title = row["title"] or "_(no title)_"
        desc = row["description_core_llm"] or ""
        if len(desc) > DESC_TRUNCATE_CHARS:
            desc = desc[:DESC_TRUNCATE_CHARS].rstrip() + " …"
        regex_skill = sorted(row["regex_skill_topics"]) if row["regex_skill_topics"] is not None else []

        md.append(f"## #{idx}  [{row['stratum_v3']}]  {title}")
        md.append(f"**uid:** `{row['uid']}` · **source:** `{row['source']}/{row['period']}` "
                  f"· **company:** {row.get('company_name_effective','—') or '—'} "
                  f"· **role-family heuristic:** `{row['role_family_heuristic']}`")
        md.append("")
        md.append("**Description (truncated):**")
        md.append(f"> {desc.replace(chr(10), ' ')}")
        md.append("")
        md.append(f"**Regex skill topics:** {fmt_set(regex_skill)}\n")

        # SKILL AXIS — skill variant vs combined variant
        md.append("**Skill axis (majority labels per variant × model):**\n")
        md.append("| Model | `skill` variant | `combined` variant |")
        md.append("|---|---|---|")
        for m in models:
            stand = grouped_skill.get((row["uid"], "skill", m), [])
            comb = grouped_skill.get((row["uid"], "combined", m), [])
            md.append(f"| `{m}` | {fmt_set(majority_labels(stand))} | "
                      f"{fmt_set(majority_labels(comb))} |")
        md.append("")

        # ROLE FAMILY AXIS — role_family variant vs combined variant
        md.append("**Role-family axis (majority labels per variant × model):**\n")
        md.append("| Model | `role_family` variant | `combined` variant |")
        md.append("|---|---|---|")
        for m in models:
            stand = grouped_role.get((row["uid"], "role_family", m), [])
            comb = grouped_role.get((row["uid"], "combined", m), [])
            md.append(f"| `{m}` | {fmt_set(majority_labels(stand))} | "
                      f"{fmt_set(majority_labels(comb))} |")
        md.append("")
        md.append("---")
        md.append("")

    OUT.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
