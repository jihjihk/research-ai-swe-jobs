"""Analyze pilot_results.jsonl — inter-model agreement, self-agreement, regex comparison.

Outputs:
  - analysis.md  (human-readable narrative + tables)
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
RESULTS = DIR / "pilot_results.jsonl"
SAMPLE = DIR / "pilot_sample.parquet"
OUT = DIR / "analysis.md"

LABELS = (
    "people_management", "orchestration", "verification", "mentorship",
    "performance", "process_scaffolding", "legacy_stack", "context_infrastructure",
)


def cohen_kappa_binary(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two parallel 0/1 vectors. NaN if degenerate."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pa0 = 1 - pa1
    pb1 = sum(b) / n
    pb0 = 1 - pb1
    pe = pa0 * pb0 + pa1 * pb1
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def f1_binary(true: list[int], pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(true, pred) if t and p)
    fp = sum(1 for t, p in zip(true, pred) if not t and p)
    fn = sum(1 for t, p in zip(true, pred) if t and not p)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")
    return prec, rec, f1


def jaccard(a: set, b: set) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def majority_labels(label_lists: list[list[str]]) -> set[str]:
    """For self-agreed labels across reps: label appears in >=ceil(n/2) reps."""
    if not label_lists:
        return set()
    counts = defaultdict(int)
    for lst in label_lists:
        for l in lst:
            counts[l] += 1
    threshold = (len(label_lists) + 1) // 2  # ceil
    return {l for l, c in counts.items() if c >= threshold}


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} call results.")

    # parse-failure rate
    parse_fail = df[~df["parsed_ok"]]
    print(f"Parse failures: {len(parse_fail)} of {len(df)} ({len(parse_fail)/len(df):.1%})")

    # ---- Aggregate to per-(uid, model) majority labels ----
    grouped = df.groupby(["uid", "model"])["labels"].apply(list).reset_index()
    grouped["majority"] = grouped["labels"].apply(lambda lsts: majority_labels(lsts))

    # latency / token stats per model
    lat_stats = df.groupby("model")["latency_s"].agg(["mean", "median", "max", "count"])
    tok_stats = df.groupby("model")[["input_tokens", "output_tokens", "cached_tokens"]].agg(["mean", "sum"])

    # uids in canonical order from sample (for stable rows)
    sample = pd.read_parquet(SAMPLE)
    uids = sample["uid"].tolist()
    models = sorted(df["model"].unique())

    # ---- 1. Self-agreement (Jaccard across reps within a model) ----
    self_agreement: dict[str, list[float]] = {m: [] for m in models}
    for (uid, model), g in df.groupby(["uid", "model"]):
        rep_label_sets = [set(lst) for lst in g["labels"].tolist()]
        if len(rep_label_sets) < 2:
            continue
        # mean pairwise Jaccard
        sims = [jaccard(a, b) for a, b in combinations(rep_label_sets, 2)]
        self_agreement[model].append(sum(sims) / len(sims))

    # ---- 2. Inter-model agreement (per-label Cohen's kappa, on majority labels) ----
    # Build per-(uid, model) majority label sets indexed by uid order
    by_um: dict[tuple[str, str], set[str]] = {}
    for _, row in grouped.iterrows():
        by_um[(row["uid"], row["model"])] = row["majority"]

    inter_kappa = {}  # (m1, m2) -> {label: kappa, "macro": ...}
    for m1, m2 in combinations(models, 2):
        per_label = {}
        for label in LABELS:
            v1 = [int(label in by_um.get((u, m1), set())) for u in uids]
            v2 = [int(label in by_um.get((u, m2), set())) for u in uids]
            per_label[label] = cohen_kappa_binary(v1, v2)
        valid = [k for k in per_label.values() if k == k]  # filter nan
        per_label["_macro"] = sum(valid) / len(valid) if valid else float("nan")
        # set-Jaccard (overall)
        jacs = [jaccard(by_um.get((u, m1), set()), by_um.get((u, m2), set())) for u in uids]
        per_label["_set_jaccard"] = sum(jacs) / len(jacs)
        inter_kappa[(m1, m2)] = per_label

    # ---- 3. LLM vs regex comparison (per-label F1) ----
    regex_by_uid = {row["uid"]: set(row["regex_topics"]) for _, row in sample.iterrows()}
    llm_vs_regex = {}  # model -> {label: (prec, rec, f1)}
    for m in models:
        per_label = {}
        for label in LABELS:
            true_v = [int(label in regex_by_uid[u]) for u in uids]
            pred_v = [int(label in by_um.get((u, m), set())) for u in uids]
            per_label[label] = f1_binary(true_v, pred_v)
        llm_vs_regex[m] = per_label

    # ---- 4. Confusion: where do models disagree? ----
    # For each posting, list (model, majority labels). Flag postings where set-jaccard between any pair < 0.5.
    disagreement_postings = []
    for u in uids:
        sets = [(m, by_um.get((u, m), set())) for m in models]
        worst = 1.0
        for (m1, s1), (m2, s2) in combinations(sets, 2):
            j = jaccard(s1, s2)
            if j < worst:
                worst = j
        if worst < 0.6:
            disagreement_postings.append((u, worst, sets))

    # ============================================================
    # Render markdown
    # ============================================================
    md = []
    md.append("# LLM classification pilot — analysis\n")
    md.append(f"_Calls analyzed: {len(df)} from {len(uids)} postings × {len(models)} models × "
              f"{df.groupby(['uid','model']).size().max()} reps._\n")
    md.append("")

    md.append("## Latency and tokens per model\n")
    md.append("| Model | Mean latency (s) | Median (s) | Max (s) | Mean input tok | Mean output tok | "
              "Cached tokens (sum) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        lat = lat_stats.loc[m]
        in_mean = tok_stats.loc[m, ("input_tokens", "mean")]
        out_mean = tok_stats.loc[m, ("output_tokens", "mean")]
        cached_sum = tok_stats.loc[m, ("cached_tokens", "sum")]
        md.append(f"| `{m}` | {lat['mean']:.2f} | {lat['median']:.2f} | {lat['max']:.2f} | "
                  f"{in_mean:.0f} | {out_mean:.0f} | {int(cached_sum) if cached_sum == cached_sum else 0} |")
    md.append("")

    md.append("## Self-agreement (Jaccard across reps within a model)\n")
    md.append("Higher is better. <0.95 = unstable on this prompt.\n")
    md.append("| Model | Mean self-Jaccard (across postings) | Min |")
    md.append("|---|---:|---:|")
    for m in models:
        vals = self_agreement[m]
        if vals:
            mn = min(vals)
            mean = sum(vals) / len(vals)
            md.append(f"| `{m}` | {mean:.3f} | {mn:.3f} |")
        else:
            md.append(f"| `{m}` | n/a | n/a |")
    md.append("")

    md.append("## Inter-model agreement\n")
    md.append("Per-label Cohen's κ on majority-vote labels (3 reps → label is positive if it appears in ≥2 reps). "
              "Higher = more agreement; >0.7 = substantial; >0.4 = moderate.\n")
    md.append("| Model A | Model B | Macro κ | Set Jaccard | "
              + " | ".join(f"{l[:6]}κ" for l in LABELS) + " |")
    md.append("|" + "---|" * (4 + len(LABELS)))
    for (m1, m2), per_label in inter_kappa.items():
        cells = [f"{per_label[l]:.2f}" if per_label[l] == per_label[l] else "n/a"
                 for l in LABELS]
        md.append(f"| `{m1}` | `{m2}` | {per_label['_macro']:.2f} | "
                  f"{per_label['_set_jaccard']:.2f} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## LLM vs regex (per-label P / R / F1)\n")
    md.append("Treats regex topic-density positives as 'true' and the model's majority labels as 'predicted'. "
              "Disagreements are interesting on both sides: the regex has known false-positive issues; the LLM "
              "is independent. Eyeball both columns rather than treating either as ground truth.\n")
    for m in models:
        md.append(f"### `{m}`\n")
        md.append("| Label | P | R | F1 |")
        md.append("|---|---:|---:|---:|")
        for label in LABELS:
            prec, rec, f1 = llm_vs_regex[m][label]
            def fmt(x):
                return f"{x:.2f}" if x == x else "n/a"
            md.append(f"| `{label}` | {fmt(prec)} | {fmt(rec)} | {fmt(f1)} |")
        md.append("")

    md.append("## Postings with cross-model disagreement\n")
    md.append(f"{len(disagreement_postings)} postings show worst pairwise set-Jaccard < 0.6.\n")
    if disagreement_postings:
        md.append("| uid | worst pairwise Jaccard | per-model labels |")
        md.append("|---|---:|---|")
        for uid, worst, sets in sorted(disagreement_postings, key=lambda x: x[1])[:10]:
            cell = " · ".join(f"`{m}`={sorted(s)}" for m, s in sets)
            md.append(f"| `{uid[:12]}` | {worst:.2f} | {cell} |")
    md.append("")

    md.append("## Notes\n")
    md.append("- Cohen's κ is computed on majority-vote labels per (posting × model). Reps were 3.")
    md.append("- 'Set Jaccard' is `|A∩B| / |A∪B|` averaged across postings.")
    md.append("- LLM-vs-regex F1 is descriptive, not validation. Use spot_check.md to read example disagreements.")

    OUT.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
