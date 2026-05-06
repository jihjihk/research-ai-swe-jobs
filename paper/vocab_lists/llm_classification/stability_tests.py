"""Stability tests on the chosen model — order-shuffling and prompt-paraphrase.

Inputs: paper/vocab_lists/llm_classification/pilot_sample.parquet (the same 25 postings).

Outputs:
  - stability_results.jsonl
  - stability_report.md

Tests:
  T1. ORDER — original alphabetical, reverse, by-regex-frequency, random shuffle.
  T2. PARAPHRASE — original prompt + 2 paraphrased system prompts.

For each variant we run the 25 postings once, then compute Jaccard similarity to
the pilot's mini majority labels (the reference). If Jaccard stays high, the prompt
is robust.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
SAMPLE = DIR / "pilot_sample.parquet"
PILOT_RESULTS = DIR / "pilot_results.jsonl"
OUT_JSONL = DIR / "stability_results.jsonl"
OUT_MD = DIR / "stability_report.md"

sys.path.insert(0, str(Path(__file__).parent))
import classifier  # noqa: E402

# ---- Order variants ----
LABELS = list(classifier.LABELS)  # canonical order
ORDER_VARIANTS = {
    "alphabetical": sorted(LABELS),
    "reverse_alphabetical": sorted(LABELS, reverse=True),
    "regex_frequency_desc": [  # rough order based on prior calibration hit rates
        "verification",          # ~40%
        "orchestration",         # ~55%
        "context_infrastructure",
        "process_scaffolding",
        "mentorship",
        "performance",
        "legacy_stack",
        "people_management",
    ],
    "random_seed_7": None,  # filled below
}
_rng = random.Random(7)
_shuf = list(LABELS)
_rng.shuffle(_shuf)
ORDER_VARIANTS["random_seed_7"] = _shuf

# ---- Paraphrased system prompts ----
PARAPHRASE_VARIANTS = {
    "original": classifier.SYSTEM_PROMPT,
    "paraphrase_terse": """Classify which of 8 themes a SWE job posting explicitly states as a duty, requirement, or named skill. Skip boilerplate, passing references, or company descriptions.

Themes:
- people_management — formal management authority: direct reports, performance reviews, hiring/firing, headcount, 1:1 cadence.
- orchestration — authoring specifications, task decomposition, system architecture, or AI/agent orchestration (context engineering, multi-agent, agent harnesses).
- verification — tests, CI/CD, code review for correctness, evals, observability for correctness, compliance/audit work.
- mentorship — peer/IC-level teaching, growing, or shepherding of other engineers (not management).
- performance — low-level performance work (profiling, latency, throughput, kernel/network internals) or explicit deep-technical-depth requirements.
- process_scaffolding — agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack — required experience with legacy enterprise stacks (.NET Framework, COBOL, mainframe, Java EE, VMware/vSphere, Active Directory, similar).
- context_infrastructure — documentation, runbooks, telemetry/observability hygiene, ADRs/RFCs, cross-functional coordination as substrate work.

Return a JSON array of slug strings. Empty array means none. No commentary.""",

    "paraphrase_active_voice": """Read the SWE job posting below and decide which of the 8 themes it explicitly describes as a stated responsibility, listed requirement, or named skill. Treat boilerplate, passing mentions, and company-description language as evidence of nothing.

The themes:
- people_management: the role grants formal people-management authority — managing direct reports, running performance reviews, hiring or firing, owning headcount, holding 1:1s.
- orchestration: the role involves writing specs, decomposing work, doing system architecture, or orchestrating AI/agents (context engineering, multi-agent coordination, agent harnesses).
- verification: the role involves testing, CI/CD, correctness-focused code review, evaluations, observability-for-correctness, or compliance/audit.
- mentorship: the role involves peer or IC-level teaching, growing, or shepherding of other engineers — distinct from management.
- performance: the role demands low-level performance work (profiling, latency, throughput, kernel or network internals) or explicit deep technical depth.
- process_scaffolding: the role involves agile/scrum/sprints, requirements engineering, V&V, project coordination, or SDLC governance.
- legacy_stack: the role requires experience with legacy enterprise stacks (.NET Framework, COBOL, mainframe, Java EE, VMware/vSphere, Active Directory, etc.).
- context_infrastructure: the role involves documentation, runbooks, telemetry/observability hygiene, ADRs/RFCs, or cross-functional coordination as substrate work.

Output a JSON array of the matching slug strings. If none match, output an empty array. Do not write any other text.""",
}


def majority_labels(label_lists):
    if not label_lists:
        return set()
    counts = defaultdict(int)
    for lst in label_lists:
        for l in lst:
            counts[l] += 1
    threshold = (len(label_lists) + 1) // 2
    return {l for l, c in counts.items() if c >= threshold}


def jaccard(a, b):
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def load_pilot_reference(model: str) -> dict[str, set[str]]:
    rows = [json.loads(l) for l in PILOT_RESULTS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["model"] == model]
    grouped = df.groupby("uid")["labels"].apply(list).to_dict()
    return {uid: majority_labels(reps) for uid, reps in grouped.items()}


def run_variant(*, sample, model: str, system_prompt: str, label_order: list[str],
                variant_name: str, workers: int) -> list[dict]:
    plan = list(sample.itertuples(index=False))

    def worker(row):
        result = classifier.classify(
            title=row.title or "",
            description=row.description_core_llm or "",
            model=model,
            system_prompt_override=system_prompt,
            label_order_override=tuple(label_order),
        )
        return {
            "uid": row.uid,
            "stratum": row.stratum,
            "variant": variant_name,
            "model": model,
            "labels": result.labels,
            "parsed_ok": result.parsed_ok,
            "latency_s": result.latency_s,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "error": result.error,
        }

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, row) for row in plan]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--max-workers", type=int, default=6)
    args = p.parse_args()

    sample = pd.read_parquet(SAMPLE)
    print(f"Stability tests on {args.model} against {len(sample)} postings.")

    # truncate output
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT_JSONL.open("a", encoding="utf-8")

    # ---- ORDER variants (original system prompt) ----
    for vname, order in ORDER_VARIANTS.items():
        print(f"  ORDER variant: {vname} ...", flush=True)
        recs = run_variant(
            sample=sample,
            model=args.model,
            system_prompt=classifier.SYSTEM_PROMPT,
            label_order=order,
            variant_name=f"order:{vname}",
            workers=args.max_workers,
        )
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()

    # ---- PARAPHRASE variants (alphabetical order) ----
    for vname, sp in PARAPHRASE_VARIANTS.items():
        if vname == "original":
            # already covered by ORDER:alphabetical; skip duplicate
            continue
        print(f"  PARAPHRASE variant: {vname} ...", flush=True)
        recs = run_variant(
            sample=sample,
            model=args.model,
            system_prompt=sp,
            label_order=sorted(LABELS),
            variant_name=f"paraphrase:{vname}",
            workers=args.max_workers,
        )
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
    fh.close()

    # ---- Analysis ----
    print(f"\nLoading reference from pilot ({args.model})...")
    reference = load_pilot_reference(args.model)

    rows = [json.loads(l) for l in OUT_JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} stability calls.")

    # Per-variant: mean Jaccard vs reference
    md = []
    md.append(f"# LLM classification — stability tests ({args.model})\n")
    md.append(f"_Variants tested against pilot reference (3-rep majority on `{args.model}`)._\n")

    md.append("## Order-shuffle variants\n")
    md.append("Same prompt, different label-list order in the structured-output enum. "
              "If the model is robust, Jaccard vs. reference stays >=0.95.\n")
    md.append("| Variant | Mean Jaccard vs reference | Min | Postings perfectly stable |")
    md.append("|---|---:|---:|---:|")
    order_subset = df[df["variant"].str.startswith("order:")]
    for v in order_subset["variant"].unique():
        sub = order_subset[order_subset["variant"] == v]
        sims = []
        n_perfect = 0
        for _, r in sub.iterrows():
            ref = reference.get(r["uid"], set())
            this = set(r["labels"])
            j = jaccard(ref, this)
            sims.append(j)
            if j == 1.0:
                n_perfect += 1
        mean = sum(sims) / len(sims) if sims else 0
        mn = min(sims) if sims else 0
        md.append(f"| `{v}` | {mean:.3f} | {mn:.3f} | {n_perfect}/{len(sims)} |")
    md.append("")

    md.append("## Paraphrase variants\n")
    md.append("Same label set, different system-prompt phrasing. If the model "
              "is robust to surface phrasing, Jaccard vs. reference stays high.\n")
    md.append("| Variant | Mean Jaccard vs reference | Min | Postings perfectly stable |")
    md.append("|---|---:|---:|---:|")
    par_subset = df[df["variant"].str.startswith("paraphrase:")]
    for v in par_subset["variant"].unique():
        sub = par_subset[par_subset["variant"] == v]
        sims = []
        n_perfect = 0
        for _, r in sub.iterrows():
            ref = reference.get(r["uid"], set())
            this = set(r["labels"])
            j = jaccard(ref, this)
            sims.append(j)
            if j == 1.0:
                n_perfect += 1
        mean = sum(sims) / len(sims) if sims else 0
        mn = min(sims) if sims else 0
        md.append(f"| `{v}` | {mean:.3f} | {mn:.3f} | {n_perfect}/{len(sims)} |")
    md.append("")

    # Cross-variant stability: how often do all variants agree on the same label set?
    md.append("## Cross-variant agreement\n")
    md.append("For each posting, fraction of (variant pairs) with identical label sets.\n")
    by_uid = defaultdict(dict)
    for _, r in df.iterrows():
        by_uid[r["uid"]][r["variant"]] = set(r["labels"])
    cross_agreements = []
    for uid, vmap in by_uid.items():
        variants = list(vmap.values())
        if len(variants) < 2:
            continue
        sims = [jaccard(a, b) for a, b in combinations(variants, 2)]
        cross_agreements.append((uid, sum(sims) / len(sims), min(sims)))
    cross_agreements.sort(key=lambda x: x[1])
    md.append("**Worst-stability postings** (lowest mean cross-variant Jaccard):\n")
    md.append("| uid | mean cross-var Jaccard | min |")
    md.append("|---|---:|---:|")
    for uid, mean, mn in cross_agreements[:10]:
        md.append(f"| `{uid[:14]}` | {mean:.2f} | {mn:.2f} |")
    md.append("")
    md.append(f"**Overall**: mean cross-variant Jaccard = "
              f"{sum(c[1] for c in cross_agreements)/len(cross_agreements):.3f}")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSONL}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
