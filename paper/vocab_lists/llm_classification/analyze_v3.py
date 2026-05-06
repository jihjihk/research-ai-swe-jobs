"""V3 pilot analysis.

Per-variant inter-model κ (Cohen's kappa), self-stability (Jaccard across reps),
and the headline metric: standalone-vs-combined Jaccard per axis.

Outputs:
  - analysis_v3.md  (human-readable narrative + tables)
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DIR = REPO / "paper/vocab_lists/llm_classification"
RESULTS = DIR / "pilot_v3_results.jsonl"
SAMPLE = DIR / "pilot_sample_v3.parquet"
OUT = DIR / "analysis_v3.md"

SKILL_LABELS = (
    "people_management", "orchestration", "verification", "mentorship",
    "performance", "process_scaffolding", "legacy_stack", "context_infrastructure",
)
ROLE_FAMILY_LABELS = (
    "software_engineer_general", "frontend_web", "backend_api", "mobile",
    "embedded", "data_engineer", "ml_engineer", "ai_llm_engineer",
    "devops_sre_platform", "security", "qa_test", "solutions_field",
    "legacy_specialist", "data_analytics", "research", "infra_ops_admin",
    "people_manager",
)


def cohen_kappa_binary(a: list[int], b: list[int]) -> float:
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


def jaccard(a: set, b: set) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def majority_labels(label_lists: list[list[str]]) -> set[str]:
    if not label_lists:
        return set()
    counts = defaultdict(int)
    for lst in label_lists:
        for l in lst:
            counts[l] += 1
    threshold = (len(label_lists) + 1) // 2
    return {l for l, c in counts.items() if c >= threshold}


def fmt_or_na(x, prec=2):
    if x is None or x != x:
        return "n/a"
    return f"{x:.{prec}f}"


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    sample = pd.read_parquet(SAMPLE)
    uids = sample["uid"].tolist()
    print(f"Loaded {len(df)} call results.")

    parse_fail = df[~df["parsed_ok"]]
    print(f"Parse failures: {len(parse_fail)} / {len(df)} ({len(parse_fail)/len(df):.1%})")

    variants = sorted(df["variant"].unique())
    models = sorted(df["model"].unique())

    # ---- Per-(uid, variant, model) majority labels for each axis ----
    by_uvm_skill: dict[tuple[str, str, str], set[str]] = {}
    by_uvm_role: dict[tuple[str, str, str], set[str]] = {}
    grouped = df.groupby(["uid", "variant", "model"])
    for (uid, variant, model), g in grouped:
        skill_lists = [list(x) for x in g["skill_themes"].tolist()]
        role_lists = [list(x) for x in g["role_families"].tolist()]
        # only use skill_themes for variants that emit it; same for role
        if variant in ("skill", "combined"):
            by_uvm_skill[(uid, variant, model)] = majority_labels(skill_lists)
        if variant in ("role_family", "combined"):
            by_uvm_role[(uid, variant, model)] = majority_labels(role_lists)

    # ---- 1. Self-stability per (variant, model) ----
    self_stab_skill: dict[tuple[str, str], list[float]] = defaultdict(list)
    self_stab_role: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (uid, variant, model), g in grouped:
        if variant in ("skill", "combined"):
            sets = [set(lst) for lst in g["skill_themes"].tolist()]
            if len(sets) >= 2:
                jacs = [jaccard(a, b) for a, b in combinations(sets, 2)]
                self_stab_skill[(variant, model)].append(sum(jacs) / len(jacs))
        if variant in ("role_family", "combined"):
            sets = [set(lst) for lst in g["role_families"].tolist()]
            if len(sets) >= 2:
                jacs = [jaccard(a, b) for a, b in combinations(sets, 2)]
                self_stab_role[(variant, model)].append(sum(jacs) / len(jacs))

    # ---- 2. Inter-model κ per variant per axis ----
    inter_skill: dict[tuple[str, str, str], dict[str, float]] = {}
    inter_role: dict[tuple[str, str, str], dict[str, float]] = {}
    for variant in variants:
        for m1, m2 in combinations(models, 2):
            if variant in ("skill", "combined"):
                per_label = {}
                for label in SKILL_LABELS:
                    v1 = [int(label in by_uvm_skill.get((u, variant, m1), set())) for u in uids]
                    v2 = [int(label in by_uvm_skill.get((u, variant, m2), set())) for u in uids]
                    per_label[label] = cohen_kappa_binary(v1, v2)
                jacs = [jaccard(by_uvm_skill.get((u, variant, m1), set()),
                                 by_uvm_skill.get((u, variant, m2), set()))
                        for u in uids]
                valid = [k for k in per_label.values() if k == k]
                per_label["_macro"] = sum(valid) / len(valid) if valid else float("nan")
                per_label["_set_jaccard"] = sum(jacs) / len(jacs)
                inter_skill[(variant, m1, m2)] = per_label
            if variant in ("role_family", "combined"):
                per_label = {}
                for label in ROLE_FAMILY_LABELS:
                    v1 = [int(label in by_uvm_role.get((u, variant, m1), set())) for u in uids]
                    v2 = [int(label in by_uvm_role.get((u, variant, m2), set())) for u in uids]
                    per_label[label] = cohen_kappa_binary(v1, v2)
                jacs = [jaccard(by_uvm_role.get((u, variant, m1), set()),
                                 by_uvm_role.get((u, variant, m2), set()))
                        for u in uids]
                valid = [k for k in per_label.values() if k == k]
                per_label["_macro"] = sum(valid) / len(valid) if valid else float("nan")
                per_label["_set_jaccard"] = sum(jacs) / len(jacs)
                inter_role[(variant, m1, m2)] = per_label

    # ---- 3. STANDALONE vs COMBINED Jaccard per axis (the headline metric) ----
    skill_combined_vs_standalone: dict[str, list[float]] = defaultdict(list)
    role_combined_vs_standalone: dict[str, list[float]] = defaultdict(list)
    for model in models:
        for u in uids:
            stand_skill = by_uvm_skill.get((u, "skill", model), set())
            comb_skill = by_uvm_skill.get((u, "combined", model), set())
            stand_role = by_uvm_role.get((u, "role_family", model), set())
            comb_role = by_uvm_role.get((u, "combined", model), set())
            skill_combined_vs_standalone[model].append(jaccard(stand_skill, comb_skill))
            role_combined_vs_standalone[model].append(jaccard(stand_role, comb_role))

    # ---- 4. Title-heuristic match for role_family ----
    title_match: dict[tuple[str, str], list[float]] = defaultdict(list)
    sample_by_uid = {r["uid"]: r for _, r in sample.iterrows()}
    for variant in ("role_family", "combined"):
        for model in models:
            for u in uids:
                heur = sample_by_uid[u]["role_family_heuristic"]
                if heur == "software_engineer_general":
                    continue  # heuristic returned the residual; uninformative
                pred = by_uvm_role.get((u, variant, model), set())
                title_match[(variant, model)].append(int(heur in pred))

    # ---- 5. software_engineer_general usage rate ----
    seg_misuse: dict[tuple[str, str], int] = defaultdict(int)
    seg_total: dict[tuple[str, str], int] = defaultdict(int)
    for variant in ("role_family", "combined"):
        for model in models:
            for u in uids:
                pred = by_uvm_role.get((u, variant, model), set())
                seg_total[(variant, model)] += 1
                # misuse = software_engineer_general appears alongside another family
                if "software_engineer_general" in pred and len(pred) > 1:
                    seg_misuse[(variant, model)] += 1

    # ============================================================
    # Render markdown
    # ============================================================
    md = []
    md.append("# V3 pilot — analysis\n")
    md.append(f"_Calls analyzed: {len(df)} from {len(uids)} postings × {len(variants)} variants × "
              f"{len(models)} models × {df.groupby(['uid','variant','model']).size().max()} reps. "
              f"Parse failures: {len(parse_fail)} ({len(parse_fail)/len(df):.1%})._\n")

    # ---- Headline: standalone vs combined per axis ----
    md.append("## Headline metric — does combining hurt? (set-Jaccard, standalone vs combined)\n")
    md.append("Threshold: ≥0.95 means combining is safe; 0.85-0.95 minor degradation; <0.85 ship two prompts.\n")
    md.append("| Model | Skill axis (skill ↔ combined) | Role family axis (role_family ↔ combined) |")
    md.append("|---|---:|---:|")
    for model in models:
        s = skill_combined_vs_standalone[model]
        r = role_combined_vs_standalone[model]
        s_mean = sum(s) / len(s) if s else float("nan")
        r_mean = sum(r) / len(r) if r else float("nan")
        s_min = min(s) if s else float("nan")
        r_min = min(r) if r else float("nan")
        md.append(f"| `{model}` | {fmt_or_na(s_mean,3)} (min {fmt_or_na(s_min,3)}) | "
                  f"{fmt_or_na(r_mean,3)} (min {fmt_or_na(r_min,3)}) |")
    md.append("")

    # ---- Self-stability ----
    md.append("## Self-stability (Jaccard across 3 reps within a (variant × model))\n")
    md.append("| Variant | Model | Skill axis mean | Role-family axis mean |")
    md.append("|---|---|---:|---:|")
    for variant in variants:
        for model in models:
            s_vals = self_stab_skill.get((variant, model), [])
            r_vals = self_stab_role.get((variant, model), [])
            s_mean = sum(s_vals) / len(s_vals) if s_vals else float("nan")
            r_mean = sum(r_vals) / len(r_vals) if r_vals else float("nan")
            md.append(f"| `{variant}` | `{model}` | {fmt_or_na(s_mean,3)} | {fmt_or_na(r_mean,3)} |")
    md.append("")

    # ---- Inter-model κ ----
    md.append("## Inter-model agreement (per variant)\n")
    md.append("Macro κ = mean of valid per-label kappas (skips labels with no positives).\n")

    md.append("### Skill axis\n")
    md.append("| Variant | Model A | Model B | Macro κ | Set Jaccard |")
    md.append("|---|---|---|---:|---:|")
    for (variant, m1, m2), per_label in inter_skill.items():
        md.append(f"| `{variant}` | `{m1}` | `{m2}` | {fmt_or_na(per_label['_macro'])} | "
                  f"{fmt_or_na(per_label['_set_jaccard'])} |")
    md.append("")

    md.append("### Role-family axis\n")
    md.append("| Variant | Model A | Model B | Macro κ | Set Jaccard |")
    md.append("|---|---|---|---:|---:|")
    for (variant, m1, m2), per_label in inter_role.items():
        md.append(f"| `{variant}` | `{m1}` | `{m2}` | {fmt_or_na(per_label['_macro'])} | "
                  f"{fmt_or_na(per_label['_set_jaccard'])} |")
    md.append("")

    # ---- Per-label inter-model κ for each variant (full, mini) only ----
    if "gpt-5.4" in models and "gpt-5.4-mini" in models:
        md.append("### Skill axis — per-label κ (`gpt-5.4` vs `gpt-5.4-mini`)\n")
        md.append("| Variant | " + " | ".join(f"{l[:6]}" for l in SKILL_LABELS) + " |")
        md.append("|" + "---|" * (1 + len(SKILL_LABELS)))
        for variant in variants:
            if variant not in ("skill", "combined"):
                continue
            per_label = inter_skill.get((variant, "gpt-5.4", "gpt-5.4-mini"), {})
            cells = " | ".join(fmt_or_na(per_label.get(l)) for l in SKILL_LABELS)
            md.append(f"| `{variant}` | {cells} |")
        md.append("")

        md.append("### Role-family axis — per-label κ (`gpt-5.4` vs `gpt-5.4-mini`)\n")
        md.append("Sparse families may show NaN κ if neither model tags any posting.\n")
        md.append("| Variant | " + " | ".join(f"{l[:8]}" for l in ROLE_FAMILY_LABELS) + " |")
        md.append("|" + "---|" * (1 + len(ROLE_FAMILY_LABELS)))
        for variant in variants:
            if variant not in ("role_family", "combined"):
                continue
            per_label = inter_role.get((variant, "gpt-5.4", "gpt-5.4-mini"), {})
            cells = " | ".join(fmt_or_na(per_label.get(l)) for l in ROLE_FAMILY_LABELS)
            md.append(f"| `{variant}` | {cells} |")
        md.append("")

    # ---- Title-heuristic match ----
    md.append("## Title-heuristic match (role family)\n")
    md.append("For postings whose title heuristic returned a specific (non-fallback) family, the rate "
              "at which the model also tagged that family. Higher is better.\n")
    md.append("| Variant | Model | Match rate |")
    md.append("|---|---|---:|")
    for variant in ("role_family", "combined"):
        for model in models:
            vals = title_match.get((variant, model), [])
            rate = sum(vals) / len(vals) if vals else float("nan")
            md.append(f"| `{variant}` | `{model}` | {fmt_or_na(rate, 3)} |")
    md.append("")

    # ---- software_engineer_general misuse ----
    md.append("## `software_engineer_general` misuse\n")
    md.append("`software_engineer_general` is supposed to be the residual fallback — the model is "
              "instructed never to pair it with another family. Misuse rate = % of postings where it "
              "appears alongside another family. Lower is better.\n")
    md.append("| Variant | Model | Misuse rate |")
    md.append("|---|---|---:|")
    for variant in ("role_family", "combined"):
        for model in models:
            t = seg_total[(variant, model)]
            m = seg_misuse[(variant, model)]
            rate = m / t if t else 0
            md.append(f"| `{variant}` | `{model}` | {rate:.1%} ({m}/{t}) |")
    md.append("")

    md.append("## Notes\n")
    md.append("- Sample size is 31 postings; per-label kappa is statistically noisy — treat as directional.")
    md.append("- For sparse role families (e.g., `research`, `solutions_field`), a NaN or zero κ usually "
              "means agreement-by-mutual-zero rather than disagreement.")
    md.append("- Self-stability target ≥ 0.85; inter-model macro κ target ≥ 0.7.")
    md.append("- The headline standalone-vs-combined Jaccard is computed model-wise; if any model "
              "shows a meaningful gap, ship two prompts for that axis.")

    OUT.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
