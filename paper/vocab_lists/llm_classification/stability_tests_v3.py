"""V3 stability tests — order-shuffle and prompt-paraphrase robustness on the chosen tier.

Runs ONE variant (default: combined) on the chosen tier (default: gpt-5.4-mini)
under 4 order-variant configurations and 2 paraphrased system prompts. Then computes
Jaccard vs the v3 pilot reference (3-rep majority of the same model under the same variant).

Outputs:
  - stability_v3_results.jsonl
  - stability_report_v3.md
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
SAMPLE = DIR / "pilot_sample_v3.parquet"
PILOT_RESULTS = DIR / "pilot_v3_results.jsonl"
OUT_JSONL = DIR / "stability_v3_results.jsonl"
OUT_MD = DIR / "stability_report_v3.md"

sys.path.insert(0, str(Path(__file__).parent))
from classifier_v3 import (  # noqa: E402
    classify_v3, SKILL_LABELS, ROLE_FAMILY_LABELS, VARIANT_PROMPTS,
)


# ---- Order variants ----
def _shuf(labels, seed):
    r = random.Random(seed)
    s = list(labels)
    r.shuffle(s)
    return tuple(s)

SKILL_ORDER_VARIANTS = {
    "alphabetical": tuple(sorted(SKILL_LABELS)),
    "reverse": tuple(sorted(SKILL_LABELS, reverse=True)),
    "shuffled_seed_7": _shuf(SKILL_LABELS, 7),
    "shuffled_seed_42": _shuf(SKILL_LABELS, 42),
}
ROLE_ORDER_VARIANTS = {
    "alphabetical": tuple(sorted(ROLE_FAMILY_LABELS)),
    "reverse": tuple(sorted(ROLE_FAMILY_LABELS, reverse=True)),
    "shuffled_seed_7": _shuf(ROLE_FAMILY_LABELS, 7),
    "shuffled_seed_42": _shuf(ROLE_FAMILY_LABELS, 42),
}


# ---- Paraphrased COMBINED system prompts (for combined variant only) ----
PARAPHRASE_TERSE_COMBINED = """Classify a SWE job posting on two axes — skill themes (8) and role families (17). Both multi-label. Tag only what is explicitly named as a duty, requirement, or skill. Skip boilerplate.

Skill themes:
- people_management — direct reports, perf reviews, hiring/firing, headcount, 1:1s. NOT bare "tech lead".
- orchestration — authoring specs/ADRs/RFCs/design docs; decomposing work for engineers or AI agents; multi-agent or context engineering. NOT design patterns as required knowledge.
- verification — CI/CD, named test frameworks, code-review processes, evals, observability for regressions, compliance/audit, static analysis. NOT generic "writes tests".
- mentorship — mentoring/teaching/growing/onboarding other engineers; pair programming. NOT managing a team.
- performance — profiling, latency, throughput, kernel/network internals, low-level optimization, "deep understanding of [a technical area]". NOT recruiter fluff.
- process_scaffolding — agile, scrum, sprints, requirements engineering, V&V, project coordination, SDLC governance.
- legacy_stack — old-paradigm enterprise frameworks regardless of version: Java/Java EE, .NET/.NET Framework, ASP.NET, COBOL, mainframe, VMware, AD, BizTalk, ColdFusion.
- context_infrastructure — runbooks, ADRs, RFCs, dashboards/SLOs, telemetry hygiene, schema docs, technical writing. NOT generic "cross-functional" / "communication skills".

Role families:
- software_engineer_general — residual fallback only; never alongside another family.
- frontend_web — UI/UX, web/mobile front-end. Full-stack defaults here.
- backend_api — server-side, APIs, services, DBs.
- mobile — native or cross-platform mobile apps.
- embedded — firmware, IoT, hardware-near.
- data_engineer — data pipelines, warehouses, ingestion infra.
- ml_engineer — pre-LLM ML pipelines, training, MLOps.
- ai_llm_engineer — LLM/agent/RAG/foundation-model work.
- devops_sre_platform — CI/CD, IaC, reliability, platform.
- security — AppSec, InfoSec, cyber.
- qa_test — QA/SDET/Test Automation as the primary role.
- solutions_field — Solutions Engineer, Forward-Deployed, Customer Engineer.
- legacy_specialist — primary purpose is maintaining a legacy stack.
- data_analytics — Data Scientist/Analyst/BI; not production ML.
- research — Research Scientist, Applied Scientist, Research Engineer.
- infra_ops_admin — DBA, sysadmin, network, cloud admin.
- people_manager — Engineering Manager and similar (excludes director/VP/C-level).

Output JSON with two arrays: skill_themes and role_families. Empty arrays mean none. No commentary."""

PARAPHRASE_ACTIVE_VOICE_COMBINED = """Read the SWE job posting below and decide which of the following themes apply. Tag only what the posting explicitly states as a stated responsibility, listed requirement, or named skill. Treat boilerplate, passing mentions, and company-description language as no evidence.

The skill themes (any subset apply):
- people_management: the role grants formal people-management authority — direct reports, performance reviews, hiring/firing, headcount, 1:1 cadence.
- orchestration: the role authors specs, ADRs, RFCs, or design documents; decomposes work for other engineers or AI agents to execute; orchestrates multi-agent or context-engineering systems. Don't tag for design patterns mentioned as required knowledge or for "architecture experience" without authoring.
- verification: the role names CI/CD pipelines, test frameworks, code-review processes, evals, observability for regressions, compliance/audit, or static analysis. Don't tag for generic "writes tests".
- mentorship: the role mentors, teaches, grows, or onboards other engineers; pair programming; knowledge transfer. Distinct from managing a team.
- performance: the role names low-level performance work — profiling, latency, throughput, kernel/network internals, low-level optimization — or demands deep technical depth in a specific area. Don't tag for "high-performing team" or "expert in [tool]" recruiter fluff.
- process_scaffolding: agile, scrum, sprints, requirements engineering, V&V, project coordination, or SDLC governance.
- legacy_stack: requires experience with old-paradigm enterprise frameworks regardless of version — Java / Java EE, .NET / .NET Framework, ASP.NET, COBOL, mainframe, VMware / vSphere, Active Directory, BizTalk, ColdFusion, similar.
- context_infrastructure: authoring or maintaining runbooks, ADRs, RFCs, dashboards / SLOs, telemetry, schema documentation, or technical writing as a substrate. Don't tag for generic "cross-functional" or "communication".

The role families (any subset apply, multi-label allowed):
- software_engineer_general: residual fallback only — never alongside another family.
- frontend_web: user-facing web/mobile UI/UX. Full-stack defaults here.
- backend_api: server-side, APIs, databases, services.
- mobile: native or cross-platform mobile apps.
- embedded: firmware, IoT, hardware-near software.
- data_engineer: pipelines, warehouses, ingestion infra.
- ml_engineer: pre-LLM ML pipelines, training, MLOps.
- ai_llm_engineer: LLM, agent, RAG, foundation-model work.
- devops_sre_platform: CI/CD, IaC, reliability, platform.
- security: application or infrastructure security.
- qa_test: software quality, testing, validation as the role's primary focus.
- solutions_field: deploys/integrates with customers — Solutions Engineer, Forward-Deployed, Field Engineer.
- legacy_specialist: role's primary purpose is maintaining a legacy stack.
- data_analytics: analytical or decision-support outputs (Data Scientist, Data Analyst, Analytics Engineer); not production ML.
- research: applied or basic research bridging academia and product.
- infra_ops_admin: operates / administrates DBs, servers, networks rather than authoring software.
- people_manager: engineering management focused on people, hiring, team operations. Excludes director/VP/C-level.

Output a JSON object with two arrays — `skill_themes` and `role_families`. Empty arrays mean none. No commentary."""

PARAPHRASE_VARIANTS_COMBINED = {
    "original": VARIANT_PROMPTS["combined"],
    "terse": PARAPHRASE_TERSE_COMBINED,
    "active_voice": PARAPHRASE_ACTIVE_VOICE_COMBINED,
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


def load_pilot_reference(model: str, variant: str):
    rows = [json.loads(l) for l in PILOT_RESULTS.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[(df["model"] == model) & (df["variant"] == variant)]
    skill_ref = {}
    role_ref = {}
    for uid, g in df.groupby("uid"):
        skill_ref[uid] = majority_labels([list(x) for x in g["skill_themes"].tolist()])
        role_ref[uid] = majority_labels([list(x) for x in g["role_families"].tolist()])
    return skill_ref, role_ref


def run_one_variant(*, sample, model: str, variant_name: str, system_prompt: str,
                    skill_order, role_order, workers: int):
    plan = list(sample.itertuples(index=False))

    def worker(row):
        result = classify_v3(
            variant="combined",  # always combined for stability tests
            title=row.title or "",
            description=row.description_core_llm or "",
            model=model,
            system_prompt_override=system_prompt,
            skill_order_override=skill_order,
            role_order_override=role_order,
        )
        return {
            "uid": row.uid,
            "stratum_v3": row.stratum_v3,
            "variant_test": variant_name,
            "model": model,
            "skill_themes": result.skill_themes,
            "role_families": result.role_families,
            "parsed_ok": result.parsed_ok,
            "latency_s": result.latency_s,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
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
    print(f"V3 stability tests on `{args.model}`, combined variant, {len(sample)} postings.")

    if OUT_JSONL.exists():
        OUT_JSONL.unlink()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    fh = OUT_JSONL.open("a", encoding="utf-8")

    # ORDER variants — vary skill order (role default), role order (skill default), both
    print("\n--- ORDER variants ---", flush=True)
    base_skill = SKILL_ORDER_VARIANTS["alphabetical"]
    base_role = ROLE_ORDER_VARIANTS["alphabetical"]
    for sk_name, sk_order in SKILL_ORDER_VARIANTS.items():
        if sk_name == "alphabetical":
            continue
        vname = f"order:skill_{sk_name}"
        print(f"  {vname} ...", flush=True)
        recs = run_one_variant(sample=sample, model=args.model, variant_name=vname,
                                system_prompt=VARIANT_PROMPTS["combined"],
                                skill_order=sk_order, role_order=base_role,
                                workers=args.max_workers)
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
    for ro_name, ro_order in ROLE_ORDER_VARIANTS.items():
        if ro_name == "alphabetical":
            continue
        vname = f"order:role_{ro_name}"
        print(f"  {vname} ...", flush=True)
        recs = run_one_variant(sample=sample, model=args.model, variant_name=vname,
                                system_prompt=VARIANT_PROMPTS["combined"],
                                skill_order=base_skill, role_order=ro_order,
                                workers=args.max_workers)
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
    # Both axes shuffled together
    vname = "order:both_shuffled_seed_7"
    print(f"  {vname} ...", flush=True)
    recs = run_one_variant(sample=sample, model=args.model, variant_name=vname,
                            system_prompt=VARIANT_PROMPTS["combined"],
                            skill_order=SKILL_ORDER_VARIANTS["shuffled_seed_7"],
                            role_order=ROLE_ORDER_VARIANTS["shuffled_seed_7"],
                            workers=args.max_workers)
    for r in recs:
        fh.write(json.dumps(r) + "\n")
    fh.flush()

    # PARAPHRASE variants
    print("\n--- PARAPHRASE variants ---", flush=True)
    for pname, sp in PARAPHRASE_VARIANTS_COMBINED.items():
        if pname == "original":
            continue
        vname = f"paraphrase:{pname}"
        print(f"  {vname} ...", flush=True)
        recs = run_one_variant(sample=sample, model=args.model, variant_name=vname,
                                system_prompt=sp,
                                skill_order=base_skill, role_order=base_role,
                                workers=args.max_workers)
        for r in recs:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
    fh.close()

    # ---- Analysis ----
    print(f"\nLoading reference from v3 pilot ({args.model}, combined)...")
    skill_ref, role_ref = load_pilot_reference(args.model, "combined")

    rows = [json.loads(l) for l in OUT_JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} stability calls.")

    md = []
    md.append(f"# V3 stability tests ({args.model}, combined variant)\n")
    md.append(f"_Variants tested against pilot reference (3-rep majority of `{args.model}` on `combined`)._\n")

    md.append("## Order-shuffle variants\n")
    md.append("| Variant | Skill axis Jaccard | Role axis Jaccard | Min skill | Min role |")
    md.append("|---|---:|---:|---:|---:|")
    for vname in df["variant_test"].unique():
        if not vname.startswith("order:"):
            continue
        sub = df[df["variant_test"] == vname]
        s_jacs, r_jacs = [], []
        for _, r in sub.iterrows():
            s_jacs.append(jaccard(skill_ref.get(r["uid"], set()), set(r["skill_themes"])))
            r_jacs.append(jaccard(role_ref.get(r["uid"], set()), set(r["role_families"])))
        s_mean = sum(s_jacs)/len(s_jacs) if s_jacs else 0
        r_mean = sum(r_jacs)/len(r_jacs) if r_jacs else 0
        md.append(f"| `{vname}` | {s_mean:.3f} | {r_mean:.3f} | {min(s_jacs):.3f} | {min(r_jacs):.3f} |")
    md.append("")

    md.append("## Paraphrase variants\n")
    md.append("| Variant | Skill axis Jaccard | Role axis Jaccard | Min skill | Min role |")
    md.append("|---|---:|---:|---:|---:|")
    for vname in df["variant_test"].unique():
        if not vname.startswith("paraphrase:"):
            continue
        sub = df[df["variant_test"] == vname]
        s_jacs, r_jacs = [], []
        for _, r in sub.iterrows():
            s_jacs.append(jaccard(skill_ref.get(r["uid"], set()), set(r["skill_themes"])))
            r_jacs.append(jaccard(role_ref.get(r["uid"], set()), set(r["role_families"])))
        s_mean = sum(s_jacs)/len(s_jacs) if s_jacs else 0
        r_mean = sum(r_jacs)/len(r_jacs) if r_jacs else 0
        md.append(f"| `{vname}` | {s_mean:.3f} | {r_mean:.3f} | {min(s_jacs):.3f} | {min(r_jacs):.3f} |")
    md.append("")

    md.append("## Notes\n")
    md.append("- The reference is the 3-rep majority of the same model+variant from the v3 pilot.")
    md.append("- Each stability variant is one rep, so Jaccard is bounded above by the model's "
              "single-rep-vs-majority noise floor (typically ~0.85-0.90 on `mini`).")
    md.append("- Order changes that don't degrade beyond the noise floor → prompt is robust to ordering.")
    md.append("- Paraphrase changes test surface-phrasing robustness.")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
