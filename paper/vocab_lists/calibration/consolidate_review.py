"""Consolidate the 8 per-topic /tmp/vocab_review_*.json files plus collisions.json
into:
  - paper/vocab_lists/calibration/edit_recommendations.json (machine-readable)
  - paper/vocab_lists/calibration/review.md (human-readable narrative)

Idempotent. Re-runnable.
"""
from __future__ import annotations

import datetime
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAL = REPO / "paper/vocab_lists/calibration"
TMP_DIR = Path("/tmp")

SLUGS = [
    "people_management", "orchestration", "verification", "mentorship",
    "performance", "process_scaffolding", "legacy_stack", "context_infrastructure",
]

PRETTY = {
    "people_management": "People-management markers",
    "orchestration": "Orchestration",
    "verification": "Verification",
    "mentorship": "Mentorship markers",
    "performance": "Performance & deep technical understanding",
    "process_scaffolding": "Process-scaffolding markers",
    "legacy_stack": "Legacy-stack markers",
    "context_infrastructure": "Context infrastructure",
}

# Hand-curated cross-list reconciliation rules. Derived from the collision pattern
# (see collisions.json) and from the topic definitions.
CROSS_LIST_RECONCILIATION = {
    "design_artifacts_cluster": {
        "description": "ADR / PRD / design doc / spec doc as written artifacts",
        "example_keywords": ["adr", "architecture decision record", "prd",
                             "product requirements document", "design doc",
                             "design docs", "design documents", "tech spec",
                             "rfc"],
        "found_across": ["orchestration", "process_scaffolding", "context_infrastructure"],
        "proposed_canonical": "context_infrastructure",
        "rationale": "Artifacts are the substrate; orchestration is about authoring the work, process_scaffolding is about governance. The artifact noun lives with the substrate.",
        "alias_in": ["orchestration", "process_scaffolding"],
    },
    "spec_authoring_activity": {
        "description": "Writing / decomposing / authoring specs as an activity",
        "example_keywords": ["write specs", "author specifications",
                             "decompose requirements", "task decomposition",
                             "scope decomposition"],
        "found_across": ["orchestration", "process_scaffolding"],
        "proposed_canonical": "orchestration",
        "rationale": "Authoring activity for AI/agent consumption is the new senior archetype the paper tracks; classical SDLC verbs ('requirements gathering', 'change request') stay in process_scaffolding.",
        "alias_in": ["process_scaffolding"],
    },
    "code_review": {
        "description": "Code review",
        "example_keywords": ["code review", "code reviews", "pr review",
                             "pull request review", "review pull requests"],
        "found_across": ["verification", "mentorship"],
        "proposed_canonical": "verification",
        "rationale": "Code review's primary JD framing is correctness gating. 'Mentor through code review' / 'use reviews to teach' stays in mentorship as a more specific phrase, but bare 'code review' belongs in verification.",
        "alias_in": ["mentorship"],
    },
    "design_review": {
        "description": "Design review",
        "example_keywords": ["design review", "design reviews",
                             "architecture review", "architecture reviews"],
        "found_across": ["verification", "mentorship", "context_infrastructure"],
        "proposed_canonical": "context_infrastructure",
        "rationale": "Design/architecture review is most often framed as a governance/quality artifact-process. Mentorship version captures 'teach through architecture review' — keep there as specific phrase only.",
        "alias_in": ["verification", "mentorship"],
    },
    "observability": {
        "description": "Observability",
        "example_keywords": ["observability", "telemetry", "metrics, logs, traces",
                             "distributed tracing", "monitoring"],
        "found_across": ["verification", "performance", "context_infrastructure"],
        "proposed_canonical": "context_infrastructure",
        "rationale": "Observability as a substrate (dashboards, telemetry hygiene) lives in context_infrastructure. Verification claims 'post-deployment observability for catching regressions' — a specific framing, keep narrowly. Performance claims 'profiling/perf telemetry' — a specific framing, keep narrowly.",
        "alias_in": ["verification", "performance"],
    },
    "leadership_vs_management": {
        "description": "Lead/leadership/principal language",
        "example_keywords": ["technical lead", "tech lead", "lead engineer",
                             "principal engineer", "staff engineer"],
        "found_across": ["mentorship", "people_management"],
        "proposed_canonical": "people_management",
        "rationale": "These are role/seniority titles, not mentorship verbs. Mentorship should keep the verbs ('mentor junior engineers', 'grow the team') and yield the title nouns to people_management — but note that 'lead' is highly polysemous and needs a separate guard.",
        "alias_in": ["mentorship"],
    },
}


def main():
    reviews = {}
    for slug in SLUGS:
        p = TMP_DIR / f"vocab_review_{slug}.json"
        reviews[slug] = json.loads(p.read_text())

    summary_path = CAL / "summary.json"
    summary = json.loads(summary_path.read_text())
    collisions = json.loads((CAL / "collisions.json").read_text())

    total_drops = sum(len(r["drops"]) for r in reviews.values())
    total_guards = sum(len(r["guards"]) for r in reviews.values())
    total_adds = sum(len(r["adds"]) for r in reviews.values())
    total_redefines = sum(len(r.get("concept_redefines", [])) for r in reviews.values())

    edit_recs = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "calibration_sample_size": summary["sample_size"],
        "calibration_strata": summary["strata"],
        "totals": {
            "topics": len(reviews),
            "input_keywords": summary["n_keywords"],
            "drops": total_drops,
            "guards": total_guards,
            "adds": total_adds,
            "concept_redefines": total_redefines,
            "cross_list_collisions": collisions["n_collisions"],
        },
        "topics": {slug: reviews[slug] for slug in SLUGS},
        "cross_list_reconciliation": CROSS_LIST_RECONCILIATION,
    }
    (CAL / "edit_recommendations.json").write_text(
        json.dumps(edit_recs, indent=2, ensure_ascii=False)
    )

    # ----- review.md -----
    md = []
    md.append("# Vocab Lists — Calibration Review\n")
    md.append(f"_Generated {edit_recs['generated_at']}._\n")
    md.append("")
    md.append(f"Calibration sample: **{summary['sample_size']:,} SWE postings** stratified across "
              + ", ".join(f"`{s['source']}/{s['period']}`" for s in summary["strata"])
              + ", matched against `description_core_llm`.\n")
    md.append("")

    # Executive summary
    md.append("## Executive summary\n")
    md.append(f"- Input vocabulary: **{summary['n_keywords']:,} keywords** across "
              f"**{sum(t['n_concepts'] for t in summary['topics'].values())} core concepts** in "
              f"**{len(reviews)} topics**.")
    md.append(f"- Recommended **drops:** {total_drops:,} keywords "
              f"(~{total_drops/summary['n_keywords']*100:.0f}% of input — mostly zero-hit entries).")
    md.append(f"- Recommended **guards:** {total_guards} keywords with high false-positive risk visible in example matches.")
    md.append(f"- Recommended **additions:** {total_adds} new keywords surfaced from corpus inspection.")
    md.append(f"- **Cross-list collisions:** {collisions['n_collisions']} keywords appear in 2+ topics; "
              f"6 reconciliation rules proposed below.")
    md.append("")

    # Per-topic state table
    md.append("### Per-topic state\n")
    md.append("| Topic | Concepts | Keywords | Zero-hit | Drop | Guard | Add | Concept hit-rate range |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for slug in SLUGS:
        ts = summary["topics"][slug]
        rates = [c["rate"] for c in ts["concept_hit_rates"]]
        rate_str = f"{min(rates):.1%} — {max(rates):.1%}" if rates else "n/a"
        r = reviews[slug]
        md.append(f"| **{slug}** | {ts['n_concepts']} | {ts['n_keywords']} | "
                  f"{ts['n_keywords_zero_hits']} | {len(r['drops'])} | "
                  f"{len(r['guards'])} | {len(r['adds'])} | {rate_str} |")
    md.append("")

    # Per-topic narrative
    md.append("## Per-topic findings\n")
    for slug in SLUGS:
        r = reviews[slug]
        ts = summary["topics"][slug]
        md.append(f"### {PRETTY[slug]} (`{slug}`)\n")
        md.append(r["headline_findings"].strip())
        md.append("")
        md.append(f"_{len(r['drops'])} drops · {len(r['guards'])} guards · "
                  f"{len(r['adds'])} adds · {len(r.get('concept_redefines', []))} concept redefines._")
        md.append("")
        # Concept hit rates
        md.append("**Concept hit rates** (% of sampled SWE postings):")
        for c in sorted(ts["concept_hit_rates"], key=lambda x: -x["rate"]):
            md.append(f"- `{c['name']}` — {c['rate']:.1%}")
        md.append("")
        # Top guard examples
        if r["guards"]:
            md.append("**Top guards** (false-positive risks worth fixing):")
            for g in r["guards"][:5]:
                md.append(f"- `{g.get('keyword','?')}` ({g.get('concept','?')}): "
                          f"{g.get('false_positive_pattern','')} → "
                          f"_guard_: {g.get('suggested_guard','')}")
            md.append("")
        # Top adds
        if r["adds"]:
            md.append("**Top suggested additions** (grounded in corpus snippets):")
            for a in r["adds"][:5]:
                kw = a.get("suggested_keyword") or a.get("keyword") or a.get("keyword_or_pattern") or "?"
                md.append(f"- `{kw}` → {a.get('concept','?')}")
            md.append("")
        # Concept redefines
        if r.get("concept_redefines"):
            md.append("**Concept-level recommendations:**")
            for cr in r["concept_redefines"][:5]:
                md.append(f"- `{cr.get('concept','?')}`: {cr.get('issue','')} → "
                          f"_{cr.get('suggested_fix','')}_")
            md.append("")

    # Cross-list reconciliation
    md.append("## Cross-list reconciliation\n")
    md.append("`collisions.json` lists every keyword appearing in ≥2 topics. The patterns below "
              "cluster the most common collisions and propose a canonical home for each.\n")
    # top topic-pair counts
    pair_counts = Counter()
    for entry in collisions["keywords"]:
        topics = sorted({loc["topic"] for loc in entry["locations"]})
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                pair_counts[(topics[i], topics[j])] += 1
    md.append("**Top topic-pair collision counts:**\n")
    md.append("| Topic A | Topic B | # collisions |")
    md.append("|---|---|---:|")
    for (a, b), c in pair_counts.most_common(10):
        md.append(f"| {a} | {b} | {c} |")
    md.append("")

    md.append("**Proposed reconciliation rules:**\n")
    for name, rule in CROSS_LIST_RECONCILIATION.items():
        md.append(f"### `{name}`")
        md.append(f"- **Pattern:** {rule['description']}")
        md.append(f"- **Examples:** " + ", ".join(f"`{k}`" for k in rule["example_keywords"]))
        md.append(f"- **Currently in:** " + ", ".join(rule["found_across"]))
        md.append(f"- **Canonical home:** `{rule['proposed_canonical']}`")
        md.append(f"- **Rationale:** {rule['rationale']}")
        md.append(f"- **Action:** drop these keywords from {', '.join(rule['alias_in'])}, "
                  f"keep in `{rule['proposed_canonical']}`. (Or, for a *specific phrase* "
                  f"that genuinely captures the alias topic — e.g., 'mentor through code review' — keep narrowly.)")
        md.append("")

    # Action plan
    md.append("## Action plan\n")
    md.append("Suggested order, cheapest first.\n")
    md.append("**1. Apply hard drops** (zero-hit keywords). ~830 keywords. Risk: near-zero — "
              "if a keyword fires on 0 of 19,433 SWE postings, it cannot affect headline rates. "
              "Saves regex overhead and reduces noise in the spec.\n")
    md.append("**2. Apply cross-list reconciliation.** Use the 6 rules above to deduplicate. "
              "Update each topic's `notes` field to point to the canonical home of any aliased phrase.\n")
    md.append("**3. Apply guards** to flagged false-positive keywords. The largest contributors "
              "by topic are mentorship (90), process_scaffolding (40), context_infrastructure (32), "
              "legacy_stack (29). For each guard, decide: drop, narrow to specific phrase, or "
              "wrap with negative-lookahead at count time.\n")
    md.append("**4. Apply additions.** 86 candidates, all grounded in corpus snippets. Cheap to apply.\n")
    md.append("**5. Re-run calibration.** `./.venv/bin/python paper/vocab_lists/calibration/run_calibration.py`. "
              "Expected outcome: dramatically smaller per-topic JSONs (zero-hit entries gone), "
              "saturation outliers wrapped, no in-list duplicates.\n")
    md.append("**6. Layer-4 human grounding** (out of agent scope). Hand-label 100–200 postings "
              "per topic on a binary 'does this posting express ⟨topic⟩' rubric and compute "
              "concept-level F1 against the keyword-density labels. This is the alt-test the "
              "paper's appendix already commits to.\n")

    # Files
    md.append("## Files\n")
    md.append("- `vocab_lists.json` — original consolidated vocab (input).")
    md.append("- `calibration/<slug>_calibration.json` — per-topic per-keyword corpus hits and examples.")
    md.append("- `calibration/summary.json` — top-level calibration summary.")
    md.append("- `calibration/collisions.json` — full cross-list collision index.")
    md.append("- `calibration/edit_recommendations.json` — machine-readable consolidated edits (this run).")
    md.append("- `calibration/review.md` — this document.")
    md.append("- `calibration/run_calibration.py` — re-runnable calibration script.")
    md.append("- `calibration/consolidate_review.py` — consolidates per-topic reviews into review.md + edit_recommendations.json.")

    (CAL / "review.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {CAL / 'review.md'} ({sum(len(l) for l in md)//1000} KB)")
    print(f"Wrote {CAL / 'edit_recommendations.json'}")
    print(f"\nTotals: drops={total_drops}, guards={total_guards}, adds={total_adds}, "
          f"redefines={total_redefines}, collisions={collisions['n_collisions']}")


if __name__ == "__main__":
    main()
