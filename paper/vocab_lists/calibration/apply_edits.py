"""Apply edits from edit_recommendations.json to vocab_lists.json IN PLACE.

Edit categories applied:
  - drops: remove keyword from concept
  - guards classified as 'drop' (suggested_guard contains drop/remove/delete):
      treat as additional drops
  - guards classified as 'guard': append to topic-level `exclusions` list
      with the false-positive pattern and suggested guard text — keyword stays
      in the list (downstream matching layer must honor exclusions)
  - cross-list reconciliation: for each rule, drop its example_keywords from
      every concept in the rule's `alias_in` topics
  - adds: append to concept's keywords list
  - within-concept duplicates: dedupe (case-insensitive)
  - concept_redefines: append `redefine_proposal` field at concept level

NOT auto-applied (left for human judgment):
  - concept-level structural changes (split / merge / restructure)

Updates the top-level vocab_lists.json metadata with edit history.
"""
from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VOCAB = REPO / "paper/vocab_lists/vocab_lists.json"
EDITS = REPO / "paper/vocab_lists/calibration/edit_recommendations.json"


def is_drop_guard(suggested_guard: str) -> bool:
    s = suggested_guard.lower()
    return any(verb in s for verb in (" drop", " remove", " delete", "drop ", "remove ", "delete "))


def main():
    vocab = json.loads(VOCAB.read_text())
    edits = json.loads(EDITS.read_text())

    # Build: per-topic concept lookup (case-insensitive name -> concept dict)
    topic_concept_idx = {}
    for slug, t in vocab["topics"].items():
        idx = {}
        for c in t["core_concepts"]:
            idx[c["name"]] = c
        topic_concept_idx[slug] = idx

    stats = defaultdict(lambda: {"drops": 0, "guards_as_drops": 0, "guards_as_exclusions": 0,
                                  "adds": 0, "redefines": 0, "reconcile_drops": 0,
                                  "dedupe": 0, "missing_concept": 0, "missing_keyword": 0})

    # ---------- 1. DROPS ----------
    for slug, t_edits in edits["topics"].items():
        for d in t_edits["drops"]:
            cname = d.get("concept")
            kw = d.get("keyword")
            if not cname or not kw:
                stats[slug]["missing_keyword"] += 1
                continue
            concept = topic_concept_idx[slug].get(cname)
            if concept is None:
                stats[slug]["missing_concept"] += 1
                continue
            kw_lc = kw.lower().strip()
            new = [k for k in concept["keywords"] if k.lower().strip() != kw_lc]
            if len(new) < len(concept["keywords"]):
                concept["keywords"] = new
                stats[slug]["drops"] += 1
            else:
                stats[slug]["missing_keyword"] += 1

    # ---------- 2. GUARDS ----------
    for slug, t_edits in edits["topics"].items():
        topic = vocab["topics"][slug]
        if "exclusions" not in topic or topic["exclusions"] is None:
            topic["exclusions"] = []
        for g in t_edits["guards"]:
            cname = g.get("concept")
            kw = g.get("keyword")
            sg = g.get("suggested_guard", "")
            if not kw:
                stats[slug]["missing_keyword"] += 1
                continue
            concept = topic_concept_idx[slug].get(cname) if cname else None
            if is_drop_guard(sg):
                if concept is None:
                    stats[slug]["missing_concept"] += 1
                    continue
                kw_lc = kw.lower().strip()
                new = [k for k in concept["keywords"] if k.lower().strip() != kw_lc]
                if len(new) < len(concept["keywords"]):
                    concept["keywords"] = new
                    stats[slug]["guards_as_drops"] += 1
            else:
                topic["exclusions"].append({
                    "pattern": kw,
                    "concept": cname or "",
                    "reason": g.get("false_positive_pattern", ""),
                    "guard": sg,
                })
                stats[slug]["guards_as_exclusions"] += 1

    # ---------- 3. CROSS-LIST RECONCILIATION ----------
    for rule_name, rule in edits["cross_list_reconciliation"].items():
        canonical = rule["proposed_canonical"]
        alias_topics = rule["alias_in"]
        example_kws = [k.lower().strip() for k in rule["example_keywords"]]
        for slug in alias_topics:
            if slug not in vocab["topics"]:
                continue
            for concept in vocab["topics"][slug]["core_concepts"]:
                kept = []
                for k in concept["keywords"]:
                    if k.lower().strip() in example_kws:
                        stats[slug]["reconcile_drops"] += 1
                    else:
                        kept.append(k)
                concept["keywords"] = kept
        # add a note to canonical topic if not already there
        canon_topic = vocab["topics"][canonical]
        notes = canon_topic.get("notes", "") or ""
        marker = f"[{rule_name}]"
        if marker not in notes:
            canon_topic["notes"] = (notes + (" " if notes else "") +
                                    f"{marker} canonical home for: "
                                    f"{', '.join(rule['example_keywords'])}. "
                                    f"Aliased-out from: {', '.join(alias_topics)}.").strip()

    # ---------- 4. ADDS ----------
    # Agents used inconsistent field names. Try suggested_keyword, then keyword,
    # then keyword_or_pattern. Skip if no concept was named (some agents emitted
    # adds without concept assignment — those need human placement).
    for slug, t_edits in edits["topics"].items():
        for a in t_edits["adds"]:
            cname = a.get("concept")
            kw = a.get("suggested_keyword") or a.get("keyword") or a.get("keyword_or_pattern")
            if not cname or not kw:
                stats[slug]["missing_keyword"] += 1
                continue
            concept = topic_concept_idx[slug].get(cname)
            if concept is None:
                stats[slug]["missing_concept"] += 1
                continue
            kw_lc = kw.lower().strip()
            existing = {k.lower().strip() for k in concept["keywords"]}
            if kw_lc not in existing:
                concept["keywords"].append(kw)
                stats[slug]["adds"] += 1

    # ---------- 5. DEDUPE WITHIN CONCEPT (case-insensitive) ----------
    for slug, t in vocab["topics"].items():
        for concept in t["core_concepts"]:
            seen = set()
            new = []
            for k in concept["keywords"]:
                k_lc = k.lower().strip()
                if k_lc in seen:
                    stats[slug]["dedupe"] += 1
                    continue
                seen.add(k_lc)
                new.append(k)
            concept["keywords"] = new

    # ---------- 6. CONCEPT REDEFINE NOTES ----------
    for slug, t_edits in edits["topics"].items():
        for cr in t_edits.get("concept_redefines", []):
            cname = cr.get("concept")
            if not cname:
                continue
            concept = topic_concept_idx[slug].get(cname)
            if concept is None:
                stats[slug]["missing_concept"] += 1
                continue
            concept["redefine_proposal"] = {
                "issue": cr.get("issue", ""),
                "suggested_fix": cr.get("suggested_fix", ""),
                "applied": False,
            }
            stats[slug]["redefines"] += 1

    # ---------- 7. METADATA ----------
    vocab["edit_history"] = vocab.get("edit_history", []) + [{
        "applied_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "source": "paper/vocab_lists/calibration/edit_recommendations.json",
        "summary": dict(stats),
        "totals": {
            "drops": sum(s["drops"] for s in stats.values()),
            "guards_as_drops": sum(s["guards_as_drops"] for s in stats.values()),
            "guards_as_exclusions": sum(s["guards_as_exclusions"] for s in stats.values()),
            "reconcile_drops": sum(s["reconcile_drops"] for s in stats.values()),
            "adds": sum(s["adds"] for s in stats.values()),
            "redefines": sum(s["redefines"] for s in stats.values()),
            "dedupe": sum(s["dedupe"] for s in stats.values()),
        },
    }]
    vocab["generated_at"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    # ---------- WRITE ----------
    VOCAB.write_text(json.dumps(vocab, indent=2, ensure_ascii=False))

    # report
    print(f"Wrote {VOCAB}")
    print()
    print(f"{'topic':28s}  drop  g→drop  g→excl  recon   add  redef dedupe  miss-c  miss-k")
    for slug in sorted(stats):
        s = stats[slug]
        print(f"{slug:28s}  {s['drops']:4d}  {s['guards_as_drops']:6d}  "
              f"{s['guards_as_exclusions']:6d}  {s['reconcile_drops']:5d}  "
              f"{s['adds']:4d}  {s['redefines']:5d}  {s['dedupe']:6d}  "
              f"{s['missing_concept']:6d}  {s['missing_keyword']:6d}")

    # Per-topic concept keyword counts after edits
    print("\nPost-edit keyword counts:")
    for slug, t in vocab["topics"].items():
        n_kw = sum(len(c["keywords"]) for c in t["core_concepts"])
        n_concepts = len(t["core_concepts"])
        empty = sum(1 for c in t["core_concepts"] if not c["keywords"])
        print(f"  {slug:28s}  {n_concepts} concepts  {n_kw:4d} keywords  "
              f"{empty} empty concepts")


if __name__ == "__main__":
    main()
