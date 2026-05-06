"""Produce bias-free review-input files for the v2 review round.

Strips fields that reflect prior reviews and could bias the new agents:
  - topic_exclusions (was populated from v1 guard recommendations)
  - topic_notes (was populated from cross-list reconciliation)
  - topic_calibration_recommendations (carry-over)
  - concept-level redefine_proposal (added by apply_edits.py from v1)

Output: paper/vocab_lists/calibration/review_input_<slug>.json
Each contains only: topic_slug, topic_name, topic_definition, sample_size,
strata, and the concepts array with each concept's name/definition/hit-rate
data and keyword-level data (keyword, hit_rate, total_occurrences, examples).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CAL = REPO / "paper/vocab_lists/calibration"

SLUGS = [
    "people_management", "orchestration", "verification", "mentorship",
    "performance", "process_scaffolding", "legacy_stack", "context_infrastructure",
]


def main():
    summary = json.loads((CAL / "summary.json").read_text())
    for slug in SLUGS:
        cal = json.loads((CAL / f"{slug}_calibration.json").read_text())
        clean = {
            "topic_slug": cal["topic_slug"],
            "topic_name": cal["topic_name"],
            "topic_definition": cal.get("topic_definition", ""),
            "sample_size": cal["sample_size"],
            "strata": summary["strata"],
            "concepts": [
                {
                    "name": c["name"],
                    "definition": c["definition"],
                    "concept_postings_hit": c["concept_postings_hit"],
                    "concept_hit_rate": c["concept_hit_rate"],
                    "n_keywords": c["n_keywords"],
                    "n_keywords_with_zero_hits": c["n_keywords_with_zero_hits"],
                    "n_keywords_dominant": c["n_keywords_dominant"],
                    "keywords": [
                        {
                            "keyword": k["keyword"],
                            "postings_hit": k["postings_hit"],
                            "hit_rate": k["hit_rate"],
                            "total_occurrences": k["total_occurrences"],
                            "examples": k["examples"],
                            # cross_list_collisions intentionally retained — useful
                            # signal for the reviewer's own judgment
                            "cross_list_collisions": k.get("cross_list_collisions", []),
                        }
                        for k in c["keywords"]
                    ],
                }
                for c in cal["concepts"]
            ],
        }
        out = CAL / f"review_input_{slug}.json"
        out.write_text(json.dumps(clean, indent=2, ensure_ascii=False))
        print(f"Wrote {out.name}: {sum(len(c['keywords']) for c in clean['concepts'])} keywords across "
              f"{len(clean['concepts'])} concepts")


if __name__ == "__main__":
    main()
