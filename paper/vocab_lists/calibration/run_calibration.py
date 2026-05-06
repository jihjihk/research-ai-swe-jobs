"""Run keyword-density calibration for paper/vocab_lists/vocab_lists.json.

For each keyword in every (topic, concept):
- corpus hit rate (% of sampled postings with >=1 occurrence)
- total occurrences
- up to N example postings with the matched sentence

Plus per-concept aggregate stats and a cross-list collision index.

Sample: stratified by (source, period) over SWE rows in unified_core.parquet,
matching against description_core_llm (fallback: description). One JSON per topic
under paper/vocab_lists/calibration/<slug>_calibration.json. Re-runnable.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

# ---------- config ----------
REPO = Path(__file__).resolve().parents[3]
VOCAB = REPO / "paper/vocab_lists/vocab_lists.json"
OUT_DIR = REPO / "paper/vocab_lists/calibration"
PARQUET = REPO / "data/unified_core.parquet"

SAMPLE_PER_BUCKET = 7000  # ~21k total across 3 source-period strata; capped by avail
EXAMPLES_PER_KEYWORD = 5
EXAMPLE_CONTEXT_CHARS = 180  # window around match
RNG = random.Random(20260505)

# strata: (source, period) — pull SWE rows from each
STRATA = [
    ("kaggle_arshkon", "2024-04"),
    ("kaggle_asaniczka", "2024-01"),
    ("scraped", "2026-04"),  # most recent month of scrape
]

WORD_LIKE = re.compile(r"^[A-Za-z0-9 \-']+$")


def compile_pattern(kw: str) -> re.Pattern:
    """Word-boundary if alphanumeric/space/hyphen; otherwise substring with edge guards."""
    kw_lc = kw.lower()
    if WORD_LIKE.match(kw_lc):
        return re.compile(rf"\b{re.escape(kw_lc)}\b", re.IGNORECASE)
    # non-word-char keywords: .NET, C++, 1:1, AS/400, etc. Use lookarounds requiring
    # non-letter on both sides where the keyword starts/ends with a letter/digit.
    starts_word = kw_lc[0].isalnum()
    ends_word = kw_lc[-1].isalnum()
    pre = r"(?<![A-Za-z0-9])" if starts_word else r""
    post = r"(?![A-Za-z0-9])" if ends_word else r""
    return re.compile(pre + re.escape(kw_lc) + post, re.IGNORECASE)


def find_examples(desc: str, pat: re.Pattern, n: int) -> list[str]:
    out = []
    for m in pat.finditer(desc):
        s = max(0, m.start() - EXAMPLE_CONTEXT_CHARS // 2)
        e = min(len(desc), m.end() + EXAMPLE_CONTEXT_CHARS // 2)
        snippet = desc[s:e].replace("\n", " ").strip()
        out.append(snippet)
        if len(out) >= n:
            break
    return out


def load_sample(con) -> list[dict]:
    rows = []
    for src, period in STRATA:
        q = f"""
        SELECT uid, source, period, title, description_core_llm, description
        FROM read_parquet('{PARQUET}')
        WHERE source = ? AND period = ? AND is_swe = TRUE
        """
        df = con.execute(q, [src, period]).fetchdf()
        n_avail = len(df)
        n_take = min(SAMPLE_PER_BUCKET, n_avail)
        if n_take < n_avail:
            df = df.sample(n=n_take, random_state=20260505)
        rows.extend(df.to_dict("records"))
        print(f"  stratum {src}/{period}: {n_take:,} of {n_avail:,} available", flush=True)
    return rows


def main():
    print("Loading vocab...", flush=True)
    vocab = json.loads(VOCAB.read_text())
    topics = vocab["topics"]

    print("Loading sample from unified_core.parquet...", flush=True)
    con = duckdb.connect()
    sample = load_sample(con)
    n_sample = len(sample)
    print(f"  sample size: {n_sample:,}", flush=True)

    # Pre-lowercase descriptions, prefer description_core_llm
    docs = []
    for r in sample:
        text = r.get("description_core_llm") or r.get("description") or ""
        docs.append({
            "uid": r["uid"],
            "source": r["source"],
            "period": r["period"],
            "title": r.get("title") or "",
            "text": text,
            "text_lc": text.lower(),
        })

    # Build collision index (keyword -> [(topic, concept)])
    keyword_to_locs = defaultdict(list)
    for slug, topic in topics.items():
        for ci, concept in enumerate(topic["core_concepts"]):
            for kw in concept["keywords"]:
                kw_norm = kw.lower().strip()
                keyword_to_locs[kw_norm].append((slug, concept["name"]))
    collisions = {kw: locs for kw, locs in keyword_to_locs.items() if len(locs) > 1}
    print(f"Cross-list collisions: {len(collisions)} keywords appear in 2+ concepts", flush=True)

    # Run matching per topic
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_summary = {
        "sample_size": n_sample,
        "strata": [{"source": s, "period": p} for s, p in STRATA],
        "n_keywords": sum(len(c["keywords"]) for t in topics.values() for c in t["core_concepts"]),
        "n_collisions": len(collisions),
        "topics": {},
    }

    for slug, topic in topics.items():
        print(f"\n[{slug}] {len(topic['core_concepts'])} concepts...", flush=True)
        per_concept = []
        # track posting-level concept hits (for concept hit rate)
        for concept in topic["core_concepts"]:
            kw_results = []
            posting_hit_set = set()  # uids matching any keyword in concept
            for kw in concept["keywords"]:
                pat = compile_pattern(kw)
                postings_matched = 0
                total_occurrences = 0
                examples = []
                # fast prefilter: substring `in` check on lowercase before regex
                kw_lc = kw.lower()
                for d in docs:
                    if kw_lc not in d["text_lc"]:
                        continue
                    matches = pat.findall(d["text"])
                    if not matches:
                        continue  # substring matched but boundary didn't
                    postings_matched += 1
                    total_occurrences += len(matches)
                    posting_hit_set.add(d["uid"])
                    if len(examples) < EXAMPLES_PER_KEYWORD:
                        ex = find_examples(d["text"], pat, EXAMPLES_PER_KEYWORD - len(examples))
                        for sn in ex:
                            examples.append({
                                "uid": d["uid"],
                                "source": d["source"],
                                "period": d["period"],
                                "title": d["title"][:120],
                                "snippet": sn,
                            })
                            if len(examples) >= EXAMPLES_PER_KEYWORD:
                                break
                kw_results.append({
                    "keyword": kw,
                    "postings_hit": postings_matched,
                    "hit_rate": round(postings_matched / n_sample, 5),
                    "total_occurrences": total_occurrences,
                    "examples": examples,
                    "cross_list_collisions": [
                        {"topic": t, "concept": c}
                        for (t, c) in collisions.get(kw.lower().strip(), [])
                        if not (t == slug and c == concept["name"])
                    ],
                })
            per_concept.append({
                "name": concept["name"],
                "definition": concept["definition"],
                "concept_postings_hit": len(posting_hit_set),
                "concept_hit_rate": round(len(posting_hit_set) / n_sample, 5),
                "n_keywords": len(concept["keywords"]),
                "n_keywords_with_zero_hits": sum(1 for k in kw_results if k["postings_hit"] == 0),
                "n_keywords_dominant": sum(1 for k in kw_results if k["hit_rate"] > 0.4),
                "keywords": kw_results,
            })
        topic_total_hits = sum(c["concept_postings_hit"] for c in per_concept)
        out_path = OUT_DIR / f"{slug}_calibration.json"
        out_path.write_text(json.dumps({
            "topic_slug": slug,
            "topic_name": topic["topic"],
            "topic_definition": topic["definition"],
            "sample_size": n_sample,
            "concepts": per_concept,
            "topic_exclusions": topics[slug].get("exclusions", []),
            "topic_calibration_recommendations": topics[slug].get("calibration_recommendations", ""),
            "topic_notes": topics[slug].get("notes", ""),
        }, indent=2, ensure_ascii=False))
        n_zero = sum(c["n_keywords_with_zero_hits"] for c in per_concept)
        n_kw = sum(c["n_keywords"] for c in per_concept)
        print(f"  {n_kw} keywords, {n_zero} with 0 hits, written to {out_path.name}", flush=True)
        overall_summary["topics"][slug] = {
            "n_concepts": len(per_concept),
            "n_keywords": n_kw,
            "n_keywords_zero_hits": n_zero,
            "n_keywords_above_40pct": sum(c["n_keywords_dominant"] for c in per_concept),
            "concept_hit_rates": [
                {"name": c["name"], "rate": c["concept_hit_rate"]} for c in per_concept
            ],
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(overall_summary, indent=2))
    (OUT_DIR / "collisions.json").write_text(json.dumps({
        "n_collisions": len(collisions),
        "keywords": [
            {"keyword": kw, "locations": [{"topic": t, "concept": c} for t, c in locs]}
            for kw, locs in sorted(collisions.items())
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\nDone. Per-topic files + summary.json + collisions.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
