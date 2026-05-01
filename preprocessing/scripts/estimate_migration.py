#!/usr/bin/env python3
"""Estimate the LLM-call migration cost from the SWE/SWE-adjacent → swe_combined collapse.

Read-only: does not modify the manifest, cache DB, or any pipeline artifact.

For each scenario we report how many rows the new 2-way frame would select,
how many of those hashes are already in the extraction / classification caches,
and how many would need fresh LLM calls.

Scenarios:
  1. Sticky next-run (preserves existing manifest, tops up under new seeding)
  2. Clean rebuild (selects from scratch under the new analysis_group seeds)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = PROJECT_ROOT / "preprocessing" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import stage9_llm_prefilter as stage9  # noqa: E402
import llm_shared  # noqa: E402

INPUT_PATH = PROJECT_ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet"
MANIFEST_PATH = PROJECT_ROOT / "preprocessing" / "intermediate" / "stage9_core_frame_manifest.json"
CACHE_DB = PROJECT_ROOT / "preprocessing" / "cache" / "llm_responses.db"


def cached_hashes(task_name: str) -> set[str]:
    con = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT DISTINCT input_hash FROM responses WHERE task_name = ? AND prompt_version = ?",
        (task_name, _prompt_version_for(task_name)),
    ).fetchall()
    con.close()
    return {h for (h,) in rows}


def _prompt_version_for(task_name: str) -> str:
    if task_name == llm_shared.EXTRACTION_TASK_NAME:
        return llm_shared.EXTRACTION_PROMPT_VERSION
    if task_name == llm_shared.CLASSIFICATION_TASK_NAME:
        return llm_shared.CLASSIFICATION_PROMPT_VERSION
    raise ValueError(task_name)


def report_frame(label: str, selected_rows: list[dict], extraction_cache: set[str]) -> dict:
    n = len(selected_rows)
    by_group: dict[str, int] = {}
    cached_by_group: dict[str, int] = {}
    for row in selected_rows:
        g = row.get("analysis_group", "unknown")
        by_group[g] = by_group.get(g, 0) + 1
        if str(row["extraction_input_hash"]) in extraction_cache:
            cached_by_group[g] = cached_by_group.get(g, 0) + 1

    deferred_total = n - sum(cached_by_group.values())
    print(f"\n=== {label} ===")
    print(f"Selected rows: {n:,}")
    for group in sorted(by_group):
        cached = cached_by_group.get(group, 0)
        total = by_group[group]
        deferred = total - cached
        print(
            f"  {group:<14} total={total:>7,}  cached={cached:>7,}  deferred={deferred:>7,}  "
            f"cache_hit_rate={cached / total * 100:5.1f}%"
        )
    print(
        f"  TOTAL DEFERRED: {deferred_total:,} "
        f"(fresh extraction calls needed for this frame)"
    )
    return {
        "selected": n,
        "cached": sum(cached_by_group.values()),
        "deferred": deferred_total,
        "by_group": by_group,
        "cached_by_group": cached_by_group,
    }


def main() -> None:
    print("Stage 9 migration estimation")
    print("=" * 70)
    print(f"Input: {INPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Cache DB: {CACHE_DB}")

    t0 = time.time()
    print("\nLoading extraction + classification cache hash sets…")
    extraction_cache = cached_hashes(llm_shared.EXTRACTION_TASK_NAME)
    classification_cache = cached_hashes(llm_shared.CLASSIFICATION_TASK_NAME)
    print(f"  extraction cache hashes: {len(extraction_cache):,}")
    print(f"  classification cache hashes: {len(classification_cache):,}")

    manifest = llm_shared.load_core_frame_manifest(MANIFEST_PATH)
    selection_target = len(manifest.get("selected_hashes", []))
    print(f"\nExisting manifest size (selection_target): {selection_target:,}")

    print("\nBuilding candidate records (this scans the 7GB stage8 parquet)…")
    candidates = stage9.build_candidate_records(INPUT_PATH)
    print(f"  in-frame eligible candidates (LinkedIn × English × ≥15w × analysis_group): {len(candidates):,}")
    by_group = {}
    for row in candidates:
        g = row.get("analysis_group", "unknown")
        by_group[g] = by_group.get(g, 0) + 1
    for group, n in sorted(by_group.items()):
        print(f"    {group}: {n:,}")

    # Scenario 1: sticky (no reset) — what next normal run would do
    sticky_selected, sticky_summary = llm_shared.select_sticky_task_frame(
        candidates,
        selection_target=selection_target,
        hash_key="extraction_input_hash",
        manifest_path=MANIFEST_PATH,
        groups=llm_shared.ANALYSIS_GROUP_PRIORITY,
        reset=False,
    )
    print(
        f"\nSticky frame summary: retained={sticky_summary.get('retained_count', 0):,} "
        f"top_up={sticky_summary.get('top_up_count', 0):,} selected={sticky_summary.get('selected_count', 0):,}"
    )
    sticky_report = report_frame(
        "Scenario 1 — Sticky next-run (Stage 9 extraction)",
        sticky_selected,
        extraction_cache,
    )
    # Stage 10 classification needs the classification_input_hash, which
    # depends on description_core_llm (the LLM-cleaned text). For rows that
    # were previously routed, we read it from stage10_llm_integrated.parquet.
    print("\nReading Stage 10 classification hashes from stage10_llm_integrated.parquet…")
    import duckdb

    integrated_path = (
        PROJECT_ROOT / "preprocessing" / "intermediate" / "stage10_llm_integrated.parquet"
    )
    integrated_rows = duckdb.execute(
        f"""
        SELECT extraction_input_hash, classification_input_hash, llm_classification_coverage
        FROM read_parquet('{integrated_path}')
        WHERE selected_for_llm_frame = TRUE
        """
    ).fetchall()
    integrated_lookup = {
        str(r[0]): {"classification_input_hash": str(r[1]) if r[1] else None,
                    "coverage": r[2]}
        for r in integrated_rows
    }
    print(f"  rows previously in frame: {len(integrated_lookup):,}")

    # Sticky Stage 10: for rows whose extraction hash is in the prior frame,
    # we know the classification hash. For rows newly entering, we'd need to
    # run extraction first.
    sticky_known = sum(
        1 for row in sticky_selected
        if str(row["extraction_input_hash"]) in integrated_lookup
    )
    sticky_class_cached = sum(
        1 for row in sticky_selected
        if (entry := integrated_lookup.get(str(row["extraction_input_hash"])))
        and entry["classification_input_hash"] in classification_cache
    )
    sticky_new = len(sticky_selected) - sticky_known
    print("\n=== Scenario 1 — Sticky next-run (Stage 10 classification) ===")
    print(f"  rows reusable (prior classification hash known): {sticky_known:,}")
    print(f"    of those, classification cache hit: {sticky_class_cached:,}")
    print(f"    of those, classification deferred: {sticky_known - sticky_class_cached:,}")
    print(f"  rows newly entering (no prior hash): {sticky_new:,} — would need fresh extraction + classification")

    # Scenario 2: reset (clean rebuild)
    reset_selected, reset_summary = llm_shared.select_task_frame(
        candidates,
        selection_target=selection_target,
        hash_key="extraction_input_hash",
        groups=llm_shared.ANALYSIS_GROUP_PRIORITY,
    )
    print(f"\nReset frame summary: selected={reset_summary['selected_count']:,}")
    report_frame(
        "Scenario 2 — Clean rebuild (Stage 9 extraction)",
        reset_selected,
        extraction_cache,
    )
    reset_known = sum(
        1 for row in reset_selected
        if str(row["extraction_input_hash"]) in integrated_lookup
    )
    reset_class_cached = sum(
        1 for row in reset_selected
        if (entry := integrated_lookup.get(str(row["extraction_input_hash"])))
        and entry["classification_input_hash"] in classification_cache
    )
    reset_new = len(reset_selected) - reset_known
    print("\n=== Scenario 2 — Clean rebuild (Stage 10 classification) ===")
    print(f"  rows reusable (prior classification hash known): {reset_known:,}")
    print(f"    of those, classification cache hit: {reset_class_cached:,}")
    print(f"    of those, classification deferred: {reset_known - reset_class_cached:,}")
    print(f"  rows newly entering (no prior hash): {reset_new:,} — would need fresh extraction + classification")

    # Overlap: how many sticky-frame hashes match the existing manifest?
    sticky_hashes = {str(row["extraction_input_hash"]) for row in sticky_selected}
    reset_hashes = {str(row["extraction_input_hash"]) for row in reset_selected}
    old_manifest_hashes = set(manifest.get("selected_hashes", []))
    print("\n=== Hash-set overlap with existing manifest ===")
    print(f"  sticky ∩ old: {len(sticky_hashes & old_manifest_hashes):,}  "
          f"(rows kept from existing frame)")
    print(f"  sticky \\ old: {len(sticky_hashes - old_manifest_hashes):,}  "
          f"(new top-ups under the 2-way seeding)")
    print(f"  reset  ∩ old: {len(reset_hashes & old_manifest_hashes):,}  "
          f"(rows that survive a clean rebuild)")
    print(f"  reset  \\ old: {len(reset_hashes - old_manifest_hashes):,}  "
          f"(rows newly entering the frame)")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
