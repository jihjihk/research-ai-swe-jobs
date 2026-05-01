"""T09 Step 12: Save the shared archetype labels artifact.

Writes: exploration/artifacts/shared/swe_archetype_labels.parquet
  Columns: uid, archetype_id, archetype_name

Verifies:
 - One row per uid, no nulls
 - All uids present in shared cleaned text artifact
"""
import numpy as np
import pandas as pd
import duckdb

OUTDIR = "exploration/artifacts/T09"
SHARED = "exploration/artifacts/shared"


def main():
    sample = pd.read_parquet(f"{OUTDIR}/sample_with_assignments.parquet")
    names = pd.read_csv(f"{OUTDIR}/archetype_names.csv")
    name_map = dict(zip(names.archetype_id, names.archetype_name))
    sample["archetype_name"] = sample["topic_reduced"].map(name_map)

    artifact = sample[["uid", "topic_reduced"]].rename(columns={"topic_reduced": "archetype_id"}).copy()
    artifact["archetype_name"] = sample["archetype_name"].values
    print(f"Artifact rows: {len(artifact)}")
    print(f"Null archetype_name: {artifact.archetype_name.isna().sum()}")
    print(f"Null archetype_id: {artifact.archetype_id.isna().sum()}")
    print(f"Unique uids: {artifact.uid.nunique()}")

    # Verify uids are in shared cleaned text
    con = duckdb.connect()
    shared_uids = set(con.execute(
        "SELECT uid FROM 'exploration/artifacts/shared/swe_cleaned_text.parquet'"
    ).fetchdf().uid.tolist())
    missing = set(artifact.uid) - shared_uids
    print(f"uids missing from shared cleaned text: {len(missing)}")

    # Write
    artifact_path = f"{SHARED}/swe_archetype_labels.parquet"
    artifact.to_parquet(artifact_path, index=False)
    print(f"Wrote {artifact_path}")

    # Verify by round-trip read
    rt = pd.read_parquet(artifact_path)
    print(f"Round-trip read: {len(rt)} rows, {rt.archetype_name.nunique()} archetypes")
    print("\nArchetype distribution:")
    print(rt.archetype_name.value_counts().to_string())


if __name__ == "__main__":
    main()
