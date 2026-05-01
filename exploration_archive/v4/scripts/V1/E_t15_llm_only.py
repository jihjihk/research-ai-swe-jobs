"""V1 Part E — T15 null-result replication on LLM-text-only subset.

Loads shared embeddings and filters to rows with text_source = 'llm'. Computes
per-period junior/senior centroid cosine. Reports whether the null finding
holds.
"""
import duckdb
import numpy as np

con = duckdb.connect()
con.execute(
    "CREATE VIEW ct AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet')"
)
con.execute(
    "CREATE VIEW u AS SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/data/unified.parquet')"
)
idx = con.execute(
    "SELECT * FROM read_parquet('/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embedding_index.parquet')"
).fetchdf()
emb = np.load("/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_embeddings.npy")
assert len(idx) == emb.shape[0]

# Join: uid -> text_source, period2, seniority (combined best-available)
meta = con.execute(
    """
WITH swe AS (
  SELECT uid,
         CASE WHEN source IN ('kaggle_arshkon','kaggle_asaniczka') THEN '2024'
              WHEN source='scraped' THEN '2026' END AS period2,
         CASE WHEN llm_classification_coverage='labeled' THEN seniority_llm
              WHEN llm_classification_coverage='rule_sufficient' THEN seniority_final
              ELSE NULL END AS seniority_best,
         yoe_extracted
  FROM u WHERE source_platform='linkedin' AND is_english=TRUE AND date_flag='ok' AND is_swe=TRUE
)
SELECT s.uid, s.period2, s.seniority_best, s.yoe_extracted, ct.text_source
FROM swe s JOIN ct USING(uid)
"""
).fetchdf()

# Align to embedding order
uid_to_row = {uid: i for i, uid in enumerate(idx["uid"].tolist())}
meta = meta.set_index("uid").reindex(idx["uid"]).reset_index()

# Sanity: text_source distribution by period
print("\n=== text_source × period2 distribution (full SWE) ===")
print(meta.groupby(["period2", "text_source"]).size().unstack(fill_value=0))


def cosine_centroid(mask_a, mask_b):
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("nan")
    ca = emb[mask_a].mean(axis=0)
    cb = emb[mask_b].mean(axis=0)
    ca /= np.linalg.norm(ca) + 1e-12
    cb /= np.linalg.norm(cb) + 1e-12
    return float((ca * cb).sum())


def junior_senior_centroid_cosine(mask_period, label_variant):
    if label_variant == "combined":
        j = mask_period & (meta["seniority_best"] == "entry").values
        s = mask_period & (meta["seniority_best"] == "mid-senior").values
    elif label_variant == "yoe":
        yoe = meta["yoe_extracted"].values
        has_yoe = meta["yoe_extracted"].notna().values
        j = mask_period & has_yoe & (yoe <= 2)
        s = mask_period & has_yoe & (yoe >= 5)
    return cosine_centroid(j, s), int(j.sum()), int(s.sum())


periods = ["2024", "2026"]
print("\n=== Centroid cosine (junior↔senior) within each period ===")
for label in ["combined", "yoe"]:
    print(f"\n-- label variant: {label} --")
    print(f"{'period':8s} {'subset':20s} {'n_junior':>10s} {'n_senior':>10s} {'cos_j_s':>10s}")
    for p in periods:
        mask = (meta["period2"].values == p)
        cos, nj, ns = junior_senior_centroid_cosine(mask, label)
        print(f"{p:8s} {'all':20s} {nj:>10d} {ns:>10d} {cos:>10.4f}")

        mask_llm = mask & (meta["text_source"].values == "llm")
        cos, nj, ns = junior_senior_centroid_cosine(mask_llm, label)
        print(f"{p:8s} {'llm-only':20s} {nj:>10d} {ns:>10d} {cos:>10.4f}")

        mask_rule = mask & (meta["text_source"].values == "rule")
        cos, nj, ns = junior_senior_centroid_cosine(mask_rule, label)
        print(f"{p:8s} {'rule-only':20s} {nj:>10d} {ns:>10d} {cos:>10.4f}")
