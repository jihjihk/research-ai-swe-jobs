"""
T18 - Step 3: Boundary shift analysis (SWE ↔ SWE-adjacent TF-IDF cosine by period)
Also emits top distinguishing/migrating terms.
"""
import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

PARQUET = "data/unified.parquet"
OUT_DIR = "exploration/tables/T18"
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(20260409)

# Company stoplist
STOPLIST = set()
try:
    with open("exploration/artifacts/shared/company_stoplist.txt") as f:
        STOPLIST = set(line.strip().lower() for line in f if line.strip())
    print(f"Loaded {len(STOPLIST)} company stoplist tokens")
except FileNotFoundError:
    pass

# Markdown backslash escape fix (per T12 / V1 finding)
_MD_ESCAPE_RE = re.compile(r"\\([+#\-./])")


def clean_text(s):
    if not s:
        return ""
    s = _MD_ESCAPE_RE.sub(r"\1", s)
    return s.lower()


def fetch(con, group_col, period2):
    q = f"""
    SELECT uid, coalesce(description_core_llm, description_core, description) AS text,
           period
    FROM '{PARQUET}'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND {group_col} = true
      AND ({"period IN ('2024-01','2024-04')" if period2=='2024' else "period IN ('2026-03','2026-04')"})
      AND description IS NOT NULL
      AND LENGTH(coalesce(description_core_llm, description_core, description)) >= 500
    """
    return con.execute(q).df()


def balanced_sample(df, n):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=RNG.integers(0, 2**31))


def cosine_groups(docs_a, docs_b, stopwords):
    # Train on combined corpus
    combined = docs_a + docs_b
    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=list(stopwords) if stopwords else None,
        min_df=3,
        max_df=0.8,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b",
    )
    X = vec.fit_transform(combined)
    Xa = X[: len(docs_a)]
    Xb = X[len(docs_a) :]
    ca = np.asarray(Xa.mean(axis=0)).ravel()
    cb = np.asarray(Xb.mean(axis=0)).ravel()
    sim = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    return sim, vec, ca, cb


def top_differences(vec, ca, cb, k=20):
    diff = ca - cb  # positive = more in group A
    vocab = np.array(vec.get_feature_names_out())
    top_a = vocab[np.argsort(-diff)[:k]]
    top_b = vocab[np.argsort(diff)[:k]]
    return list(top_a), list(top_b)


con = duckdb.connect()

results = []
migration_rows = []

# Stopwords: basic English + company stoplist (only first tokens to avoid killing everything)
basic_sw = set(
    """the a an and or but of to in for with on at by from as is are was were be been
       will can could should would may might must shall do does did have has had you your our
       we they them it their us if then when while this that these those""".split()
)
# Keep short company tokens only
comp_sw = {w for w in STOPLIST if len(w) >= 4 and w.isalpha() and len(w) <= 20}
stopwords = basic_sw | comp_sw

SAMPLE_SIZE = 400  # larger than 200 to improve stability

for period2 in ["2024", "2026"]:
    print(f"\n=== Period {period2} ===")
    swe = fetch(con, "is_swe", period2)
    adj = fetch(con, "is_swe_adjacent", period2)
    ctrl = fetch(con, "is_control", period2)
    print(f"  fetched SWE={len(swe)} adj={len(adj)} ctrl={len(ctrl)}")

    swe_s = balanced_sample(swe, SAMPLE_SIZE)
    adj_s = balanced_sample(adj, SAMPLE_SIZE)
    ctrl_s = balanced_sample(ctrl, SAMPLE_SIZE)

    swe_docs = [clean_text(t) for t in swe_s.text.tolist()]
    adj_docs = [clean_text(t) for t in adj_s.text.tolist()]
    ctrl_docs = [clean_text(t) for t in ctrl_s.text.tolist()]

    # SWE vs adjacent
    sim, vec, ca, cb = cosine_groups(swe_docs, adj_docs, stopwords)
    top_swe, top_adj = top_differences(vec, ca, cb, k=25)
    results.append({"pair": "SWE_vs_adjacent", "period": period2, "cosine": sim,
                    "n_a": len(swe_docs), "n_b": len(adj_docs)})
    print(f"  SWE↔adjacent centroid cosine: {sim:.4f}")
    migration_rows.append({"period": period2, "pair": "SWE_vs_adjacent",
                           "top_in_swe": ", ".join(top_swe),
                           "top_in_adjacent": ", ".join(top_adj)})

    # SWE vs control
    sim2, vec2, ca2, cb2 = cosine_groups(swe_docs, ctrl_docs, stopwords)
    top_swe2, top_ctrl2 = top_differences(vec2, ca2, cb2, k=25)
    results.append({"pair": "SWE_vs_control", "period": period2, "cosine": sim2,
                    "n_a": len(swe_docs), "n_b": len(ctrl_docs)})
    print(f"  SWE↔control centroid cosine: {sim2:.4f}")

    # adjacent vs control
    sim3, _, _, _ = cosine_groups(adj_docs, ctrl_docs, stopwords)
    results.append({"pair": "adjacent_vs_control", "period": period2, "cosine": sim3,
                    "n_a": len(adj_docs), "n_b": len(ctrl_docs)})
    print(f"  adjacent↔control centroid cosine: {sim3:.4f}")

pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "boundary_similarity.csv"), index=False)
pd.DataFrame(migration_rows).to_csv(os.path.join(OUT_DIR, "boundary_top_terms.csv"), index=False)

print("\nBoundary similarity summary:")
print(pd.DataFrame(results).to_string())

# Did SWE and adjacent become more similar or less similar?
r = pd.DataFrame(results)
sim24 = r[(r.pair=="SWE_vs_adjacent") & (r.period=="2024")].cosine.iloc[0]
sim26 = r[(r.pair=="SWE_vs_adjacent") & (r.period=="2026")].cosine.iloc[0]
print(f"\nSWE↔adjacent cosine 2024={sim24:.4f} -> 2026={sim26:.4f} (delta={sim26-sim24:+.4f})")
