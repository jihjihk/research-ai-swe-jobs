"""V2.8 — T20 boundary AUC re-derivation.

For each seniority boundary (entry↔associate, associate↔mid-senior,
mid-senior↔director) and the J3/S4 yoe panel, train logistic regression with
5-fold CV and report AUC for 2024 and 2026.

Features: yoe_numeric (median-imputed at 3), tech_count, ai_binary (strict),
org_scope_density (per 1K), mgmt_density (per 1K), desc_length_cleaned,
education_level (0-3 ordinal via simple regex).
"""
import re
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
MGMT_STRICT = re.compile(
    r"\b(mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)?|(?:hire\s+and\s+(?:develop|manage|grow))|(?:hire(?:\s+a)?\s+team)|(?:hiring\s+(?:and|manager|engineers|team|plan))|headcount|performance[- ]review)\b",
    re.IGNORECASE,
)
ORG_SCOPE = re.compile(
    r"\b(stakeholder(?:s)?|cross[- ]functional|leadership|strategic|vision|roadmap|mentor(?:ing|ship)?|influence|partner(?:ship|ing)?\s+with|executive|c[- ]?suite|align(?:ment)?|org(?:anization)?)\b",
    re.IGNORECASE,
)
TECH_PATS = [
    r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bgolang\b", r"\brust\b", r"\bc\+\+\b", r"\bc#\b", r"\bruby\b",
    r"\bkotlin\b", r"\bswift\b", r"\bscala\b", r"\bphp\b", r"\bsql\b",
    r"\breact\b", r"\bangular\b", r"\bvue\.?js\b", r"\bnode\.?js\b",
    r"\bdjango\b", r"\bflask\b", r"\bspring\b", r"\b\.net\b",
    r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bkubernetes\b", r"\bdocker\b",
    r"\bterraform\b", r"\bpostgres\w*\b", r"\bmongo\w*\b", r"\bredis\b",
    r"\bkafka\b", r"\bspark\b", r"\bsnowflake\b", r"\btensorflow\b",
    r"\bpytorch\b", r"\bpandas\b", r"\blangchain\b", r"\brag\b",
    r"\bcopilot\b", r"\bcursor\b", r"\bagile\b", r"\bscrum\b",
    r"\bjenkins\b", r"\blinux\b", r"\bnumpy\b",
]
TECH_COMP = [re.compile(p, re.IGNORECASE) for p in TECH_PATS]
EDU_PATS = {
    "phd": re.compile(r"\bph[.]?\s?d\b", re.IGNORECASE),
    "ms": re.compile(r"\bmaster\b|\bm[.]?s[.]?(\s+in|\s+degree)\b", re.IGNORECASE),
    "bs": re.compile(r"\bbachelor\b|\bb[.]?s[.]?(\s+in|\s+degree)\b", re.IGNORECASE),
    "hs": re.compile(r"\bhigh\s+school\b|\bhs\s+diploma\b", re.IGNORECASE),
}


def edu_level(text: str) -> int:
    if not text:
        return 0
    if EDU_PATS["phd"].search(text):
        return 3
    if EDU_PATS["ms"].search(text):
        return 2
    if EDU_PATS["bs"].search(text):
        return 1
    return 0


def features(text: str) -> dict:
    if not text:
        return {"tech_count": 0, "ai_binary": 0, "org_scope_density": 0.0,
                "mgmt_density": 0.0, "desc_len": 0, "edu_level": 0}
    L = max(1, len(text))
    tc = sum(1 for p in TECH_COMP if p.search(text))
    ai_b = int(bool(AI_STRICT.search(text)))
    scope = len(ORG_SCOPE.findall(text)) / L * 1000
    mgmt = len(MGMT_STRICT.findall(text)) / L * 1000
    return {
        "tech_count": tc, "ai_binary": ai_b,
        "org_scope_density": scope, "mgmt_density": mgmt,
        "desc_len": L, "edu_level": edu_level(text),
    }


def boundary_auc(df_sub, pos_class, neg_class):
    """5-fold CV AUC on L2 logistic regression."""
    mask = df_sub["seniority_final"].isin([pos_class, neg_class])
    sub = df_sub[mask].copy()
    if len(sub) < 50:
        return float("nan"), 0
    y = (sub["seniority_final"] == pos_class).astype(int).values
    if y.sum() < 15 or (1 - y).sum() < 15:
        return float("nan"), 0
    feat_cols = ["yoe_numeric", "tech_count", "ai_binary", "org_scope_density",
                 "mgmt_density", "desc_len", "edu_level"]
    X = sub[feat_cols].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(penalty="l2", max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), len(sub)


def main():
    con = duckdb.connect()
    q = """
    SELECT uid, description, description_core_llm, seniority_final, yoe_extracted,
           CASE WHEN period LIKE '2024%' THEN '2024'
                WHEN period LIKE '2026%' THEN '2026' ELSE 'other' END AS period_bucket,
           CASE WHEN description_core_llm IS NOT NULL AND description_core_llm != '' THEN 'llm' ELSE 'raw' END AS text_source
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok' AND is_swe=true
      AND seniority_final IS NOT NULL
    """
    print("[V2.8] loading SWE rows...")
    df = con.execute(q).df()
    # LLM-only primary (T20 spec)
    df = df[df["text_source"] == "llm"]
    df["text"] = df["description_core_llm"].fillna("")
    df = df[df["period_bucket"].isin(["2024", "2026"])]
    df = df[df["text"] != ""]
    print(f"[V2.8] rows with text: {len(df)}")

    # Feature extraction
    print("[V2.8] extracting features...")
    f_rows = [features(t) for t in df["text"]]
    f_df = pd.DataFrame(f_rows, index=df.index)
    df = pd.concat([df, f_df], axis=1)
    df["yoe_numeric"] = df["yoe_extracted"].fillna(3.0)

    # Boundaries
    pairs = [
        ("associate", "entry", "associate_vs_entry"),
        ("mid-senior", "associate", "mid-senior_vs_associate"),
        ("director", "mid-senior", "director_vs_mid-senior"),
    ]
    rows = []
    for per in ["2024", "2026"]:
        sub = df[df["period_bucket"] == per]
        for pos, neg, name in pairs:
            auc, n = boundary_auc(sub, pos, neg)
            rows.append({"boundary": name, "period": per, "auc": auc, "n": n})
            print(f"  {name} {per}: AUC={auc:.4f} n={n}")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT_DIR / "V2_8_boundary_auc.csv", index=False)

    # J3 vs S4 yoe panel (yoe<=2 vs yoe>=5), without yoe in features
    print("\n[V2.8] J3 vs S4 yoe panel (yoe excluded from features):")
    feat_cols_noyoe = ["tech_count", "ai_binary", "org_scope_density",
                        "mgmt_density", "desc_len", "edu_level"]
    rows2 = []
    for per in ["2024", "2026"]:
        sub = df[(df["period_bucket"] == per) & (df["yoe_extracted"].notna())]
        yoe = sub["yoe_extracted"].values
        mask = (yoe <= 2) | (yoe >= 5)
        sub2 = sub[mask].copy()
        y = (sub2["yoe_extracted"] >= 5).astype(int).values
        if y.sum() < 15 or (1 - y).sum() < 15:
            print(f"  {per}: insufficient n")
            continue
        X = sub2[feat_cols_noyoe].values
        X = StandardScaler().fit_transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(penalty="l2", max_iter=500, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        auc = float(np.mean(aucs))
        rows2.append({"panel": "J3_vs_S4", "period": per, "auc": auc, "n": len(sub2)})
        print(f"  J3_vs_S4 {per}: AUC={auc:.4f} n={len(sub2)}")
    pd.DataFrame(rows2).to_csv(OUT_DIR / "V2_8_j3_s4_panel.csv", index=False)
    print(f"\n  T20 claims: associate/entry +0.054, mid-sr/associate +0.084, dir/mid-sr +0.003; J3/S4 +0.14")


if __name__ == "__main__":
    main()
