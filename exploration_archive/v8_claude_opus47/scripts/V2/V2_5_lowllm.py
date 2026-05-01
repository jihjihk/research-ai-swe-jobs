"""V2.5 — Independent authorship score + low-LLM subset re-test.

Build independent authorship score from:
  - signature vocab density
  - em-dash density
  - sentence length SD

Z-score sum → bottom-40% within-period subset. Re-compute Δ on:
  - AI-strict binary
  - AI-broad binary
  - Mentor-strict binary (senior-only AND corpus-wide)
  - Breadth-resid (length-residualized)

Compare to full-corpus Δ. Verify T29's retention (77%, 86%, 72%, 71%).
"""
import re
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


LLM_SIG_WORDS = [
    "delve", "leverage", "robust", "tapestry", "cutting-edge", "seamless",
    "stakeholders", "trade-offs", "enhance", "utilize", "paradigm",
    "synergize", "innovative", "transformative", "groundbreaking", "dynamic",
    "empower", "holistic", "comprehensive", "nuanced", "intricate",
    "underscore", "facilitate", "paramount", "myriad", "plethora",
    "multifaceted", "embark", "navigate", "fosters", "streamline",
    "cultivate", "elevates", "unwavering",
]
EM_DASH_RE = re.compile(r"[—]|(?:\s--\s)")
SENT_SPLIT = re.compile(r"[.!?]\s")

AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)
AI_BROAD = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face|ai|artificial intelligence|ml|machine learning|llm|large language model|generative ai|genai|anthropic)\b",
    re.IGNORECASE,
)
MGMT_STRICT = re.compile(
    r"\b(mentor(?:s|ed|ing|ship)?|coach(?:es|ed|ing)?|(?:hire\s+and\s+(?:develop|manage|grow))|(?:hire(?:\s+a)?\s+team)|(?:hiring\s+(?:and|manager|engineers|team|plan))|headcount|performance[- ]review)\b",
    re.IGNORECASE,
)
# Tech counts for breadth
TECH_PATS = [
    r"\bpython\b", r"\bjava\b", r"\bjavascript\b", r"\btypescript\b",
    r"\bgolang\b", r"\brust\b", r"\bc\+\+\b", r"\bc#\b", r"\bruby\b",
    r"\bkotlin\b", r"\bswift\b", r"\bscala\b", r"\bphp\b", r"\bsql\b",
    r"\breact\b", r"\bangular\b", r"\bvue\.?js\b", r"\bnext\.?js\b",
    r"\bhtml\b", r"\bcss\b", r"\bnode\.?js\b", r"\bdjango\b", r"\bflask\b",
    r"\bspring\b", r"\b\.net\b", r"\baws\b", r"\bazure\b", r"\bgcp\b",
    r"\bkubernetes\b", r"\bdocker\b", r"\bterraform\b", r"\bpostgres\w*\b",
    r"\bmysql\b", r"\bmongo\w*\b", r"\bredis\b", r"\bkafka\b", r"\bspark\b",
    r"\bsnowflake\b", r"\btensorflow\b", r"\bpytorch\b", r"\bpandas\b",
    r"\bjupyter\b", r"\blangchain\b", r"\brag\b", r"\bcopilot\b", r"\bcursor\b",
    r"\bopenai\b", r"\bclaude\b", r"\bagile\b", r"\bscrum\b",
    r"\bmicroservices?\b", r"\bjenkins\b", r"\blinux\b", r"\bbash\b", r"\bnumpy\b",
]
TECH_COMPILED = [re.compile(p, re.IGNORECASE) for p in TECH_PATS]
# org-scope for breadth
ORG_SCOPE = re.compile(
    r"\b(stakeholder(?:s)?|cross[- ]functional|leadership|strategic|vision|roadmap|mentor(?:ing|ship)?|influence|partner(?:ship|ing)?\s+with|executive|c[- ]?suite|align(?:ment)?|org(?:anization)?)\b",
    re.IGNORECASE,
)


def authorship_features(text: str) -> dict:
    if not text:
        return {
            "vocab_density": 0.0,
            "em_dash_density": 0.0,
            "sentence_sd": 0.0,
            "desc_len": 0,
        }
    L = max(1, len(text))
    n_sig = sum(text.lower().count(w) for w in LLM_SIG_WORDS)
    vocab_density = n_sig / L * 1000
    em_count = len(EM_DASH_RE.findall(text))
    em_density = em_count / L * 1000
    sents = SENT_SPLIT.split(text)
    sent_lens = [len(s) for s in sents if s.strip()]
    sd = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
    return {
        "vocab_density": vocab_density,
        "em_dash_density": em_density,
        "sentence_sd": sd,
        "desc_len": L,
    }


def content_features(text: str) -> dict:
    if not text:
        return {
            "ai_strict": 0, "ai_broad": 0, "mgmt_strict": 0,
            "tech_count": 0, "org_scope_count": 0,
        }
    tc = sum(1 for p in TECH_COMPILED if p.search(text))
    osm = len(ORG_SCOPE.findall(text))
    return {
        "ai_strict": int(bool(AI_STRICT.search(text))),
        "ai_broad": int(bool(AI_BROAD.search(text))),
        "mgmt_strict": int(bool(MGMT_STRICT.search(text))),
        "tech_count": tc,
        "org_scope_count": osm,
    }


def main():
    print("[V2.5] loading SWE LinkedIn rows with LLM-cleaned text...")
    # Use shared cleaned text artifact for consistency with T29 (LLM-cleaned only)
    cleaned = pd.read_parquet(REPO / "exploration/artifacts/shared/swe_cleaned_text.parquet")
    print(f"[V2.5] cleaned rows: {len(cleaned)}")
    # LLM-text subset (29,599 raw rows excluded per T29 conventions)
    llm_rows = cleaned[cleaned["text_source"] == "llm"].copy()
    print(f"[V2.5] LLM rows: {len(llm_rows)}")
    # period bucket
    llm_rows["period_bucket"] = llm_rows["period"].str[:4]
    llm_rows = llm_rows[llm_rows["period_bucket"].isin(["2024", "2026"])]

    # Feature extraction
    print("[V2.5] extracting features...")
    rows_af = []
    rows_cf = []
    for text in llm_rows["description_cleaned"]:
        rows_af.append(authorship_features(text))
        rows_cf.append(content_features(text))
    af = pd.DataFrame(rows_af, index=llm_rows.index)
    cf = pd.DataFrame(rows_cf, index=llm_rows.index)
    feats = llm_rows[["uid", "period_bucket", "seniority_final"]].join(af).join(cf)

    # Length-residualized breadth: breadth = tech + org_scope;
    # residualize on 2024 OLS of breadth ~ log(desc_len)
    feats["breadth"] = feats["tech_count"] + feats["org_scope_count"]
    log_len = np.log(feats["desc_len"].clip(lower=1))
    m24 = feats["period_bucket"] == "2024"
    X = np.column_stack([np.ones(m24.sum()), log_len[m24].values])
    y = feats.loc[m24, "breadth"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"[V2.5] breadth OLS coefs (2024): β0={beta[0]:.3f}, β1={beta[1]:.3f}")
    feats["breadth_resid"] = feats["breadth"] - (beta[0] + beta[1] * log_len)

    # Authorship score: z-score sum
    ref24 = feats[feats["period_bucket"] == "2024"]
    zcols = ["vocab_density", "em_dash_density", "sentence_sd"]
    for c in zcols:
        mu = ref24[c].mean()
        sd = ref24[c].std()
        feats[f"z_{c}"] = (feats[c] - mu) / max(sd, 1e-9)
    feats["authorship_score_v2"] = feats[[f"z_{c}" for c in zcols]].mean(axis=1)
    print(f"[V2.5] authorship median 2024 = {feats[m24]['authorship_score_v2'].median():.3f}")
    print(f"[V2.5] authorship median 2026 = {feats[~m24]['authorship_score_v2'].median():.3f}")

    feats.to_parquet(OUT_DIR / "V2_5_authorship_features.parquet", index=False)

    # Compute deltas: full corpus vs low40-within-period
    def deltas(df_sub):
        out = {}
        for m in ["ai_strict", "ai_broad", "mgmt_strict", "breadth_resid"]:
            v24 = df_sub.loc[df_sub["period_bucket"] == "2024", m].mean()
            v26 = df_sub.loc[df_sub["period_bucket"] == "2026", m].mean()
            out[m] = {"2024": v24, "2026": v26, "delta": v26 - v24}
        # senior-only mentor
        sr = df_sub[df_sub["seniority_final"].isin(["mid-senior", "director"])]
        v24 = sr.loc[sr["period_bucket"] == "2024", "mgmt_strict"].mean() if len(sr) else float("nan")
        v26 = sr.loc[sr["period_bucket"] == "2026", "mgmt_strict"].mean() if len(sr) else float("nan")
        out["mgmt_senior"] = {"2024": v24, "2026": v26, "delta": v26 - v24}
        return out

    full = deltas(feats)
    # Low-40% within-period
    low_idx = []
    for per in ["2024", "2026"]:
        sub = feats[feats["period_bucket"] == per]
        thr = sub["authorship_score_v2"].quantile(0.40)
        low_idx.extend(sub[sub["authorship_score_v2"] <= thr].index.tolist())
    low = feats.loc[low_idx]
    low_d = deltas(low)

    rows = []
    for k in full.keys():
        f = full[k]
        l = low_d[k]
        retention = l["delta"] / f["delta"] if abs(f["delta"]) > 1e-9 else float("nan")
        rows.append({
            "metric": k,
            "full_2024": f["2024"], "full_2026": f["2026"], "full_delta": f["delta"],
            "low40_2024": l["2024"], "low40_2026": l["2026"], "low40_delta": l["delta"],
            "retention": retention,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "V2_5_low40_attenuation.csv", index=False)
    print("\n[V2.5] Low-40% attenuation table:")
    print(out_df.to_string(index=False))

    print(f"\n[V2.5] Low-40% n: {len(low)} (out of {len(feats)})")
    print(f"  T29 claims: AI-strict 77%, AI-broad 86%, mentor 72%, breadth-resid 71%")


if __name__ == "__main__":
    main()
