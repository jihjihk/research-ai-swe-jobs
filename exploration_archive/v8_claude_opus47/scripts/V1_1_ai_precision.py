"""V1.1 - AI-mention semantic precision audit.

Sample 50 matches stratified 25/25 2024/2026 for each pattern (broad, strict).
Also compute per-sub-term precision for the broad pattern's highest-risk terms.
Drop any sub-term < 80% precision; rebuild compound; re-compute +33pp effect and SNR.
"""
import duckdb
import re
import pandas as pd
import random
import json
import os
from pathlib import Path

random.seed(42)

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"

# ===== Pattern definitions =====
# Broad (from gate_2.md finding 1; AI-mention rise SNR 24.7)
BROAD_SUBTERMS = {
    "ai_bare": r"\bai\b",
    "artificial_intelligence": r"\bartificial intelligence\b",
    "ml_bare": r"\bml\b",
    "machine_learning": r"\bmachine learning\b",
    "llm": r"\bllm\b",
    "large_language_model": r"\blarge language models?\b",
    "generative_ai": r"\bgenerative ai\b",
    "genai": r"\bgenai\b",
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "claude": r"\bclaude\b",
    "chatgpt": r"\bchatgpt\b",
    "openai": r"\bopenai\b",
    "anthropic": r"\banthropic\b",
    "gemini": r"\bgemini\b",
    "gpt": r"\bgpt\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "rag": r"\brag\b",
    "retrieval_augmented": r"\bretrieval augmented\b",
    "agent_bare": r"\bagent\b",
    "langchain": r"\blangchain\b",
}
STRICT_SUBTERMS = {
    "copilot": r"\bcopilot\b",
    "cursor": r"\bcursor\b",
    "claude": r"\bclaude\b",
    "chatgpt": r"\bchatgpt\b",
    "openai_api": r"\bopenai api\b",
    "gpt_num": r"\bgpt-?\d+\b",
    "gemini": r"\bgemini\b",
    "codex": r"\bcodex\b",
    "mcp": r"\bmcp\b",
    "llamaindex": r"\bllamaindex\b",
    "langchain": r"\blangchain\b",
    "prompt_engineering": r"\bprompt engineering\b",
    "fine_tuning": r"\bfine[- ]tuning\b",
    "rag": r"\brag\b",
    "vector_database": r"\bvector database\b",
    "pinecone": r"\bpinecone\b",
    "huggingface": r"\b(huggingface|hugging face)\b",
}

BROAD_COMPOUND = "(" + "|".join(BROAD_SUBTERMS.values()) + ")"
STRICT_COMPOUND = "(" + "|".join(STRICT_SUBTERMS.values()) + ")"


def load_corpus():
    """Load SWE cleaned text with period and source info. Use text_source='llm' only for text-sensitive work."""
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT uid, description_cleaned, text_source, source, period, seniority_final
        FROM '{CLEAN_PARQ}'
        WHERE description_cleaned IS NOT NULL
          AND description_cleaned != ''
    """).df()
    return df


def extract_context(text, match_start, match_end, window=100):
    """Extract +/-100 char context around match."""
    s = max(0, match_start - window)
    e = min(len(text), match_end + window)
    return text[s:e].replace("\n", " ")


def find_matches_in_text(text, subterm_name, pattern):
    """Find all match positions for this sub-term."""
    out = []
    if not text:
        return out
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        out.append({
            "subterm": subterm_name,
            "start": m.start(),
            "end": m.end(),
            "matched_text": m.group(0),
            "context": extract_context(text, m.start(), m.end()),
        })
    return out


def sample_matches_per_subterm(df, subterms, n_per_period=25, samples_per_subterm=50):
    """For each subterm, stratify 25 2024 + 25 2026 matches randomly."""
    out = {}
    # Flag period-group
    df = df.copy()
    df["period_group"] = df["period"].apply(lambda p: "2024" if p.startswith("2024") else "2026")
    for subterm_name, pattern in subterms.items():
        rows_2024 = []
        rows_2026 = []
        # Shuffle row order deterministically for each subterm
        idx = list(df.index)
        random.Random(hash(subterm_name) % (2**32)).shuffle(idx)
        for i in idx:
            if len(rows_2024) >= n_per_period and len(rows_2026) >= n_per_period:
                break
            row = df.loc[i]
            pg = row["period_group"]
            if (pg == "2024" and len(rows_2024) >= n_per_period) or (pg == "2026" and len(rows_2026) >= n_per_period):
                continue
            matches = find_matches_in_text(row["description_cleaned"], subterm_name, pattern)
            if not matches:
                continue
            # Pick a single match per posting (first)
            m = matches[0]
            rec = {
                "uid": row["uid"],
                "period": row["period"],
                "period_group": pg,
                "source": row["source"],
                "subterm": subterm_name,
                "matched_text": m["matched_text"],
                "context": m["context"],
            }
            if pg == "2024":
                rows_2024.append(rec)
            else:
                rows_2026.append(rec)
        out[subterm_name] = rows_2024 + rows_2026
        print(f"  {subterm_name}: {len(rows_2024)} 2024 + {len(rows_2026)} 2026 samples")
    return out


def auto_classify_sample(rec):
    """Heuristic classifier to label TP/FP/AMB based on context. We will still manually classify."""
    context = rec["context"].lower()
    matched = rec["matched_text"].lower()
    sub = rec["subterm"]

    # AI-tool context signals
    AI_CONTEXT = [
        "artificial intelligence", "machine learning", "deep learning", "neural network",
        "llm", "large language", "generative ai", "gen ai", "genai",
        "openai", "anthropic", "google", "claude", "gpt", "chatgpt",
        "prompt", "fine-tun", "fine tun", "rag ", "retrieval augmented",
        "copilot", "cursor", "vector database", "embedding", "transformer",
        "natural language", "nlp", "pytorch", "tensorflow", "hugging face",
        "huggingface", "langchain", "agent", "mcp",
        "data scien", "data engineer", "ml model", "ai model", "ai tool",
        "model train", "model inference", "ai-assist", "ai assist",
        "mlops", "ml ops", "feature engineer", "recommend",
    ]
    AI_TRUE = any(s in context for s in AI_CONTEXT)

    # Non-AI indicators (false positive for broad patterns)
    # \bai\b: non-AI sub-uses
    if sub == "ai_bare":
        # "ai" standalone in context like "jai alai", "aide", "aikido", "aid", "ai-series"
        if re.search(r"\b(jai alai|aide|aikido|aiko|aikido|thailand|asia|ailment|aim|ainsworth|aila)\b", context):
            return "false_positive"
        # "ai/ml" or "ai and ml" or AI-adjacent tech terms are true
        if re.search(r"\bai[/ ,-]|[/ ,-]ai\b", context) and AI_TRUE:
            return "true_positive"
        if AI_TRUE:
            return "true_positive"
        return "ambiguous"
    if sub == "ml_bare":
        # HTML often breaks here - check context
        if re.search(r"\b(html|xml|yaml|haml|kml)\b", context):
            return "false_positive"
        # .ml domain or ml shorthand for ML - check context
        if re.search(r"ml[/ ,-]|[/ ,-]ml\b", context) and AI_TRUE:
            return "true_positive"
        if AI_TRUE:
            return "true_positive"
        return "ambiguous"
    if sub == "agent_bare":
        # Very noisy - insurance/real estate/sales agent, shipping agent, travel agent
        if re.search(r"\b(insurance agent|real estate|real-estate|sales agent|travel agent|shipping agent|bookkeep|call center|customer service|property manage|mortgage|buyer|seller)\b", context):
            return "false_positive"
        if re.search(r"\b(ai agent|llm agent|autonomous agent|agentic|multi-agent|ai agents|agent framework|build agents|agent-based|react agent)\b", context):
            return "true_positive"
        # "user agent" in HTTP context = fp
        if re.search(r"\buser agent\b|useragent", context):
            return "false_positive"
        return "ambiguous"
    if sub == "rag":
        # RAG in AI context = retrieval augmented generation. FP in clothing/rags
        if re.search(r"\b(clothing|fabric|rags to|ragged|ragtime|tag rag)\b", context):
            return "false_positive"
        if AI_TRUE or re.search(r"\b(vector|embed|retrieval|generation|langchain|knowledge base|document)\b", context):
            return "true_positive"
        return "ambiguous"
    if sub == "gpt":
        if re.search(r"\b(gpt partition|gpt-?disk|gpt format|uefi)\b", context):
            return "false_positive"
        if AI_TRUE or re.search(r"\b(chat|llm|language model|openai|api|model)\b", context):
            return "true_positive"
        return "ambiguous"
    if sub == "claude":
        # Claude the person? rare; mostly AI
        if re.search(r"\bclaude\b", context) and re.search(r"\b(person|born|died|mr\.|dr\.|dear|senator)\b", context):
            return "false_positive"
        return "true_positive"
    if sub == "cursor":
        # Cursor editor AI or database cursor?
        if re.search(r"\b(sql cursor|database cursor|db cursor|cursor-based|cursor pagination|mouse cursor|cursor pointer|sql server|postgres)\b", context):
            return "false_positive"
        if re.search(r"\b(cursor editor|cursor ide|cursor ai|ai editor|ai assistant|code editor)\b", context) or AI_TRUE:
            return "true_positive"
        return "ambiguous"
    if sub == "gemini":
        # Gemini horoscope or Google Gemini?
        if re.search(r"\b(horoscope|zodiac|astrology|constellation)\b", context):
            return "false_positive"
        return "true_positive"
    if sub == "codex":
        # Medical codex? Codex alimentarius?
        if re.search(r"\b(codex alimentarius|medical|manuscript|illuminated|biblical)\b", context):
            return "false_positive"
        return "true_positive"
    if sub == "mcp":
        # MCP could be Microsoft Certified Professional
        if re.search(r"\b(microsoft certified|certified professional|mcp[- ]certified)\b", context):
            return "false_positive"
        # Model Context Protocol
        if re.search(r"\b(model context|anthropic|llm|agent|tool)\b", context):
            return "true_positive"
        return "ambiguous"
    # Default: if AI_TRUE, call TP
    if AI_TRUE:
        return "true_positive"
    # Sub-terms like openai, chatgpt, claude, langchain, copilot, pinecone, huggingface are essentially unambiguous
    if sub in ("openai", "chatgpt", "langchain", "copilot", "pinecone", "huggingface", "llamaindex", "openai_api",
               "llm", "large_language_model", "generative_ai", "genai", "artificial_intelligence",
               "machine_learning", "prompt_engineering", "retrieval_augmented",
               "vector_database", "fine_tuning", "anthropic", "gpt_num"):
        return "true_positive"
    return "ambiguous"


def compute_precision(classifications):
    counts = {"true_positive": 0, "false_positive": 0, "ambiguous": 0}
    for c in classifications:
        counts[c["label"]] += 1
    n = sum(counts.values())
    if n == 0:
        return 0.0, counts
    # precision = TP / (TP + FP + AMB) -- conservative (count ambiguous as non-TP)
    prec = counts["true_positive"] / n
    return prec, counts


def compute_ai_mention_share(df, pattern, text_col="description_cleaned"):
    """Compute share of postings matching pattern, by period_group."""
    df = df.copy()
    # Flag period_group
    df["period_group"] = df["period"].apply(lambda p: "2024" if p.startswith("2024") else "2026")
    # Use regex
    df["match"] = df[text_col].str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
    summary = df.groupby("period_group")["match"].agg(["mean", "sum", "count"])
    return summary


def compute_snr(df, pattern, text_col="description_cleaned"):
    """Bootstrap-style SNR: (2024 mean - 2026 mean) / sd of bootstrap samples.
    Here we use the simpler formula:
        SNR = delta / sqrt( var(2024) / n2024 + var(2026) / n2026 )
    which is a Welch z-ish."""
    df = df.copy()
    df["period_group"] = df["period"].apply(lambda p: "2024" if p.startswith("2024") else "2026")
    df["match"] = df[text_col].str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
    g24 = df[df.period_group == "2024"]["match"]
    g26 = df[df.period_group == "2026"]["match"]
    p24 = g24.mean()
    p26 = g26.mean()
    # Variance of Bernoulli
    v24 = p24 * (1 - p24) / len(g24)
    v26 = p26 * (1 - p26) / len(g26)
    import math
    se = math.sqrt(v24 + v26)
    delta = p26 - p24
    snr = delta / se if se > 0 else float("inf")
    return {
        "p24": p24, "p26": p26, "delta": delta,
        "n24": len(g24), "n26": len(g26),
        "se": se, "snr": snr,
    }


def main():
    print("Loading corpus...")
    df = load_corpus()
    print(f"  {len(df)} rows")
    # For text-sensitive: use text_source == 'llm' (as recommended in prompt)
    df_llm = df[df.text_source == "llm"].copy()
    df_all = df.copy()
    print(f"  {len(df_llm)} LLM-labeled; {len(df_all)} all")

    # === STEP 1: Sample 50 matches per sub-pattern ===
    print("\n=== Sampling broad sub-terms ===")
    broad_samples = sample_matches_per_subterm(df_llm, BROAD_SUBTERMS, n_per_period=25)
    # Auto-classify
    all_classifications = []
    for subterm, samples in broad_samples.items():
        for s in samples:
            label = auto_classify_sample(s)
            all_classifications.append({**s, "label": label, "pattern_class": "broad"})

    print("\n=== Sampling strict sub-terms ===")
    strict_samples = sample_matches_per_subterm(df_llm, STRICT_SUBTERMS, n_per_period=25)
    for subterm, samples in strict_samples.items():
        for s in samples:
            label = auto_classify_sample(s)
            all_classifications.append({**s, "label": label, "pattern_class": "strict"})

    # Save all classifications
    df_cls = pd.DataFrame(all_classifications)
    df_cls.to_csv(OUT_DIR / "V1_1_classifications.csv", index=False)
    print(f"Saved {len(df_cls)} classifications to {OUT_DIR / 'V1_1_classifications.csv'}")

    # === STEP 2: Per-sub-term precision ===
    subterm_stats = {}
    for subterm in list(BROAD_SUBTERMS.keys()) + list(STRICT_SUBTERMS.keys()):
        sub_rows = [r for r in all_classifications if r["subterm"] == subterm]
        if not sub_rows:
            continue
        prec, counts = compute_precision(sub_rows)
        subterm_stats[subterm] = {"precision": prec, "n": len(sub_rows), "counts": counts}
    print("\n=== Sub-term precision ===")
    for sub, s in sorted(subterm_stats.items(), key=lambda x: x[1]["precision"]):
        star = " <-- FAIL" if s["precision"] < 0.80 else ""
        print(f"  {sub:30s}  p={s['precision']:.2f}  n={s['n']:>3}  tp={s['counts']['true_positive']}  fp={s['counts']['false_positive']}  amb={s['counts']['ambiguous']}{star}")

    # === STEP 3: Drop <80% precision sub-terms; rebuild pattern ===
    KEEP_BROAD = {name: pat for name, pat in BROAD_SUBTERMS.items()
                  if name in subterm_stats and subterm_stats[name]["precision"] >= 0.80}
    DROP_BROAD = [name for name in BROAD_SUBTERMS if name not in KEEP_BROAD]
    KEEP_STRICT = {name: pat for name, pat in STRICT_SUBTERMS.items()
                   if name in subterm_stats and subterm_stats[name]["precision"] >= 0.80}
    DROP_STRICT = [name for name in STRICT_SUBTERMS if name not in KEEP_STRICT]

    print(f"\nDropped from broad: {DROP_BROAD}")
    print(f"Dropped from strict: {DROP_STRICT}")

    REFINED_BROAD = "(" + "|".join(KEEP_BROAD.values()) + ")" if KEEP_BROAD else None
    REFINED_STRICT = "(" + "|".join(KEEP_STRICT.values()) + ")" if KEEP_STRICT else None

    # === STEP 4: Compute AI-mention effect on original + refined patterns ===
    print("\n=== AI-mention share (full pattern) ===")
    broad_orig = compute_snr(df_llm, BROAD_COMPOUND)
    strict_orig = compute_snr(df_llm, STRICT_COMPOUND)
    print("Broad (orig):  ", broad_orig)
    print("Strict (orig): ", strict_orig)

    print("\n=== AI-mention share (refined after dropping <80% sub-terms) ===")
    if REFINED_BROAD:
        broad_refined = compute_snr(df_llm, REFINED_BROAD)
        print("Broad (refined):  ", broad_refined)
    else:
        broad_refined = None
    if REFINED_STRICT:
        strict_refined = compute_snr(df_llm, REFINED_STRICT)
        print("Strict (refined): ", strict_refined)
    else:
        strict_refined = None

    # Summary JSON
    summary = {
        "broad_subterms": BROAD_SUBTERMS,
        "strict_subterms": STRICT_SUBTERMS,
        "subterm_precision": subterm_stats,
        "dropped_broad": DROP_BROAD,
        "dropped_strict": DROP_STRICT,
        "effects": {
            "broad_orig": broad_orig,
            "strict_orig": strict_orig,
            "broad_refined": broad_refined,
            "strict_refined": strict_refined,
        },
        "n_rows_llm": len(df_llm),
    }
    with open(OUT_DIR / "V1_1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {OUT_DIR / 'V1_1_summary.json'}")


if __name__ == "__main__":
    main()
