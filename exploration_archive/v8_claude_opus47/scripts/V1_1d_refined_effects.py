"""V1.1d - Rebuild compound with failed sub-terms dropped; compute effect + SNR."""
import duckdb
import re
import math
import json
from pathlib import Path

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")
CLEAN_PARQ = "/home/jihgaboot/gabor/job-research/exploration/artifacts/shared/swe_cleaned_text.parquet"

# Original (from prompt)
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

# Failed sub-terms (V1.1c finding)
FAILED = {"agent_bare", "mcp"}

REFINED_BROAD = {k: v for k, v in BROAD_SUBTERMS.items() if k not in FAILED}
REFINED_STRICT = {k: v for k, v in STRICT_SUBTERMS.items() if k not in FAILED}


def build_compound(subterms):
    return "(" + "|".join(subterms.values()) + ")"


BROAD_ORIG_RE = build_compound(BROAD_SUBTERMS)
BROAD_REFINED_RE = build_compound(REFINED_BROAD)
STRICT_ORIG_RE = build_compound(STRICT_SUBTERMS)
STRICT_REFINED_RE = build_compound(REFINED_STRICT)


def load_corpus():
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT uid, description_cleaned, text_source, source, period, seniority_final
        FROM '{CLEAN_PARQ}'
        WHERE description_cleaned IS NOT NULL AND description_cleaned != ''
    """).df()
    return df


def compute_snr(df, pattern):
    df = df.copy()
    df["period_group"] = df["period"].apply(lambda p: "2024" if p.startswith("2024") else "2026")
    # Use vectorized regex match
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    df["match"] = df["description_cleaned"].map(lambda t: bool(compiled.search(t)) if t else False)
    g24 = df[df.period_group == "2024"]["match"]
    g26 = df[df.period_group == "2026"]["match"]
    p24 = g24.mean()
    p26 = g26.mean()
    v24 = p24 * (1 - p24) / len(g24)
    v26 = p26 * (1 - p26) / len(g26)
    se = math.sqrt(v24 + v26)
    delta = p26 - p24
    snr = delta / se if se > 0 else float("inf")
    return {"p24": float(p24), "p26": float(p26), "delta": float(delta),
            "n24": int(len(g24)), "n26": int(len(g26)),
            "se": float(se), "snr": float(snr)}


def main():
    print("Loading corpus...")
    df = load_corpus()
    print(f"  {len(df)} rows; llm={len(df[df.text_source=='llm'])}")

    # Use the LLM-cleaned subset as primary (stable across descriptions)
    df_llm = df[df.text_source == "llm"].copy()
    df_all = df.copy()

    print("\n=== Broad pattern ===")
    b_orig_llm = compute_snr(df_llm, BROAD_ORIG_RE)
    b_refined_llm = compute_snr(df_llm, BROAD_REFINED_RE)
    print("  orig (LLM): ", b_orig_llm)
    print("  refined (LLM): ", b_refined_llm)
    # Also compute on all rows (raw+llm)
    b_orig_all = compute_snr(df_all, BROAD_ORIG_RE)
    b_refined_all = compute_snr(df_all, BROAD_REFINED_RE)
    print("  orig (ALL): ", b_orig_all)
    print("  refined (ALL): ", b_refined_all)

    print("\n=== Strict pattern ===")
    s_orig_llm = compute_snr(df_llm, STRICT_ORIG_RE)
    s_refined_llm = compute_snr(df_llm, STRICT_REFINED_RE)
    print("  orig (LLM): ", s_orig_llm)
    print("  refined (LLM): ", s_refined_llm)
    s_orig_all = compute_snr(df_all, STRICT_ORIG_RE)
    s_refined_all = compute_snr(df_all, STRICT_REFINED_RE)
    print("  orig (ALL): ", s_orig_all)
    print("  refined (ALL): ", s_refined_all)

    summary = {
        "refined_broad_subterms": list(REFINED_BROAD.keys()),
        "refined_strict_subterms": list(REFINED_STRICT.keys()),
        "dropped": list(FAILED),
        "effects": {
            "broad_orig_llm": b_orig_llm,
            "broad_refined_llm": b_refined_llm,
            "broad_orig_all": b_orig_all,
            "broad_refined_all": b_refined_all,
            "strict_orig_llm": s_orig_llm,
            "strict_refined_llm": s_refined_llm,
            "strict_orig_all": s_orig_all,
            "strict_refined_all": s_refined_all,
        }
    }
    with open(OUT_DIR / "V1_1d_refined_effects.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'V1_1d_refined_effects.json'}")


if __name__ == "__main__":
    main()
