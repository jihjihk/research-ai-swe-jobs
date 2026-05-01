"""V2.9a — AI-mention precision across occupation groups.

Sample 20 SWE + 20 adjacent + 20 control postings that AI-strict matched in
scraped 2026. Extract ±150-char context windows. Rule-based classify TP/FP/AMB
with keyword heuristics matching V1 protocol. Flag if control precision differs.
"""
import re
import random
import pandas as pd
import duckdb
from pathlib import Path

REPO = Path("/home/jihgaboot/gabor/job-research")
OUT_DIR = REPO / "exploration/artifacts/V2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AI_STRICT = re.compile(
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\b",
    re.IGNORECASE,
)


def classify(text: str, match_term: str) -> str:
    """Heuristic rule-based TP/FP/AMB classifier per V1 protocol."""
    t = text.lower()
    term = match_term.lower()
    # Clear TP signals within the window
    ai_signals = [
        "ai", "ml", "machine learning", "llm", "language model", "genai",
        "generative", "openai", "anthropic", "hugging", "nvidia", "tensor",
        "chatbot", "engineer", "developer", "assistant", "coding", "tool",
        "fine-tun", "inference", "retrieval", "prompt", "model", "rag",
        "agent",
    ]
    # FP signals
    fp_signals = []
    if term == "cursor":
        # 'cursor' = database cursor (DB), mouse cursor, lifecycle cursor
        # TP if near AI terms; FP if near SQL, mouse, scroll
        fp_signals = ["database cursor", "sql cursor", "mouse cursor", "pagination cursor", "fetch"]
    if term == "rag":
        fp_signals = ["raggedy", "rag (", "ragbag"]  # rare
    if term == "gemini":
        # could be zodiac; rare in JDs
        fp_signals = ["zodiac", "horoscope"]
    if term == "claude":
        # rare FP (person name); check for context
        fp_signals = ["monet", "painter", "debussy"]
    # Count
    for fp in fp_signals:
        if fp in t:
            return "FP"
    for ai in ai_signals:
        if ai in t:
            return "TP"
    # If the matched term itself is a high-confidence AI tool, call TP
    if term in {"copilot", "chatgpt", "openai api", "llamaindex", "langchain",
                "prompt engineering", "fine-tuning", "vector database",
                "pinecone", "huggingface", "hugging face"}:
        return "TP"
    return "AMB"


def main():
    con = duckdb.connect()
    # All scraped 2026 rows with AI-strict hit
    q = f"""
    SELECT uid, is_swe, is_swe_adjacent, is_control,
           COALESCE(description_core_llm, description, '') AS text
    FROM 'data/unified.parquet'
    WHERE source_platform='linkedin' AND is_english=true AND date_flag='ok'
      AND period LIKE '2026%'
      AND (is_swe OR is_swe_adjacent OR is_control)
      AND regexp_matches(lower(COALESCE(description_core_llm, description, '')),
                         '\\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\\d+|gemini|codex|llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|vector database|pinecone|huggingface|hugging face)\\b')
    """
    print("[V2.9a] loading AI-hit rows...")
    df = con.execute(q).df()
    print(f"[V2.9a] rows: {len(df)}")

    # Split by group
    swe = df[df["is_swe"]]
    adj = df[df["is_swe_adjacent"]]
    ctrl = df[df["is_control"]]
    print(f"  SWE: {len(swe)}, ADJACENT: {len(adj)}, CONTROL: {len(ctrl)}")

    def sample_and_class(sub, n, label):
        random.seed(42)
        s = sub.sample(min(n, len(sub)), random_state=42).copy()
        s["term"] = s["text"].apply(lambda t: AI_STRICT.search(t).group(0) if AI_STRICT.search(t) else "")
        s["window"] = s["text"].apply(lambda t: _window(t))
        s["verdict"] = s.apply(lambda r: classify(r["window"], r["term"]), axis=1)
        s["group"] = label
        return s[["uid", "group", "term", "verdict", "window"]]

    def _window(text):
        m = AI_STRICT.search(text)
        if not m:
            return ""
        s = max(0, m.start() - 150)
        e = min(len(text), m.end() + 150)
        return text[s:e]

    swe_s = sample_and_class(swe, 20, "swe")
    adj_s = sample_and_class(adj, 20, "adjacent")
    ctrl_s = sample_and_class(ctrl, 20, "control")
    all_s = pd.concat([swe_s, adj_s, ctrl_s])
    all_s.to_csv(OUT_DIR / "V2_9a_ai_precision_samples.csv", index=False)

    # Summary
    for label, g in [("SWE", swe_s), ("ADJACENT", adj_s), ("CONTROL", ctrl_s)]:
        tp = (g["verdict"] == "TP").sum()
        fp = (g["verdict"] == "FP").sum()
        amb = (g["verdict"] == "AMB").sum()
        print(f"\n[V2.9a] {label}: TP={tp}, FP={fp}, AMB={amb}, "
              f"strict_precision={tp/len(g)*100:.1f}%, relaxed={tp/(tp+fp if tp+fp else 1)*100:.1f}%")
        print("  Matched terms:", g["term"].value_counts().to_dict())

    # Print windows for hand review
    for label, g in [("SWE", swe_s), ("ADJACENT", adj_s), ("CONTROL", ctrl_s)]:
        print(f"\n=== {label} samples (first 10) ===")
        for _, r in g.head(10).iterrows():
            print(f"  [{r['verdict']}] {r['uid']} term={r['term']}")
            print(f"    ...{r['window'][:300]}...")


if __name__ == "__main__":
    main()
