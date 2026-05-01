#!/usr/bin/env python3
"""
T29 step 1: Compute LLM-authorship features per SWE LinkedIn posting.

Uses the RAW `description` column (not cleaned) because cleaning strips
formatting features like bullets, paragraphs, em-dashes. Also stores
`text_source` so downstream analyses can control for the 2024/2026
text-source composition confound.

Signals:
  1. Vocabulary density (per 1K chars): classic LLM tells
  2. Em-dash density (Unicode — and ASCII --)
  3. Sentence length mean and std (only for raw text w/ punctuation)
  4. Type-token ratio (TTR) within posting
  5. Bullet marker density (• - * → and numbered list markers)
  6. Paragraph count (\n\n splits) and mean length
  7. Boilerplate closer indicator ("in conclusion", "we are an equal", "in summary")
  8. Hedging phrase indicator ("but not limited to", "such as but")
  9. Punctuation stats (comma density, exclamation density)
"""
from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
import duckdb

OUT_T = "exploration/tables/T29"
os.makedirs(OUT_T, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM signature vocabulary (normalized, lowercase)
# Reference set based on widely-reported LLM tells + corporate boilerplate words.
# Flagged as "suggestive" not "diagnostic" — precision check in step 2.
# ---------------------------------------------------------------------------
LLM_SIGNATURE_WORDS = [
    # Classic LLM tells
    "delve", "delving", "tapestry", "leverage", "leveraging",
    "robust", "unleash", "embark", "navigate", "navigating",
    "cutting-edge", "cutting edge",
    "in the realm of", "comprehensive", "seamless", "seamlessly",
    "furthermore", "moreover", "notably", "nonetheless",
    "align with", "at the forefront",
    "pivotal", "harness", "harnessing",
    "vibrant", "dynamic",
    "it's worth noting", "worth noting",
    # LLM-preferred descriptive style
    "ever-evolving", "ever evolving", "landscape", "foster",
    "fostering", "empower", "empowering",
    "holistic", "streamline", "streamlining",
    "spearhead", "spearheading",
    "transformative", "paradigm",
    "underscore", "underscores",
    "multifaceted", "synergy", "synergies",
    "cornerstone", "bedrock", "hallmark",
    "drive innovation", "driving innovation",
    "world-class", "world class",
    "state-of-the-art", "state of the art",
    "best-in-class",
    "unlock",
]

# Compile a single regex for speed. Use word boundaries where sensible.
# For multi-word phrases, include them as literal substrings.
LLM_VOCAB_RE = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(w) for w in LLM_SIGNATURE_WORDS) + r")\b"
)

# Separately count em-dashes
EMDASH_RE = re.compile(r"—|(?<!-)--(?!-)")  # — or standalone --

# Bullets — markdown common forms
BULLET_RE = re.compile(r"(?:^|\n)\s*(?:[•\*\-\u2022]|\d+\.)\s")

# Paragraph splits (double newline or \n\n)
PARA_RE = re.compile(r"\n\s*\n")

# Hedging phrases (LLM tendency)
HEDGE_PHRASES = [
    "but not limited to",
    "such as but",
    "including but not",
    "not limited to",
    "as well as any",
]
HEDGE_RE = re.compile(
    r"(?i)(?:" + "|".join(re.escape(p) for p in HEDGE_PHRASES) + r")"
)

# Boilerplate closer patterns
CLOSER_PATTERNS = [
    r"in conclusion",
    r"in summary",
    r"to summarize",
    r"we look forward to",
    r"join (?:our|us)",
    r"if (?:this sounds|you are|you're) .{0,50}(?:reach out|apply|contact)",
]
CLOSER_RE = re.compile(r"(?i)(?:" + "|".join(CLOSER_PATTERNS) + r")")

# Simple sentence splitter (not perfect, but stable)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Word tokenizer (ASCII-ish; we lowercase)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ---------------------------------------------------------------------------
# SANITY TESTS (assert-driven)
# ---------------------------------------------------------------------------
def _tests():
    t = "Leverage cutting-edge LLMs to embark on the journey. We foster a dynamic environment."
    assert len(LLM_VOCAB_RE.findall(t)) >= 5, f"expected >=5 LLM hits, got {LLM_VOCAB_RE.findall(t)}"

    t2 = "This is a test — with em-dash. And another one -- here."
    assert len(EMDASH_RE.findall(t2)) == 2, f"expected 2 em-dashes, got {EMDASH_RE.findall(t2)}"

    t3 = "Responsibilities:\n• Build things\n• Ship fast\n- Also this\n1. Numbered\n2. Lists"
    assert len(BULLET_RE.findall(t3)) == 5, f"expected 5 bullets, got {len(BULLET_RE.findall(t3))}"

    t4 = "Para one.\n\nPara two.\n\nPara three."
    # 2 splits -> 3 paras
    assert len(PARA_RE.split(t4)) == 3

    t5 = "Skills including but not limited to: Python, SQL."
    assert HEDGE_RE.search(t5) is not None

    t6 = "We look forward to hearing from you!"
    assert CLOSER_RE.search(t6) is not None

    t7 = "First sentence. Second sentence! Third one? Fourth end."
    sents = SENTENCE_RE.split(t7)
    assert len(sents) == 4, f"expected 4 sentences, got {len(sents)}: {sents}"

    print("T29 regex tests pass.")


_tests()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def compute_features(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "n_chars": 0, "n_words": 0, "ttr": 0.0,
            "llm_vocab_count": 0, "llm_vocab_density_1k": 0.0,
            "emdash_count": 0, "emdash_density_1k": 0.0,
            "bullet_count": 0, "bullet_density_1k": 0.0,
            "paragraph_count": 0, "mean_para_len": 0.0,
            "sentence_count": 0, "mean_sent_len_chars": 0.0, "std_sent_len_chars": 0.0,
            "hedge_phrase_count": 0, "closer_count": 0,
            "comma_density_1k": 0.0, "exclaim_density_1k": 0.0,
        }

    n_chars = len(text)
    per_1k = 1000.0 / max(n_chars, 1)

    # Vocabulary
    llm_hits = len(LLM_VOCAB_RE.findall(text))
    emdash = len(EMDASH_RE.findall(text))
    bullets = len(BULLET_RE.findall(text))
    hedges = len(HEDGE_RE.findall(text))
    closers = len(CLOSER_RE.findall(text))

    # Paragraphs
    paras = PARA_RE.split(text)
    paras = [p for p in paras if p.strip()]
    para_count = len(paras)
    mean_para_len = np.mean([len(p) for p in paras]) if paras else 0.0

    # Sentences (approximate)
    sents = SENTENCE_RE.split(text)
    sents = [s for s in sents if s.strip()]
    sent_lens = [len(s) for s in sents]
    sent_count = len(sents)
    mean_sent = float(np.mean(sent_lens)) if sent_lens else 0.0
    std_sent = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0

    # TTR
    words = [w.lower() for w in WORD_RE.findall(text)]
    n_words = len(words)
    ttr = (len(set(words)) / n_words) if n_words > 0 else 0.0

    # Comma / exclaim
    n_commas = text.count(",")
    n_excl = text.count("!")

    return {
        "n_chars": n_chars, "n_words": n_words, "ttr": ttr,
        "llm_vocab_count": llm_hits, "llm_vocab_density_1k": llm_hits * per_1k,
        "emdash_count": emdash, "emdash_density_1k": emdash * per_1k,
        "bullet_count": bullets, "bullet_density_1k": bullets * per_1k,
        "paragraph_count": para_count, "mean_para_len": float(mean_para_len),
        "sentence_count": sent_count, "mean_sent_len_chars": mean_sent, "std_sent_len_chars": std_sent,
        "hedge_phrase_count": hedges, "closer_count": closers,
        "comma_density_1k": n_commas * per_1k, "exclaim_density_1k": n_excl * per_1k,
    }


# ---------------------------------------------------------------------------
# Load SWE LinkedIn rows — raw description
# ---------------------------------------------------------------------------
print("Loading SWE LinkedIn rows...")
con = duckdb.connect()
df = con.execute("""
    SELECT uid, source, period,
           description,
           CASE
               WHEN description_core_llm IS NOT NULL THEN 'llm'
               WHEN description_core IS NOT NULL THEN 'rule'
               ELSE 'raw'
           END AS text_source_best,
           is_aggregator,
           company_name_canonical,
           seniority_final,
           yoe_extracted,
           llm_extraction_coverage
    FROM read_parquet('data/unified.parquet')
    WHERE source_platform = 'linkedin'
      AND is_english = true
      AND date_flag = 'ok'
      AND is_swe = true
""").df()
print(f"Loaded {len(df):,} SWE LinkedIn rows")

df["year"] = np.where(df["period"].astype(str).str.startswith("2024"), "2024", "2026")
df["text_source_actual"] = np.where(
    df["llm_extraction_coverage"].isin(["labeled"]),
    "llm",
    "rule"
)

print(f"Text source (actual llm vs rule for description_core):")
print(df["text_source_actual"].value_counts())
print(f"\nBy year:")
print(pd.crosstab(df["year"], df["text_source_actual"]))

# ---------------------------------------------------------------------------
# Extract features on RAW description
# ---------------------------------------------------------------------------
print("\nExtracting features on raw description...")
from tqdm import tqdm
tqdm.pandas()

feats = df["description"].progress_apply(compute_features)
feat_df = pd.DataFrame(list(feats))
out = pd.concat([df[["uid", "source", "year", "period", "is_aggregator",
                     "company_name_canonical", "seniority_final", "yoe_extracted",
                     "text_source_actual"]].reset_index(drop=True),
                 feat_df.reset_index(drop=True)], axis=1)

print(f"\nFeature summary (means by year):")
agg_cols = [c for c in feat_df.columns if c not in {"n_chars", "n_words"}]
print(out.groupby("year")[["n_chars", "n_words"] + agg_cols].mean().round(3).T)

out.to_parquet(f"{OUT_T}/authorship_scores.parquet", index=False)
print(f"\nSaved: {OUT_T}/authorship_scores.parquet ({len(out):,} rows)")
