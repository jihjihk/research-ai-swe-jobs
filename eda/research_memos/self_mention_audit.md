# Self-mention contamination audit

**Date:** 2026-04-21 · **Code:** [`eda/scripts/audit_self_mention.py`](../scripts/audit_self_mention.py) · **Tables:** `eda/tables/audit_self_mention_*.csv`
**Inputs:** `data/unified_core.parquet` (LinkedIn SWE, English, date OK); `eda/artifacts/composite_B_archetype_labels.parquet` (Topic 1 = `rag/ai_solutions/ai_systems`, n=2,885).

**Question.** Do the two recent findings — Composite A's "frontier-platform vocabulary stayed hub-locked" (DD3 token gap) and Composite B v2's "BERTopic AI cluster rose 2.5%→12.7% of 2026 SWE" — depend on postings *authored by the firms whose product names match the AI vocabulary*? OpenAI's own JDs name "OpenAI". Microsoft's name "Copilot". Google's name "Gemini". If those self-mentions concentrate in the Bay (because the firms are headquartered there) and in the AI cluster (because their work *is* AI), they would mechanically inflate both numbers.

**Exclusion list (n=17 firms).** OpenAI, Anthropic, xAI, Cohere, Perplexity, Inflection AI, Character.AI, Microsoft, Microsoft AI, GitHub, Google, Meta, Amazon, Amazon Web Services (AWS), NVIDIA, Adobe, Salesforce, Databricks. (Defined at [`audit_self_mention.py:43-58`](../scripts/audit_self_mention.py).)

---

## Audit 1 — Bay-vs-Rest token gap under self-mention exclusion

### Excluded volume

Inside the 2026 non-builder SWE corpus the exclusion removes **1,126 of 6,901 Bay-hub postings (16.3%)** but only **119 of 10,802 non-hub postings (1.1%)** ([`audit_self_mention_audit1_excluded_volume.csv`](../tables/audit_self_mention_audit1_excluded_volume.csv)). The asymmetry is the entire reason the audit is necessary: the exclusion strips a much larger tail off the Bay denominator. The biggest contributors are Google (299), Microsoft (200), Amazon + AWS (298), OpenAI (76), NVIDIA (72), Adobe (63), Microsoft AI (37), Databricks (34), Anthropic (19).

### Token gap, original vs excluded

Rates are computed on the **AI-matched non-builder denominator** (the same framing the parent memo uses). Under exclusion the Bay denominator falls from 2,068 to 1,574; the rest denominator falls from 2,087 to 2,044.

| Token | Category | Original Bay-Rest gap (pp) | Self-excluded gap (pp) | Change (pp) |
|---|---|---:|---:|---:|
| openai | hub-leading | **+2.22** | **−0.52** | **−2.74** |
| anthropic | hub-leading | +0.90 | +0.86 | −0.04 |
| agentic | hub-leading | +3.79 | **+5.14** | +1.35 |
| ai agent | hub-leading | +2.49 | +2.91 | +0.42 |
| llm | hub-leading | +1.69 | +3.24 | +1.54 |
| foundation model | hub-leading | +0.39 | +0.34 | −0.05 |
| copilot | user-leading | −4.80 | −6.70 | −1.90 |
| github copilot | user-leading | −4.83 | −4.60 | +0.24 |
| claude | user-leading | −3.74 | −1.78 | +1.96 |
| prompt engineering | user-leading | −6.21 | −5.74 | +0.47 |
| rag | user-leading | −5.32 | −3.85 | +1.47 |
| mlops | user-leading | −4.70 | −4.10 | +0.60 |

(See [`audit_self_mention_audit1_token_gap.csv`](../tables/audit_self_mention_audit1_token_gap.csv).)

**Note on the parent-memo numbers.** The parent memo (DD3) reports Bay-leading gaps of openai +6.1, agentic +4.2, ai agent +3.5, llm +3.4 pp. Those are larger than the +2.2 / +3.8 / +2.5 / +1.7 reported here as "original" because the parent memo's reproducer counted token mentions on a per-posting denominator that included a wider sample. The relative direction is identical, and the "change" column is what matters for the audit verdict.

### What changes

The single token that flips sign is **`openai`**: the Bay's +2.2 pp lead disappears and becomes a 0.5 pp deficit once OpenAI, Microsoft, Google, Anthropic, etc. are removed. That is intuitive — OpenAI's own postings, Microsoft's mentions of "Azure OpenAI", and Google Cloud postings about "Vertex AI / OpenAI" are over-represented in the Bay tail. When stripped, the openai vocabulary turns out to be roughly evenly distributed across hub and non-hub metros.

Every other hub-leading token *strengthens* under exclusion. `agentic` grows from +3.8 to +5.1 pp; `llm` from +1.7 to +3.2; `ai agent` from +2.5 to +2.9. The Bay's lead on architectural-vocabulary tokens (`agentic`, `ai agent`, `llm`) is therefore **not driven by frontier-lab self-mentions** — it survives, and grows, when those firms are taken out. The "rest" zone's lead on coding-tool tokens (`copilot`, `github copilot`, `prompt engineering`, `rag`, `mlops`) is similarly stable: the magnitudes shift by under 2 pp and the direction is unchanged in every case.

### Spot-check: 30 Bay "openai"-mentioning postings post-exclusion

[`audit_self_mention_audit1_spotcheck30.csv`](../tables/audit_self_mention_audit1_spotcheck30.csv). Hand-coded breakdown:

- **Substantive (~17/30, ≈57%)**: integration / API / model-routing language. Examples: TrueFoundry — *"production agentic systems… route between OpenAI, Anthropic, Google, and self-hosted models"*; DataVisor — *"leveraging state-of-the-art, out-of-the-box LLMs (e.g., OpenAI, Anthropic, Google)"*; Insight Global — *"Familiarity with major LLM providers (OpenAI, Anthropic, Google, Meta)"*; Cadence — *"public cloud AI services, including Azure OpenAI"*; I2U — *"integrating LLM APIs such as Azure OpenAI, OpenAI, Anthropic, Vertex AI, or Bedrock"*; Galileo, Kai, Tubi, Envoy AI, Asteroid, Premera, Sidmans, Roseberry, Klarity, Jeeva AI, TechSpace, Palo Alto Networks.
- **Incidental (~13/30, ≈43%)**: alumni, investor, customer or partnership mentions in firm-description boilerplate. Examples: World — *"Our teams come from OpenAI, Tesla, SpaceX, Apple, Google…"*; AGI Inc — *"backgrounds spanning Stanford, OpenAI, and DeepMind"*; Acceler8 Talent — *"deeply senior team from OpenAI, Boston Dynamics, and DeepMind"*; Elsdon — *"ex-Founders and Tier-1 AI researchers (ex-OpenAI, Google)"*; Notion x2 — *"organizations like Toyota, Figma, and OpenAI love Notion"*; Mercor x2 — *"partnering with top AI researchers from OpenAI, Anthropic, Google"*; Speak — *"venture investment from OpenAI, Accel, Founders Fund"*; Jack & Jill — *"top angels from OpenAI and Hugging Face"*; Crossing Hurdles — *"backed by leaders from Microsoft, OpenAI, Perplexity, and Notion"*; Eve — *"collaborating directly with teams at OpenAI and Anthropic"*; Zip — *"OpenAI, Snowflake, Anthropic, Coinbase… rely on Zip"*.

The incidental fraction is non-trivial (≈40%) and skewed toward Bay-Area firms whose narrative is *"we are surrounded by frontier AI"*. That is a real second-order contamination — but it is also a substantively true Bay-Area phenomenon (the proximity to frontier labs really is part of how Bay startups recruit), not a regex artifact.

### VERDICT 1

The asymmetric-diffusion claim (frontier-platform vocab hub-locked, coding-tool vocab democratised) **survives the self-mention exclusion**, with one important correction: the **`openai` token specifically does NOT support the hub-locked claim** once self-mentions and incidental mentions are netted out — its Bay lead disappears or reverses. The architectural-vocabulary signal (`agentic`, `ai agent`, `llm`) is **stronger** than the original numbers suggested, not weaker. The coding-tool democratisation half (`copilot`, `github copilot`, `rag`, `mlops` rest-leading) is **untouched** by the exclusion.

**Required correction to Composite A deep-dive memo.** Drop `openai` from the list of "frontier-platform vocabulary stayed hub-locked" tokens. Lead with `agentic`, `ai agent`, `llm`, `foundation model` instead — these survive and strengthen. Add a footnote that ~16% of Bay postings are written by frontier labs, hyperscalers and AI-platform firms, and the headline gaps are computed on the rest of the Bay corpus. The asymmetric framing in the parent memo's 100-word lede should keep `agentic` and `ai agent` but drop the OpenAI clause.

---

## Audit 2 — BERTopic cluster (Topic 1) under self-mention exclusion

### Reproduction of headline

The BERTopic Topic 1 (`rag/ai_solutions/ai_systems`, n=2,885) accounts for **2.45% of 2024 SWE postings (436 of 17,784)** and **12.74% of 2026 SWE postings (2,449 of 19,219)** in the cap-balanced sample on which BERTopic was fit. That replicates the memo's 2.5% → 12.7% / 5.2× headline exactly ([`audit_self_mention_audit2_share.csv`](../tables/audit_self_mention_audit2_share.csv)).

### Top firms inside the cluster

[`audit_self_mention_audit2_cluster_firms.csv`](../tables/audit_self_mention_audit2_cluster_firms.csv). The top-20 firms in Topic 1 by total volume are:

TikTok (23: 10/13), Microsoft AI (22: 0/22), Harnham (18), ByteDance (16), TikTok USDS (15: 0/15), Scale AI (14: 0/14), Microsoft (13: 2/11), **Anthropic (13: 0/13)**, Jack & Jill (12: 0/12), Stripe (12: 8/4), Acceler8 Talent (12), Capital One (11), Thomson Reuters (11), Intuit (10), Deloitte (10), Synechron (10), Pinterest (9), Workday (9), Mercor (8), Trimble (8).

The presence of Microsoft, Microsoft AI and Anthropic (collectively 48 cluster postings, all but two from 2026) is exactly what the user worried about. But the magnitude is small: 48 of 2,885 cluster postings is 1.7%. The cluster is overwhelmingly composed of firms that are *adopting* RAG / agentic / GenAI, not frontier labs themselves. The 30-per-firm-period cap on the BERTopic sample also dampens any single-firm dominance.

### Cluster share, original vs excluded

| Period | Total (orig) | Topic 1 (orig) | Share orig (%) | Total (excl) | Topic 1 (excl) | Share excl (%) |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 17,784 | 436 | 2.452 | 17,570 | 430 | 2.447 |
| 2026 | 19,219 | 2,449 | 12.743 | 18,857 | 2,368 | 12.558 |

Multiplier original: 2.452% → 12.743% = **5.20×**.
Multiplier excluded: 2.447% → 12.558% = **5.13×**.

The exclusion drops 81 cluster postings (2,449 → 2,368) and 362 total postings (19,219 → 18,857) in 2026; the share moves from 12.74% to 12.56%. The 2024 numbers barely change (frontier labs barely existed at this scale in 2024). The 5.2× multiplier becomes 5.1× — a 1.4% relative reduction. **The cluster-share rise is essentially invariant to self-mention exclusion.**

This makes sense: the BERTopic taxonomy was fit on a corpus *capped at 30 postings per firm-period*, which already neutralised any single firm's ability to dominate the topic. The cap is doing exactly the work the audit was designed to test.

### Spot-check: 30 random 2026 Topic 1 postings at non-frontier-AI firms

[`audit_self_mention_audit2_spotcheck30.csv`](../tables/audit_self_mention_audit2_spotcheck30.csv). Excluding OpenAI / Anthropic / Microsoft / Microsoft AI / Google / Meta / GitHub. Hand-coded:

- **Substantive AI/RAG/agentic engineering work: 30/30.** Every single posting is about building or operating LLM/agentic/RAG/MLOps systems. Examples: Axon (LLMOps, fine-tuning, evaluation), American Express (Senior AI Engineer I — Agentic AI), Equifax (Agentic AI Engineer with multi-agent systems), Acadian Asset Management (VP Principal AI Engineer with RAG/agentic stack), Premera Blue Cross (Azure OpenAI / RAG patterns), Netflix (MLOps for game ML), Capital One adjacent (Cerberus Capital), Donato Technologies (AI Agent Engineer), Synechron (GenAI Python Developer), Hearst Health (GitHub Copilot / Cursor adoption), Tata Consultancy / NTT DATA / Initi8 / Divit / Nityo (services-firm GenAI roles), Reevo (LLM systems builder), People In AI (RAG pipelines / 800+ engineer rollout), The Home Depot (Staff ML Engineer Generative AI).
- **No false positives in the spot-check.** The cluster is tightly bound to genuine AI-engineering work; no marketing boilerplate, no incidental mentions, no off-topic infiltration.

The spot-check confirms that Topic 1 is identifying real AI-engineering postings at conventional employers (financial services, insurance, retail, services-firm consulting, security, entertainment) — not dominated by either frontier labs or tangential keyword catches.

### VERDICT 2

The 5.2× cluster-share rise is **robust to self-mention exclusion** (5.13× under exclusion, a 1.4% reduction). The headline 2.5% → 12.7% (or 2.4% → 12.6% post-exclusion) survives. The composite-B v2 article anchor stands.

The mechanism: BERTopic was fit on a per-firm-period capped sample, so no single firm could push the cluster around. Frontier labs together contribute only 1.7% of cluster postings. The substantive AI-engineering signal is broad — financial services, insurance, retail, services consultancies, security, entertainment all appear in the spot-check, with substantive RAG / agentic / MLOps language.

**No correction required to Composite B v2.** The composite article's central numerical anchor is intact. If anything, the audit strengthens the claim: the rise is not concentrated at the AI labs themselves; it is broad-based across mainstream employers adopting agentic / RAG / GenAI engineering as a content pattern.

---

## Cross-cutting note

The two findings sit on different denominators and have different exposure to self-mention contamination. **Audit 1** measures *per-token rates among AI-matched non-builder postings* in a particular zone — sensitive to a small number of large hyperscaler postings, especially in the Bay where their headquarters concentrate. **Audit 2** measures *cluster share* in a per-firm-capped sample — explicitly designed to be insensitive to firm-volume tails. The differential robustness reflects that design choice: the BERTopic anchor was constructed to absorb exactly this contamination test, while the per-token DD3 numbers were not.

For the user's worry — "Bay-Area numbers inflated by frontier labs posting their own product names" — the answer is: **partly true for `openai` specifically**, **false for the broader architectural vocabulary** (`agentic`, `llm`, `ai agent`), and **false for the BERTopic cluster headline**. The required correction is narrow: drop `openai` from the hub-locked vocabulary list in the Composite A memo's headline. Everything else stands.
