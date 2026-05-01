# Pattern validation

## Why this matters

Studies of posting content routinely claim "management language fell" or "AI tool mentions rose" on the strength of a regex that matches the word "manager" or the word "Copilot". The regex gets run against a corpus, the count goes up or down, and the number becomes a finding. The step that almost nobody does is to check, by reading a sample of matches, whether the regex is actually picking up the thing it claims to pick up.

An early audit in this study exposed how badly that step can be missed. A widely-used "broad management" regex was hand-audited on a stratified 50-posting sample. Its precision (the fraction of its matches that are genuinely about management responsibility) was 28%. Seventy-two of every hundred matches were false positives: generic uses of "team", "stakeholder", or "coordinate" in contexts that had nothing to do with managing anyone.

After that, every pattern used in a published claim on this site was audited the same way before it could carry a headline.

## The audit protocol

1. **Draw a stratified 50-row sample** from the corpus for each pattern (25 pre-period, 25 post-period).
2. **Manually label each match** as a true positive or false positive, based on whether the matched span is semantically consistent with the pattern's intended meaning. This is a human reading step; there is no shortcut.
3. **Threshold for acceptance:**
   - Precision 0.80 or higher → approved as a **primary** pattern, usable in headline claims.
   - Precision between 0.60 and 0.79 → approved as **diagnostic** only, not for headline use.
   - Precision below 0.60 → **failed**, the pattern is retired.
4. **Break the precision out by sub-pattern** (e.g., within the management pattern, compute precision separately for "hire", "performance review", etc.), not just the aggregate. Aggregate precision can hide a dead sub-pattern that happens to be small.
5. **Break the precision out by period** to catch period-specific contamination. The canonical example: "MCP" in 2024 text almost always meant Microsoft Certified Professional; in 2026 text it almost always means Model Context Protocol. A pattern matching "MCP" will have very different precision in the two periods, and pooling them hides it.

## Results of the first audit pass

| Pattern | Precision | Period split | Worst sub-pattern | Recommendation |
|---|---|---|---|---|
| AI-mention (strict) | 0.86 | 2024=0.78, 2026=0.94 | fine-tuning in 2024 (0.47) | Primary; restrict fine-tuning to LLM-adjacent contexts in 2024 |
| AI-mention (broad) | 0.72 | 2024=0.63, 2026=0.82 | **MCP in 2024 (0.15)**, agent (0.75) | Borderline; drop "mcp" for 2024 baselines |
| Management (strict) | 0.55 | — | **hire (0.07)**, **performance review (0.25)** | **Failed.** Use the rebuilt version. |
| Management (broad) | **0.28** | — | all four broad tokens fail | **Failed.** Retire entirely. |
| Scope | 0.89 | — | autonomous (0.55) | Primary; drop "autonomous" for robotics postings |
| Soft skills | 0.94 | — | leadership (0.86) | Primary; use as-is |

### The worst false-positive classes

- **Broad management at 28%**: the tokens "team", "stakeholder", and "coordinate" match in generic collaboration contexts, not management responsibility. If an IC engineer's posting says "works with the team to coordinate deployments", the regex fires. It should not.
- **Strict-management "hire" at 7%**: the overwhelming majority of "hire" matches are "contract-to-hire", "direct-hire", and "how-we-hire / accommodations" language. That is HR metadata, not "manages hiring".
- **Strict-management "performance review" at 25%**: the primary false-positive class is "code review" and "peer review" in QA contexts.
- **Broad-AI "MCP" in 2024 at 15%**: in 2024 text, MCP is overwhelmingly Microsoft Certified Professional. In 2026 text, it is Model Context Protocol. A pattern that doesn't split by period is measuring a different thing in each half of the corpus.

## The rebuilt patterns

After the first audit, the failing patterns were rewritten to be more specific. The rebuilt versions:

```regex
mgmt_strict_v1_rebuilt:
\b(?:mentor(?:s|ed|ing)? (?:junior|engineers?|team(?:s)?|others|the team|engineering|peers|sd(?:e|es))|coach(?:es|ed|ing)? (?:team|engineers?|junior|peers)|direct reports?|headcount|hiring manager|hiring decisions?)\b

ai_strict_v1_rebuilt:
\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|llamaindex|langchain|prompt engineering|rag|vector databas(?:e|es)|pinecone|huggingface|hugging face|(?:fine[- ]tun(?:e|ed|ing))\s+(?:the\s+)?(?:model|llm|gpt|base model|foundation model|embeddings))\b

scope_v1_rebuilt:
\b(ownership|end[\s\-]to[\s\-]end|cross[\s\-]functional|initiative(?:s)?|stakeholder(?:s)?)\b
```

## Re-validation after rebuilding

A second audit pass, this time with larger samples and additional patterns, confirmed the rebuilt versions and approved four new patterns:

| Pattern | Original precision | Re-validated on 50 | Re-validated on 30 | Recommendation |
|---|---|---|---|---|
| AI-mention (strict, rebuilt) | 0.86 → — | **0.96** | — | Primary |
| Management (strict, rebuilt) | 0.55 → — | **0.98** | — | Primary |
| Scope (rebuilt) | 0.89 → — | **1.00** | — | Primary |
| Aspiration hedging | — | 0.92 | **1.00** | Primary |
| Firm requirement | — (0.54 initial) | **1.00** | **1.00** | Primary (rebuilt by dropping the generic noun "requirements") |
| Scope (broader definition) | — | 0.96 | — | Primary |
| Senior scope terms | — | ≥ 0.89 by construction | — | Primary |

All seven patterns have their measured precision stored as a field in the shared pattern registry and are flagged as having cleared the 80% threshold.

## What changed in the headlines after the audit

### Claims that had to be corrected

1. The original "management language density fell" claim was corrected to "flat" once the broad management pattern was retired and the rebuilt strict pattern was used.
2. The "MCP 29x acceleration" claim was restricted to the top-level AI pattern, since the broad pattern's MCP contamination in 2024 made the ratio unreliable.
3. The within-firm AI rewriting finding retains a pattern-provenance caveat: the report text claims to use the rebuilt AI pattern, but the code as shipped uses the top-level pattern. The direction is the same under both; the magnitude is 10 to 15% smaller under the rebuilt pattern.

### Claims that survived unchanged

All six of the main cross-checks the first replication ran landed within 5% of the reported magnitudes:

| Check | Claim | Replicated value | Within 5% of original |
|---|---|---|---|
| 1 | Title-versus-seniority information ratio | 8.88x (original 8.80x) | yes |
| 2 | Junior-senior vocabulary-overlap change | 0.950 → 0.871 (original 0.946 → 0.863) | yes |
| 3 | Junior requirements-characters change | -5.4% | exact |
| 4 | Within-company junior share, across panels | +3.95 (pooled), -0.22 (arshkon) | within band |
| 5 | Accelerating ratios (RAG, MCP) | RAG 75.3x, MCP 28.8x | exact |
| 6 | Length-residualized requirement breadth | J3 +1.56, S4 +2.60 | yes |

## The artifact

All validated patterns live in the shared pattern registry with this schema:

```json
{
  "pattern": "<regex string>",
  "precision": 0.96,
  "sub_pattern_precisions": { "...": 0.98 },
  "by_period_precision": { "2024": 0.94, "2026": 0.98 },
  "fp_classes": ["HR metadata", "..."],
  "recommendation": "PRIMARY",
  "semantic_precision_measured": true,
  "precision_threshold_80_pass": true
}
```

Every downstream task reads from this registry; every published claim names the pattern it used and that pattern's measured precision.

## The underlying principle

A regex can have high "token-level" precision (it matches tokens related to its target) while having low semantic precision, because those token matches are not actually instances of the target concept. Word presence and meaning fidelity are different things.

The methodological contribution here is mostly this one protocol: **cite measured semantic precision per pattern, per period, per sub-pattern, based on a stratified 50-row manual audit**. Without this step, a content finding on a regex is an unverified claim about a regex.
