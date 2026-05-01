"""V1.1b - Improved semantic classifier with manual-review-style rules.

Re-classify all previously-collected samples with deeper rules. Flag ambiguous
samples that need human eyes.
"""
import pandas as pd
import re
import json
from pathlib import Path

OUT_DIR = Path("/home/jihgaboot/gabor/job-research/exploration/artifacts/V1")

# AI context strong signals (single-term ideally confirms AI usage)
AI_STRONG = [
    r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bdeep learning\b",
    r"\bneural network\b", r"\bllm\b", r"\blarge language", r"\bgenerative",
    r"\bgenai\b", r"\bopenai\b", r"\banthropic\b", r"\bclaude\b", r"\bgpt\b",
    r"\bchatgpt\b", r"\bcopilot\b", r"\bcursor\b", r"\bhuggingface\b", r"\bhugging face\b",
    r"\bpytorch\b", r"\btensorflow\b", r"\btransformer\b", r"\brag\b",
    r"\bvector (database|search|store|embed)\b", r"\bembedding\b",
    r"\bfine[- ]tun", r"\bprompt engineer", r"\bagentic\b", r"\bmulti[- ]agent\b",
    r"\bnatural language\b", r"\bnlp\b", r"\bmlops\b", r"\bml[- ]ops\b",
    r"\bai[- ]driven\b", r"\bai[- ]assist", r"\bai[- ]power",
    r"\bai[- ]model", r"\bai[- ]tool", r"\bai[- ]system", r"\bai[- ]platform",
    r"\bai[- ]team\b", r"\bai[- ]engineer\b", r"\bml[- ]engineer\b",
    r"\bai[/ ]ml\b", r"\bml[/ ]ai\b", r"\bai engineer\b", r"\bai scient",
    r"\bml model\b", r"\bml engineer\b", r"\bml platform\b", r"\bml algorithm\b",
    r"\bml pipeline\b", r"\bml system\b", r"\bml framework\b",
    r"\bdata scien", r"\bdata engin",
    r"\brecommender\b", r"\brecommendation system",
    r"\bcomputer vision\b", r"\bspeech recognition\b",
    r"\bdata lifecycle\b", r"\bmodel (train|infer|deploy|serving)",
    r"\bkubeflow\b", r"\bmlflow\b", r"\bsagemaker\b", r"\bbedrock\b",
    r"\bairflow.*data\b", r"\bdata.*pipeline",
    r"\bjupyter\b", r"\bnotebook",
    r"\bai governance\b", r"\bai safety\b", r"\bai ethics\b",
    r"\bresponsible ai\b", r"\bethical ai\b",
    r"\bai product\b", r"\bai service\b", r"\bai application\b",
    r"\bai offering", r"\bai capabili", r"\bai benchmark", r"\bai-based\b",
    r"\brag\b", r"\blangchain\b", r"\bllamaindex\b", r"\bpinecone\b",
]

# Non-AI false-positive signals per sub-term
FP_SIGNALS = {
    "ai_bare": [
        r"\bjai alai\b", r"\bbaik-ai\b", r"\bcheng-ai\b", r"\bhua-ai\b",
        r"\bthai ?land\b", r"\bshang(-?)hai\b", r"\bdubai\b",
    ],
    "ml_bare": [
        r"\bhtml\b", r"\bxml\b", r"\byaml\b", r"\bhaml\b", r"\bkml\b",
        r"\b\.ml\b(?!\s*(ai|machine|model))",  # .ml domain
    ],
    "agent_bare": [
        r"\binsurance agent\b", r"\breal[- ]?estate agent\b", r"\bsales agent\b",
        r"\btravel agent\b", r"\bshipping agent\b", r"\bcall center agent\b",
        r"\bcustomer service agent\b", r"\bmortgage agent\b",
        r"\buser agent\b", r"\buseragent\b", r"\buser-agent\b",
        r"\bnessus agent\b", r"\bsql agent\b", r"\bbrowser agent\b",
        r"\bfree agent\b", r"\bservice agent\b",
        r"\bin[- ]?service engineering agent\b", r"\bisea\b",
        r"\blaw enforcement agent\b",
    ],
    "mcp": [
        r"\bmcp/mcsa\b", r"\bmcse\b", r"\bmcp\+\b",
        r"\bmicrosoft certified\b",
        r"\bmcps\b",  # mechanical plural
        r"\bmcp, mcsa\b",
        r"\bmcp/mce\b", r"\bmcp in c#\b", r"\bmcp in .net\b",
        r"\bmcp in asp\b",
        r"\b(ccna|ccnp|ccea|ccia|cssp|security\+|itil|rhsa|vcp|vca)\b.*\bmcp\b|\bmcp\b.*\b(ccna|ccnp|ccea|ccia|cssp|security\+|itil|rhsa|vcp|vca)\b",
        r"\bmarketing cloud personalization\b",
        r"\bmcp (certified|certification)\b",
    ],
    "rag": [
        r"\brag(s|ged|time)\b", r"\bclothing rag\b", r"\btags\b",
    ],
    "gpt": [
        r"\bgpt partition\b", r"\bgpt[- ]disk\b", r"\buefi\b", r"\bmbr\b",
    ],
    "cursor": [
        r"\bsql cursor\b", r"\bdatabase cursor\b", r"\bdb cursor\b",
        r"\bcursor[- ]based pagination\b", r"\bmouse cursor\b",
        r"\bcursor pointer\b",
    ],
    "codex": [
        r"\bcodex alimentarius\b", r"\billuminated manuscript\b",
        r"\bbiblical codex\b",
    ],
    "gemini": [
        r"\bgemini (horoscope|zodiac)\b", r"\bgemini constellation\b",
    ],
}

# Load
df = pd.read_csv(OUT_DIR / "V1_1_classifications.csv")


def manual_classify(row):
    context = str(row["context"]).lower()
    sub = row["subterm"]
    matched = str(row["matched_text"]).lower()

    # Check for FP signals specific to the subterm
    fp_patterns = FP_SIGNALS.get(sub, [])
    for fp in fp_patterns:
        if re.search(fp, context):
            return "false_positive"

    # Check for strong AI signals
    for s in AI_STRONG:
        if re.search(s, context):
            return "true_positive"

    # Fallback: subterm-specific heuristics
    if sub in ("ai_bare",):
        # "ai" as standalone in tech-listing: "cloud, ai, and devops", "ai and ml", "ai/ml"
        if re.search(r"ai[ ,/]+ml|ml[ ,/]+ai|ai[/ ]driven|ai[/ ]based|ai[/ ]power", context):
            return "true_positive"
        # "ai and <tech>": "ai and data", "ai and machine", "ai and cloud"
        if re.search(r"\bai (and|or) (ml|ml[/ ]|machine|data|cloud|analytic|automation|platform|tool|technology|model|engineering|system)", context):
            return "true_positive"
        # "(tech) and ai": "cloud and ai", "python and ai", "data and ai"
        if re.search(r"\b(cloud|python|data|analytic|ml|java|scala|software|engineer|platform|system|technolog|automat) (and|or) ai\b", context):
            return "true_positive"
        # "ai team", "ai product", "ai service", "ai role", "ai application"
        if re.search(r"\bai (team|product|service|role|application|capabilit|feature|stack|engineer|architect|research|platform|offering|use case|space|solution|tool|model|workflow|pipeline|based)\b", context):
            return "true_positive"
        # leverage ai / using ai / building ai / with ai
        if re.search(r"\b(leverage|using|build|with|into|integrate|involve|powered by|powered-by|around|through|drive|driven by) ai\b", context):
            return "true_positive"
        # "in ai" or "on ai"
        if re.search(r"\b(experience|expertise|knowledge|work|research|skill|career) (in|on|with) ai\b", context):
            return "true_positive"
        # "ai industry", "ai space"
        if re.search(r"\bai (industry|space|market|field|community)\b", context):
            return "true_positive"
        # "advanc(e/ing/ed) ai"
        if re.search(r"\bai (advance|progress|innovat|transform|revolut)", context):
            return "true_positive"
        # fallback ambiguous
        return "ambiguous"
    if sub in ("ml_bare",):
        # "ml" as standalone in tech-listing
        if re.search(r"ml[ ,/]+ai|ai[ ,/]+ml|\bml algorithm\b|\bml team\b|\bml engineer\b|\bml model\b|\bml system\b|\bml platform\b|\bml framework\b|\bml pipeline\b|\bml ops\b|\bmlops\b", context):
            return "true_positive"
        if re.search(r"\b(generative|applied|advanced) ml\b", context):
            return "true_positive"
        # "ml tasks", "ml lifecycle", "ml training"
        if re.search(r"\bml (task|lifecycle|training|service|product|workload|research|infrastructur)", context):
            return "true_positive"
        # in ml
        if re.search(r"\b(in|on|with) ml\b", context):
            return "true_positive"
        if re.search(r"\bml[/ ]", context) and re.search(r"\b(algorithm|model|data|train|infer|predict|pipeline|framework|engineer)\b", context):
            return "true_positive"
        # "data or ml tasks", "ai/ml"
        return "ambiguous"
    if sub in ("agent_bare",):
        # "ai agent", "llm agent", "agentic", "autonomous agent", "agent framework"
        if re.search(r"\b(ai|llm) agent\b|\bagentic\b|\bautonomous agent\b|\bagent framework\b|\bbuild agent|\bmulti[- ]agent\b|\bagent benchmark|\bagent workflow\b|\bweb-based agent\b|\bagent (system|service|platform|tool|orchestr|network|protocol|architecture|based|handler|behavior|based)\b", context):
            return "true_positive"
        # "develop agents", "agent capabilities"
        if re.search(r"\b(develop|build|create|design|deploy|train|evaluate|orchestrate) (ai )?agent", context):
            return "true_positive"
        if re.search(r"\bagent (api|capabilit|workflow|interact|system|platform|infrastructure|ecosystem|strategy)\b", context):
            return "true_positive"
        if re.search(r"\b(agent|agents) with (llm|ai|gpt|claude|gemini|mcp)\b", context):
            return "true_positive"
        if re.search(r"\bai[/ ]", context) and re.search(r"\bagent\b", context):
            return "true_positive"
        if re.search(r"\bagent benchmark", context):
            return "true_positive"
        return "ambiguous"
    if sub in ("mcp",):
        # Model Context Protocol: "mcp protocol", "mcp server", "mcp anthropic", "mcp tool"
        if re.search(r"\bmcp (server|protocol|tool|agent|anthropic|claude|ai|llm|based|integration|client|compatible|ecosystem|interface|specification)\b", context):
            return "true_positive"
        if re.search(r"\bmodel context protocol\b", context):
            return "true_positive"
        if re.search(r"\banthropic.*mcp\b|\bmcp.*anthropic\b", context):
            return "true_positive"
        if re.search(r"\bmcp[- ]enabled\b|\bmcp[- ]compatible\b", context):
            return "true_positive"
        return "ambiguous"
    # Default
    return "ambiguous"


df["label_v2"] = df.apply(manual_classify, axis=1)


def compute_precision(sub_rows, label_col="label_v2"):
    tp = (sub_rows[label_col] == "true_positive").sum()
    fp = (sub_rows[label_col] == "false_positive").sum()
    amb = (sub_rows[label_col] == "ambiguous").sum()
    n = tp + fp + amb
    # Conservative: precision = TP / (TP + FP + AMB)
    p_cons = tp / n if n > 0 else 0
    # Non-conservative: precision = TP / (TP + FP), ambiguous excluded
    p_excl = tp / (tp + fp) if (tp + fp) > 0 else 0
    return p_cons, p_excl, (tp, fp, amb)


print("=== V2 per-sub-term precision ===")
print(f"{'subterm':30s}  {'p_cons':>7s}  {'p_excl':>7s}  {'TP':>4s} {'FP':>4s} {'AMB':>4s} {'N':>4s}")
stats = {}
for sub in df["subterm"].unique():
    sub_rows = df[df["subterm"] == sub]
    p_cons, p_excl, (tp, fp, amb) = compute_precision(sub_rows)
    star = " <-- FAIL" if p_cons < 0.80 else ""
    print(f"{sub:30s}  {p_cons:7.2f}  {p_excl:7.2f}  {tp:>4d} {fp:>4d} {amb:>4d} {len(sub_rows):>4d}{star}")
    stats[sub] = {"p_cons": p_cons, "p_excl": p_excl, "tp": int(tp), "fp": int(fp), "amb": int(amb), "n": int(len(sub_rows))}

# Save
df.to_csv(OUT_DIR / "V1_1b_classifications_v2.csv", index=False)

# Also save ambiguous for manual review
amb_df = df[df["label_v2"] == "ambiguous"]
amb_df.to_csv(OUT_DIR / "V1_1b_ambiguous_for_review.csv", index=False)
print(f"\nSaved {len(amb_df)} ambiguous for manual review")

with open(OUT_DIR / "V1_1b_precision_v2.json", "w") as f:
    json.dump(stats, f, indent=2)
