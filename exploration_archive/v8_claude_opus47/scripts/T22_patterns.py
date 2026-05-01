"""T22 pattern definitions.

Canonical V1-refined AI patterns plus T22-specific aspiration / firm /
org-scope / credential-contradiction regexes. Every pattern in this file
is what we CITE. Precision numbers are MEASURED on stratified 50-sample
semantic checks and written into the JSON artifact downstream.
"""

# V1-refined strict AI-tool pattern (SNR 38.7 in V1). Dropped: mcp, agent_bare.
AI_STRICT_REGEX = (
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face)\b"
)

# V1-refined broad AI pattern. Dropped: agent_bare, mcp.
AI_BROAD_REGEX = (
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|codex|"
    r"llamaindex|langchain|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face|"
    r"ai|artificial intelligence|ml|machine learning|llm|large language model|"
    r"generative ai|genai|anthropic)\b"
)

# AI-as-tool: specific tools that developers use day-to-day (narrow).
AI_TOOL_REGEX = (
    r"\b(copilot|cursor|claude|chatgpt|openai api|gpt-?\d+|gemini|"
    r"langchain|llamaindex|prompt engineering|fine[- ]tuning|rag|"
    r"vector database|pinecone|huggingface|hugging face)\b"
)

# AI-as-domain: traditional ML/NLP/CV vocabulary.
AI_DOMAIN_REGEX = (
    r"\b(machine learning|ml|deep learning|nlp|natural language processing|"
    r"computer vision|cv|data science|model training|"
    r"neural network|gradient|transformer|embedding)\b"
)

# Hedging / aspirational language. NOTE: unrestricted "plus" is too noisy
# (common as a math/tool modifier). We require a requirement-context phrase.
ASPIRATION_REGEX = (
    r"\b("
    r"nice to have|"
    r"would be a plus|"
    r"is a plus|"
    r"are a plus|"
    r"big plus|"
    r"a plus but not required|"
    r"plus but not required|"
    r"bonus points|"
    r"a bonus|"
    r"is a bonus|"
    r"preferred(?:\s+qualification)?|"
    r"preferred but not required|"
    r"ideally|"
    r"ideal candidate|"
    r"would be nice|"
    r"helpful but not required|"
    r"we would love|"
    r"we'd love|"
    r"exposure to|"
    r"familiarity with|"
    r"working knowledge of|"
    r"nice-to-have|"
    r"desirable|"
    r"beneficial"
    r")\b"
)

# Firm-requirement language.
FIRM_REGEX = (
    r"\b("
    r"must have|"
    r"must be|"
    r"must possess|"
    r"you must|"
    r"required(?!\s+(?:by|to\s+apply))|"
    r"requirement(?:s)?|"
    r"minimum (?:qualification|requirement|of|experience|years)|"
    r"mandatory|"
    r"essential(?:\s+qualification)?|"
    r"non[- ]negotiable|"
    r"strictly required|"
    r"hard requirement|"
    r"basic qualification"
    r")\b"
)

# Organizational scope cues (mirror V1-era requirement_breadth components).
ORG_SCOPE_REGEX = (
    r"\b("
    r"stakeholder(?:s)?|"
    r"cross[- ]functional|"
    r"leadership|"
    r"strategic|"
    r"vision|"
    r"roadmap|"
    r"mentor(?:ing|ship)?|"
    r"influence|"
    r"drive\s+(?:the|our|business|product|strategy)|"
    r"partner(?:ship|ing)?\s+with|"
    r"executive|"
    r"c[- ]?suite|"
    r"align(?:ment)?|"
    r"org(?:anization)?\b"
    r")\b"
)

# Strict management pattern — V1-refined. Narrowed "hire" to forms that
# clearly denote the job-holder hiring others (not being hired).
MGMT_STRICT_REGEX = (
    r"\b("
    r"mentor(?:s|ed|ing|ship)?|"
    r"coach(?:es|ed|ing)?|"
    r"(?:hire\s+and\s+(?:develop|manage|grow))|"
    r"(?:hire(?:\s+a)?\s+team)|"
    r"(?:hiring\s+(?:and|manager|engineers|team|plan))|"
    r"headcount|"
    r"performance[- ]review"
    r")\b"
)

# Credential impossibility patterns (used in rule-based contradictions).
DEGREE_NO_REGEX = r"\b(no degree required|degree not required|degree optional)\b"
DEGREE_MS_REGEX = r"\b(m\.?s\.?\s+required|master[’']?s\s+required|master[’']?s\s+degree\s+required|phd required)\b"
DEGREE_ANY_REGEX = (
    r"\b(bachelor(?:'?s)?|master(?:'?s)?|b\.?s\.?|m\.?s\.?|phd|doctorate|ph\.?d\.?)\b"
)
