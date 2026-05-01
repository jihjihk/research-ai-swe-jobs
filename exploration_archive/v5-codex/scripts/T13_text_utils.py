from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


SECTION_PHRASES: dict[str, list[str]] = {
    "role_summary": [
        "role summary",
        "summary",
        "about the role",
        "about this role",
        "job summary",
        "position summary",
        "overview",
        "the role",
    ],
    "responsibilities": [
        "responsibilities",
        "responsibility",
        "what you'll do",
        "what you will do",
        "what you'll be doing",
        "duties",
        "your responsibilities",
        "day to day",
        "what the job involves",
    ],
    "requirements": [
        "requirements",
        "requirement",
        "qualifications",
        "qualification",
        "what you'll need",
        "what you will need",
        "what you'll bring",
        "what you bring",
        "minimum qualifications",
        "basic qualifications",
        "required qualifications",
        "essential qualifications",
    ],
    "preferred": [
        "preferred qualifications",
        "preferred",
        "nice to have",
        "nice-to-have",
        "desired experience",
        "bonus qualifications",
        "ideal candidate",
        "desired qualifications",
    ],
    "benefits": [
        "benefits",
        "perks",
        "compensation",
        "salary",
        "pay",
        "equity",
        "bonus",
        "401k",
        "dental",
        "pto",
        "insurance",
        "health insurance",
    ],
    "about_company": [
        "about the company",
        "about us",
        "who we are",
        "company overview",
        "our company",
        "why us",
        "mission",
        "values",
        "employees",
    ],
    "legal": [
        "equal opportunity",
        "equal employment opportunity",
        "eeo",
        "eeoc",
        "accommodation",
        "sponsorship",
        "visa",
        "privacy",
        "disclaimer",
        "background check",
    ],
}

CORE_SECTION_LABELS = {"role_summary", "responsibilities", "requirements", "preferred"}
BOILERPLATE_SECTION_LABELS = {"benefits", "about_company", "legal"}

SPECIAL_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bc\+\+\b"), "cplusplus"),
    (re.compile(r"(?i)\bc#\b"), "csharp"),
    (re.compile(r"(?i)\b\.net\b"), "dotnet"),
    (re.compile(r"(?i)\bnode\.?js\b"), "nodejs"),
    (re.compile(r"(?i)\bnext\.?js\b"), "nextjs"),
    (re.compile(r"(?i)\bci\s*/\s*cd\b"), "cicd"),
    (re.compile(r"(?i)\bai\s*/\s*ml\b"), "aiml"),
    (re.compile(r"(?i)\br&d\b"), "rnd"),
]

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[+#./-][a-z0-9]+)*")

TERM_CATEGORY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("boilerplate", ("salary", "benefit", "compensation", "pay", "equity", "bonus", "dental", "pto", "insurance", "culture", "mission", "values", "diversity", "inclusion", "sponsorship", "visa", "employees", "people")),
    ("ai_tool", ("copilot", "cursor", "claude", "gpt", "chatgpt", "llm", "rag", "mcp", "openai", "prompt", "fine tuning", "finetuning", "langchain", "langgraph", "gemini", "agentic", "ai agent", "vector db", "vector database", "hugging face", "codex")),
    ("ai_domain", ("machine learning", "deep learning", "nlp", "computer vision", "ai/ml", "aiml", "ai", "ml")),
    ("credential", ("years", "experience", "degree", "bachelor", "master", "phd", "certification", "certified", "license")),
    ("mgmt", ("manage", "management", "mentor", "coach", "hire", "hiring", "lead", "leadership", "team lead", "people", "direct reports", "performance review", "headcount")),
    ("org_scope", ("ownership", "own", "end to end", "end-to-end", "cross functional", "cross-functional", "stakeholder", "roadmap", "partner", "collaborate", "accountable", "deliverables", "scope", "autonomy")),
    ("sys_design", ("architecture", "architect", "distributed systems", "scalability", "scalable", "microservices", "system design", "fault tolerant", "high availability", "large scale", "platform")),
    ("method", ("agile", "scrum", "cicd", "ci cd", "test driven", "tdd", "kanban", "testing", "debugging", "code review", "devops", "release", "git", "version control")),
    ("tech_stack", ("python", "java", "javascript", "typescript", "go", "golang", "rust", "csharp", "cplusplus", "ruby", "kotlin", "swift", "scala", "php", "react", "angular", "vue", "nextjs", "nodejs", "django", "flask", "spring", "dotnet", "rails", "fastapi", "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "snowflake", "databricks", "dbt", "elasticsearch", "tensorflow", "pytorch", "scikit", "pandas", "numpy")),
    ("soft_skill", ("communication", "collaboration", "problem solving", "problem-solving", "teamwork", "interpersonal", "adaptable", "self starter", "self-starter", "critical thinking")),
]

TERM_PRETTY = {
    "cplusplus": "C++",
    "csharp": "C#",
    "dotnet": ".NET",
    "nodejs": "Node.js",
    "nextjs": "Next.js",
    "cicd": "CI/CD",
    "aiml": "AI/ML",
    "ai": "AI",
    "ml": "ML",
    "llm": "LLM",
    "gpt": "GPT",
    "mcp": "MCP",
    "ci": "CI",
    "cd": "CD",
}


def normalize_text(s: str) -> str:
    s = s.replace("’", "'").replace("`", "'")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_token(token: str) -> str:
    if token is None:
        return ""
    text = token.lower().replace("’", "'").replace("`", "'")
    for pattern, replacement in SPECIAL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if " " in text:
        return text
    return text


def normalize_stop_tokens(tokens: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for token in tokens:
        norm = normalize_token(token)
        if not norm:
            continue
        out.add(norm)
    return out


def load_stop_tokens(company_stoplist_path, location_tokens: Iterable[str] | None = None) -> set[str]:
    raw: set[str] = set()
    with open(company_stoplist_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw.add(line)
    if location_tokens:
        raw.update(location_tokens)
    protected = {"go", "r", "c", "dotnet", "nodejs", "nextjs", "cplusplus", "csharp", "cicd", "aiml"}
    normalized = normalize_stop_tokens(raw)
    return {tok for tok in normalized if tok not in protected}


def location_tokens_from_values(values: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = normalize_text(str(value))
        tokens.update(normalized.split())
    return tokens


def tokenize_for_terms(text: str, stop_tokens: set[str]) -> list[str]:
    if not text:
        return []
    text = text.lower()
    for pattern, replacement in SPECIAL_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = text.replace("’", "'").replace("`", "'")
    tokens = []
    for token in TOKEN_RE.findall(text):
        norm = normalize_token(token)
        if not norm:
            continue
        if norm in stop_tokens:
            continue
        if norm.isdigit():
            continue
        tokens.append(norm)
    return tokens


def tokenize_for_term_counts(text: str, stop_tokens: set[str], bigrams: bool = False) -> list[str]:
    tokens = tokenize_for_terms(text, stop_tokens)
    if not bigrams:
        return tokens
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def pretty_term(term: str) -> str:
    return " ".join(TERM_PRETTY.get(tok, tok) for tok in term.split())


def is_html_artifact(term: str) -> bool:
    raw = term.strip()
    if not raw:
        return True
    if any(x in raw for x in ("nbsp", "amp", "quot", "http", "www", "<", ">", "href", "class=")):
        return True
    if re.search(r"[<>]{2,}|_{2,}|-{3,}", raw):
        return True
    if re.fullmatch(r"[a-z]{1,3}[0-9]{3,}[a-z0-9]*", raw):
        return True
    if len(raw) > 30 and raw not in {"systems engineering", "machine learning", "distributed systems"}:
        return True
    return False


def term_category(term: str) -> str:
    normalized = normalize_text(term)
    if not normalized:
        return "noise"

    # Phrase-level checks first.
    for category, phrases in TERM_CATEGORY_PATTERNS:
        for phrase in phrases:
            p = normalize_text(phrase)
            if not p:
                continue
            if p in normalized:
                return category

    if normalized in {"ai", "ml"}:
        return "ai_domain"
    if normalized in {"llm", "gpt", "chatgpt", "claude", "copilot", "cursor", "rag", "mcp", "agent", "agents", "langchain", "langgraph"}:
        return "ai_tool"
    if normalized in {"python", "java", "javascript", "typescript", "go", "golang", "rust", "csharp", "cplusplus", "ruby", "kotlin", "swift", "scala", "php", "react", "angular", "vue", "nextjs", "nodejs", "django", "flask", "spring", "dotnet", "rails", "fastapi", "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "postgresql", "mysql", "mongodb", "redis", "kafka", "spark", "snowflake", "databricks", "dbt", "elasticsearch", "tensorflow", "pytorch", "pandas", "numpy"}:
        return "tech_stack"
    return "noise"


def make_header_regexes() -> list[tuple[str, re.Pattern[str]]]:
    regexes: list[tuple[str, re.Pattern[str]]] = []
    for label, phrases in SECTION_PHRASES.items():
        ordered = sorted(phrases, key=lambda x: len(normalize_text(x)), reverse=True)
        for phrase in ordered:
            phrase_norm = normalize_text(phrase)
            if not phrase_norm:
                continue
            parts = [re.escape(part) for part in phrase_norm.split()]
            phrase_pat = r"\s+".join(parts)
            regex = re.compile(
                rf"(?is)^\s*[\*\-\|:>•]*\s*(?:{phrase_pat})\b(?:\s*[:\-\|–—]\s*|\s+|$)(?P<rest>.*)$"
            )
            regexes.append((label, regex))
    return regexes


HEADER_REGEXES = make_header_regexes()


def detect_header_chunk(chunk: str) -> tuple[str, str] | None:
    if not chunk:
        return None
    raw = chunk.strip()
    if not raw:
        return None
    if len(raw) > 220:
        return None
    for label, regex in HEADER_REGEXES:
        m = regex.match(raw)
        if m:
            rest = m.group("rest").strip()
            if label == "role_summary" and normalize_text(raw) == "overview":
                # Keep generic overview bodies from false-triggering when they are clearly not headers.
                if len(rest.split()) > 15:
                    continue
            return label, rest
    return None


def _append_segment(segments: list[dict], label: str, parts: list[str], order: int) -> int:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return order
    segments.append(
        {
            "segment_order": order,
            "section_label": label,
            "section_text": text,
            "section_chars": len(text),
        }
    )
    return order + 1


def extract_sections(text: str) -> list[dict]:
    if not text:
        return []

    parts = re.split(r"(\*\*[^*]{0,120}?\*\*|\|)", text)
    segments: list[dict] = []
    current_label = "unclassified"
    current_parts: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order, current_parts
        order = _append_segment(segments, current_label, current_parts, order)
        current_parts = []

    for part in parts:
        if not part:
            continue
        stripped = part.strip()
        if not stripped or stripped == "|":
            continue

        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) >= 4:
            inner = stripped[2:-2].strip()
            if not inner:
                continue
            header = detect_header_chunk(inner)
            if header:
                label, rest = header
                flush()
                current_label = label
                if rest:
                    current_parts.append(rest)
                continue
            current_parts.append(inner)
            continue

        header = detect_header_chunk(stripped)
        if header:
            label, rest = header
            flush()
            current_label = label
            if rest:
                current_parts.append(rest)
            continue

        current_parts.append(stripped)

    flush()
    return segments


def section_group(label: str) -> str:
    if label in CORE_SECTION_LABELS:
        return "core"
    if label in BOILERPLATE_SECTION_LABELS:
        return "boilerplate"
    return "unclassified"


def core_text_from_sections(sections: list[dict]) -> str:
    pieces = [seg["section_text"] for seg in sections if seg["section_label"] in CORE_SECTION_LABELS]
    return " ".join(pieces).strip()


def count_section_chars(sections: list[dict]) -> dict[str, int]:
    out = defaultdict(int)
    for seg in sections:
        out[seg["section_label"]] += int(seg["section_chars"])
    return dict(out)

