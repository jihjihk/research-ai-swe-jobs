"""T22 step 0 — Unit tests for ghost pattern regexes.

Edge-case assertions per the analytical preamble. These patterns are also the
ones saved to validated_mgmt_patterns.json and used in step 2 feature
extraction.
"""
import re

HEDGE = {
    "nice_to_have": r"nice[- ]to[- ]have",
    "preferred_hedge": r"(skills?|experience|knowledge|familiarity|qualification|background)[^.\n]{0,30}(preferred|a plus)",
    "preferred_alt": r"preferred[^.\n]{0,30}(skills?|experience|qualification|but not required)",
    "bonus_plus": r"\ba plus\b|\bis a plus\b|would be a plus",
    "ideally": r"\bideally\b",
    "familiarity_with": r"familiarity with",
}
FIRM = {
    "must_have": r"must[- ]have|must have",
    "required_req": r"(experience|knowledge|skills?|degree|qualification)[^.\n]{0,10}(is |are )?required|(required[^.\n]{0,10})(experience|knowledge|skills?|qualifications?)",
    "minimum_req": r"\bminimum[- ]of\b|\bat[- ]minimum\b|\bminimum qualification",
    "mandatory_req": r"\bmandatory\b",
}
MGMT = {
    "strict_mentor": r"mentor (engineers?|juniors?|team|developers?|interns?|the team)|coach engineers?|mentor(ing|ship) (engineers?|juniors?|team|developers?)",
    "strict_people_mgr": r"direct reports?|\bpeople manager\b|\bpeople management\b|performance reviews?",
    "strict_hire_mgmt": r"hire engineers?|hiring engineers?|grow the team|build (the |a |out )?team|lead hiring|own hiring|manage a team of|lead a team of",
}
AI_TOOL = r"\bcopilot\b|\bcursor\b|\bllm(s)?\b|prompt engineering|\blangchain\b|ai pair program|model context protocol"
AI_GENERAL = r"\bai\b|artificial intelligence"
AGENTIC = r"\bagentic\b"


def m(pat: str, text: str) -> bool:
    return bool(re.search(pat, text, flags=re.I))


# --- Hedge patterns ---
# Positive
assert m(HEDGE["preferred_hedge"], "experience with Kubernetes is preferred")
assert m(HEDGE["preferred_hedge"], "knowledge of React preferred")
assert m(HEDGE["preferred_alt"], "preferred skills: Kubernetes and Terraform")
assert m(HEDGE["bonus_plus"], "Kubernetes is a plus")
assert m(HEDGE["bonus_plus"], "Kubernetes a plus")
assert m(HEDGE["bonus_plus"], "Go experience would be a plus")
assert m(HEDGE["ideally"], "ideally with 2 years of Python")
assert m(HEDGE["familiarity_with"], "familiarity with React")
assert m(HEDGE["nice_to_have"], "nice to have: Rust")
assert m(HEDGE["nice_to_have"], "nice-to-have: Rust")
# Negative — we do NOT want customer/vendor to fire
assert not m(HEDGE["preferred_hedge"], "preferred customer rewards program")
assert not m(HEDGE["preferred_hedge"], "preferred vendor list")
# "a plus" alone without context is a hedge marker we accept
assert m(HEDGE["bonus_plus"], "Rust experience is a plus")

# --- Firm patterns ---
assert m(FIRM["must_have"], "must have 5 years of experience")
assert m(FIRM["must_have"], "must-have: Python")
assert m(FIRM["required_req"], "experience required: 5+ years")
assert m(FIRM["required_req"], "Required skills: Python, Go")
assert m(FIRM["required_req"], "required qualifications")
assert m(FIRM["minimum_req"], "minimum of 5 years")
assert m(FIRM["minimum_req"], "minimum qualifications include...")
assert m(FIRM["mandatory_req"], "mandatory skills: Python")
# Negative — "required field" is form boilerplate
assert not m(FIRM["required_req"], "* required field")

# --- Mgmt strict patterns ---
assert m(MGMT["strict_mentor"], "mentor junior engineers")
assert m(MGMT["strict_mentor"], "mentor developers")
assert m(MGMT["strict_mentor"], "coach engineers")
assert m(MGMT["strict_mentor"], "mentoring junior engineers")
assert m(MGMT["strict_people_mgr"], "you will have 3 direct reports")
assert m(MGMT["strict_people_mgr"], "people management experience")
assert m(MGMT["strict_people_mgr"], "conduct performance reviews")
assert m(MGMT["strict_hire_mgmt"], "lead a team of 5 engineers")
assert m(MGMT["strict_hire_mgmt"], "manage a team of 10")
assert m(MGMT["strict_hire_mgmt"], "hire engineers for the platform team")
assert m(MGMT["strict_hire_mgmt"], "help grow the team")
# Negative — naive "hire" boilerplate should NOT fire
assert not m(MGMT["strict_hire_mgmt"], "our hiring process takes 4 weeks")
assert not m(MGMT["strict_hire_mgmt"], "experience hiring vendors")

# --- AI patterns ---
assert m(AI_TOOL, "experience with Copilot and Cursor")
assert m(AI_TOOL, "LLM and prompt engineering")
assert m(AI_TOOL, "LangChain experience")
assert m(AI_TOOL, "Model Context Protocol (MCP)")
# Negative — bare mcp should NOT fire under refined ai_tool
assert not m(AI_TOOL, "Microsoft MCP certification")
assert not m(AI_TOOL, "Cisco MCP network management")

assert m(AI_GENERAL, "ai-driven perception")
assert m(AI_GENERAL, "Artificial Intelligence team")
assert m(AI_GENERAL, "ai/ml background")
# Negative — openai and naive words should not match since \bai\b requires word boundary
# Actually openai has no word boundary; test it doesn't match
assert not m(AI_GENERAL, "openai the company is a nonprofit")  # Wait, "openai" shouldn't match \bai\b BUT "the" has \ba... we need to check the actual text
# Actually "openai" followed by space -> the \ba\b is false at position i. But the text also contains 'a' as word (a nonprofit). Actually "ai" is NOT in "openai" with word boundaries (ai is inside, not at word boundary). Let me verify:
# The regex \bai\b requires a word boundary before 'a' and after 'i'. In "openai the", positions of 'a' in 'openai' are inside the word, so \b before 'a' is false. Good.
# But the text "a nonprofit" contains 'a' not 'ai'. Ok.
# Test the tricky case:
assert not m(r"\bai\b", "openai")  # \bai\b should not match inside openai

assert m(AGENTIC, "agentic AI engineer")
assert not m(AGENTIC, "agent code")
assert not m(AGENTIC, "insurance agent")

print("All pattern unit tests passed.")
