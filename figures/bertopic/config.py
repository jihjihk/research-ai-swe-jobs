"""
Frozen single-source-of-truth for the BERTopic discovery project.

Hyperparameters, seeds, sample-cap rules, LLM model pins, anchor sets, and
paths used by every script in `figures/bertopic/`. The design doc
(`figures/bertopic/design.md`) is the canonical specification; this file is
the pre-registered, code-loadable form of §4.2, §4.5, §4.6, §5.1, §11.4,
§11.5, §11.7, and §11.9.

No script in this package may override a value defined here; figures and
analyses import from this module. Any change to a frozen value requires a
paper-visible erratum entry in `prereg_log.md`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (PROJECT_ROOT-derived; no hard-coded absolute paths anywhere)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UNIFIED_CORE_PATH = DATA_DIR / "unified_core.parquet"

BERTOPIC_DATA_DIR = DATA_DIR / "bertopic"
EMBEDDINGS_CACHE_PATH = BERTOPIC_DATA_DIR / "embeddings_cache.npy"
EMBEDDINGS_INDEX_PATH = BERTOPIC_DATA_DIR / "embeddings_cache.index.parquet"
MODEL_PATH = BERTOPIC_DATA_DIR / "model.bertopic"
ASSIGNMENTS_PATH = BERTOPIC_DATA_DIR / "assignments.parquet"
TOPIC_INFO_PATH = BERTOPIC_DATA_DIR / "topic_info.parquet"

PACKAGE_DIR = PROJECT_ROOT / "figures" / "bertopic"
INTERMEDIATE_DIR = PACKAGE_DIR / "intermediate"
MEMOS_DIR = PACKAGE_DIR / "memos"

SAMPLE_A_PATH = INTERMEDIATE_DIR / "sample_a.parquet"
SAMPLE_B_PATH = INTERMEDIATE_DIR / "sample_b.parquet"
SAMPLE_SIZES_CSV = INTERMEDIATE_DIR / "sample_sizes.csv"
STAGE1_FREEZE_JSON = INTERMEDIATE_DIR / "stage1_freeze.json"
MCS_SWEEP_CSV = INTERMEDIATE_DIR / "mcs_sweep.csv"
K_SWEEP_CSV = INTERMEDIATE_DIR / "k_sweep.csv"
RAW_FIT_PATH = INTERMEDIATE_DIR / "raw_fit.bertopic"

PREREG_LOG_PATH = PACKAGE_DIR / "prereg_log.md"

OPENAI_ENV_FILE = Path("~/.config/job-research/openai.env").expanduser()


# ---------------------------------------------------------------------------
# Substrate convention (§2.3 + 2026-05-05 user clarification)
# ---------------------------------------------------------------------------

SUBSTRATE_COLUMN = "description_core_llm"
SUBSTRATE_MIN_LENGTH = 200  # §2.4 length floor


# ---------------------------------------------------------------------------
# Embedding model (§2.1, §10.6)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072
EMBEDDING_NORM_TOLERANCE = (0.99, 1.01)  # §13.1 S0.4 pre-flight


# ---------------------------------------------------------------------------
# LLM models (§5.1, §11.6)
# ---------------------------------------------------------------------------

LLM_MODEL_PRIMARY = "gpt-5.5-2026-04-23"  # cluster naming, < 100 calls per pass
LLM_MODEL_SECONDARY = "gpt-5.4-mini"  # cross-model sanity (§7.11), large passes
LLM_MAX_CALLS_PRIMARY = 100  # threshold above which we switch to LLM_MODEL_SECONDARY


# ---------------------------------------------------------------------------
# Seeds (§11.4) — no silent reseeding
# ---------------------------------------------------------------------------

SEED_PRIMARY = 42
SEED_STABILITY = (1337, 2026)
ALL_SEEDS = (SEED_PRIMARY, *SEED_STABILITY)


# ---------------------------------------------------------------------------
# Sample-cap rules (§3.1)
# ---------------------------------------------------------------------------

PER_BUCKET_CAP = 5  # per (canonical_co, period, title_normalized)


# ---------------------------------------------------------------------------
# UMAP hyperparameters (§4.2)
# ---------------------------------------------------------------------------

UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"


# ---------------------------------------------------------------------------
# HDBSCAN hyperparameters (§4.2, §4.6)
# ---------------------------------------------------------------------------

HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"
HDBSCAN_PREDICTION_DATA = True

# §4.6 sweep grid; headline value is committed after the sweep
MCS_SWEEP_GRID = (10, 20, 30, 50, 70)
MCS_INITIAL = 30
MCS_NOISE_RANGE = (0.15, 0.35)  # acceptable raw noise rate
MCS_PLATEAU_ARI = 0.7  # post-reduction ARI vs neighbors

# min_samples = round(0.5 * min_cluster_size); pinned rule, not swept


# ---------------------------------------------------------------------------
# CountVectorizer / c-TF-IDF (§4.2)
# ---------------------------------------------------------------------------

NGRAM_RANGE = (1, 3)
COUNT_MIN_DF = 10
COUNT_MAX_DF = 0.4
TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-\+/\.]+\b"


# ---------------------------------------------------------------------------
# Custom stopwords (§4.5) — frozen pre-fit; no additions during analysis
# ---------------------------------------------------------------------------

CUSTOM_STOPWORDS = (
    # Substrate residue
    "description", "responsibilities", "qualifications", "requirements",
    "position", "role", "team", "work", "looking", "seeking", "candidate",
    "experience",
    # Boilerplate that survived L9 stripping
    "equal", "opportunity", "affirmative", "disability", "veteran",
    "protected", "sexual", "gender", "race", "religion", "applicant",
    "employee",
    # Generic recruiter-CTA
    "apply", "resume", "interview", "hiring", "recruit", "recruiting",
    # Pure-noise tokens observed in the v3 run
    "etc", "eg", "ie", "including", "e.g", "i.e",
)


# ---------------------------------------------------------------------------
# Representation models (§4.2)
# ---------------------------------------------------------------------------

MMR_DIVERSITY = 0.3
KEYBERT_TOP_N_WORDS = 10


# ---------------------------------------------------------------------------
# BERTopic top-level (§4.2)
# ---------------------------------------------------------------------------

CALCULATE_PROBABILITIES = False
APPROX_DIST_WINDOW = 8
APPROX_DIST_STRIDE = 4
APPROX_DIST_MIN_SIM = 0.1
APPROX_DIST_PADDING = False


# ---------------------------------------------------------------------------
# K sweep grid (§4.4)
# ---------------------------------------------------------------------------

K_SWEEP_GRID = (10, 15, 20, 25, 30, 40, 50, 75)
SUPER_FAMILY_K_RANGE = (10, 15)


# ---------------------------------------------------------------------------
# Validation thresholds (§11.9)
# ---------------------------------------------------------------------------

PERIOD_REPRO_MIN = 0.85
SEED_ARI_MIN = 0.4
WITHIN_2024_MIN = 0.85
OUTLIER_RATE_MAX = 0.40
LARGEST_CLUSTER_SHARE_MAX = 0.30
AXIS_LOO_SPREAD_MAX = 0.10
HELDOUT_HIT_RATE_MIN = 0.80
WEAT_COHEN_D_MIN = 0.5
WEAT_P_BONF_MAX = 0.01

# Headline-K selection criteria (§4.4)
HEADLINE_INTERP_MIN = 3.5  # author-rated, 1–5
HEADLINE_OUTLIER_MAX = 0.40

# Effect-size thresholds for §6 (§13.5 Gate 2)
AXIS_PERIOD_SHIFT_MIN = 0.05  # cosine units
AXIS_LOO_RATIO_MIN = 3.0  # period-shift / LOO sensitivity
BOUNDARY_FRACTION_DELTA_MIN = 0.05  # 5pp
BOUNDARY_PERMUTATION_P_MAX = 0.05
DRIFT_VS_CONTROL_MIN = 2.0  # SWE drift / control drift on same axis


# ---------------------------------------------------------------------------
# §6.5 anchor-neighborhood thresholds
# ---------------------------------------------------------------------------

ANCHOR_NEIGHBORHOOD_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


# ---------------------------------------------------------------------------
# §11.7 anchor sets — ANCHOR STRINGS BELOW MUST MATCH design.md §11.7
#   character-for-character. Any change is a paper-visible erratum.
# ---------------------------------------------------------------------------

# §6.1 axis anchors — six per pole, embedded individually, axis is PC1 of the
# difference set per §6.1.

AXIS_ANCHORS: dict[str, dict[str, tuple[str, ...]]] = {
    "ai_native_vs_traditional": {
        "positive": (
            "Build LLM agents and RAG pipelines.",
            "Fine-tune foundation models for production.",
            "Develop multi-agent orchestration systems.",
            "Build evals for LLM applications.",
            "Integrate vector databases for retrieval.",
            "Apply prompt engineering to production systems.",
        ),
        "negative": (
            "Maintain enterprise CRUD applications.",
            "On-call rotation for legacy services.",
            "Bug fixes in established codebase.",
            "Operate monitoring for existing systems.",
            "Implement features in mature framework.",
            "Maintain integration with internal tools.",
        ),
    },
    "ic_vs_management": {
        "positive": (
            "Design and implement features end-to-end.",
            "Write production code daily.",
            "Debug and ship.",
            "Pair-program with peers.",
            "Own technical decisions for a service.",
            "Build and ship a feature alone.",
        ),
        "negative": (
            "Manage a team of engineers.",
            "Hire and grow staff.",
            "Conduct performance reviews.",
            "Set roadmap and priorities.",
            "Lead 1:1s and career development.",
            "Drive headcount planning.",
        ),
    },
    "builder_vs_operator": {
        "positive": (
            "Build new systems from scratch.",
            "Greenfield architecture.",
            "Prototype and iterate quickly.",
            "Ship novel features to users.",
            "Design new APIs.",
            "Architect a new platform.",
        ),
        "negative": (
            "Operate and maintain existing systems.",
            "Reliability engineering.",
            "Incident response and post-mortems.",
            "Capacity planning for production.",
            "Migrate legacy infrastructure.",
            "Patch and harden existing services.",
        ),
    },
    "concrete_vs_abstract": {
        "positive": (
            "Implement specific feature using named tools.",
            "Write code in Python for a service.",
            "Build a recommendation component.",
            "Add OAuth flow to the user portal.",
            "Use Kubernetes for deployment.",
            "Ship the dashboard by end of quarter.",
        ),
        "negative": (
            "Drive technical strategy and direction.",
            "Evangelize engineering culture.",
            "Set company-wide standards.",
            "Define long-term technical vision.",
            "Influence cross-functional priorities.",
            "Champion best practices org-wide.",
        ),
    },
    "generalist_vs_specialist": {
        "positive": (
            "Polyglot fullstack across frontend, backend, infra.",
            "Comfortable across the stack.",
            "Wear multiple hats in a startup.",
            "Handle anything from UI to database.",
            "Ship across product surfaces.",
            "Cross-domain delivery.",
        ),
        "negative": (
            "Deep expert in Kubernetes.",
            "Specialist in distributed databases.",
            "10+ years on the iOS platform.",
            "Recognized authority on COBOL mainframe.",
            "Focus exclusively on the search infrastructure.",
            "Master of CUDA optimization.",
        ),
    },
}


# §6.5 anchor descriptions — single-string anchors, embedded individually.
NEIGHBORHOOD_ANCHORS: dict[str, str] = {
    "ai_engineer": "A SWE who builds RAG agents and integrates LLMs into production.",
    "frontend_engineer": "A SWE who builds React UIs and ships features to web users.",
    "backend_engineer": "A SWE who designs APIs and operates server infrastructure.",
    "legacy_specialist": "A SWE who maintains a COBOL mainframe codebase.",
    "sre": "A SWE who runs reliability for production systems.",
}


# §6.4 WEAT attribute sets — six anchor sentences per set, averaged per the
# WEAT-style attribute construction.
WEAT_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "innovation": (
        "Pioneer new technology.",
        "Prototype novel ideas.",
        "Push the state of the art.",
        "Build greenfield products.",
        "Explore new methods.",
        "Pioneer ML applications.",
    ),
    "maintenance": (
        "Keep existing systems running.",
        "Patch and stabilize.",
        "Reduce technical debt.",
        "Operate at high reliability.",
        "Maintain backward compatibility.",
        "Sustain mature systems.",
    ),
    "orchestration_mentorship": (
        "Decompose work for engineers.",
        "Author specs and ADRs.",
        "Mentor junior engineers.",
        "Onboard new team members.",
        "Pair-program for knowledge transfer.",
        "Lead through influence.",
    ),
    "solo_ic": (
        "Ship features individually.",
        "Own a service end-to-end.",
        "Code without supervision.",
        "Independently debug production issues.",
        "Self-direct daily work.",
        "Single-author technical decisions.",
    ),
    "architecture_design": (
        "Design system architecture.",
        "Author technical RFCs.",
        "Decompose monoliths.",
        "Design APIs for scale.",
        "Plan platform evolution.",
        "Architect cross-service flows.",
    ),
    "implementation": (
        "Write production code.",
        "Implement well-defined tickets.",
        "Add features to an existing module.",
        "Refactor specified code.",
        "Ship implementation tasks.",
        "Code against a written spec.",
    ),
    "growth": (
        "Scale the platform.",
        "Expand customer base.",
        "Drive product growth.",
        "Launch new markets.",
        "Grow capacity.",
        "Increase throughput.",
    ),
    "stability": (
        "Maintain existing customers.",
        "Preserve uptime.",
        "Reduce churn.",
        "Stabilize releases.",
        "Sustain SLAs.",
        "Lock in current revenue.",
    ),
    "exploration": (
        "Investigate new approaches.",
        "Experiment with techniques.",
        "Pilot novel architectures.",
        "Explore new vendors.",
        "Try new tools.",
        "Investigate the unknown.",
    ),
    "exploitation": (
        "Optimize known approach.",
        "Tune existing systems.",
        "Refine current methods.",
        "Squeeze performance from production code.",
        "Improve known metrics.",
        "Maximize current returns.",
    ),
}


# §6.4 WEAT pre-registered tests — name : (target_X, target_Y, attr_A, attr_B)
WEAT_TESTS: dict[str, tuple[str, str, str, str]] = {
    "ai_vs_legacy_innovation_vs_maintenance": (
        "ai_clusters",  # X — populated at run time from cluster catalog
        "legacy_clusters",
        "innovation",
        "maintenance",
    ),
    "period_orchestration_vs_solo_ic": (
        "period_2026",
        "period_2024",
        "orchestration_mentorship",
        "solo_ic",
    ),
    "seniority_architecture_vs_implementation": (
        "senior_swe",
        "junior_swe",
        "architecture_design",
        "implementation",
    ),
    "ai_growth_vs_stability": (
        "ai_clusters",
        "non_ai_clusters",
        "growth",
        "stability",
    ),
    "ai_exploration_vs_exploitation": (
        "ai_clusters",
        "non_ai_clusters",
        "exploration",
        "exploitation",
    ),
}


# Stable string IDs for every anchor stored in `embeddings_cache.npy`. Used as
# the lookup key in `embeddings_cache.index.parquet`. Each axis anchor is keyed
# `axis::<axis_name>::<pole>::<index>`; each neighborhood anchor is keyed
# `neighborhood::<role>`; each WEAT attribute is keyed `weat::<set>::<index>`.
ANCHOR_ID_PREFIX_AXIS = "axis"
ANCHOR_ID_PREFIX_NEIGHBORHOOD = "neighborhood"
ANCHOR_ID_PREFIX_WEAT = "weat"


def all_anchor_strings() -> dict[str, str]:
    """Return every anchor string keyed by its stable anchor-id.

    The set of strings returned here is the complete, frozen set of anchors;
    `embedding_cache.py` calls this once and embeds every value.
    """
    out: dict[str, str] = {}
    for axis, poles in AXIS_ANCHORS.items():
        for pole, strings in poles.items():
            for i, s in enumerate(strings):
                out[f"{ANCHOR_ID_PREFIX_AXIS}::{axis}::{pole}::{i}"] = s
    for role, s in NEIGHBORHOOD_ANCHORS.items():
        out[f"{ANCHOR_ID_PREFIX_NEIGHBORHOOD}::{role}"] = s
    for set_name, strings in WEAT_ATTRIBUTES.items():
        for i, s in enumerate(strings):
            out[f"{ANCHOR_ID_PREFIX_WEAT}::{set_name}::{i}"] = s
    return out


# ---------------------------------------------------------------------------
# Smoke-test slice (§13.1 S0.5)
# ---------------------------------------------------------------------------

SMOKE_TEST_FRACTION = 0.05
