"""Agent Prep step 2: build the cleaned-text artifact.

For SWE LinkedIn rows under the default filter:
- description_cleaned = description_core_llm if labeled, else raw description
- text_source in {'llm','raw'}
- unescape markdown-escaped punctuation, lowercase, collapse whitespace
- strip company name tokens from the stoplist

Save parquet with the full schema required by Wave 2.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/jihgaboot/gabor/job-research")
STOPLIST_PATH = ROOT / "exploration/artifacts/shared/company_stoplist.txt"
OUT_PATH = ROOT / "exploration/artifacts/shared/swe_cleaned_text.parquet"
PARQUET_IN = ROOT / "data/unified.parquet"

# Markdown-escape regex: unescape \+ \- \# \. \& \_ \( \) \[ \] \{ \} \! \*
UNESCAPE_RE = re.compile(r"\\([+\-#.&_()\[\]\{\}!*])")
# Whitespace collapse
WS_RE = re.compile(r"\s+")
# Tokenizer for stoplist stripping — preserves punctuation like . + # by only
# splitting on whitespace. We only remove words that exactly match company
# tokens; the tech regex works on word-internal punctuation which will survive.
SPLIT_WS_RE = re.compile(r"(\s+)")


def _unescape(text: str) -> str:
    return UNESCAPE_RE.sub(r"\1", text)


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = _unescape(text)
    text = text.lower()
    text = WS_RE.sub(" ", text).strip()
    return text


def _make_stripper(stopset: frozenset[str]) -> callable:
    """Build a function that strips whitespace-tokens whose punctuation-
    stripped form is in the global stopset OR in the per-row company-name
    token set passed as `local_set`. Tech-like characters (`+#.`) are kept
    inside tokens, so `c++` and `.net` survive.
    """
    strip_edge = re.compile(r"^[^a-z0-9+#.]+|[^a-z0-9+#.]+$")

    def strip_tokens(text: str, local_set: frozenset[str] = frozenset()) -> str:
        if not text:
            return text
        parts = text.split(" ")
        out = []
        for p in parts:
            base = strip_edge.sub("", p)
            if base and (base in stopset or base in local_set):
                continue
            out.append(p)
        return " ".join(out)

    return strip_tokens


# Cache of per-company token frozensets — canonical names repeat across rows.
_COMPANY_TOKEN_CACHE: dict[str, frozenset[str]] = {}
_COMPANY_SPLIT_RE = re.compile(r"[\s,.\-_&()\[\]/'\"|]+")


def _company_tokens(canonical: str | None, preserve_vocab: frozenset[str]) -> frozenset[str]:
    if not canonical:
        return frozenset()
    hit = _COMPANY_TOKEN_CACHE.get(canonical)
    if hit is not None:
        return hit
    tokens = {t for t in _COMPANY_SPLIT_RE.split(canonical.lower()) if t}
    # Keep preserve_vocab out of the local strip set as well.
    local = frozenset(t for t in tokens if len(t) >= 3 and t not in preserve_vocab)
    _COMPANY_TOKEN_CACHE[canonical] = local
    return local


def _sanity_check() -> None:
    assert _unescape(r"C\+\+") == "C++", _unescape(r"C\+\+")
    assert _unescape(r"C\#") == "C#"
    assert _unescape(r"\.NET") == ".NET"
    assert _normalize_text(r"C\+\+   and  \.NET") == "c++ and .net"


def main() -> None:
    _sanity_check()
    print("Sanity asserts on markdown unescape passed.")

    # Load stoplist. The on-disk stoplist file keeps every token (as step 1
    # specifies) — Wave 2 tasks read that file and apply their own guards.
    #
    # For the per-row stripping we do in this step we use a hybrid approach:
    # (a) a *global* stoplist of tokens that appear in company names AND are
    #     NOT common English words AND NOT tech tokens AND NOT common SWE-JD
    #     vocabulary. This is the frequency-restricted conservative set.
    # (b) a *per-row* strip that removes tokens from the posting's own
    #     `company_name_canonical`, which handles one-off rare company names
    #     that don't pass the frequency filter.
    with STOPLIST_PATH.open() as f:
        stoplist_tokens = {line.strip() for line in f if line.strip()}

    # Broad English + SWE-JD vocabulary guard list. Any token on this list
    # MUST NOT be stripped from descriptions even if it appears in a company
    # name. This covers common verbs, nouns, and adjectives that routinely
    # appear in SWE job descriptions.
    PRESERVE_VOCAB = {
        # English function words
        "a","about","above","after","again","against","all","am","an","and","any","are","aren","as","at",
        "be","because","been","before","being","below","between","both","but","by",
        "can","cannot","could","did","do","does","doing","don","down","during",
        "each","few","for","from","further",
        "had","has","have","having","he","her","here","hers","herself","him","himself","his","how",
        "i","if","in","into","is","it","its","itself",
        "just","me","more","most","my","myself","no","nor","not","now","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own",
        "same","she","should","so","some","such",
        "than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too",
        "under","until","up","us","very","was","we","were","what","when","where","which","while","who","whom","why","will","with","you","your","yours","yourself","yourselves",
        "also","may","might","must","shall","would","like","want","need","use","used","using","get","got","make","made","help","helps","helped",
        # Tech tokens / core taxonomy
        "python","java","javascript","typescript","go","golang","rust","ruby","kotlin","swift","scala","php","perl","bash","shell","sql",
        "react","angular","vue","svelte","jquery","html","css","redux","tailwind",
        "node","django","flask","spring","rails","fastapi","express","laravel",
        "aws","azure","gcp","kubernetes","docker","terraform","ansible","jenkins","gitlab","github","helm","argocd",
        "postgresql","postgres","mysql","mongodb","redis","kafka","spark","snowflake","databricks","dbt","elasticsearch","oracle","dynamodb","cassandra","bigquery",
        "tensorflow","pytorch","sklearn","pandas","numpy","jupyter","keras","xgboost",
        "langchain","langgraph","rag","pinecone","chromadb","huggingface","openai","claude","gemini","mcp","llamaindex","anthropic","ollama",
        "copilot","cursor","chatgpt","codex","llm","fine","agent",
        "jest","pytest","selenium","cypress","junit","mocha","playwright",
        "agile","scrum","tdd","devops","sre","microservices",
        # Generic SWE JD vocabulary — preserve
        "ai","ml","data","cloud","mobile","web","backend","frontend","fullstack","full","stack","engineering","software","developer","engineer","systems","security","network","linux","unix","windows","mac","ios","android",
        "api","rest","graphql","grpc","http","json","xml","yaml","git","svn",
        "design","build","develop","implement","manage","lead","mentor","architect","deliver","own","create","maintain","optimize","improve","scale","deploy","test","review","collaborate","communicate","solve","analyze","write","code","debug","document","research","investigate","evaluate","plan","prioritize","execute","ship","iterate","refactor","automate","monitor","measure","report","track",
        "experience","knowledge","skills","qualifications","requirements","responsibilities","bachelor","master","phd","degree","years","year",
        "team","teams","product","products","system","service","services","project","projects","customer","customers","client","clients","stakeholder","stakeholders","user","users","partner","partners","vendor","vendors",
        "senior","junior","mid","lead","principal","staff","director","manager","engineer","scientist","analyst",
        "role","job","position","company","business","technology","technologies","tool","tools","platform","platforms","framework","frameworks","library","libraries","application","applications","feature","features","function","functions","method","methods","class","classes","module","modules","component","components","service","services",
        "work","working","worked","working","ability","abilities","passion","strong","good","great","high","quality","best","deep","broad","solid","proven","demonstrated","excellent","effective","efficient","clear","concise","detailed","rigorous","thorough",
        "office","onsite","hybrid","remote","location","locations","city","state","country","us","usa","united","states","america","canada","european","europe",
        "offer","offers","benefits","salary","compensation","equity","bonus","insurance","vacation","retirement","stock","options","package","perks","wellness","health","medical","dental","vision","parental","leave","pto",
        "inclusive","diversity","equity","equal","opportunity","employer","disability","veteran","gender","race","religion","sexual","orientation","identity",
        "full","time","part","time","contract","internship","permanent","temporary","full-time","part-time",
        "etc","eg","ie","including","include","includes","such","etc",
    }

    # Keep tokens length>=3 that aren't preserved vocabulary. The resulting
    # stopset strips genuine company names like "anthropic" (still ok — the
    # name isn't in jd corpus outside of the company's own postings) or rare
    # multi-word brand names.
    MIN_LEN = 3
    stopset = frozenset(
        t for t in stoplist_tokens
        if len(t) >= MIN_LEN and t not in PRESERVE_VOCAB
    )
    print(
        f"Stoplist: {len(stoplist_tokens):,} total tokens, "
        f"{len(stopset):,} in global stripping set "
        f"(len>={MIN_LEN}, excluding {len(PRESERVE_VOCAB)} preserved words)"
    )

    preserve_vocab = frozenset(PRESERVE_VOCAB)
    strip_tokens = _make_stripper(stopset)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    print("Reading SWE LinkedIn rows from parquet...")
    df = con.execute(
        f"""
        SELECT
            uid,
            description,
            description_core_llm,
            llm_extraction_coverage,
            source,
            period,
            seniority_final,
            seniority_3level,
            seniority_final_source,
            is_aggregator,
            company_name_canonical,
            metro_area,
            yoe_extracted,
            swe_classification_tier
        FROM read_parquet('{PARQUET_IN}')
        WHERE source_platform='linkedin'
          AND is_english=true
          AND date_flag='ok'
          AND is_swe=true
        """
    ).df()
    print(f"Fetched {len(df):,} rows")

    # Build description_cleaned and text_source
    def pick_text(row) -> tuple[str, str]:
        if row["llm_extraction_coverage"] == "labeled" and row["description_core_llm"]:
            return row["description_core_llm"], "llm"
        return row["description"] or "", "raw"

    print("Picking text source...")
    texts_and_sources = df.apply(pick_text, axis=1)
    df["description_cleaned"] = [t for t, _ in texts_and_sources]
    df["text_source"] = [s for _, s in texts_and_sources]

    print("Normalizing text (unescape, lowercase, collapse whitespace)...")
    df["description_cleaned"] = df["description_cleaned"].apply(_normalize_text)

    print("Stripping global stoplist + per-row company-name tokens...")
    def _strip_row(row):
        local = _company_tokens(row["company_name_canonical"], preserve_vocab)
        return strip_tokens(row["description_cleaned"], local)

    df["description_cleaned"] = df.apply(_strip_row, axis=1)
    print(f"Company-token cache size: {len(_COMPANY_TOKEN_CACHE):,}")

    # Drop the raw columns we no longer need
    out_cols = [
        "uid",
        "description_cleaned",
        "text_source",
        "source",
        "period",
        "seniority_final",
        "seniority_3level",
        "seniority_final_source",
        "is_aggregator",
        "company_name_canonical",
        "metro_area",
        "yoe_extracted",
        "swe_classification_tier",
    ]
    df = df[out_cols]

    # text_source breakdown
    print("\ntext_source split per (source, period):")
    print(df.groupby(["source", "period", "text_source"]).size().unstack(fill_value=0))

    print(f"\nWriting parquet to {OUT_PATH}")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, OUT_PATH, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(df):,} rows, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
