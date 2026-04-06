from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

from viewer_shared import ROOT, inject_css, friendly_bytes, render_value

# Import hash function from pipeline code to join candidates with cache.
sys.path.insert(0, str(ROOT / "preprocessing" / "scripts"))
from llm_shared import (  # noqa: E402
    compute_extraction_input_hash,
    segment_description_into_units,
    join_retained_units,
    format_numbered_units,
)

CACHE_DIR = ROOT / "preprocessing" / "cache"
DEFAULT_CACHE_DB = CACHE_DIR / "llm_responses.db"
CANDIDATES_PATH = ROOT / "preprocessing" / "intermediate" / "stage9_llm_candidates.parquet"

DEFAULT_PAGE_SIZE = 50


def reconstruct_cleaned_description(description: str, response_json: str) -> dict[str, Any]:
    """Replay the extraction logic: segment into units, apply the cached LLM
    decision, and return the cleaned text plus annotated unit breakdown."""
    units = segment_description_into_units(description)
    try:
        payload = json.loads(response_json)
    except (json.JSONDecodeError, TypeError):
        return {"cleaned": None, "units": units, "drop_ids": [], "uncertain_ids": [], "error": "invalid JSON"}

    drop_ids = payload.get("boilerplate_unit_ids") or []
    uncertain_ids = payload.get("uncertain_unit_ids") or []
    task_status = payload.get("task_status", "")

    if task_status != "ok":
        return {"cleaned": None, "units": units, "drop_ids": drop_ids, "uncertain_ids": uncertain_ids, "error": task_status}

    cleaned = join_retained_units(units, drop_ids)
    return {"cleaned": cleaned, "units": units, "drop_ids": drop_ids, "uncertain_ids": uncertain_ids, "error": None}


def format_annotated_units(units: list[dict], drop_ids: list[int], uncertain_ids: list[int]) -> str:
    """Format units with markers showing which were dropped/uncertain."""
    drop_set = set(drop_ids)
    uncertain_set = set(uncertain_ids)
    parts = []
    for unit in units:
        uid = unit["unit_id"]
        if uid in drop_set:
            marker = "DROP"
        elif uid in uncertain_set:
            marker = "UNCERTAIN"
        else:
            marker = "KEEP"
        parts.append(f"[{uid}] ({marker})\n{unit['text']}")
    return "\n\n".join(parts)


def _list_cache_dbs() -> list[Path]:
    if not CACHE_DIR.is_dir():
        return []
    return sorted(p for p in CACHE_DIR.glob("*.db") if p.stat().st_size > 0)


@st.cache_data(show_spinner=False, ttl=120)
def cache_summary(db_path: str) -> dict[str, Any]:
    with duckdb.connect() as con:
        con.execute("INSTALL sqlite; LOAD sqlite")
        con.execute(f"ATTACH '{db_path}' AS cache (TYPE SQLITE, READ_ONLY)")
        total = con.execute("SELECT count(*) FROM cache.responses").fetchone()[0]
        by_task = con.execute(
            "SELECT task_name, count(*) as cnt FROM cache.responses GROUP BY 1 ORDER BY cnt DESC"
        ).fetchdf()
        by_model = con.execute(
            "SELECT model, count(*) as cnt FROM cache.responses GROUP BY 1 ORDER BY cnt DESC"
        ).fetchdf()
        ts_range = con.execute(
            "SELECT min(timestamp) as earliest, max(timestamp) as latest FROM cache.responses"
        ).fetchone()
    return {
        "total": int(total),
        "by_task": by_task,
        "by_model": by_model,
        "earliest": ts_range[0],
        "latest": ts_range[1],
        "file_size": Path(db_path).stat().st_size,
    }


@st.cache_data(show_spinner=False, ttl=30)
def fetch_cache_page(
    db_path: str,
    task_filter: list[str],
    model_filter: list[str],
    page_size: int,
    page_number: int,
) -> tuple[pd.DataFrame, int]:
    clauses: list[str] = []
    params: list[Any] = []

    if task_filter:
        placeholders = ", ".join("?" for _ in task_filter)
        clauses.append(f"task_name IN ({placeholders})")
        params.extend(task_filter)
    if model_filter:
        placeholders = ", ".join("?" for _ in model_filter)
        clauses.append(f"model IN ({placeholders})")
        params.extend(model_filter)

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with duckdb.connect() as con:
        con.execute("INSTALL sqlite; LOAD sqlite")
        con.execute(f"ATTACH '{db_path}' AS cache (TYPE SQLITE, READ_ONLY)")

        count_params = list(params)
        total = con.execute(
            f"SELECT count(*) FROM cache.responses {where_sql}", count_params
        ).fetchone()[0]

        offset = max(0, page_number - 1) * page_size
        query = f"""
            SELECT
                input_hash,
                task_name,
                model,
                timestamp,
                json_extract_string(response_json, '$.task_status') AS task_status,
                json_array_length(json_extract(response_json, '$.boilerplate_unit_ids')) AS boilerplate_count,
                json_array_length(json_extract(response_json, '$.uncertain_unit_ids')) AS uncertain_count,
                json_extract_string(response_json, '$.reason') AS reason,
                response_json
            FROM cache.responses
            {where_sql}
            ORDER BY timestamp DESC
            LIMIT {page_size} OFFSET {offset}
        """
        df = con.execute(query, params).fetchdf()

    return df, int(total)


@st.cache_data(show_spinner="Computing input hashes for candidates...", ttl=600)
def load_candidate_hashes(candidates_path: str) -> pd.DataFrame:
    with duckdb.connect() as con:
        cand = con.execute(
            f"""SELECT description_hash, title, company_name, description,
                       needs_llm_classification, needs_llm_extraction, llm_route_group,
                       source_row_count
                FROM read_parquet('{candidates_path}')"""
        ).fetchdf()

    cand["input_hash"] = [
        compute_extraction_input_hash(row["title"], row["company_name"], row["description"])
        for _, row in cand.iterrows()
    ]
    return cand


@st.cache_data(show_spinner=False, ttl=60)
def load_cache_index(db_path: str) -> pd.DataFrame:
    with duckdb.connect() as con:
        con.execute("INSTALL sqlite; LOAD sqlite")
        con.execute(f"ATTACH '{db_path}' AS cache (TYPE SQLITE, READ_ONLY)")
        df = con.execute("""
            SELECT
                input_hash,
                task_name,
                model,
                json_extract_string(response_json, '$.task_status') AS task_status,
                json_array_length(json_extract(response_json, '$.boilerplate_unit_ids')) AS boilerplate_count,
                json_array_length(json_extract(response_json, '$.uncertain_unit_ids')) AS uncertain_count,
                json_extract_string(response_json, '$.reason') AS reason,
                response_json
            FROM cache.responses
        """).fetchdf()
    return df


@st.cache_data(show_spinner="Building merged candidate + cache view...", ttl=60)
def build_merged_view(candidates_path: str, db_path: str) -> pd.DataFrame:
    candidates = load_candidate_hashes(candidates_path)
    cache_idx = load_cache_index(db_path)

    merged = candidates.merge(
        cache_idx,
        on="input_hash",
        how="left",
        suffixes=("", "_cache"),
    )
    merged["cache_status"] = merged["task_status"].apply(
        lambda v: "cached" if pd.notna(v) else "pending"
    )
    return merged


def render_cache_browser(db_path: str, summary: dict[str, Any]) -> None:
    st.subheader("Cache Responses")

    filter_left, filter_right = st.columns(2)
    with filter_left:
        all_tasks = list(summary["by_task"]["task_name"])
        task_filter = st.multiselect("Task name", options=all_tasks, default=all_tasks)
    with filter_right:
        all_models = list(summary["by_model"]["model"])
        model_filter = st.multiselect("Model", options=all_models, default=all_models)

    page_size = st.slider("Rows per page", min_value=10, max_value=200, value=DEFAULT_PAGE_SIZE, step=10, key="cache_page_size")

    df, total = fetch_cache_page(db_path, task_filter, model_filter, page_size, st.session_state.get("cache_page", 1))

    max_page = max(1, math.ceil(total / page_size))
    page_number = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="cache_page")
    st.caption(f"Showing page {page_number} of {max_page} ({total:,} matching responses)")

    if df.empty:
        st.warning("No responses match the current filters.")
        return

    display_cols = [c for c in df.columns if c != "response_json"]
    st.dataframe(df[display_cols], hide_index=True, width="stretch", height=480)

    st.subheader("Response Detail")
    row_idx = st.selectbox(
        "Select a response to inspect",
        options=list(range(len(df))),
        format_func=lambda i: f"{i + 1}. {df.iloc[i]['input_hash'][:16]}... | {df.iloc[i]['model']} | {df.iloc[i]['reason'] or ''}",
        key="cache_detail_select",
    )
    row = df.iloc[row_idx]
    try:
        parsed = json.loads(row["response_json"])
        st.json(parsed)
    except (json.JSONDecodeError, TypeError):
        st.code(str(row["response_json"]), language="json")


def render_candidates_tab(db_path: str) -> None:
    if not CANDIDATES_PATH.exists():
        st.warning(f"Candidates file not found: `{CANDIDATES_PATH}`")
        return

    merged = build_merged_view(str(CANDIDATES_PATH), db_path)

    cached_count = (merged["cache_status"] == "cached").sum()
    pending_count = (merged["cache_status"] == "pending").sum()

    cols = st.columns(3)
    cols[0].metric("Total candidates", f"{len(merged):,}")
    cols[1].metric("Cached", f"{cached_count:,}")
    cols[2].metric("Pending", f"{pending_count:,}")

    filter_left, filter_right = st.columns(2)
    with filter_left:
        status_filter = st.multiselect(
            "Cache status",
            options=["cached", "pending"],
            default=["cached", "pending"],
            key="cand_status_filter",
        )
    with filter_right:
        route_groups = sorted(merged["llm_route_group"].dropna().unique())
        route_filter = st.multiselect(
            "Route group",
            options=route_groups,
            default=route_groups,
            key="cand_route_filter",
        )

    mask = merged["cache_status"].isin(status_filter) & merged["llm_route_group"].isin(route_filter)
    filtered = merged[mask].reset_index(drop=True)
    st.caption(f"Filtered: {len(filtered):,} candidates")

    if filtered.empty:
        st.warning("No candidates match the current filters.")
        return

    page_size = st.slider("Rows per page", min_value=10, max_value=200, value=DEFAULT_PAGE_SIZE, step=10, key="cand_page_size")
    max_page = max(1, math.ceil(len(filtered) / page_size))
    page_number = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="cand_page")
    st.caption(f"Page {page_number} of {max_page}")

    start = (page_number - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]

    table_cols = [
        "title", "company_name", "llm_route_group", "cache_status",
        "task_status", "model", "boilerplate_count", "uncertain_count", "reason",
    ]
    table_cols = [c for c in table_cols if c in page_df.columns]
    st.dataframe(page_df[table_cols], hide_index=True, width="stretch", height=480)

    st.subheader("Candidate Detail")
    detail_idx = st.selectbox(
        "Select a candidate to inspect",
        options=list(range(len(page_df))),
        format_func=lambda i: f"{i + 1}. {page_df.iloc[i]['title']} | {page_df.iloc[i]['company_name'] or ''} | {page_df.iloc[i]['cache_status']}",
        key="cand_detail_select",
    )
    row = page_df.iloc[detail_idx]
    desc = row.get("description", "")
    desc_str = str(desc) if pd.notna(desc) else ""
    resp_json = row.get("response_json")
    has_cache = pd.notna(resp_json)

    # Reconstruct cleaned description if cache entry exists
    if has_cache and desc_str:
        result = reconstruct_cleaned_description(desc_str, resp_json)
    else:
        result = None

    if has_cache and result and result["cleaned"] is not None:
        # Show three tabs: cleaned result, annotated units, original
        detail_tabs = st.tabs(["Cleaned Description", "Annotated Units", "Original Description", "Cache Response"])

        with detail_tabs[0]:
            st.text_area(
                "cleaned",
                value=result["cleaned"],
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )
            drop_count = len(result["drop_ids"])
            total_units = len(result["units"])
            kept = total_units - drop_count
            st.caption(f"{kept} of {total_units} units kept, {drop_count} dropped, {len(result['uncertain_ids'])} uncertain")

        with detail_tabs[1]:
            annotated = format_annotated_units(result["units"], result["drop_ids"], result["uncertain_ids"])
            st.text_area(
                "annotated_units",
                value=annotated,
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )

        with detail_tabs[2]:
            st.text_area(
                "original",
                value=desc_str,
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )

        with detail_tabs[3]:
            try:
                st.json(json.loads(resp_json))
            except (json.JSONDecodeError, TypeError):
                st.code(str(resp_json), language="json")
    else:
        # No cache or reconstruction failed — show original + status
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("**Original Description**")
            st.text_area(
                "description",
                value=desc_str,
                height=500,
                disabled=True,
                label_visibility="collapsed",
            )
        with right:
            if not has_cache:
                st.info("No cache entry for this candidate yet.")
            elif result and result["error"]:
                st.warning(f"Reconstruction failed: {result['error']}")
                try:
                    st.json(json.loads(resp_json))
                except (json.JSONDecodeError, TypeError):
                    st.code(str(resp_json), language="json")

    # Metadata row at the bottom
    with st.expander("Metadata"):
        meta = {
            "input_hash": row.get("input_hash", ""),
            "description_hash": row.get("description_hash", ""),
            "model": row.get("model", "") if has_cache else "",
            "task_name": row.get("task_name", "") if has_cache else "",
            "llm_route_group": row.get("llm_route_group", ""),
            "needs_llm_classification": row.get("needs_llm_classification"),
            "needs_llm_extraction": row.get("needs_llm_extraction"),
            "source_row_count": row.get("source_row_count"),
        }
        st.dataframe(pd.DataFrame([meta]), hide_index=True, width="stretch")


def main() -> None:
    inject_css()
    st.markdown('<div class="viewer-kicker">Spot Check Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="viewer-title">LLM Cache Browser</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="viewer-subtitle">
        Browse LLM extraction and classification cache responses, and see which stage 9
        candidates have been processed vs. are still pending.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cache_dbs = _list_cache_dbs()
    if not cache_dbs:
        st.error(f"No cache databases found in `{CACHE_DIR}`.")
        return

    with st.sidebar:
        st.header("Cache Database")
        default_idx = next((i for i, p in enumerate(cache_dbs) if p.name == "llm_responses.db"), 0)
        selected_db = st.selectbox(
            "Database",
            options=cache_dbs,
            index=default_idx,
            format_func=lambda p: f"{p.name} ({friendly_bytes(p.stat().st_size)})",
        )
        db_path = str(selected_db)

    try:
        summary = cache_summary(db_path)
    except Exception as exc:
        st.error(f"Failed to read cache database: {exc}")
        return

    metrics = st.columns(4)
    metrics[0].metric("Responses", f"{summary['total']:,}")
    metrics[1].metric("File size", friendly_bytes(summary["file_size"]))
    metrics[2].metric("Earliest", str(summary["earliest"])[:10] if summary["earliest"] else "—")
    metrics[3].metric("Latest", str(summary["latest"])[:10] if summary["latest"] else "—")

    with st.sidebar:
        st.subheader("By task")
        for _, row in summary["by_task"].iterrows():
            st.caption(f"{row['task_name']}: {int(row['cnt']):,}")
        st.subheader("By model")
        for _, row in summary["by_model"].iterrows():
            st.caption(f"{row['model']}: {int(row['cnt']):,}")

    tab_cache, tab_candidates = st.tabs(["Cache Browser", "Candidates + Cache Status"])

    with tab_cache:
        render_cache_browser(db_path, summary)

    with tab_candidates:
        render_candidates_tab(db_path)


main()
