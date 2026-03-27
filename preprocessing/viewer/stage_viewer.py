from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8501
MAX_FACET_VALUES = 200
DEFAULT_PAGE_SIZE = 50
MAX_FILTERS = 4
TEXT_DETAIL_COLUMNS = [
    "description_raw",
    "description",
    "description_core",
    "description_core_llm",
]
DEFAULT_VISIBLE_COLUMNS = [
    "uid",
    "source",
    "source_platform",
    "title",
    "company_name_effective",
    "company_name",
    "location",
    "metro_area",
    "seniority_final",
    "seniority_llm",
    "is_swe",
    "swe_classification_llm",
]
DEFAULT_SEARCH_COLUMNS = [
    "uid",
    "title",
    "company_name_effective",
    "company_name",
    "location",
    "description",
    "description_core",
    "description_core_llm",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    description: str
    path: Path


DATASET_SPECS = [
    DatasetSpec(
        key="stage8",
        label="Stage 8 final",
        description="Primary current exploration artifact before LLM augmentation.",
        path=ROOT / "preprocessing" / "intermediate" / "stage8_final.parquet",
    ),
    DatasetSpec(
        key="stage10",
        label="Stage 10 integrated",
        description="Post-LLM integrated artifact that feeds final canonical output.",
        path=ROOT / "preprocessing" / "intermediate" / "stage10_llm_integrated.parquet",
    ),
    DatasetSpec(
        key="unified",
        label="Final unified",
        description="Canonical posting-level output in data/.",
        path=ROOT / "data" / "unified.parquet",
    ),
    DatasetSpec(
        key="observations",
        label="Final observations",
        description="Daily observation panel in data/.",
        path=ROOT / "data" / "unified_observations.parquet",
    ),
]


st.set_page_config(
    page_title="Parquet Stage Viewer",
    page_icon=":material/table_rows:",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .viewer-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #6b7280;
            margin-bottom: 0.15rem;
        }
        .viewer-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .viewer-subtitle {
            color: #4b5563;
            max-width: 66rem;
            margin-bottom: 1rem;
        }
        .viewer-card {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            background: linear-gradient(180deg, rgba(252, 250, 247, 1) 0%, rgba(248, 244, 238, 1) 100%);
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parquet_expr(path_str: str) -> str:
    escaped = path_str.replace("'", "''")
    return f"read_parquet('{escaped}')"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def friendly_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def family_for_type(type_name: str) -> str:
    kind = type_name.upper()
    if "BOOL" in kind:
        return "boolean"
    if any(token in kind for token in ["INT", "DECIMAL", "DOUBLE", "FLOAT", "HUGEINT", "UBIGINT", "BIGINT"]):
        return "numeric"
    if any(token in kind for token in ["DATE", "TIMESTAMP", "TIME"]):
        return "temporal"
    return "text"


def render_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<NULL>"
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def dataset_options() -> tuple[list[DatasetSpec], list[DatasetSpec]]:
    available = [spec for spec in DATASET_SPECS if spec.path.exists()]
    missing = [spec for spec in DATASET_SPECS if not spec.path.exists()]
    return available, missing


@st.cache_data(show_spinner=False)
def dataset_profile(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    relation = parquet_expr(path_str)
    with duckdb.connect() as con:
        schema_df = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchdf()
        row_count = con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0]
    return {
        "row_count": int(row_count),
        "schema": schema_df.to_dict("records"),
        "file_size": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
    }


def normalize_filters(raw_filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for item in raw_filters:
        if not item or not item.get("column") or not item.get("operator"):
            continue
        operator = item["operator"]
        if operator == "contains" and item.get("text"):
            active.append({"column": item["column"], "operator": operator, "text": item["text"]})
        elif operator == "in_values" and item.get("values"):
            active.append({"column": item["column"], "operator": operator, "values": item["values"]})
        elif operator in {"is_null", "is_not_null"}:
            active.append({"column": item["column"], "operator": operator})
        elif operator == "boolean_is" and item.get("value") in {True, False}:
            active.append({"column": item["column"], "operator": operator, "value": item["value"]})
    return active


def build_where_clause(
    filters: list[dict[str, Any]],
    search_term: str,
    search_columns: list[str],
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    for item in filters:
        column_sql = quote_ident(item["column"])
        operator = item["operator"]

        if operator == "contains":
            clauses.append(f"CAST({column_sql} AS VARCHAR) ILIKE ?")
            params.append(f"%{item['text']}%")
        elif operator == "in_values":
            values = item["values"]
            non_null_values = [value for value in values if value is not None]
            subclauses: list[str] = []
            if non_null_values:
                placeholders = ", ".join("?" for _ in non_null_values)
                subclauses.append(f"{column_sql} IN ({placeholders})")
                params.extend(non_null_values)
            if any(value is None for value in values):
                subclauses.append(f"{column_sql} IS NULL")
            if subclauses:
                clauses.append("(" + " OR ".join(subclauses) + ")")
        elif operator == "is_null":
            clauses.append(f"{column_sql} IS NULL")
        elif operator == "is_not_null":
            clauses.append(f"{column_sql} IS NOT NULL")
        elif operator == "boolean_is":
            clauses.append(f"{column_sql} = ?")
            params.append(item["value"])

    search_term = search_term.strip()
    if search_term and search_columns:
        search_clauses = []
        for column in search_columns:
            search_clauses.append(f"CAST({quote_ident(column)} AS VARCHAR) ILIKE ?")
            params.append(f"%{search_term}%")
        clauses.append("(" + " OR ".join(search_clauses) + ")")

    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


@st.cache_data(show_spinner=False, ttl=30)
def filtered_row_count(
    path_str: str,
    filters_json: str,
    search_term: str,
    search_columns_json: str,
) -> int:
    filters = json.loads(filters_json)
    search_columns = json.loads(search_columns_json)
    where_sql, params = build_where_clause(filters, search_term, search_columns)
    relation = parquet_expr(path_str)
    query = f"SELECT COUNT(*) FROM {relation} {where_sql}"
    with duckdb.connect() as con:
        return int(con.execute(query, params).fetchone()[0])


@st.cache_data(show_spinner=False, ttl=30)
def distinct_values(
    path_str: str,
    column: str,
    filters_json: str,
    search_term: str,
    search_columns_json: str,
    value_search: str,
) -> pd.DataFrame:
    filters = json.loads(filters_json)
    search_columns = json.loads(search_columns_json)
    where_sql, params = build_where_clause(filters, search_term, search_columns)
    column_sql = quote_ident(column)
    clauses = [where_sql] if where_sql else []
    if value_search.strip():
        prefix = "AND" if clauses else "WHERE"
        clauses.append(f"{prefix} CAST({column_sql} AS VARCHAR) ILIKE ?")
        params.append(f"%{value_search.strip()}%")
    filters_sql = " ".join(clauses)
    relation = parquet_expr(path_str)
    query = f"""
        SELECT {column_sql} AS value, COUNT(*) AS row_count
        FROM {relation}
        {filters_sql}
        GROUP BY 1
        ORDER BY row_count DESC, CAST(value AS VARCHAR)
        LIMIT {MAX_FACET_VALUES}
    """
    with duckdb.connect() as con:
        return con.execute(query, params).fetchdf()


@st.cache_data(show_spinner=False, ttl=30)
def fetch_page(
    path_str: str,
    columns_json: str,
    filters_json: str,
    search_term: str,
    search_columns_json: str,
    mode: str,
    sort_column: str,
    sort_desc: bool,
    page_size: int,
    page_number: int,
    random_seed: int,
) -> pd.DataFrame:
    columns = json.loads(columns_json)
    filters = json.loads(filters_json)
    search_columns = json.loads(search_columns_json)
    where_sql, params = build_where_clause(filters, search_term, search_columns)
    relation = parquet_expr(path_str)
    select_sql = ", ".join(quote_ident(column) for column in columns)

    if mode == "random":
        random_base = quote_ident("uid" if "uid" in columns else columns[0])
        query = f"""
            SELECT {select_sql}
            FROM {relation}
            {where_sql}
            ORDER BY hash(COALESCE(CAST({random_base} AS VARCHAR), '') || '{random_seed}')
            LIMIT {page_size}
        """
    else:
        direction = "DESC" if sort_desc else "ASC"
        sort_sql = quote_ident(sort_column)
        offset = max(0, page_number - 1) * page_size
        query = f"""
            SELECT {select_sql}
            FROM {relation}
            {where_sql}
            ORDER BY {sort_sql} {direction} NULLS LAST
            LIMIT {page_size} OFFSET {offset}
        """

    with duckdb.connect() as con:
        return con.execute(query, params).fetchdf()


def default_visible_columns(columns: list[str]) -> list[str]:
    preferred = [column for column in DEFAULT_VISIBLE_COLUMNS if column in columns]
    if preferred:
        return preferred
    return columns[: min(10, len(columns))]


def default_search_columns(columns: list[str]) -> list[str]:
    preferred = [column for column in DEFAULT_SEARCH_COLUMNS if column in columns]
    if preferred:
        return preferred
    return columns[: min(4, len(columns))]


def augment_query_columns(visible_columns: list[str], all_columns: list[str]) -> list[str]:
    ordered = list(visible_columns)
    for column in TEXT_DETAIL_COLUMNS:
        if column in all_columns and column not in ordered:
            ordered.append(column)
    return ordered


def format_dataset_label(spec: DatasetSpec) -> str:
    return f"{spec.label} ({spec.path.name})"


def filter_editor(
    slot_index: int,
    path_str: str,
    columns: list[str],
    type_map: dict[str, str],
    prior_filters: list[dict[str, Any]],
    search_term: str,
    search_columns: list[str],
) -> dict[str, Any] | None:
    column = st.selectbox(
        f"Column {slot_index + 1}",
        options=[""] + columns,
        index=0,
        key=f"filter_column_{slot_index}",
    )
    if not column:
        return None

    family = family_for_type(type_map[column])
    if family == "boolean":
        operator_label = st.selectbox(
            "Operator",
            options=[
                ("boolean_is", "is true / false"),
                ("is_null", "is null"),
                ("is_not_null", "is not null"),
            ],
            format_func=lambda item: item[1],
            key=f"filter_operator_{slot_index}",
        )[0]
        if operator_label == "boolean_is":
            bool_label = st.radio(
                "Value",
                options=["True", "False"],
                horizontal=True,
                key=f"filter_bool_value_{slot_index}",
            )
            return {"column": column, "operator": operator_label, "value": bool_label == "True"}
        return {"column": column, "operator": operator_label}

    operator = st.selectbox(
        "Operator",
        options=[
            ("in_values", "in distinct values"),
            ("contains", "contains text"),
            ("is_null", "is null"),
            ("is_not_null", "is not null"),
        ],
        format_func=lambda item: item[1],
        key=f"filter_operator_{slot_index}",
    )[0]

    if operator == "contains":
        text = st.text_input(
            "Contains",
            key=f"filter_contains_{slot_index}",
            placeholder="substring match",
        ).strip()
        if not text:
            return None
        return {"column": column, "operator": operator, "text": text}

    if operator in {"is_null", "is_not_null"}:
        return {"column": column, "operator": operator}

    value_search = st.text_input(
        "Filter distinct values",
        key=f"filter_value_search_{slot_index}",
        placeholder="narrow the distinct-value list",
    )
    options_df = distinct_values(
        path_str=path_str,
        column=column,
        filters_json=json.dumps(prior_filters, sort_keys=True),
        search_term=search_term,
        search_columns_json=json.dumps(search_columns, sort_keys=True),
        value_search=value_search,
    )

    if options_df.empty:
        st.caption("No distinct values found under the current filters.")
        return None

    options = list(range(len(options_df)))
    selected = st.multiselect(
        "Values",
        options=options,
        format_func=lambda idx: f"{render_value(options_df.iloc[idx]['value'])} ({int(options_df.iloc[idx]['row_count']):,})",
        key=f"filter_values_{slot_index}",
    )
    if not selected:
        return None
    values = [options_df.iloc[idx]["value"] for idx in selected]
    return {"column": column, "operator": operator, "values": values}


def row_label(df: pd.DataFrame, row_idx: int) -> str:
    row = df.iloc[row_idx]
    parts = []
    if "uid" in df.columns:
        parts.append(str(row["uid"]))
    if "title" in df.columns:
        parts.append(str(row["title"]))
    if "company_name_effective" in df.columns and pd.notna(row["company_name_effective"]):
        parts.append(str(row["company_name_effective"]))
    elif "company_name" in df.columns and pd.notna(row["company_name"]):
        parts.append(str(row["company_name"]))
    return " | ".join(parts[:3]) if parts else f"Row {row_idx + 1}"


def render_row_details(df: pd.DataFrame) -> None:
    if df.empty:
        return

    st.subheader("Row Detail")
    selected_index = st.selectbox(
        "Inspect a row from the current page",
        options=list(range(len(df))),
        format_func=lambda idx: f"{idx + 1}. {row_label(df, idx)}",
    )
    row = df.iloc[selected_index]

    metadata_cols = [
        column
        for column in df.columns
        if column not in TEXT_DETAIL_COLUMNS and not column.lower().endswith("_raw")
    ]
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        preview = pd.DataFrame(
            [{"column": column, "value": render_value(row[column])} for column in metadata_cols]
        )
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            height=520,
            column_order=["column", "value"],
        )

    with right:
        long_text_fields = [
            column for column in TEXT_DETAIL_COLUMNS if column in df.columns and pd.notna(row[column])
        ]
        if not long_text_fields:
            st.info("No long-text columns are visible for the selected row.")
        else:
            tabs = st.tabs(long_text_fields)
            for tab, column in zip(tabs, long_text_fields):
                with tab:
                    st.text_area(
                        column,
                        value=str(row[column]),
                        height=520,
                        disabled=True,
                        label_visibility="collapsed",
                    )

    with st.expander("Full row as JSON"):
        payload = {}
        for column in df.columns:
            value = row[column]
            if pd.isna(value):
                payload[column] = None
            else:
                payload[column] = value.item() if hasattr(value, "item") else value
        st.json(payload)


def main() -> None:
    inject_css()
    st.markdown('<div class="viewer-kicker">Spot Check Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="viewer-title">Parquet Stage Viewer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="viewer-subtitle">
        Server-side table browser for the large preprocessing artifacts. It queries Parquet lazily with DuckDB,
        supports random sampling, column toggles, text search, and distinct-value facet filters, and keeps long
        descriptions readable in a dedicated row-detail panel.
        </div>
        """,
        unsafe_allow_html=True,
    )

    available, missing = dataset_options()
    if not available:
        st.error("No known Parquet artifacts were found. Check the expected stage paths in preprocessing/intermediate/ and data/.")
        return

    default_index = next((idx for idx, spec in enumerate(available) if spec.key == "stage8"), 0)

    with st.sidebar:
        st.header("Dataset")
        selected_spec = st.selectbox(
            "Artifact",
            options=available,
            index=default_index,
            format_func=format_dataset_label,
        )
        st.caption(selected_spec.description)
        st.code(str(selected_spec.path), language="text")

        if missing:
            with st.expander("Missing expected artifacts"):
                for spec in missing:
                    st.write(f"- `{spec.path}`")

    profile = dataset_profile(str(selected_spec.path))
    schema = profile["schema"]
    columns = [item["column_name"] for item in schema]
    type_map = {item["column_name"]: item["column_type"] for item in schema}

    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", f"{profile['row_count']:,}")
    summary_columns[1].metric("Columns", f"{len(columns):,}")
    summary_columns[2].metric("File Size", friendly_bytes(profile["file_size"]))
    summary_columns[3].metric("Port", str(DEFAULT_PORT))

    st.markdown('<div class="viewer-card">Use the sidebar to pick the artifact. Controls below only query the current Parquet file.</div>', unsafe_allow_html=True)

    control_left, control_right = st.columns([1.15, 1.0], gap="large")

    with control_left:
        st.subheader("Search and Columns")
        search_term = st.text_input("Search text", placeholder="title, company, uid, description...")
        search_columns = st.multiselect(
            "Search columns",
            options=columns,
            default=default_search_columns(columns),
        )
        visible_columns = st.multiselect(
            "Visible table columns",
            options=columns,
            default=default_visible_columns(columns),
        )
        if not visible_columns:
            st.warning("Select at least one visible column.")
            return

    with control_right:
        st.subheader("Browse Mode")
        mode = st.radio(
            "Retrieval mode",
            options=["page", "random"],
            format_func=lambda value: "Ordered page" if value == "page" else "Random sample",
            horizontal=True,
        )
        page_size = st.slider("Rows per result batch", min_value=10, max_value=200, value=DEFAULT_PAGE_SIZE, step=10)
        sort_candidates = columns
        default_sort = "uid" if "uid" in columns else columns[0]
        sort_column = st.selectbox(
            "Sort column",
            options=sort_candidates,
            index=sort_candidates.index(default_sort),
            disabled=mode == "random",
        )
        sort_desc = st.toggle("Descending sort", value=False, disabled=mode == "random")

        if "random_seed" not in st.session_state:
            st.session_state.random_seed = 42
        if mode == "random":
            if st.button("Reshuffle sample", use_container_width=True):
                st.session_state.random_seed += 1
            st.caption(f"Seed: {st.session_state.random_seed}")

    st.subheader("Facet Filters")
    filter_count = st.slider("Number of filter slots", min_value=0, max_value=MAX_FILTERS, value=2)
    filter_specs: list[dict[str, Any]] = []
    if filter_count:
        filter_columns = st.columns(filter_count)
        for idx in range(filter_count):
            with filter_columns[idx]:
                spec = filter_editor(
                    slot_index=idx,
                    path_str=str(selected_spec.path),
                    columns=columns,
                    type_map=type_map,
                    prior_filters=filter_specs,
                    search_term=search_term,
                    search_columns=search_columns,
                )
                if spec:
                    filter_specs.append(spec)

    filters_json = json.dumps(normalize_filters(filter_specs), sort_keys=True)
    search_columns_json = json.dumps(search_columns, sort_keys=True)
    query_columns = augment_query_columns(visible_columns, columns)
    query_columns_json = json.dumps(query_columns)

    filtered_count = filtered_row_count(
        path_str=str(selected_spec.path),
        filters_json=filters_json,
        search_term=search_term,
        search_columns_json=search_columns_json,
    )
    st.caption(f"Filtered rows: {filtered_count:,} of {profile['row_count']:,}")

    if filtered_count == 0:
        st.warning("The current filters returned zero rows.")
        return

    if mode == "page":
        max_page = max(1, math.ceil(filtered_count / page_size))
        page_number = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
        st.caption(f"Page {page_number} of {max_page}")
    else:
        page_number = 1

    results = fetch_page(
        path_str=str(selected_spec.path),
        columns_json=query_columns_json,
        filters_json=filters_json,
        search_term=search_term,
        search_columns_json=search_columns_json,
        mode=mode,
        sort_column=sort_column,
        sort_desc=sort_desc,
        page_size=page_size,
        page_number=int(page_number),
        random_seed=int(st.session_state.random_seed),
    )

    if results.empty:
        st.warning("No rows were returned for the current page.")
        return

    table_results = results[visible_columns]
    selection = st.dataframe(
        table_results,
        hide_index=True,
        width="stretch",
        height=520,
        on_select="rerun",
        selection_mode="single-row",
        row_height=32,
        key="results_table",
    )

    selected_rows: list[int] = []
    if isinstance(selection, dict):
        selected_rows = selection.get("selection", {}).get("rows", [])
    elif hasattr(selection, "selection") and hasattr(selection.selection, "rows"):
        selected_rows = list(selection.selection.rows)
    if selected_rows:
        selected_df = results.iloc[selected_rows].reset_index(drop=True)
    else:
        selected_df = results
    render_row_details(selected_df)

    with st.expander("Schema and types"):
        schema_df = pd.DataFrame(schema).rename(
            columns={"column_name": "column", "column_type": "type", "null": "nullable"}
        )
        st.dataframe(schema_df, hide_index=True, width="stretch", height=420)

    with st.expander("Operational notes"):
        st.markdown(
            f"""
            - Bind this app to the network with `--server.address 0.0.0.0` and browse to `http://<tailscale-ip>:{DEFAULT_PORT}`.
            - `Stage 8 final` is the current primary artifact in this repo. `Stage 11 integrated` appears automatically once the LLM integration output exists.
            - Distinct-value filters query the file directly and intentionally cap the list at the top {MAX_FACET_VALUES} values under the current filter context.
            - Random mode uses a deterministic hash shuffle over the filtered result set, which is stable for a given seed and does not materialize the full file into pandas.
            """
        )


if __name__ == "__main__":
    main()
