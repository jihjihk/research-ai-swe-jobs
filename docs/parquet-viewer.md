# Parquet Viewer Proposal

Date: 2026-03-25

## Goal

Provide a lightweight browser-based spot-check tool for large preprocessing artifacts, especially:

- `preprocessing/intermediate/stage8_final.parquet`
- `preprocessing/intermediate/stage10_llm_integrated.parquet` when present
- `data/unified.parquet`
- `data/unified_observations.parquet`

The tool should be reachable over Tailscale from another machine, avoid loading multi-GB parquet files into pandas, and make it easy to inspect long text such as job descriptions.

## Decision

Build and run a narrow custom app on top of DuckDB and Streamlit.

Why this is the best fit here:

- DuckDB can query parquet directly and lazily, which matches the repo's parquet-first workflow and RAM constraints.
- Streamlit is fast to ship for an internal tool and can be bound to `0.0.0.0` for Tailscale access.
- A custom app lets us expose the exact controls needed for spot checking: stage selection, random sampling, column hiding, text search, distinct-value facet filters, and a dedicated long-text reader.

## Evaluated Options

### 1. Datasette

Applicability:

- Strong fit for server-side table exploration, facets, and search.
- Official docs emphasize facets, full-text search, table pages, page sizing, and truncated-cell settings with row pages for long values.

Why not the default choice here:

- Datasette is naturally SQLite-shaped, while this workflow is parquet-shaped.
- Adopting it cleanly would likely mean either duplicating the 7.1 GB Stage 8 artifact into SQLite or adding a plugin layer around DuckDB/parquet.
- The customization needed for random row retrieval and repo-specific stage switching would still require extra work.

### 2. Perspective

Applicability:

- Excellent grid/analytics component for large and streaming datasets.
- Official project docs describe a framework-agnostic data grid and DuckDB integration points.

Why not the default choice here:

- Perspective is a powerful component, not a complete ready-to-run local app for this repo.
- It would still require building a backend, app shell, and all of the repo-specific controls.
- That is more engineering than this need requires.

### 3. marimo

Applicability:

- Strong notebook/app hybrid.
- Official docs describe notebooks as deployable as apps with SQL and dataframe filtering/search.

Why not the default choice here:

- The notebook metaphor is not the best shape for a persistent spot-check browser for large stage artifacts.
- It is better suited to analyst-authored notebooks than to a focused QA viewer with explicit facet and row-detail controls.

### 4. Streamlit + DuckDB

Applicability:

- Direct fit for an internal tool hosted on this machine.
- Streamlit provides an interactive table and row selection support.
- DuckDB keeps filtering and sampling server-side against parquet.

Why this wins:

- Lowest implementation overhead for the exact workflow needed.
- Easy to bind to `0.0.0.0` and reach over Tailscale.
- Lets us add a readable long-text inspection panel instead of relying on truncated cells alone.

## Implemented Tool

Files:

- `preprocessing/viewer/stage_viewer.py`
- `preprocessing/scripts/run_stage_viewer.sh`
- `preprocessing/viewer/requirements.txt`

Capabilities:

- Select among the known stage/final parquet artifacts that exist locally
- Query parquet lazily with DuckDB
- Toggle visible columns
- Search across selected columns
- Add multiple facet filters from a column's distinct values
- Shuffle to retrieve random rows using a deterministic DuckDB hash shuffle
- Page through ordered results
- Inspect long text in a dedicated row-detail panel
- Review schema and column types inline

## Hosting

Run:

```bash
./preprocessing/scripts/run_stage_viewer.sh
```

Then browse from another Tailscale-connected machine:

```text
http://<tailscale-ip>:8501
```

If you want a different port:

```bash
STAGE_VIEWER_PORT=8510 ./preprocessing/scripts/run_stage_viewer.sh
```

## Operational Notes

- The current primary exploration artifact remains `stage8_final.parquet`.
- The app auto-surfaces `stage10_llm_integrated.parquet` once it exists.
- Distinct-value facet lists are intentionally capped to keep UI queries responsive on very large text-heavy columns.
- For long descriptions, use the row-detail panel instead of trying to read directly from the table grid.

## Sources

- Streamlit `st.dataframe` docs: https://docs.streamlit.io/develop/api-reference/data/st.dataframe
- Streamlit deploy docs showing `--server.address=0.0.0.0`: https://docs.streamlit.io/deploy/tutorials/docker
- Datasette docs: https://docs.datasette.io/en/stable/
- Perspective project README: https://github.com/perspective-dev/perspective
- marimo docs: https://docs.marimo.io/
