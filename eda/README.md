# eda/ — Open-ended senior-DS EDA on `data/unified.parquet`

Branch-scoped, self-contained exploratory analysis. Produces findings organized around seven pre-registered hypotheses (H1–H7, see `memos/priors.md`).

This is **not** a replacement for a full `exploration/` orchestrator run. It is a targeted audit + three extensions (H4 industry dispersion, H6 Big Tech stratification, H2 new-AI-title emergence) that v8 did not explicitly publish.

## Prereqs

- Python virtualenv at `../.venv/` (DuckDB, pyarrow, matplotlib, pandas)
- `data/unified.parquet` (1.45M × 96) and `data/unified_observations.parquet` (daily panel)
- v8 archive at `exploration-archive/v8_claude_opus47/` for reconciliation

## How to re-run

From the repo root:

```bash
# Phase A — corpus profile
./.venv/bin/python eda/scripts/profile.py

# Phase B — 12 hypothesis-driven scans (S1–S11 + Sv)
./.venv/bin/python eda/scripts/scans.py

# Phase C — stress-test finalists against 4 independent slices
./.venv/bin/python eda/scripts/triangulate.py
```

Each phase writes to `eda/figures/` and `eda/tables/` idempotently (no appending). Phase B depends on Phase A outputs (for v8 row-count cross-check); Phase C depends on Phase B outputs.

## Layout

```
memos/              priors.md (blind-to-v8 H1–H7) + archived references
scripts/            profile.py, scans.py, triangulate.py
notebooks/          open_ended_v1.ipynb (optional driver; scripts are source of truth)
figures/            A_*.png, S1_*.png … Sv_*.png, C_*.png
tables/             A_*.csv, S1_*.csv … Sv_*.csv
reports/            open_ended_v1.md (final findings, one verdict per hypothesis)
```

## Guardrails

- DuckDB for aggregations (reads parquet directly).
- No `pd.read_parquet()` on `unified.parquet`; it does not fit in memory.
- Every figure has `n=` and filter text in caption.
- Every metric and filter is pre-committed before the scan runs (see `memos/priors.md` and the scan table in `scripts/scans.py`).

## Branch + identity

Work is committed on `eda/open-ended-v1` with local git identity configured to jihjihk. Do not push without explicit approval.
