# Fact-check: Piece 09 "Forward-deployed, finally"

Source tables: `exploration/tables/journalist/fde_{counts,companies,exemplars}.csv`, `fde_summary.json`.
Script: `exploration/scripts/journalist/fde_prevalence.py`.
Corpus used by script: **`data/unified_core.parquet`** (de-duplicated core), filter `source_platform='linkedin' AND is_swe AND is_english AND date_flag='ok'`; 2024 = arshkon+asaniczka, 2026 = scraped.

## Independent rerun (DuckDB)

Title match: `lower(title) LIKE '%forward deployed%' OR '%forward-deployed%' OR \\bfde\\b`.

| Metric | Piece claim | unified_core (rerun) | unified.parquet (per task spec) |
|---|---|---|---|
| 2024 FDE n | 3 | **3** (of 22,812) | 3 (of 22,820) |
| 2026 FDE n | 59 | **59** (of 25,822) | 101 (of 45,317) |
| 2024 share | 0.013% | **0.0132%** | 0.0131% |
| 2026 share | 0.229% | **0.2285%** | 0.2229% |
| Share growth | 17x | **17.37x** | 17.02x |
| Distinct firms 2026 | **38** | **42** | 66 |
| AI-strict FDE | 32.2% | **32.20%** (n=59) | n/a (rerun not done) |
| AI-strict overall | 13.8% | **13.78%** (n=25,822) | n/a |
| AI-strict ratio | 2.3x | **2.34x** | n/a |
| Median YOE FDE | 5.0 | **5.0** (n=41 labeled) | 5.0 (n=41) |
| Median YOE overall | 5.0 | **5.0** (n=20,510) | 5.0 (n=20,511) |

Named firms (Saronic, CACI, Govini, Mach Industries, FOX Tech, OpenAI, Palantir, Ramp): **all 8 present** in the 2026 FDE-title list in unified_core.

## Verdict: **Qualifies**

Every headline number replicates exactly on `unified_core.parquet`, which is the corpus the script and piece use. The caller's pointer to `data/unified.parquet` refers to the pre-dedup superset (45k rows, 66 distinct firms, 101 FDE-titled rows — same 17x growth pattern, same ~0.22% share, conclusion unchanged).

## Material discrepancy

"**38 distinct firms**" is wrong. The correct count in unified_core is **42** (`fde_companies.csv` only shows the top-10 by volume, which may be the source of the 38 miscount). Not a directional problem, but factually off.

## Caveat on Anthropic's absence

Piece claims Anthropic does not post FDE-titled roles in our window. Confirmed: no "forward deployed" / "fde" titles at Anthropic. But near-synonyms exist — **Anthropic posts 5 "Applied AI Engineer" SWE roles** in 2026 scraped (e.g., "Applied AI Engineer (Startups)", "Applied AI Engineer, Beneficial Deployments"). If the FDE archetype is defined functionally (customer-facing deployment), Anthropic is present under a different label, and the "absent" framing would be misleading.
