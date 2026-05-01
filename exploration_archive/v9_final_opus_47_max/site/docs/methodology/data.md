# Data sources and samples

Every number on this site is computed over one of seven slices of one pooled dataset. This page names the sources, defines each slice, and says what each slice can support.

## The three data sources

The study pools three separate collections of LinkedIn job postings. Each fills a role the others cannot:

- **the 2024 arshkon dataset**: a public Kaggle release of roughly 5,000 software-engineering LinkedIn postings collected in April 2024. Small, but it comes with LinkedIn's own "entry", "associate", "mid-senior" seniority labels attached, which makes it the cleanest surface for checking whether the seniority patterns depend on how we define seniority.
- **the 2024 asaniczka dataset**: a second Kaggle release, roughly 18,000 LinkedIn postings from January 2024. Four times the size of arshkon, but with no native entry-level labels at all (LinkedIn's labels were already stripped at the time of collection). Useful for volume; not usable for any claim that depends on LinkedIn's own entry-level tagging.
- **our 2026 collection**: roughly 45,000 LinkedIn and Indeed postings our team scraped between 20 March and 18 April 2026. This is the 2026 side of every 2024-to-2026 comparison on the site.

Under the study's default filter (English-language software-engineering postings on LinkedIn, with a valid posting date), the three sources combine to 68,137 postings: 4,691 from arshkon, 18,129 from asaniczka, 45,317 from the 2026 scrape. LinkedIn is the primary surface. Indeed postings are only used in sensitivity checks, because both Kaggle sources are LinkedIn-only and any cross-period comparison needs to compare like with like.

The three snapshots are separated by roughly 791 days. That gap spans eight major frontier-model releases (GPT-4o, Claude 3.5 Sonnet, o1, DeepSeek-V3, GPT-4.5, Claude 3.6 Sonnet, Claude 4 Opus, Gemini 2.5 Pro). The temporal windows are narrow: asaniczka covers six days in January 2024, arshkon covers fifteen days in April 2024, the 2026 scrape covers 30-plus days and is still growing.

| Source | Platform | Temporal role | Strength | Key gap |
|---|---|---|---|---|
| the arshkon dataset | LinkedIn | 2024 snapshot | Has LinkedIn's native entry-level labels | Small software-engineering count |
| the asaniczka dataset | LinkedIn | 2024 snapshot | Large volume | **No native entry-level labels** |
| our 2026 scrape | LinkedIn + Indeed | 2026 current window | Fresh; carries search metadata | Query-stratified by design |

A practical consequence of the asaniczka gap: any claim that uses LinkedIn's own seniority tagging has to fall back to arshkon alone, because asaniczka simply does not carry those tags. When a finding on this site reports a pooled 2024 magnitude next to an arshkon-only magnitude, this is why.

## The seven analytical slices

Different findings need different slices of the data. The numbered labels (Sample 1 through Sample 7) are internal shorthand. Throughout the rest of the site, the slices are referred to by the content names given here.

### The full software-engineering frame (Sample 1)

68,137 postings under the default filter (English-language software-engineering LinkedIn postings with valid dates).

Used for: descriptive and prevalence claims, the signal-to-noise calibration computed within 2024, and raw 2024-to-2026 percentage-point changes.

Key caveat: the 2026 scrape's 45,000 postings are query-stratified by the scraper's design. The scraper searched for specific roles rather than sweeping everything on the platform. Claims about shares *within* software-engineering are safe; raw claims about software-engineering as a share of *all* postings are not.

### The returning-firms cohort (Sample 2)

The 2,109 firms that posted software-engineering roles in both 2024 and 2026. This is 13,437 postings from 2024 and 24,927 from 2026 (38,364 total), or 55% of 2026 postings and 25% of 2026 unique firms.

Used for: checking whether headline findings survive when you restrict to firms the 2024 and 2026 data both capture, which rules out most "different firms are in 2026" concerns. Thirteen of fifteen tested headlines retain a ratio of at least 0.80 on this cohort (see the [sensitivity framework](sensitivity.md)).

### The within-firm panels (Sample 3)

Three overlapping panels of firms that appear in both 2024 and 2026 at different volume thresholds. These support within-firm longitudinal analysis (did the *same firm* change what it posts?) and are the basis for the central AI-rewriting claim.

| Panel name | Threshold | Firms |
|---|---|---|
| arshkon_min3 | 3 or more postings in arshkon AND 3 or more in 2026 | 243 |
| arshkon_min5 | 5 or more postings in arshkon AND 5 or more in 2026 | 125 |
| pooled_min5 | 5 or more across pooled 2024 AND 5 or more in 2026 | 356 |

Within-firm AI-mention rewriting on these panels: +8.34 percentage points on arshkon_min5, +7.65 on pooled_min5, +7.91 on the returning-firms cohort. Consolidated range: **+7.7 to +8.3 percentage points**.

Caveat: the pooled panel is dominated by asaniczka (four times the size of arshkon). Because asaniczka has a selection characteristic that affects senior-level labels, senior claims are always reported against arshkon-only magnitudes as well as pooled.

### The same-title pair panel (Sample 4)

Pairs of postings from the same firm for the same job title, one from 2024 and one from 2026. Twenty-three such pairs under the main filter, with replication attempts producing 37 pairs under a relaxed filter (+9.98 pp AI-mention rewriting) or 12 pairs under strict arshkon-only filtering (+13.3 pp).

Used for: measuring whether AI language appears in rewrites of the *same job* at the *same firm*. The pattern used is our pattern-based AI-mention detector (see [pattern validation](pattern-validation.md)).

Verdict: the magnitude is reported as a **range of +10 to +13 percentage points**; the direction (same-title drift exceeds firm-level drift) is robust across replications.

### The LLM-labeled subset (Sample 5)

The 48,277 postings where our language-model seniority classifier produced a confident label (not "unknown", not skipped). Required for any analysis that uses the LLM's extracted years-of-experience value as its seniority operationalization.

Caveat: for the 2026 scrape, the classifier labeled only 56.9% of postings; 43% were not selected for routing in the budgeted run. Analyses sensitive to posting text must either restrict to labeled postings or report results on labeled-versus-unlabeled splits.

### The archetype-labeled subset (Sample 6)

An archetype panel with 8,000 postings directly labeled by a topic-modelling pass, 40,223 with labels projected to similar postings, and 19,914 left unassigned.

Used for: archetype-stratified analyses (the "applied AI" and "forward-deployed" senior archetypes surfaced in A5, for example).

Caveat: only 17% of the 48,000 labels are direct; the rest are projected from similar postings. The topic model used is BERTopic (a topic-modelling library that groups documents by shared meaning rather than exact-word overlap). The single largest cluster it produced holds 47% of its output, and on inspection that cluster is a noise bucket labeled "stack-agnostic senior SWE", not a substantive archetype. Archetype claims in the site come from the smaller, cleaner clusters.

### The interview exemplar set (Sample 7)

Forty postings drawn from two concentration points (20 top-ghost-signal postings and 20 exemplars of a specific archetype cluster). Used to build interview artifacts for a later research phase, not for statistical claims.

## The default filter

Unless a finding specifies otherwise, every number on the site comes from postings matching this filter:

```sql
WHERE is_swe = TRUE
  AND source_platform = 'linkedin'
  AND is_english = TRUE
  AND date_flag = 'ok'
```

In English: software-engineering roles, on LinkedIn, in English, with a plausible posting date.

## Seniority definitions

The study measures seniority five different ways on the junior side and five on the senior side, then tests whether findings survive under each. Short codes ("J3", "S4") are used as shorthand across the site; they expand as follows.

| Short code | Plain-English definition | Database column |
|---|---|---|
| J1 | LinkedIn tagged the posting 'entry' natively | `seniority_final = 'entry'` |
| J2 | LinkedIn tagged it 'entry' or 'associate' natively | `seniority_final IN ('entry','associate')` |
| **J3** | The posting asks for two or fewer years of experience (the study's **primary junior definition**) | `yoe_min_years_llm <= 2` |
| J4 | Rule-based years-of-experience floor of two or fewer | `yoe_extracted <= 2` |
| J5 | Title keyword: intern, junior, jr, new grad | (title regex) |
| J3_rule | J3 OR rule-based title keyword | (composite) |
| S1 | LinkedIn tagged it 'mid-senior' natively | `seniority_final = 'mid-senior'` |
| S2 | LinkedIn tagged it 'director' (less than 1% of the corpus; diagnostic only) | `seniority_final = 'director'` |
| S3 | LinkedIn tagged it 'mid-senior' or 'director' | `seniority_final IN ('mid-senior','director')` |
| **S4** | The posting asks for five or more years of experience (the study's **primary senior definition**) | `yoe_min_years_llm >= 5` |
| S5 | Rule-based years-of-experience floor of five or more | `yoe_extracted >= 5` |
| S4_rule | S4 OR rule-based title keyword (senior/staff/principal/lead/architect) | (composite) |

The primary definitions (J3 and S4) are years-of-experience floors, because those are the signals postings actually use and they do not depend on LinkedIn's own tagging conventions. A thirteen-definition robustness panel cross-tabulates these against six analysis groups (78 rows total). Twelve of thirteen definitions point the same direction across periods. See the [sensitivity framework](sensitivity.md) for the full panel.
