# Stage 1 freeze — 2026-05-06

## 1. Headline K and the §4.4 sweep

| k_target | n_clusters | noise_rate | seed_pair_ari_mean | per_period_centroid_alignment_mean |
|---|---|---|---|---|
| 10 | 9 | 0.314 | 0.699 | 0.913 |
| 15 | 14 | 0.314 | 0.638 | 0.942 |
| 20 | 19 | 0.314 | 0.682 | 0.932 |
| 25 | 24 | 0.314 | 0.703 | 0.940 |
| 30 | 29 | 0.314 | 0.690 | 0.946 |
| 40 | 39 | 0.314 | 0.701 | 0.946 |
| 50 | 49 | 0.314 | 0.696 | 0.935 |
| 75 | 74 | 0.314 | 0.700 | 0.940 |


**Headline K = 10**, super-family K = 10.

## 2. `min_cluster_size` sweep

| min_cluster_size | n_clusters_raw | noise_rate_raw | n_clusters_k30 | adjacent_ari_k30 |
|---|---|---|---|---|
| 10 | 873 | 0.382 | 29 | 0.381 |
| 20 | 396 | 0.346 | 29 | 0.534 |
| 30 | 267 | 0.338 | 29 | 0.496 |
| 50 | 154 | 0.306 | 29 | 0.722 |
| 70 | 115 | 0.314 | 29 | nan |


**Headline mcs = 70**.

## 3. Noise rate (raw HDBSCAN at headline)

**31.4 %** before any reduce_outliers.

## 4. Seed-pair ARI at headline K

| pair | ari | centroid_alignment |
|---|---|---|
| 42_vs_1337 | 0.753 | 1.000 |
| 42_vs_2026 | 0.675 | 0.989 |
| 1337_vs_2026 | 0.670 | 0.989 |


## 5. Mega-cluster check

Largest cluster share at headline K: **26.1 %**. Gate (≤ 30 %) passed.

## 6. AI-region structure

AI-flavoured clusters detected by c-TF-IDF top-words: [0, 6, 7].

## 7. Determinism

Double-run identical: **True**, ARI = 1.0000.

## 8. Flags for Stage 2

- T-l1l2 queued: `role_family_l1` and `skill_theme_*` are not yet populated in `unified_core.parquet`.
- Author interpretability rating deferred (autonomous-run authorisation): K selection used three of four §4.4 criteria.
- AI-cluster sub-structure (§1.4.4): 3 cluster(s) flagged AI-flavoured at headline K; T-axis and T-anchor will quantify.

## 9. Stage 1 cluster catalog


### Cluster 0 — AI Software Engineering (n = 15055, 2024 = 4367, 2026 = 10688)

- top words: ai, software engineering, automation, engineers, software development, engineer, expertise, engineering, devops, architecture
- top firms: Google, Capital One, Amazon Web Services (AWS)
- top metros: San Francisco Bay Area, Seattle Metro, Washington DC Metro

### Cluster 1 — Test Automation Engineer (n = 7843, 2024 = 3658, 2026 = 4185)

- top words: software engineer, software development, test automation, agile, testing, development, software, engineers, tests, programming
- top firms: Jobs via Dice, ClearanceJobs, Anduril Industries
- top metros: Washington DC Metro, San Francisco Bay Area, Los Angeles Metro

### Cluster 2 — Data Engineer (n = 6782, 2024 = 2460, 2026 = 4322)

- top words: data engineer, data engineering, data scientist, data pipelines, data science, databricks, data, analytics, aws, datasets
- top firms: Jobs via Dice, Amazon, Intuit
- top metros: Washington DC Metro, San Francisco Bay Area, Dallas-Fort Worth Metro

### Cluster 3 — Salesforce Cloud Developer (n = 4232, 2024 = 2462, 2026 = 1770)

- top words: developer, cloud, salesforce, azure, aws, developing, development, develop, frameworks, agile
- top firms: Jobs via Dice, Tata Consultancy Services, Apex Systems
- top metros: Dallas-Fort Worth Metro, Washington DC Metro, Charlotte Metro

### Cluster 4 — Full Stack Developer (n = 2807, 2024 = 1598, 2026 = 1209)

- top words: developer, developers, front-end, engineer, frontend, web applications, skills, frameworks, backend, framework
- top firms: Jobs via Dice, Syrinx Consulting, Motion Recruitment
- top metros: San Francisco Bay Area, Dallas-Fort Worth Metro, Washington DC Metro

### Cluster 5 — Mobile Application Developer (n = 1058, 2024 = 415, 2026 = 643)

- top words: mobile development, react native, mobile applications, mobile apps, ios android, mobile application, developer, mobile, ios, mobile app
- top firms: Jobs via Dice, Capgemini, Motion Recruitment
- top metros: San Francisco Bay Area, Dallas-Fort Worth Metro, Atlanta Metro

### Cluster 6 — E-commerce Software Engineering (n = 856, 2024 = 141, 2026 = 715)

- top words: tiktok, programming, computer science, e-commerce, internship, software, ai, machine learning, automation, degree computer science
- top firms: TikTok, ByteDance, TikTok USDS Joint Venture
- top metros: San Francisco Bay Area, Seattle Metro, Washington DC Metro

### Cluster 7 — Application Systems Analyst (n = 607, 2024 = 482, 2026 = 125)

- top words: systems analyst, application, applications, software, development, analyst, programs, documentation, duties, maintenance
- top firms: Fidelity Investments, Jobs via Dice, Broadcom
- top metros: Chicago Metro, Dallas-Fort Worth Metro, Washington DC Metro

### Cluster 8 — ServiceNow Developer (n = 382, 2024 = 209, 2026 = 173)

- top words: servicenow development, servicenow developer, servicenow platform, servicenow, servicenow modules, servicenow applications, service management, servicenow certified, service, years servicenow
- top firms: Wells Fargo, ServiceNow, Alldus
- top metros: Washington DC Metro, Dallas-Fort Worth Metro, San Francisco Bay Area



## Self-sign

Per the 2026-05-06 user authorisation (memory: `project_bertopic_run.md`), the orchestrator self-signs and proceeds to Stage 2. Sign-off rationale: the seed gate, mega-cluster gate, and determinism check completed; per-period centroid alignment at headline K is recorded in §1; the cluster catalog is sanity-coherent on c-TF-IDF top-words. If any gate had failed we would have fallen back to super-family granularity rather than launched Stage 2.
