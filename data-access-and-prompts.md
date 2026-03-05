# Data Access Without Lightcast: Public Datasets, Scraping Strategy & LLM Prompts

---

## Part 1: Public / Free Datasets Worth Using

### Tier A — Directly usable for your RQs

| Dataset | What it gives you | Coverage | Link |
|---------|-------------------|----------|------|
| **Kaggle: LinkedIn Job Postings** | ~33K structured postings with title, description, seniority, skills, company, industry | 2023–2024 | [kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| **OpenDataBay: Job Posting Skills** | Postings with extracted hard/soft skills via RecAI | Varies | [opendatabay.com](https://www.opendatabay.com/data/ai-ml/d307e964-107d-4754-87df-85c06de4215a) |
| **Indeed Hiring Lab** | Aggregate posting volume by sector (trend lines, not text) | 2020–present | [hiringlab.org](https://www.hiringlab.org) |
| **JOLTS via FRED** | Official monthly job openings by sector (macro control variable) | 2000–present | [fred.stlouisfed.org/series/JTSJOL](https://fred.stlouisfed.org/series/JTSJOL) |
| **Stack Overflow Developer Survey** | Annual survey: tools used, job search, demographics, compensation | 2011–2025 | [survey.stackoverflow.co](https://survey.stackoverflow.co) |

### Tier B — Supplementary / triangulation

| Dataset | Use case | Link |
|---------|----------|------|
| **JobHop (Hugging Face)** | 1.67M career trajectories mapped to ESCO codes — useful for tracking *who gets hired* into junior roles, not just what's posted | [huggingface.co/datasets/aida-ugent/JobHop](https://huggingface.co/datasets/aida-ugent/JobHop) |
| **BigQuery: Stack Overflow Q&A** | Free SQL access. Join with GitHub data. Proxy for junior developer activity (question complexity, tag trends) | [cloud.google.com/blog](https://cloud.google.com/blog/products/gcp/google-bigquery-public-datasets-now-include-stack-overflow-q-a) |
| **GitHub Archive (BigQuery)** | New contributor rate, PR complexity, repo creation — behavioral proxies for Appendix A | [gharchive.org](https://www.gharchive.org) |
| **O*NET Online** | Standardized task/skill taxonomy per occupation — your crosswalk for mapping posting text to task categories | [onetonline.org](https://www.onetonline.org) |

### Tier C — Free samples from commercial providers

| Provider | What you get free | Link |
|----------|-------------------|------|
| **Bright Data** | Sample datasets (Indeed, LinkedIn, Glassdoor) | [brightdata.com/products/datasets/jobs](https://brightdata.com/products/datasets/jobs) |
| **Coresignal** | Free sample of historical multi-source postings | [coresignal.com](https://coresignal.com/alternative-data/job-postings-data/) |
| **Revelio Labs RPLS** | Public aggregate labor statistics (not posting-level, but free) | [reveliolabs.com/public-labor-statistics](https://www.reveliolabs.com/public-labor-statistics/employment/) |

---

## Part 2: Scraping Your Own Dataset

### Recommended tool: JobSpy (open source)

[github.com/speedyapply/JobSpy](https://github.com/speedyapply/JobSpy) — scrapes LinkedIn, Indeed, Glassdoor, Google Jobs, ZipRecruiter concurrently. Returns structured fields (title, company, description, location, salary, date posted).

### What to scrape and how to structure it

For your research design, you need these fields per posting:

```
- date_posted (monthly resolution minimum)
- title (raw — you'll classify seniority from this + description)
- seniority_level (if available; otherwise infer)
- description (full text — this is where the ML value is)
- skills (parsed or raw)
- experience_required (years)
- company_name
- company_size (if available)
- industry
- location
- remote_flag
- SOC_code (map later via O*NET crosswalk)
```

### Scraping strategy for temporal coverage

Your research needs data from 2020–2026. No single scraping run gives you history. Combine:

1. **Kaggle LinkedIn dataset** → covers 2023–2024 (structured, ready to go)
2. **JobSpy scraping** → covers present day forward (start now, accumulate monthly)
3. **Common Crawl** → covers 2020–2023 (archived snapshots of job boards; requires parsing)
4. **Wayback Machine** → spot-check historical company career pages

For Common Crawl: filter WARC files by domain (indeed.com, linkedin.com/jobs, greenhouse.io, lever.co) and extract posting pages. This is engineering-heavy but gives you the deepest historical reach for free.

### Legal notes

The hiQ Labs v. LinkedIn ruling (2022) established that scraping publicly available data does not violate the CFAA. For academic research, you're on solid ground scraping public job boards as long as you don't bypass authentication or CAPTCHA. Don't scrape data behind logins. Respect robots.txt where reasonable, and document your methodology for IRB/ethics review.

---

## Part 3: LLM Prompts

### Prompt 1: Finding and evaluating public datasets

Use this with Claude, ChatGPT, or any reasoning model when you're hunting for data:

```
I am conducting academic labor economics research on how AI coding agents
are restructuring junior software engineer roles. I need job posting data
with the following properties:

REQUIRED FIELDS: date posted, job title, full description text,
experience requirements, skills listed, seniority level, company name,
industry, location, remote/hybrid/onsite flag.

TEMPORAL COVERAGE: Ideally 2020–2026. At minimum, 2022–2026 with monthly
resolution.

OCCUPATIONAL SCOPE: Software engineering roles across all seniority
levels. I also need non-SWE control occupations (civil engineering,
nursing, mechanical engineering) for difference-in-differences analysis.

CONSTRAINTS: I cannot access Lightcast/Burning Glass. I need free or
low-cost datasets, or datasets accessible through academic data libraries.

For each dataset you recommend:
1. Exact name, URL, and access method
2. Which of my required fields it covers (and which it's missing)
3. Temporal coverage and sample size
4. Known biases or limitations (e.g., overrepresents large firms,
   missing gig work, US-only)
5. Whether it has been used in published labor economics research
   (cite the paper if so)
6. How I would need to preprocess it for panel data analysis at
   monthly frequency

Do not recommend datasets I cannot actually access. If a dataset
requires a partnership or paid license, say so explicitly.
```

### Prompt 2: Building a scraping pipeline

Use this when you're ready to build:

```
I need to build a job posting scraper for academic research on junior
SWE labor market restructuring. Here are my requirements:

TARGET SITES: Indeed, LinkedIn (public listings only), Glassdoor,
Google Jobs, Greenhouse/Lever career pages for top 200 tech companies
(by headcount).

DATA TO EXTRACT PER POSTING:
- date_posted, title, full_description, company, location,
  remote_flag, salary_range (if listed), experience_years,
  skills_listed, seniority_level, source_url

TECHNICAL CONSTRAINTS:
- Python-based
- Must respect rate limits and robots.txt
- No login bypass or CAPTCHA solving
- Output: parquet files partitioned by month and source
- Deduplication logic (same role posted on multiple boards)
- Error handling for anti-bot blocks (retry with backoff,
  log failures, don't crash)

HISTORICAL DATA STRATEGY:
- I also need to extract historical postings (2020–2023) from
  Common Crawl WARC files. Provide a pipeline that:
  1. Filters Common Crawl index for job board domains
  2. Downloads relevant WARC segments
  3. Parses HTML into the same schema as live scraping
  4. Deduplicates against live-scraped data

Build this using JobSpy (github.com/speedyapply/JobSpy) for live
scraping and cdx-toolkit or comcrawl for Common Crawl access.
Provide the complete code with clear comments explaining each step.
```

### Prompt 3: Classifying and enriching raw postings with an LLM

Use this for batch classification once you have raw posting text:

```
You are a labor economist's research assistant. I will provide you with
a job posting. Extract the following structured fields. If a field cannot
be determined from the text, return null — do not guess.

{
  "inferred_seniority": "junior|mid|senior|lead|principal|unknown",
  "min_experience_years": <number or null>,
  "max_experience_years": <number or null>,
  "requires_degree": "none|bachelors|masters|phd|unspecified",
  "technical_skills": ["list of specific technologies, languages, frameworks"],
  "meta_skills": ["list of non-technical competencies: system design,
    architecture, mentorship, cross-functional leadership, etc."],
  "ai_tool_mentions": ["any mention of AI tools: Copilot, ChatGPT,
    Claude, cursor, LLM, prompt engineering, AI-augmented, etc."],
  "ownership_scope": "individual_tasks|feature_ownership|end_to_end|
    system_level|unclear",
  "remote_policy": "remote|hybrid|onsite|unspecified",
  "is_ghost_job": <boolean — true if posting appears to have
    contradictory requirements, e.g., junior title but senior
    expectations, or requirements that no real candidate could meet>
}

IMPORTANT: "inferred_seniority" should be based on the RESPONSIBILITIES
and REQUIREMENTS described, NOT the job title. A posting titled
"Junior Developer" that requires 5+ years experience and system design
ownership should be classified as "mid" or "senior".

The field "is_ghost_job" operationalizes the concept from Akanegbu (2026)
— postings where title-level and content-level seniority are mismatched
by 2+ levels, or where listed requirements are internally contradictory.

Here is the posting:
---
[PASTE POSTING TEXT]
---
```

### Prompt 4: Temporal skill migration analysis

Use this after you've classified a batch and want to build the task migration map (RQ2):

```
I have a dataset of classified job postings spanning 2020–2026, each
tagged with:
- date_posted (monthly)
- inferred_seniority (junior/mid/senior)
- technical_skills (list)
- meta_skills (list)
- ai_tool_mentions (list)

I need to construct a TASK MIGRATION MAP showing when specific
competencies first appeared in junior postings and at what prevalence.

For each skill in my taxonomy:
1. Calculate monthly prevalence in junior postings
   (% of junior postings mentioning that skill)
2. Identify the first month it crosses 5%, 10%, and 25% thresholds
3. Compare against the same skill's prevalence in senior postings
   at the same dates (convergence metric)
4. Flag skills where junior prevalence is INCREASING while senior
   prevalence is STABLE or DECREASING (these are migrating skills,
   not just trending skills)

Produce:
- A heatmap: skills × quarters, colored by prevalence in junior postings
- A migration timeline: ordered list of skills by date they crossed
  the 10% threshold in junior postings
- A convergence plot: junior-senior prevalence gap over time per skill

Use Python with pandas, matplotlib/seaborn. Provide complete,
runnable code. Handle edge cases: skills with zero prevalence in
early periods, skills that appear and then disappear, and skills
present only in one seniority band.
```

---

## Part 4: Realistic Assessment

**What you can do without Lightcast:**

| RQ | Feasibility with free data | Limiting factor |
|----|---------------------------|-----------------|
| RQ1 (disappearing vs. redefined) | **High** — Kaggle LinkedIn + JobSpy scraping gives you enough posting text to train the seniority classifier | Sample size for pre-2023 period |
| RQ2 (task migration) | **Medium-High** — depends on temporal coverage. 2023–2026 is solid. 2020–2022 requires Common Crawl or free samples | Common Crawl parsing effort |
| RQ3 (structural break) | **Medium** — you need dense monthly observations. Aggregate data (Indeed Hiring Lab, JOLTS) can detect volume breaks but not content breaks | Content-level ITS needs posting text, not aggregates |
| RQ4 (SWE-specific) | **High for volume DiD** (JOLTS covers all occupations), **Medium for content DiD** (need non-SWE posting text) | Non-SWE posting text is available on Kaggle LinkedIn |
| RQ5 (training implications) | **High** — conditional on RQ1–4, no additional data needed | N/A |

**Honest gap:** Without Lightcast, your biggest vulnerability is pre-2023 posting text at scale. The Kaggle LinkedIn dataset starts in 2023. Common Crawl can fill the gap but requires meaningful engineering. If you want a cleaner path, request academic access to Revelio Labs — they have a research API and are more accessible than Lightcast for academic partnerships.
