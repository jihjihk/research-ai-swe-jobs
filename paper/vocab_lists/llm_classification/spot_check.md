# LLM classification pilot — spot-check

_25 postings × 3 models × 3 reps. Read each block, eyeball whether the LLM-majority labels make sense given the description._

Stratification:
- **A_single** (1-8): one regex topic only — easy single-label cases.
- **B_multi** (9-13): 3+ regex topics — multi-label stress test.
- **C_zero** (14-18): zero regex topics — false-positive/recall risk.
- **D_noise** (19-23): hits persistent-fluff regex concepts — does the LLM resist the fluff?
- **E_random** (24-25): sanity baseline.

---

## #1  [A_single]  Principal Programmer Analyst

**uid:** `asaniczka_7b7a59d16b282ef0`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** City of North Las Vegas

**Description (truncated):**
> Work Schedule - | This position is scheduled to work 4 days per week, 9 hours per day, Monday through Thursday, 8:00am to 6:00pm Supervises application support and development teams that support the City’s internal and external applications that utilize SQL Server and/or Oracle Database with Microsoft.net technologies. Directs and leads by planning, organizing and evaluating the team’s work which includes a wide range of routine to complex configuration, software, connectivity and applications support services. Bachelor’s degree in Computer Science, Management Information Systems, Engineering, Mathematics, or a closely related field and seven years of developer experience with knowledge, skills and experience required to include coding and troubleshooting interfaces and extensions of an Enterprise Application running on Oracle or SQL Server Database. A minimum of two years developing in C#.Net, ASP.Net, and SQL languages is required. Enterprise Applications are those that service multiple departments and have a user base exceeding 25 concurrent users and one (1) year of experience overseeing technical projects and staff May substitute 8 years of work experience for the education bu …

**Regex topics:** `people_management`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `people_management` | {people_management} · {people_management} · {people_management} |
| `gpt-5.4-mini` | `legacy_stack`, `people_management`, `process_scaffolding` | {legacy_stack,people_management} · {people_management,process_scaffolding} · {legacy_stack,people_management,process_scaffolding} |
| `gpt-5.4-nano` | `people_management` | {legacy_stack,people_management} · {people_management} · {people_management} |

---

## #2  [A_single]  Android Developer

**uid:** `arshkon_3891025227`  · **source:** `kaggle_arshkon/2024-04`  · **company:** TechMatrix Inc

**Description (truncated):**
> - Experience in using Eclipse, SQLite, and Android Studio. - Hands-on with Kotlin and Java is a must - Good knowledge of MVVM, MVC, MVP architecture, Singleton, and Delegation design patterns. - Good knowledge of Android Architecture Components (Room, Live Data, View Model and Data binding). - Having Strong experience in integrating web services(SOAP,REST) with android applications - Excellent knowledge of Object-Oriented Concepts and Core Java Technologies.

**Regex topics:** `orchestration`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | `orchestration` | {orchestration} · {} · {orchestration} |
| `gpt-5.4-nano` | _(none)_ | {} · {} · {} |

---

## #3  [A_single]  DevOps Engineer

**uid:** `linkedin_li-4399275584`  · **source:** `scraped/2026-04`  · **company:** EPITEC

**Description (truncated):**
> **W2 CONTRACT ONLY** **OBJECTIVES** | * Setup, maintenance and update of React data collection platform. * Implementing and maintaining new Continuous Integration tools and infrastructure. * Understanding the changing needs of developers and translating them into action items. * Creating new ways to automate and improve the Continuous Integration process. * Testing and examining code written by others * Identifying technical problems and developing updates and fixes * Working with other teams in the organization to ensure adherence to our best practices **SKILLS REQUIRED** | \| 5-7\+ years Linux, GCP, GitHub, Jenkins, Cloud Infrastructure, Python, Data Analysis, Data Collection, React, JavaScript

**Regex topics:** `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `verification` | {verification} · {verification} · {verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `process_scaffolding`, `verification` | {context_infrastructure,process_scaffolding,verification} · {context_infrastructure,process_scaffolding,verification} · {context_infrastructure,verification} |
| `gpt-5.4-nano` | `process_scaffolding`, `verification` | {context_infrastructure,orchestration,verification} · {process_scaffolding,verification} · {process_scaffolding,verification} |

---

## #4  [A_single]  Lead Software Engineer, Full Stack (Java, Python, AWS)

**uid:** `asaniczka_3db8d81794a97bcd`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** Capital One

**Description (truncated):**
> This position is open to candidates who reside in and have the legal right to work in the country where the job is located. Lead Software Engineer, Full Stack (Java, Python, AWS) At Capital One, we believe in inclusivity, diversity, and innovation. We are looking for Full Stack Software Engineers who are passionate about technology and helping us solve complex business problems in a collaborative, inclusive, and fast-paced environment. As a Lead Software Engineer, you will play a key role in driving a major transformation within Capital One. What You'll Do: Lead a diverse team of developers with deep experience in distributed microservices and full stack systems to create solutions that meet regulatory needs Stay on top of tech trends and experiment with new technologies, while also mentoring other members of the engineering community Collaborate with digital product managers to deliver robust cloud-based solutions that empower millions of Americans to achieve financial stability Utilize programming languages like JavaScript, Java, HTML/CSS, TypeScript, SQL, Python, and Go, as well as AWS tools and services Basic Qualifications: Bachelor’s Degree At least 6 years of software engine …

**Regex topics:** `mentorship`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `mentorship`, `people_management` | {mentorship,people_management} · {mentorship,people_management} · {mentorship,people_management,verification} |
| `gpt-5.4-mini` | `mentorship`, `people_management` | {mentorship,people_management} · {mentorship,people_management} · {mentorship,people_management} |
| `gpt-5.4-nano` | `legacy_stack`, `mentorship`, `orchestration`, `people_management` | {legacy_stack,mentorship,orchestration,people_management,process_scaffolding} · {legacy_stack,mentorship,orchestration,people_management} · {legacy_stack,mentorship,orchestration,people_management} |

---

## #5  [A_single]  Cyber Security Engineer

**uid:** `linkedin_li-4398109111`  · **source:** `scraped/2026-04`  · **company:** Haystack

**Description (truncated):**
> The Role | - Reverse engineer software, firmware, and hardware to uncover vulnerabilities. - Collaborate on modifying device firmware and software for enhanced functionalities. - Operationalize proof-of-concept code through testing, documentation, and system integration. - Articulate complex technical concepts to diverse audiences via papers and presentations. What You'll Need - Bachelor's degree in Computer Science, Electrical Engineering, or a related field. - At least 2 years of experience in reverse engineering or offensive cyber capabilities development. - Strong understanding of operating system fundamentals, internals, and exploitation techniques. - Development experience in low-level languages like assembly or C/C\+\+. - Proficiency in using software or hardware debuggers. - Excellent written and verbal communication skills. - Ability to obtain a Secret-level security clearance, with potential for TS/SCI.

**Regex topics:** `performance`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `performance`, `verification` | {performance,verification} · {performance,verification} · {performance,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `performance`, `verification` | {context_infrastructure,performance,verification} · {context_infrastructure,performance,verification} · {context_infrastructure,performance,verification} |
| `gpt-5.4-nano` | `verification` | {verification} · {verification} · {context_infrastructure,performance,verification} |

---

## #6  [A_single]  Software Developer with React

**uid:** `linkedin_li-4396830961`  · **source:** `scraped/2026-04`  · **company:** The Marlin Alliance, Inc.

**Description (truncated):**
> **Software Developer** with React experience to join our team in San Diego, CA. This position supports our Navy customer and requires the ability to obtain a US Secret security clearance. We are looking for motivated individuals to lead and support digital transformation, data science and analytics, and automation projects for variety of Navy clients. Individuals must be able to function in a fast-paced work environment and able to adapt quickly to rapidly changing requirements and technologies. Using your comprehensive knowledge of various technologies, you will design, develop, and implement solutions to support Navy mission owners in their digital transformation journey. **Required Qualifications** * Excellent problem solver * Experienced in React 15\+ and Typescript * Takes pride in writing clean reusable code * At least one major project involving React under your belt * Experience with version control software such as Git **Preferred Qualifications** | * Experienced in Scrum * Experience contributing to open-source projects * Experience with a static typed language such as C#, C, C\+\+, or Java * Experience working on both Front-End and Back-End technologies such as web frame …

**Regex topics:** `process_scaffolding`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `process_scaffolding` | {process_scaffolding} · {process_scaffolding} · {process_scaffolding} |
| `gpt-5.4-mini` | `process_scaffolding` | {process_scaffolding} · {process_scaffolding} · {process_scaffolding} |
| `gpt-5.4-nano` | `process_scaffolding` | {context_infrastructure,orchestration,process_scaffolding,verification} · {process_scaffolding} · {process_scaffolding} |

---

## #7  [A_single]  Associate Database Administrator (226109)

**uid:** `arshkon_3903816638`  · **source:** `kaggle_arshkon/2024-04`  · **company:** Medix Technology

**Description (truncated):**
> They are seeking an Associate Database Administrator who can maintain their current Revenue Collections system (Ontario Systems) - including data maintenance, automation in VBA/Excel Macros, and integrations with other systems in the law firm. RESPONSIBILITIESProviding the day to day operations maintenance for a Revenue Management/ Collections Management database Working in a solutions-minded capacity, to identify opportunities to automate and make data processes more accurate or efficient, using VBA programming and Excel MacrosWorking with an in-depth knowledge of Microsoft Excel (current state), this role will also be participating in building and implementing a new future state database solution in Supabase (PostgreSQL)Supporting and partnering with Data Analysts on reporting requests, for both regular and ad-hoc reports Analyzing a variety of healthcare data including (but not limited to) trends and statistics on claims, account collections, and referrals REQUIRED QUALIFICATIONSRequired Bachelor Degree in a Computer Science or IT related fieldRequired 3 years of professional experience as a Data Analyst / Database Administrator Required strong capabilities in Microsoft Excel an …

**Regex topics:** `legacy_stack`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {legacy_stack} |
| `gpt-5.4-mini` | `orchestration` | {orchestration} · {orchestration} · {orchestration} |
| `gpt-5.4-nano` | `orchestration` | {orchestration} · {orchestration} · {orchestration} |

---

## #8  [A_single]  Data Engineer

**uid:** `arshkon_3891075843`  · **source:** `kaggle_arshkon/2024-04`  · **company:** AtekIT

**Description (truncated):**
> 11+ years in Data Engineering and AnalyticsExpertise in data analytical skills and handling big data along with real time streamingGraph Ontology and semantic modeling with GraphQL or SPARQL experience is desirable.Proactive, self-driven, works independently and collaborates wellExpertise in Python, PysparkUse of databricks is a mustclient - AT&T

**Regex topics:** `context_infrastructure`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | _(none)_ | {} · {} · {} |
| `gpt-5.4-nano` | _(none)_ | {} · {} · {} |

---

## #9  [B_multi]  Lead Forward Deployed Software Engineer

**uid:** `linkedin_li-4281730145`  · **source:** `scraped/2026-04`  · **company:** AMD

**Description (truncated):**
> Specialized in AI optimization, fine-tuning large language models to unlock unprecedented Generative AI efficiency. Our expertise extends beyond the hardware realm, encompassing 3P enablement, where we develop custom AI Software Solutions for Industry leading AI customers. **The Role** As a | **Forward Deployed Software Engineer** , you will work closely with our most strategic partners as a hands-on technical expert. You are responsible for turning AMD’s cutting-edge AI technology into tangible business value. This role is a unique blend of customer relationship skills and elite software engineer; you will work side-by-side with clients to help them prove out and ultimately deploy AI solutions on AMD GPUs. You will be the trusted technical advisor and hands-on developer who makes it happen. **In This Role, You Will** * Work closely with strategic customers to understand their requirements challenges and identify opportunities for AMD hardware and software to provide value. * Close gaps in the AMD software stack needed to support customer solutions. * Work hands-on as a technical expert and creative problem-solver, developing side-by-side with customers to drive projects from proof …

**Regex topics:** `orchestration`, `performance`, `process_scaffolding`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `performance`, `process_scaffolding`, `verification` | {performance,process_scaffolding,verification} · {performance,process_scaffolding,verification} · {performance,process_scaffolding,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} |
| `gpt-5.4-nano` | `orchestration`, `performance`, `process_scaffolding` | {orchestration,performance,process_scaffolding} · {performance,process_scaffolding} · {context_infrastructure,orchestration,performance,process_scaffolding} |

---

## #10  [B_multi]  Data Architect

**uid:** `asaniczka_9c580a1386e7adb4`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** Agile Staffing Groups

**Description (truncated):**
> Our client, a leading manufacturing software development company has a requirement for an experienced Data Architect in Ann Arbor, MI. Ann Arbor, MI | Duration: 12 months About The Role | As a Data Architect you will be leading the effort to establish world class data foundations for some of the largest manufacturers in the world. If working with real time streaming plant data using cutting edge ETL pipeline software tools, and love asking and answering questions using data, please apply for this position. You will make a significant impact on how company pushes forward the field of manufacturing analytics. What You Will Do |  Lead the technical delivery of the Company's data platform for large manufacturing enterprises through the customer lifecycle: discovery, implementation, expansion, and ongoing support |  Investigate, diagnose, and resolve data challenges using common data mining techniques that often involve creating custom python notebooks or constructing complex SQL queries |  Collaborate with product and platform engineering teams to define and optimize new features at scale |  Design and build complex, streaming data pipelines that synthesize disparate manufacturing …

**Regex topics:** `context_infrastructure`, `orchestration`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `orchestration` | {context_infrastructure,orchestration} · {context_infrastructure,orchestration} · {context_infrastructure,orchestration} |
| `gpt-5.4-mini` | `context_infrastructure`, `orchestration` | {context_infrastructure,orchestration} · {context_infrastructure,orchestration} · {context_infrastructure,orchestration} |
| `gpt-5.4-nano` | `context_infrastructure`, `orchestration` | {context_infrastructure,orchestration} · {context_infrastructure,orchestration,performance} · {context_infrastructure,orchestration} |

---

## #11  [B_multi]  Senior Data Engineer, WWDE

**uid:** `arshkon_3904084737`  · **source:** `kaggle_arshkon/2024-04`  · **company:** Amazon

**Description (truncated):**
> WWDE engineers solve complex problems and build scalable, cutting edge solutions to help our customers navigate through issues and eliminate systemic defects to prevent future issues. As a Senior Data Engineer, you will partner with Software Developers, Business Intelligence Engineers, Scientists, and Program Managers to develop scalable and maintainable data pipelines on both structured and unstructured (text based) data. The ideal candidate has strong business judgment, good sense of architectural design, written/documentation skills, and experience with big data technologies (Spark/Hive, Redshift, EMR, +Other AWS technologies). This role involves both overseeing existing pipelines as well as developing brand new ones for ML). Basic Qualifications 5+ years of data engineering experience Experience with data modeling, warehousing and building ETL pipelines Experience with SQL Experience in at least one modern scripting or programming language, such as Python, Java, Scala, or NodeJS Experience providing technical leadership and mentoring other engineers for best practices on data engineering Bachelor's Degree Preferred Qualifications | Experience with big data technologies such as: …

**Regex topics:** `context_infrastructure`, `mentorship`, `orchestration`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `mentorship`, `orchestration` | {context_infrastructure,mentorship,orchestration} · {mentorship,orchestration} · {context_infrastructure,mentorship,orchestration} |
| `gpt-5.4-mini` | `context_infrastructure`, `mentorship`, `orchestration` | {context_infrastructure,mentorship,orchestration} · {context_infrastructure,mentorship,orchestration} · {context_infrastructure,mentorship,orchestration} |
| `gpt-5.4-nano` | `context_infrastructure`, `mentorship`, `orchestration` | {mentorship,orchestration} · {context_infrastructure,mentorship} · {context_infrastructure,mentorship,orchestration} |

---

## #12  [B_multi]  Junior Test Automation Engineer

**uid:** `asaniczka_0c98df9033ade4a5`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** LanceSoft, Inc.

**Description (truncated):**
> Must have at least 3+ years of work Experience as Test Automation Engineer/Application Engineer Job responsibilities: | - o Analyze customer requirements, technical documentation, BOM setup and verify First Article orders are accurate to ensure customer requirements are met. o Independently develop automated process to configure hardware and software to meet customer’s technical specifications with existing automation production environment. o Create work instructions used by manufacturing to assemble and configure product. o Analyze technical data to determine appropriate setup criteria for platform programming. o Develop new testing and automation methods for leading edge products. o Daily monitoring and resolution of technical issues and validation of engineering changes that arise from revised requirements. o Required to author technical documents such as: Manufacturing Work Instruction, Workmanship Standards, Engineering Bulletins, ECO, FQR and Failure Analysis Reports o Monitors and reports on productivity along with quality of First Article/Prototype orders  4-year bachelor's degree in computer science/software engineering or related study.  Experience working with PowerSh …

**Regex topics:** `context_infrastructure`, `legacy_stack`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `legacy_stack`, `verification` | {context_infrastructure,legacy_stack,verification} · {context_infrastructure,legacy_stack,verification} · {context_infrastructure,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `legacy_stack`, `verification` | {context_infrastructure,legacy_stack,verification} · {context_infrastructure,legacy_stack,verification} · {context_infrastructure,legacy_stack,process_scaffolding,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} · {context_infrastructure,verification} |

---

## #13  [B_multi]  Staff Data Scientist - Product

**uid:** `linkedin_li-4359781208`  · **source:** `scraped/2026-04`  · **company:** Gusto

**Description (truncated):**
> **About The Role** Gusto is looking for highly skilled and motivated Data Scientists with extensive experience (7\+ years) applying their expertise in a business environment. As a Staff Data Scientist, you will play a crucial role in leveraging experimentation, statistical inference, and causal analysis to drive strategic decision making that contributes to the overall success of our organization. The ideal candidate is a trusted data storyteller with strong statistical and coding skills, and a passion for applying these skills to help small businesses thrive. **About The Team** In this role you will work closely with our Product, Engineering, Design, Finance, and other Data teams to become an expert in the data for your domain, define and track metrics that help us understand our business performance, and dive deep into our Payroll, Benefits, and HR data to deliver insights and answer questions. You’ll also integrate AI-assisted practices to accelerate analysis, enhance rigor, and expand the reach of insights across Gusto. **Here’s What You’ll Do Day-to-day** * Lead: Own ambiguous problems, design analysis frameworks, and introduce structure that scales across multiple product dom …

**Regex topics:** `context_infrastructure`, `mentorship`, `orchestration`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `mentorship` | {mentorship} · {mentorship} · {mentorship} |
| `gpt-5.4-mini` | `mentorship`, `orchestration` | {mentorship,orchestration} · {mentorship,orchestration} · {mentorship,orchestration,verification} |
| `gpt-5.4-nano` | `mentorship`, `orchestration` | {mentorship,orchestration} · {mentorship,orchestration} · {context_infrastructure,mentorship,orchestration} |

---

## #14  [C_zero]  NodeJS developer

**uid:** `arshkon_3903458140`  · **source:** `kaggle_arshkon/2024-04`  · **company:** Diamondpick

**Description (truncated):**
> Role Description This is a full-time on-site role for a NodeJS Developer at Diamondpick in Bentonville, AR. As a NodeJS Developer, you will be responsible for front-end and back-end web development, software development, and JavaScript programming. You will also work with Redux.js to develop and maintain web applications. Qualifications Front-End Development and Back-End Web Development skillsSoftware Development skillsJavaScript and Redux.js programming skillsExperience in developing and maintaining web applicationsStrong problem-solving and analytical skillsExcellent teamwork and communication skillsBachelor's degree in Computer Science, Engineering, or a related fieldRelevant certifications in web development or software development are a plus

**Regex topics:** _(none)_

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | _(none)_ | {} · {} · {} |
| `gpt-5.4-nano` | _(none)_ | {} · {} · {} |

---

## #15  [C_zero]  Embedded Sensor/EW Real-Time Software Engineer

**uid:** `arshkon_3904049939`  · **source:** `kaggle_arshkon/2024-04`  · **company:** Elegant Enterprise-Wide Solutions, Inc.

**Description (truncated):**
> Required Minimum QualificationsExcellent written and verbal communication skills.Hands-on experience with modern Radar technologies.Hands-on experience with one or more of C++, C, C#, MATLAB.Familiarity with Windows and Linux Operating Systems.Make/CMake/Studio (or similar build system). GIT and GitLab (or similar version control system). Strong analytical skills.Experience working with containerized environments. Preferred QualificationsActive Secret Clearance. Travel Requirements<10% travel Education and Length of Experience14 years of related experience with a Bachelor’s degree in Electrical Engineering, Mechanical Engineering, Physics, or Computer Science.12 years of related experience with a Masters’ degree in Electrical Engineering, Mechanical Engineering, Physics, or Computer Science.9 years of related experience with a Ph.D. in Electrical Engineering, Mechanical Engineering, Physics, or Computer Science. U.S. Citizenship RequirementsDue to our research contracts with the U.S. federal government, candidates for this position must be U.S. Citizens. Clearance Type RequiredCandidates must be able to obtain and maintain an active security clearance.

**Regex topics:** _(none)_

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | _(none)_ | {} · {} · {} |
| `gpt-5.4-nano` | _(none)_ | {} · {} · {} |

---

## #16  [C_zero]  Principal Software Engineer React/Node/AI

**uid:** `linkedin_li-4375021205`  · **source:** `scraped/2026-04`  · **company:** Motion Recruitment

**Description (truncated):**
> **Job Description** In this role, you’ll work on integrating user interfaces for a cutting-edge product built on top of an AI powered large language model (LLM). You’ll collaborate directly with technical leadership and play an active role in shaping the direction of the platform. * Highly skilled in React and Node * Skilled in Vercel * Exposure to GraphQL * Exposure to LLMs * Experience working with * 8\+ years of Software Engineering * Degree in a relevant field Daily Responsibilities | * 100% Hands On Alexander Rachalski

**Regex topics:** _(none)_

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | _(none)_ | {orchestration} · {} · {} |
| `gpt-5.4-nano` | _(none)_ | {} · {} · {orchestration} |

---

## #17  [C_zero]  Full Stack Developer - AI Lab

**uid:** `linkedin_li-4339055672`  · **source:** `scraped/2026-04`  · **company:** Stewart Title

**Description (truncated):**
> **Job Description** **Job Summary** Central information technology organization, providing the network infrastructure, hardware, software and enterprise services for offices to run their business. Provides network and database administration, device management, and administers processes, services and technical support for hardware and software for both the organization's internal and external clients. **Job Responsibilities** | * Provides comprehensive application software development services and/or technical support on moderately complex projects and initiatives * Analyzes, modifies and may develop program logic for existing applications, programs and enhancements * Competent to work at the highest technical level of some phases of applications programming activities * Performs specialized assignments; solves complex problems and develops non-traditional solutions through sophisticated analytical thinking * Interprets internal/external business environment * Recommends best practices to improve processes or services * Impacts achievements of customer, operational, project or service objectives * Communicates difficult concepts to team to generate clarity and alignment on projects …

**Regex topics:** _(none)_

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | _(none)_ | {} · {} · {} |
| `gpt-5.4-mini` | _(none)_ | {} · {} · {} |
| `gpt-5.4-nano` | _(none)_ | {} · {process_scaffolding} · {} |

---

## #18  [C_zero]  Web Application Developer Senior

**uid:** `linkedin_li-4393082303`  · **source:** `scraped/2026-04`  · **company:** Jobs via Dice

**Description (truncated):**
> ECS is seeking a | **Web Application Developer Senior** **Please Note:** | **This position is contingent upon contract award.** Responsible for designing, testing, and developing well designed web based software coding that is testable, and efficient by using best software development practices to meet user's needs; recommending software upgrades for current and future systems; as well as translating the User Interface (UI)/User Experience (UX) design wireframes, ensuring software continues to function normally through software maintenance and testing; and documenting every aspect of the application, service or environment as a reference for future maintenance and upgrades. **Required Skills** | * Bachelor's degree * TS/SCI clearance * Minimum of 2 years experience in the past 6 years in one of the following fields of expertise: + AWS Cloud based solutions + C# | + Java + Python **Desired Skills**

**Regex topics:** _(none)_

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `verification` | {context_infrastructure,verification} · {context_infrastructure,verification} · {context_infrastructure,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `verification` | {context_infrastructure,verification} · {context_infrastructure,verification} · {context_infrastructure,orchestration,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `verification` | {context_infrastructure,verification} · {context_infrastructure,verification} · {context_infrastructure,verification} |

---

## #19  [D_noise]  Sr. Software Engineer

**uid:** `asaniczka_bff9db288ace229c`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** A-Line Staffing Solutions

**Description (truncated):**
> Title: Senior Software Developer - AWS The Software Engineer Provides counsel and advice to top management on significant Engineering matters, often requiring coordination between organizations. Designs and develops a consolidated, conformed enterprise data warehouse and data lake which store all critical data across Customer, Provider, Claims, Client and Benefits data. Manages processes that are highly complex and impact the greater organization. Designs, develops and implements methods, processes, tools and analyses to sift through large amounts of data stored in a data warehouse or data mart to find relationships and patterns. May lead or manage sizable projects. Participates in the delivery of the definitive enterprise information environment that enables strategic decision-making capabilities across enterprise via an analytics and reporting. Focuses on providing thought leadership and technical expertise across multiple disciplines. Recognized internally as “the go-to person” for the most complex Information Management assignments. Job Responsibilities | The job position will report into the Provider Value Stream (PVS) solution architecture organization. This role will provide …

**Regex topics:** `context_infrastructure`, `mentorship`, `orchestration`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,orchestration,performance,process_scaffolding,verification} |

---

## #20  [D_noise]  AI Process Engineer

**uid:** `linkedin_li-4366631950`  · **source:** `scraped/2026-04`  · **company:** ICE

**Description (truncated):**
> **Job Purpose** This hands-on engineering role is responsible for designing, building, and implementing AI-powered workflow automation solutions to drive operational and financial value across Infrastructure Operations. The AI Process Engineer will lead technical initiatives, deliver scalable automation, and serve as a subject matter expert for workflow automation and AI integration across ICE’s mission-critical environments. The role requires excellent analytical, troubleshooting, and communication skills, as well as the ability to work independently, manage multiple priorities, and demonstrate a strong sense of ownership and accountability. **Responsibilities** | * Lead the end-to-end delivery of AI and automation projects, from ideation and requirements gathering through deployment and monitoring. * Design, develop, and maintain robust workflows in collaboration with other engineers and business users. * Serve as the technical SME for AI workflow automation, providing guidance on best practices. * Gather and apply lessons learned from organizational feedback, ensuring continuous improvement in AI build initiatives. * Ensure that all automation solutions comply with ICE’s securit …

**Regex topics:** `context_infrastructure`, `orchestration`, `performance`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `orchestration`, `process_scaffolding`, `verification` | {orchestration,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `orchestration`, `process_scaffolding`, `verification` | {context_infrastructure,orchestration,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `orchestration`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,orchestration,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} · {context_infrastructure,orchestration,process_scaffolding,verification} |

---

## #21  [D_noise]  OH - Sr Software Developer (Python, Flask, Azure Cloud native Services) - 798828

**uid:** `linkedin_li-4393040538`  · **source:** `scraped/2026-04`  · **company:** SR International Inc

**Description (truncated):**
> **Job Title: Sr Software Developer (Python, Flask, Azure Cloud native Services)** * The candidate will serve as a Senior Software Developer and is responsible for designing, developing, and implementing scalable applications leveraging Python, Flask, and cloud-native services within the Microsoft Azure environment. * The role includes collaborating with internal stakeholders, architects, and project teams to deliver high-quality solutions that meet business and technical requirements. **Responsibilities** | * Design, develop, test, and deploy scalable web applications using Python and Flask frameworks. * Develop and integrate solutions leveraging Microsoft Azure services, including compute, storage, and serverless components. * Utilize AI-assisted development tools, including Claude, to enhance coding efficiency, solution design, and documentation. * Collaborate with business analysts and stakeholders to translate functional requirements into technical solutions. * Participate in all phases of the software development lifecycle, including requirements analysis, design, development, testing, and production support. * Develop and maintain APIs and integrations with internal and exter …

**Regex topics:** `context_infrastructure`, `mentorship`, `orchestration`, `performance`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `mentorship`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `mentorship`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `mentorship`, `orchestration`, `performance`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,mentorship,orchestration,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,performance,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,performance,process_scaffolding,verification} |

---

## #22  [D_noise]  Google Cloud Platform Developer

**uid:** `asaniczka_19a33798789ede99`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** Experfy

**Description (truncated):**
> Key Responsibilities : 1- Design, build and configure applications to meet business process and application requirements 1-Deep knowledge of AngularJS practices and commonly used modules based on extensive work experience 2-Creating custom, general use modules and components which extend the elements and modules of core AngularJS3-Hands on experience in Qlik Sense development, dashboarding and data modeling and reporting ad hoc report generation techniques4-Experienced in application designing, architecting, development and deployment using Qlik Sense Strong database designing and SQL skills SSIS Minimum qualifications: | Bachelor's degree in Computer Science, Computer Engineering, a related field of study or equivalent practical experience 5 years of software development experience through coding in a general purpose programming language Experience writing libraries and tools used by Java Developers : | Experience participating in online platforms (such as Open Source platforms, StackOverflow) Experience working with third party developer tools such as documentation, APIs, SDKs, and client libraries Experience developing on a cloud computing platform, including full-stack or back- …

**Regex topics:** `legacy_stack`, `orchestration`, `performance`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `orchestration` | {orchestration} · {orchestration} · {orchestration} |
| `gpt-5.4-mini` | `orchestration` | {orchestration} · {context_infrastructure,orchestration} · {orchestration} |
| `gpt-5.4-nano` | `orchestration` | {orchestration} · {orchestration} · {orchestration} |

---

## #23  [D_noise]  IT Developer | Hybrid

**uid:** `asaniczka_a559aa95b0611583`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** Allianz Life

**Description (truncated):**
> Participate in the design, development and delivery of technology-enabled applications, products and services. Under general guidance and direction, oversee the delivery of technical solutions and ensure that the delivered solution meets the business requirements, design requirements and technical specifications. Planning and execution: | Participate in the conceptual design development, ensuring that the solution is viable and designed appropriately to solve the business case Provide support for the facilitation and review of estimates, development and deployment plans for all capabilities participating in the release in both Agile / Scrum and SDLC (Waterfall) projects Project process support: | Assist more Senior Developers in providing/coordinating estimates for the development components required for the solution and overseeing design efforts across capabilities/systems Provide support to Project Managers to ensure design / build activities remain within project schedule / budget Act as a technical advisor to the Project Manager, QA lead and/or business Facilitate escalation and resolution of technical issues during the development of the solutions Responsible for supporting ve …

**Regex topics:** `context_infrastructure`, `legacy_stack`, `mentorship`, `orchestration`, `process_scaffolding`, `verification`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `legacy_stack`, `mentorship`, `orchestration`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} |
| `gpt-5.4-mini` | `context_infrastructure`, `legacy_stack`, `mentorship`, `orchestration`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} |
| `gpt-5.4-nano` | `context_infrastructure`, `legacy_stack`, `mentorship`, `orchestration`, `process_scaffolding`, `verification` | {context_infrastructure,legacy_stack,mentorship,orchestration,process_scaffolding,verification} · {context_infrastructure,legacy_stack,orchestration,process_scaffolding,verification} · {context_infrastructure,mentorship,orchestration,process_scaffolding,verification} |

---

## #24  [E_random]  Mid-Level Full Stack Developer (Ruby on Rails)

**uid:** `linkedin_li-4403553540`  · **source:** `scraped/2026-04`  · **company:** Prevail.ai

**Description (truncated):**
> We are seeking a motivated and skilled Mid-Level Full Stack Developer with experience in Ruby on Rails. As a Mid-Level Full Stack Developer, you will contribute to building and maintaining web applications, collaborating with the team to implement new features, and optimizing performance. Success in this role requires a solid understanding of Git/source control and familiarity with Gen-AI tools and their practical applications in development workflows. This position reports to the Chief Technology Officer. Required Qualifications: | * 4\+ years of Ruby on Rails experience * Strong understanding of Git/source control systems * Solid understanding of object-oriented and functional programming paradigms * Ability to write clear technical documentation * Familiarity with Gen-AI tools and their practical applications in development workflows Preferred Knowledge: | * Familiarity with WebRTC * Experience writing performant JavaScript preferably Stimulus, Hotwire, or Importmaps * Understanding of asset optimization techniques * Experience with PostgreSQL (1\+ year preferred) * Exposure to AWS/Kubernetes environments Ideal Candidate: | * A passionate self-starter with strong time-management …

**Regex topics:** `context_infrastructure`, `mentorship`, `orchestration`, `performance`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `mentorship`, `performance` | {context_infrastructure,mentorship,performance} · {context_infrastructure,mentorship} · {context_infrastructure,mentorship,performance} |
| `gpt-5.4-mini` | `context_infrastructure`, `mentorship`, `performance` | {context_infrastructure,mentorship,performance} · {context_infrastructure,mentorship,performance} · {context_infrastructure,mentorship,performance} |
| `gpt-5.4-nano` | `context_infrastructure`, `mentorship`, `performance` | {context_infrastructure,mentorship,performance} · {context_infrastructure,mentorship,performance} · {context_infrastructure,mentorship,performance} |

---

## #25  [E_random]  Sr. Python Developer

**uid:** `asaniczka_00210abe2e002d87`  · **source:** `kaggle_asaniczka/2024-01`  · **company:** Altumint

**Description (truncated):**
> The Senior Python Developer reports to the Chief Technology Officer. Position is part of the R&D team and involves limited interfacing with the operations teams. This person will port C++ code and upgrade Python 3.6 code to Python 3.10, while refactoring our entire production pipeline to improve its efficiency and maintainability. She/he will also be responsible for updating the neural network models used in the existing pipeline with the latest models trained in-house by our AI team. Qualifications | Bachelor’s Degree in Software Engineering, Computer Science, or other relevant area with strong coding experience. Master degree is preferred. Education can be substituted by demonstration of strong understanding and experience in solving engineering problems and ML/AI inference. Minimum 3 years hands-on work experience coding and deploying solutions for engineering problems. Minimum 1 year hands-on work experience coding and deploying solutions that use inference with ML/AI models. Fluent in Python3 (Pandas, NumPy, OpenCV, Matplotlib, Multiprocessing). Hands-on experience with Tensorflow and PyTorch on GPU. Hands-on experience with creating Python applications with GUI. Comfortable w …

**Regex topics:** `orchestration`

**LLM majority labels (per model):**

| Model | Majority | Per-rep |
|---|---|---|
| `gpt-5.4` | `context_infrastructure`, `performance` | {context_infrastructure,performance} · {context_infrastructure,performance} · {context_infrastructure,performance} |
| `gpt-5.4-mini` | `context_infrastructure`, `orchestration`, `performance` | {context_infrastructure,orchestration,performance} · {context_infrastructure,orchestration,performance} · {context_infrastructure,legacy_stack,orchestration,performance} |
| `gpt-5.4-nano` | `context_infrastructure`, `legacy_stack`, `orchestration`, `performance` | {context_infrastructure,legacy_stack,orchestration,performance} · {context_infrastructure,legacy_stack,performance} · {context_infrastructure,orchestration,performance} |

---

