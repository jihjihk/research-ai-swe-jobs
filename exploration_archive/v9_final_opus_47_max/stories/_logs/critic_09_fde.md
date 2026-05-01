# Critic: 09_forward_deployed

**Verdict:** Numbers check, but framing overreaches in three places; defensible with targeted softening.

## Revisions needed

- **Ditto omission (n=3) is a tell.** The top-firms list in the draft names Saronic at 3 but skips Ditto at 3 — a non-defence, non-finance database firm that breaks the "AI commercialisation into regulated verticals" frame. Include Ditto or justify its exclusion; silently dropping a tied top-ranked firm is cherry-picking.

- **"38 firms" vs evidence sheet.** Draft asserts 38 distinct firms; after aggregator strip it says 37. The counts file lists only a top-10 — confirm the 38 number is auditable in `fde_companies.csv` (file shown truncates at 10) or soften to "the top-ten include…".

- **Anthropic caveat undermines headline more than draft admits.** If the single largest FDE hirer is not captured by the title-match, then the "archetype has escaped Palantir" claim is partly a measurement artifact of title-search. The Applied-AI-Engineer cluster (T34 cluster 0: 2,251 postings, 15.6× growth, 1,163 firms) is plausibly the same phenomenon rebranded — and sits in our own data. Piece should either (a) acknowledge T34 cluster 0 as the broader signal, with FDE-by-title as the narrow subset, or (b) drop "escaped" language and frame as "the title itself is spreading." Current phrasing lets readers infer the larger claim without the supporting breadth.

- **Tiny-n loudness.** The 17× number sits in paragraph 3 with one hedge; the explicit "3 postings" caveat is deferred to paragraph 6. Move the n=3 into the first numeric sentence, or lead with the share language and drop "seventeen" from the opening section.

- **Defence cluster is rhetorical, not strong.** Six firms at 1–3 postings each (Saronic 3, CACI/Govini/Mach/FOX Tech at 1 implied) is a naming pattern, not a statistical cluster. "Defence primes" oversells CACI; calling Mach/FOX/Govini "primes" is wrong. Rewrite as "a scattering of defence-adjacent firms" and name the posting counts.

- **Title-match fairness.** The piece acknowledges Applied-AI/Solutions Engineer variants once but does not quantify the miss. Add one line: "on title-match alone; the broader field-engineer / applied-engineer surface is larger and measured separately in T34."

- **"+800% YoY" tracker claim.** Currently framed as "direction consistent" in the evidence block but reads in the body as corroboration. The body sentence ("more conservative in multiplier, more informative in composition") implies the trackers and this study are measuring the same thing. They are not — trackers count Anthropic/OpenAI hires under any title; this piece counts title-match only. Either drop the tracker sentence or re-label it as non-comparable context.

- **Closing line.** "Palantir was right, ten years early" is too strong for a 59-posting sample and an acknowledged title-match limitation. Soften to "Palantir's eccentricity now reads as early" or similar.
