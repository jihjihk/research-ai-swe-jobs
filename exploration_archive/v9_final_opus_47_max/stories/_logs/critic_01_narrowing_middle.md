# Critic: Piece 01 "The narrowing middle that wasn't"

**Date:** 2026-04-21
**Reviewer posture:** Adversarial / in-house subeditor

## Revisions needed

- **§lede, Amodei paraphrase.** "Half of all entry-level white-collar jobs could vanish within five years" overstates the Axios quote, which pairs that figure with a 1-to-5-year window and a 10-20% unemployment spike. Trim or quote verbatim. Also note Amodei is a *forecast*, not a claim about what *has* happened — as written the lede conflates forecast with observation.
- **§lede, Stanford paraphrase.** The 13% figure is a *relative* employment decline for ages 22-25 in AI-exposed occupations from late-2022; the piece implies cross-sectional level. Add "since late 2022" and "relative" or the Stanford reading is strawmanned.
- **§lede, SignalFire.** Prior-art memo says "2024 vs 2023" and "new-grad share"; piece says "down 25% year-on-year" without year anchors. Add the vintage.
- **§3 headline numbers lack denominators.** "+5.0 pp" is pp of *what base*? Reader does not know the 2024 pooled J3 share was 9.15% until footnote — so +5 pp = a 55% proportional rise. Either state base-share or proportional change; otherwise "5 pp" floats.
- **§3 "Seven different junior-definition panels … all move in the same direction."** Evidence cites 7/7 junior up but 12/13 total (T30). The rhetorical "all" for junior is fine; verify no J-panel exception exists and state "seven of seven" not "seven different … all."
- **§3 "Five of six senior definitions move the opposite way."** Evidence says S2 director-only is *flat*, not moving same direction. "Opposite way" with S2 flat is accurate; the sentence implicitly treats flat as non-falsifying — acceptable but flag that one panel is null, not merely minority.
- **§4 classifier +0.15 AUC.** AUC gain is a model-fit metric, not a linguistic-distance metric. Piece glosses it as if the *postings themselves* diverged. Add one clause: the classifier was retrained per-period; improvement reflects increased separability, not drift alone.
- **§4 "directly contradicted."** Overreach. Relabelling-into-AI-engineer hypothesis is *not supported* by these data; "directly contradicted" requires a test that specifically looked for the blob. Soften to "not supported" or cite T15/T20 as the falsification test.
- **§5 causal slip.** "Senior roles gained more breadth" → "The thing being redefined is not the entry rung." Posting-content change ≠ role redefinition ≠ employment outcome. The piece elsewhere concedes postings ≠ payroll, then re-bundles them here. Tighten the hedge.
- **§5 missing segmentation.** Breadth-widening claim (9 of 13 domains) is not segmented by firm type (Big Tech vs. startup vs. enterprise) or metro, despite the dataset supporting it. A sceptic will ask whether the Mag-7 (where SignalFire's claim lives) shows the same pattern. At minimum flag as limitation.
- **§6 "Applied-AI senior median YOE 6 vs 5."** One-year gap on a single archetype; no CI, no sample size. Underpowered for the weight placed on it. Either add n and spread or demote from evidence to illustration.
- **§6 "It is the clearest rung on the ladder."** Metaphor overreach — "clearest" is asserted, not measured. The supporting claim is *widened* junior/senior gap, which makes the *boundary* clearer, not the rung. Economist-style turn cheats here.
- **§kicker "The senior one to watch."** Cute inversion but unsupported: postings-breadth change is not evidence of senior *employment* risk; Stanford/SignalFire still show juniors losing headcount. Kicker implies senior displacement the data cannot establish.
- **SNR caveat buried.** Evidence block admits pooled J3 SNR = 1.06 (near noise floor). Body treats +5 pp as headline finding without this caveat. A fact-checker will flag: the returning-cohort +6.2 pp is the defensible number; the pooled +5 pp is marginal. Lead with the robust figure or note the noise.
- **Payroll-vs-postings framing (§5).** Reasonably fair, but "postings measure what employers say they are hiring" understates that postings also include ghost posts, aggregator duplicates, and re-postings — acknowledged in T29/aggregator sensitivities but not in-body.

## Prior-art fairness

Amodei, Brynjolfsson, SignalFire each appear in single-sentence caricature. Brynjolfsson's paper explicitly brackets posting vs. payroll — not acknowledged. The piece could concede the payroll finding is real and then argue postings reveal the *composition* story payroll cannot. As drafted, it reads as a rebuttal of claims the priors did not make.

## Verdict

**Needs substantial revision.** The core finding (junior share up, senior-junior boundary sharpening) is defensible and methodologically novel, but the piece over-claims in the lede and kicker, buries the SNR caveat, skips firm-type segmentation, and slides from posting-content to employment implications. Publishable after a tighter rewrite with denominators, hedged causal language, and one firm-type cut.
