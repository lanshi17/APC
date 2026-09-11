# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts

*Working English manuscript v0.3 — full F1-F11 chain mirrored from
`paper-apc.md` (Chinese master). Experiment numbers cited here are final as of
v0.4; full English §3 (simulation + real-LLM F1-F7 incl. GEPA head-to-head & variance audit). Venue target: EMNLP/NeurIPS-style.*

**Abstract.** APC turns prompts from hand-written text into compilable engineering
objects: a task specification (TaskSpec) compiles, together with an explicit
15-dimensional model profile (ModelProfile), through a discrete 10-gene genome
(PromptGenome) into a prompt, which is then improved by budget-aware evolutionary
search (with a profile-guided mutation operator, PGAM) and carried across models
by structural migration. On a deterministic four-task benchmark (APCBench;
financial/contract/math/constraint-following × 3 simulated model profiles): search
significantly beats zero-shot (+0.009 to +0.029); evolution beats random search on
compositional tasks (+0.003) but loses on single-locus tasks (−0.013, successive-halving
selection noise); PGAM is indistinguishable from uniform mutation (pooled +0.0004,
a null result); migration reaches native-quality at 30% of the re-search budget
(KR-6 passed 12/12). On a real frontier reasoning model (qwen3.8-flash, four arms ×
three tasks, identical rule-judge): accuracy gains of genome search are ≈0 — the
base-root arm holds the zero-shot floor on all three tasks (within −0.004) while the
profile-rule root is a net liability (dev-score deficits of 0.2–0.53, recovered only
partially within budget); the strongest deployment policy is *zero-adaptation
transfer* of a cross-task champion genome (0 budget, ≥ same-task zero-shot, up to
+0.79 over cold re-search). External author-unseen benchmarks (GSM8K, MATH Level-5,
AIME 2024+25) reproduce both findings: all six champion-vs-base deltas are within
noise and the model saturates every benchmark (AIME: 60/60), bounding the room for
prompt optimization on strong reasoners. We distill these into a methodological
principle, *Compile-as-Hypothesis*: compiled artifacts are testable hypotheses,
never default deployables, and prior value is a measured quantity that can be
On the same real API we then close the baseline and regime questions: the official GEPA baseline (Algorithm 1 reproduced) is also within the paired same-day noise band (+.0024), the second strong baseline ESPO likewise (.7009), and a pre-registered token-starved discriminative experiment (F10) pins six pathways at the judge floor (.1500) when headroom is physically unreachable. F11 completes the map on a stratified 90-problem HLE-exact subset — the first REAL reachable knowledge-type headroom (zero-shot accuracy .10): all six arms tie again (.388-.412), even though GEPA demonstrably learned domain content rules (SMILES conventions, perturbation-theory regime qualifiers). The discriminator for prompt-optimization payoff is not whether headroom exists but what kind: physical-truncation and knowledge-type headroom are out of reach for any prompt pathway (ability lives in weights), while protocol/format headroom is reachable and favors structured genomes — deployment should begin with a headroom-type diagnostic (§3.4m). A deployment gate (AutoAPC-Select, F8: val-argmax + noise-band Occam tie-break, 5/7 exact-oracle on 7 replay+blind groups) turns these findings into a selection rule. Multi-model validation of PGAM awaits additional API credentials.

## 1. Problem and Claim

The same business task requires different prompts on different LLMs (Sclar et al.
2023: format sensitivity is model-specific; FLASK, Ye et al. 2024: skill profiles
differ per model). Existing evaluation suites (HELM/FLASK) rank models but do not
guide prompt construction; existing prompt compilers (DSPy) re-search per model but
keep the profile implicit and uninterpretable (see `docs/literature/lit-compiler.md`
G1–G6).

APC's claim: **an explicit capability profile (15 dims) as a compiler intermediate
representation** —
`TaskSpec → PromptGenome (10 genes) → ModelProfile → PromptCompiler (rule
compilation) → TrialResult → evolutionary search (PGAM) → model migration`,
fully traceable end-to-end (KR-3).

**Contributions** (ordered by reviewer concern, not chronology):
- **C1 · Profile-driven traceable compilation**: the TaskSpec×PromptGenome×ModelProfile
  IR attributes every prompt change to a gene locus and a profile dimension (§2.1-2.3,
  F4 decision-chain audit) — a capability DSPy-style implicit profiles cannot offer.
- **C2 · Quantified liability of priors**: the rule-root costs −.090 on a real model
  (robust across three seeds, including a bimodal seed-collapse case); "compilation
  priors can be negative and are measurable" — Compile-as-Hypothesis becomes a
  quantity, not a slogan (§3.4 F2/F7).
- **C3 · A complete discriminator for APO payoff**: four independent pathways tie in
  the saturated regime (F7-F9) → all tie under physical truncation (F10) → all tie
  under reachable knowledge-type headroom, with mechanism evidence that reflection
  learned content rules yet transferred zero (F11) → while structured pathways win
  +0.05 in simulated protocol-type headroom. Merged into a **headroom taxonomy**:
  when no APO can win — prior to "whose representation is better" (§3.4 F10/§3.4m).
- **C4 · Zero-cost deployment gate**: AutoAPC-Select spends measured same-day drift
  bands as its tie-break currency; on 7 replay+blind groups, 5/7 exact-oracle, both
  seed-collapse arms rescued at zero regret (§3.4 F8).
- **C5 · Evaluation-methodology assets**: the hardened exact-match judge v3.2 (F6),
  the paired same-day re-measurement protocol (F7), two empirical instances of the
  val/holdout common-pool clause (F10/F11), and the connection-layer reliability
  triad for long-run harnesses (F11) — all in-repo.


## 2. Method

### 2.1 PromptGenome: a discrete structural genome

Ten gene classes (role/goal/instructions/constraints/examples/reasoning/
verification/output/style/layout); `search_space` declares mutable loci (13 in
our experiments). Unlike continuous soft prompts (Lester 2021; SPoT), a discrete
genome is re-compilable and portable across models (§3.3).

### 2.2 Profile-Guided Adaptive Mutation (PGAM)

- Prior: `w = 1 + 2(1 − capability)` truncated to [0.5, 3.0]; unmapped loci get
  weight 1.0.
- Online correction: per-generation surviving elite mutation loci ×1.3 (cap 5.0),
  floor 0.3 — an ε-guarantee that lets the bandit override a mis-specified profile.
- The sole difference from uniform mutation is the locus sampling distribution;
  scheduling, budget accounting and lineage logging are identical, making the
  ablation a clean single-variable contrast.

### 2.3 Budget-aware population + Successive Halving (SHA)

`P = min(P₀, max(E+1, (B−3)/(G+2)))`; per generation dev_r1 halves the field and
dev_full re-scores. Dev selects, validation names the champion, holdout runs the
champion exactly once (the EvoPrompt discipline).

### 2.4 Migration protocol (FR-7)

`CapabilityDelta (15 dims) → MigrationMutator adjusts the seed → small-budget
re-search on target → adopt iff holdout ≥ 90% of source (KR-6)`.

### 2.5 Design principle: Compile-as-Hypothesis

Real-model evidence (§3.4 F2) forces the principle: **the output of profile-rule
compilation is a hypothesis to be tested, not a deployable default**. The pipeline
therefore always runs two roots in parallel at equal budget — rule-root
(apc-full) and base-root (apc-safe) — and deploys the winner. This makes the
implicit assumption of DSPy-style per-model re-search explicit and measurable:
prior value = Δ(rule-root, base-root), and it can be negative (measured here:
2 of 3 tasks ≤ +0.001; independently corroborated by the capability-dependent
compilation gains of arXiv 2608.02639). The principle does not arrive *ex nihilo*:
the spirit of deterministic acceptance layers over free-form judging is now
established in production-systems literature (PROCTOR catalogues 11 evaluation-
signal failure modes with a judge-overridable-proof acceptance gate,
arXiv 2609.02246; accept-or-revert gate family: TARA arXiv 2607.18724,
arXiv 2606.30840, SSO arXiv 2607.28777; a measurable-prior Bayesian cousin:
Textual Bayes arXiv 2506.10060). Our contribution is to *name and operationalize*
this practice as a compiler design principle — both roots permanently in the
pipeline — and to attach quantitative deficit readings (F2/F7). The same principle
governs migration: seeds must pass the holdout test before being carried (KR-6).

## 3. Experiments (APCBench, `experiments/apcbench/`)

> - Task A "Financial Report Analysis" (dev 40 / validation 30 / holdout 30 / perturbation 30);
>   Task B "Contract Information Extraction", Task C "Math Word Problems", and Task D
>   "Verifiable Constraint Following" (all dev 40 / validation 30 / holdout 30; generators
>   and seeds are in the respective task scripts).
>   B/C share the `_prompt_skill` gene dynamics (different domain logic); D uses in-task
>   dynamics (three channels: constraints / sentence count / numbers, activating the
>   instructions/constraints loci that were neutral until now).
>   Perturbation design (A only): distractor sentences + replacement of the first metric
>   value (the gold standard is kept on the clean version).
> - Models: glm / qwen / gpt (deterministic simulation via MockClient; ground truth in the README).
> - Methods: zero-shot / manual / random-search / apc-full (uniform evolution + SHA) /
>   apc-no-profile / apc-no-halving / apc-pgam.
> - Statistics: Task A 5 seeds × 3 models, B/C/D 3 seeds × 3 models; paired bootstrap 95% CI.

### 3.1 Main results (b100, holdout)

| Comparison | Δ | 95% CI | Conclusion |
|---|---|---|---|
| apc-full − zero-shot | +0.0093 | [0.0082, 0.0103] | significant; search works |
| apc-pgam − zero-shot | +0.0101 | [0.0095, 0.0107] | significant |
| evolution − random | +0.0030 | [0.0022, 0.0038] | significant; value of multi-step composition |
| apc-pgam − apc-full | +0.0008 | [0.0000, 0.0017] | **marginal**; variance 0.0001 vs 0.0016 |
| apc-full − manual | −0.0009 | [−0.0017, −0.0001] | a strong manual heuristic still leads |
| SHA / rules ablation | 0.0000 | — | null result (base already satisfies the rules) |

### 3.2 Budget curves (holdout mean)

| budget | pgam | full | random | manual | zero-shot |
|---|---|---|---|---|---|
| 25 | 0.7501 | 0.7495 | 0.7497 | 0.7595 | 0.7493 |
| 50 | 0.7581 | 0.7581 | 0.7581 | 0.7595 | 0.7493 |
| 100 | 0.7600 | 0.7592 | 0.7561 | 0.7600 | 0.7499 |

The plateau is reached at 50 evaluations (a consequence of the task's unimodality);
random in fact degrades at b100 (single-step space + more candidates → dev overfitting).

### 3.2b Second task: contract extraction (b50, holdout, n=9/method)

| Method | Mean | Comparison | Δ | 95% CI |
|---|---|---|---|---|
| apc-full / apc-pgam | 0.9844 | search − zero-shot | +0.0275 | [0.0130, 0.0420], significant |
| manual | 0.9814 | search − manual | +0.0031 | [0.0000, 0.0061], not significant |
| zero-shot | 0.9569 | (qwen alone 0.9300 → search repairs it to 0.9833; weak-model rescue +0.05) | | |

### 3.2c Third task: math word problems (b50, holdout, n=9/method)

| Comparison | Δ | 95% CI | Conclusion |
|---|---|---|---|
| search − zero-shot | +0.0285 | [0.0191, 0.0372] | significant |
| search − manual | −0.0109 | [−0.0154, −0.0063] | significant; manual is strong (exact count/strategy is hand-optimal) |
| pgam − full | 0.0000 | — | no difference |

Cross-task consistency: search ≫ zero-shot holds on all four tasks
(+0.009 / +0.028 / +0.029 / +0.029); search vs manual: financial −0.001,
contract +0.003 (n.s.), math −0.011, constraint following +0.024 (significant) —
strong handcrafted heuristics are hard to beat on a unimodal simulator; the value of
automation lies in no handwork, plus transfer, plus lineage.

### 3.2d Fourth task: verifiable constraint following (b50, holdout, n=9/method)

| Comparison | Δ | 95% CI | Conclusion |
|---|---|---|---|
| search − zero-shot | +0.0292 | [0.0171, 0.0399] | significant |
| search − manual | +0.0238 | [0.0116, 0.0341] | significant; search beats manual significantly for the first time |
| random − evolution | +0.0132 | [0.0035, 0.0246] | significant; direction reversed (see below) |
| pgam − full | +0.0005 | [−0.0148, 0.0162] | not significant |

**Optimizer cross-decomposition** (after adding a no-halving control for Task D):
no-halving (0.9616) > random (0.9513) > SHA evolution (0.938).
no-halving − random: +0.0103 [0.0022, 0.0205] (significant) — when selection is
accurate, the compositional value of multi-step evolution is positive;
full − no-halving: −0.0235 [−0.0357, −0.0112] (significant) — SHA's small-sample
dev_r1 screening comes at a heavy cost on this task. Conclusion: **the
speed–accuracy trade-off is quantifiable, and there is no free lunch**; SHA favors
compositionally hard tasks (financial), while full-population selection favors
single-locus tasks (constraint following). PGAM is likewise indistinguishable from
uniform on this task (+0.0005).

### 3.2e PGAM pooled across tasks (combined paired deltas, n=42)

pgam − uniform: +0.0004 [−0.0029, 0.0038] — a **null result** (the CI still crosses
zero after narrowing). The profile prior delivers no measurable gain on this
simulator (the profile itself is weakly informative: the three models differ on only
3/15 dimensions; and the optimal basin is a single one). Reasons to keep PGAM: lower
variance (b100 std 0.0001 vs 0.0016), an interpretable prior, and a bandit lower
bound guaranteeing it is no worse; confirming or falsifying it requires genuinely
multimodal real tasks.

### 3.2f Robustness (perturbation, champion genome, n=15/method)

All methods drop from holdout → perturbation by ≈ −0.010 (a numeric-substitution
offset independent of the genome); there is no evidence that search champions are
more fragile (the ranking is consistent with holdout).

### 3.3 Transfer (4 directions × 3 seeds, adapt budget 30 vs native 100)

- recover (adapted/native): 1.0000 [0.9976, 1.0025]; KR-6 pass rate 12/12;
  adapt − direct: +0.0001 (direct transfer is essentially lossless — the heterogeneity
  sits below the optimizer's resolution).
- Conclusion: the transfer protocol reaches native re-search quality at **30% of the
  budget** (a GEPA-style rollout argument), with lineage and an adopt/keep decision
  audit chain attached; the flatness of the decay matrix itself is a limitation of
  this simulator.

### 3.4 Real-LLM validation (phase one: qwen3.8-flash × 3 tasks, `bench_real_full/transfer.py`)

Setup: whitelisted-key model, reasoning model (~29 s/sample), temp=0; scoring uses the **same rule-judge** as the simulation benchmark; all three tasks share the identical dev_r/val/holdout = 5/8/20 split; budget 8 for the main comparison, budget 6 for the transfer experiments.

| Task | zero-shot | manual | apc-full(rule-root) | apc-safe(base-root) |
|---|---|---|---|---|
| contract | 0.9770 | 0.9773 | **0.9782** | 0.9773 |
| math | 0.8436 | 0.8439 | **0.8442** | 0.8436 |
| financial | **0.6712** | 0.6662 | 0.5704 | 0.6672 |

- **F1 ceiling effect (double saturation of accuracy and robustness)**: on a strong
  reasoning model, the accuracy headroom on rule-verifiable tasks is ≤0.005, and the
  robustness headroom under input perturbation is likewise ≤0.013 (F5) — the preset
  stance that "the value of prompt optimization lies in the format/constraint
  dimension" is **falsified on the real model**; the only positive value remaining is
  weak-prior rescue (F2, with no guarantee of full repair) and zero-loss transfer (F3).
- **F2 the rule prior is the dominant risk on real models; base-root search is a
  "lossless floor", rule-root search is "high-cost rescue"**: the dev scores of the
  profile-compiled root are contract 0.1778 / math 0.6335 / financial 0.1286 — all
  0.2–0.53 below zero-shot. The b8 search pulls the rule-root arm back into the
  examples-on basin, but **repair completeness decreases as task headroom grows**:
  contract 0.9782 > z0, math 0.8442 ≈ z0, financial 0.5704 still −0.101; at budget 6
  the search never even samples the repair site (cold 0.1850). The control arm
  **apc-safe (base root + the same b8 budget) lands within −0.004 of zero-shot on all
  three tasks** (0.9773/0.8436/0.6672): after spending the same budget, the search
  neither gains nor loses, and the financial champion is base + `goal.explicitness:
  high→low` (val selection noise, holdout −0.004). Conclusion: **on real models the
  profile rule prior is a net liability; structural genome search yields accuracy
  gains ≈ 0, and its only value is rescuing a bad prior (without guaranteeing a full
  repair)**. The simulator's "apc-full ≡ zero-shot" safety does not carry over to real
  models — the source of unsafety is precisely the compiled rules.
- **F3 cross-task genome transfer (two directions, b6 same-budget three arms)**:

| Transfer | cold (search) | transfer-0 (zero-adapt direct use) | transfer-ws (continued search) |
|---|---|---|---|
| contract→math | 0.6402 | **0.8439** | 0.8441 |
| math→contract | 0.1850 | **0.9778** | 0.9582 |

  The strongest arm is **zero-adapt direct use of the source champion** (transfer-0),
  reaching the target task's zero-shot level at 0 budget; warm-start continued search
  yields ≈0 gain on saturated tasks (math +0.0002) or even negative gain (contract
  −0.0196 — val_n=8 selection noise picks an overfit champion); cold start lags
  significantly because it must first pay F2's repair cost. Mechanism: the genome
  carries full structure while few-shot content is injected from the TaskSpec at
  compile time ⇒ zero-loss across tasks; the optimal basin is unique (both tasks'
  champions are examples-on), cross-validating the simulator's "single-peak" finding.
  **Engineering implication: deployment order transfer-0 → base/apc-safe →
  rule-root with budget ≥ repair cost; never default to a cold-start re-search.**
- **F4 judge-disagreement audit (20 cases from the financial champion,
  self-LLM-judge, `real_judge_check.py`)**: on the accuracy dimension, rule mean 0.206
  vs LLM 0.830 (Pearson 0.48 / Spearman 0.56; 20/20 disagree by ≥0.25); on the
  constraint dimension the direction reverses: rule constantly 1.0 vs LLM 0.50.
  Reading: the rule-judge is a gold-standard tolerance check, systematically stricter
  than the self-evaluating LLM, and the two judges are strict along *different*
  dimensions. ⇒ All real-LLM ranking conclusions in this paper are valid under the
  same-protocol rule-judge, but **absolute scores must not be read as "human quality
- [x] High-difficulty contest F11 (pre-reg git 262a39e): `bench_real_hle.py` (HLEJudge/HLEChecker atop hardened v3.2; z0/champs/gepa/espo arms; per-sample stream resume; thread-level hard timeout; stateless per-request `_fresh_http_call`); `gen_hle_dataset.py` (MIT upstream, stratified 30/30/30 seed=2026, bit-reproducible); subset committed at `datasets/hle_exact/`; results in `real_hle.json` (6 rows)
  scores"**; an independent third-party judge (not self-evaluation) remains a gap. The
  self-judge's leniency may also contain a model self-preference bias.
- **F5 perturbation-robustness audit (first 20 cases of the financial
  number-tampering set × 4 genomes, `bench_real_robust.py`)**: pert scores base 0.6580
  (drop +0.0132) / manual 0.6654 (+0.0008) / apc-full 0.5820 (−0.0116) / apc-safe
  0.6582 (+0.0090). All |drop| ≤ 0.013 — strong models are themselves insensitive to
  numeric-tampering perturbations, and format genes have no realizable robustness
  premium (apc-full's negative drop is its overall level shift falling into the
  perturbation-set noise, not "more robust"). ⇒ Merged with F1: **on real strong
  models, structural genomes are saturated in both accuracy and robustness; the gain
  budget can only come from rescue and transfer, not from optimization itself.**
- **F6 external real benchmarks (GSM8K / MATH Level-5 / AIME 2024+25,
  `bench_real_external.py`, exact-match judge with latex structural normalization,
  zero-adapt three arms)**: F1/F3 tested directly on tasks the authors did not build —
  genomes compiled via the external `external_math.yaml` spec; arms = base /
  math-champion (source task) / contract-champion (cross-domain), all evaluated on
  datasets that never participated in any genome.

| Dataset (n) | base | math-champ (transfer-0) | contract-champ (cross-domain) |
|---|---|---|---|
| GSM8K (100) | 0.96 | **0.98** | **0.98** |
| MATH-L5 (135) | 0.9704 | 0.9481 (−1.0σ) | 0.9704 |
| AIME 24+25 (60) | 1.00 | 1.00 | 1.00 |

  After offline re-judging under judge v3.2 (exact-match + latex structural
  normalization): none of the six champion-vs-base comparisons is significant, and
  **on AIME 2024+25 all three arms score 60/60** (a frontier reasoning model gets a
  perfect olympiad score ⇒ the external benchmarks fail even at being "hard"); the
  cross-domain champion (contract→math) is lossless just like the source-domain
  champion. The only consistent directional signal: math-champ is −0.012~−0.022 on
  both math sets (pooled −1.1σ, not significant) — a **faint echo of the prior
  liability on external tasks**, same direction as F2 but two orders of magnitude
  smaller (the genome has already evolved into a near-base form). Judge reliability:
  12 failed samples manually audited → three repair rounds v2→v3.1→v3.2 (matrices /
  solution sets / units / leading-zero flattening + addition-commutativity sign-term
  multiset; policy: sacrifice the strictness that (1,2)≠(2,1) in exchange for
  equivalence of all notation variants, and judge as wrong when in doubt) → 27
  positive + 4 negative regression cases + full predictions persisted for
  recomputation (`rejudge_external.py`); on AIME, one network-timeout case per arm is
  scored as wrong (n=60 ⇒ ≤0.017 downward bias).
- Honest boundary: single model, single seed, no CI; multi-model variation and the
  real validation of PGAM are still missing (credential whitelist).

**F7 GEPA official-baseline head-to-head + variance audit (gepa vs full/safe/z0, `bench_real_gepa.py` / `bench_real_reeval.py`)**

Control design: we reproduce the core of GEPA (Agrawal et al. 2025, arXiv 2507.19457) Algorithm 1 — failure-driven reflective rewriting from a minibatch (3) (free-text mutation), parent sampling from the per-instance Pareto frontier, and full-val re-evaluation to select the champion; identical to APC in starting point (the compiled z0 text), judge path (TrialScorer), budget currency (48 task rollouts ≈ APC b8's actual spend of 25–35), and final holdout20 testing. Reflective LLM calls are not charged to the budget (the same logic as the APC compiler's overhead).

The variance-audit finding precedes every comparison: **the reasoning model is
non-deterministic at temp=0** — the same z0 prompt scores .6950/.6997/.6998/.7016 on
four same-day holdout20 re-measures (full range .0066), and drifts across days
(yesterday .6712 → today .6990, Δ.028). Therefore **any cross-day comparison of
numbers is invalid**; every conclusion in this section rests on **paired same-day
re-evaluation** (the reeval batch, seed 902).

Same-day paired four arms (financial, completed within one hour window):

| Method (same-day re-measured) | holdout | vs z0 |
|---|---|---|
| z0 (same-batch ref; 4 same-day measures: .6950/.6983/.6998/.7016) | .6983 | — |
| manual | .7012 | +.0029 (noise band; last night's .6662 was a low-drift day) |
| APC-safe champion | .7008 | +.0025 (within the noise band) |
| GEPA champion | .7014 | +.0024 (within the noise band) |
| APC-full champion | .6082 | **−.090 (a real liability)** |

Seed stability of the searches (each seed carries its own within-day baseline, hence
naturally paired): APC-full across four seeds = .5704/.5705/.1334/.1318 — a
**clean bimodal distribution** (a ~.57 deficit basin and a ~.13 collapse basin,
2/4 collapse rate); collapsed arms show the failure in val too (.1452/.1708,
visible to any selector) — **rule-root mutation exhibits seed fragility**:
beyond the deficit (−.09) there is bimodal risk. APC-safe's three
seeds .6672/.6679/.6575 (full range .010, same magnitude as the day-drift) are
stably indistinguishable. GEPA on contract (generic-judge caliber — **not comparable
with APC's contract_case caliber**): .7491/.7495 vs the same-caliber z0 .7500 — on a
saturated task there is no failure signal to reflect on, zero progress, corroborating
evidence for the null.

Three conclusions: (i) **the strongest open-source baseline GEPA also achieves zero
gain on the strong reasoning model** (+.0024, within the noise band) — the §3.4 null
is not a local defect of APC's representation but a property of the "strong model +
high judge-coverage" regime; GEPA's reflection path and APC's structured-mutation
path flatline alike here. (ii) APC's differentiated value holds beyond the null:
the auditable decision trace (F4), zero-cost transfer (contract→math champion .9808),
and the schema/robustness machinery (F3, the hardened external judges of F6) — none
of which free-text mutation provides. (iii) **Methodological action**: we adopt the
established practice of same-day test-retest floors and matched controls
(arXiv 2608.00705; arXiv 2608.08239, COLM 2026; temp-0 nondeterminism quantified
in arXiv 2408.04667, 2606.26185; reporting standards in arXiv 2607.24372) and
institutionalize it for APO arm comparisons: to our knowledge the first explicit
variance decomposition (between-day .028 vs within-day ±.007) for reasoning-model
prompt optimization, from which we draw the protocol clause that any arm-level
claim within ±.01 must rest on paired same-day re-measurement — indistinguishability
inside the noise band being an information-theoretic necessity under verification
lower bounds (arXiv 2604.12951). Our main four-arm table was run in one night under
one protocol; the GEPA/reeval batches are paired same-day supplementary measures.

**F8 AutoAPC-Select: a deployment-time gating selector (`auto_apc_gate.py`,
offline replay, zero additional rollouts).** F7's two negative findings
(rule-root liability + seed fragility) yield a constructive corollary: per-arm
validation scores are free on any given day, so deployment selection can be
*automatically gated* — argmax over candidates {z0, manual, safe, full,
transfer} on validation score, with ties inside the ±.007 noise band broken by
Occam order (z0 < manual < safe < full). Offline replay over the 12 existing
arm groups plus a prospective blind-test group: 6/7 land within the noise band,
5/7 are exact oracles, and both collapsed rule-root arms (s44 → .1334, s45 →
.1318) are fully rescued (gate picks safe .6575 / z0 .6692, regret 0). The s45
group is **prospective**: the gate rule (val argmax + ±.007 Occam tie-break) was
frozen before the group ran, and the gate's pick equalled that day's oracle —
evidence, not post-hoc fitting. The single large regret
(math→contract, +.0196) exposes the gate's failure mode: val(8)/holdout(20)
distribution mismatch can inflate a candidate's validation score (transfer-ws
val .9822 > transfer-0 .8871 while holdout reverses, .9582 < .9778); the honest
corollary is to bias noise-band ties toward the simpler arm or to draw val and
holdout from a common pool. This upgrades F7's manual two-arm contrast into an
automatic non-inferiority selector with a characterized failure mode — at zero
extra evaluation cost, reusing existing validation scores. Boundaries within the gating family:
ESPO (arXiv 2609.04197, EMNLP 2026 main) buys selection stability with bootstrap
re-sampling (extra evaluation budget); AutoAPC buys it with the *measured
exogenous drift band* at zero extra calls, requiring only that same-day val
scores exist. Pure-exploration bandits (arXiv 2605.14553, ICLR 2026) give
best-feasible identification with sampling theory but consume budget; AutoAPC is
its free-score degenerate case. PROCTOR's deterministic acceptance layer guards
against judge hacking — orthogonal to, and stackable with, APC's variance guard
(APC's own judges are deterministic checkers). Indistinguishability inside the
band is not an engineering compromise: verification-tax lower bounds (arXiv
2604.12951) show 23% of frontier-model pairwise comparisons are statistically
unresolvable in principle.

**F9 ESPO, the second strong baseline (`bench_real_espo.py`, faithful
reproduction of EMNLP 2026 main's Diagnose/Propose/Select).** ESPO is currently
the strongest published GEPA successor (its pitch: fix GEPA's prompt bloat and
unreliable selection; +3.76pp over GEPA on 7 benchmarks). We reproduce its
three steps: Diagnose clusters ALL validation failures into at most 4 error
patterns in one pass (vs GEPA's incremental 3-case minibatch reflection);
Propose draws one candidate from each of four independent bias strategies
(root-cause abstraction / simplification / exemplification / constraint
hardening); Select accepts a candidate only if it beats the incumbent in >=75%
of B=200 bootstrap resamples of per-case validation scores. Budget matches the
GEPA arm (48 rollouts, 40 used); bootstrap costs zero additional calls.
Paired same-day result (seed-42 batch, z0 day band .6950-.7016): **ESPO .7009,
inside the band and statistically indistinguishable from z0/safe/GEPA** — the
fourth independent optimizer family (reflective GEPA, cluster-and-stabilize
ESPO, APC genome search, manual) all null in the same regime. A second seed
scored .7501 but its champion text was not persisted, there is no same-day z0
pair, and it crossed a day boundary — under the F7 protocol clause it is
excluded from conclusions and disclosed as a live case of why unpaired numbers
are untrustworthy. Relationship to our F8 gate: ESPO's stability currency is
endogenous (re-sampling its own evaluations); APC's is exogenous (a measured
drift band, at zero extra cost) — orthogonal and stackable, and this run
illustrates exactly that distinction.

**F10 Pre-registered discriminative experiment: the token-starved regime.**
Before running it we committed (git e8bd373, 22:20) a three-branch prediction:
capping output at max_tokens=150 collapses zero-shot holdout from .699 to .150,
opening apparent headroom; the question was whether genome search, free-text
reflection (GEPA), and cluster-and-stabilize selection (ESPO) diverge once the
signal is winnable. Result — branch (c), the pre-registered null-of-the-null:
all six pathways measured .1500 on a paired same day (z0 / manual / APC-safe /
APC-full / GEPA / ESPO). Truncation occurs mid-JSON (verified in fail samples);
the floor is the judge's format-collapse value, not model competence — **no
prompt rewrite stops a reasoning model from spending its output budget**
(instruction-level thought suppression is ineffective on this model). F10 does
not overturn F1-F9; it completes the regime map: optimizer representations are
equivalent-to-zero whenever headroom is either absent (F1-F9) or physically
unreachable by prompting (F10); where headroom is genuinely reachable, the
structured pathway wins (simulator weak-model rescue, +0.05, §3.2b). Side
finding for evaluation protocol: GEPA scored .6961 on the starved val-8 (short
documents finish inside 150 tokens) while the starved holdout-20 scored .15 —
truncation-difficulty asymmetry between a small validation set and holdout makes
reflective optimizers adapt to the wrong regime; this is a concrete instance of
the val/holdout common-pool clause of F8's failure-mode analysis.

**F11 Headroom taxonomy: the HLE-exact high-difficulty contest (pre-registered P1
branch hit).** `bench_real_hle.py`; pre-registration `docs/literature/F11-prereg.md`
(git 262a39e, locked before any gepa/espo holdout number existed); dataset generator
`scripts/gen_hle_dataset.py`.

Motivation: after F1-F9 (no headroom) and F10 (physically unreachable headroom), one
cell of the regime map remained untested on a real API — **reachable headroom**: does
any optimizer family win when there is room and nothing structural blocks it? We strat-
ify Humanities' Last Exam (cais/hle, MIT; arXiv 2501.14249) via its ungated text mirror
(datasets-server pagination), filter to `exactMatch` answers of ≤40 chars (pool 1 710),
draw 90 problems stratified over Mathematics/Physics/Chemistry/Other (seed 2026), and
split 30/30/30 dev/validation/holdout. Grading reuses the hardened exact-match judge
v3.2 from F6 verbatim; the task spec is the same strict-JSON `{answer}` protocol as
external_math.

Pilot (4 arms, same day, seed 921): zero-shot compiled base genome scores .3920
(answer accuracy 3/30 = .10); the three cross-task transfer champions land at
.3876–.4020 — statistically indistinguishable. Two priors die at once: (a) there is no
protocol headroom — a strong reasoning model's JSON compliance is ~100% (the format
gap is entirely network-error rows, not prompt defects); (b) genome structural priors
transfer zero gain onto a knowledge-hard task — format genes have nothing to grab.
The real headroom is accuracy (.10 against a ceiling near .4), putting all the
discriminative pressure on the reflective pathways.

Decisive matrix (6 arms, one paired batch, 12.5 h wall):

| arm | holdout | acc/30 | fmt/30 | note |
|---|---|---|---|---|
| z0 (base genome) | .3920 | 3 | 25 | accuracy .10 |
| math-champ (transfer) | .3876 | 2 | 26 | |
| contract-champ | .4020 | 3 | 26 | |
| financial-champ | .3876 | 2 | 26 | |
| GEPA (val[:2] minibatch search + val[:8] selection, 64 rollouts, 20 iters) | .4120 | 3 | 27 | |
| ESPO (val[:8] full evals, 4 biases, 64 rollouts) | .4020 | 3 | 26 | champion = z0 (bootstrap rejected all 4) |

Arm spread .024; accuracy is 2–3/30 everywhere (binomial SE .055 at p=.10, n=30 —
every pairwise accuracy difference is one-problem quantization noise). **Verdict:
pre-registered branch P1 — under real-API reachable knowledge-type headroom, all six
independent pathways tie again.**

**Statistical power disclosure**: at n=30 the binomial resolution on accuracy is
one problem = .033 (SE ≈ .055 at p≈.10); the "all-tie" verdict reliably excludes
large effects (≥2SE ≈ .11 acc, i.e. +3 problems), not true gains of 1-2 problems.
The conclusion's real strength is triple redundancy: the arm band (.024 wide),
problem-level agreement (GEPA's 3 hits are z0's exact 3 hits), and the mechanism
evidence (learned content rules with zero transfer). Any one alone would be
underpowered; together they point the same way. Hardening paths for the camera-ready:
expand holdout to n≥100 (pool of 1 710 suffices) or trade thinking budget for problem
count — both can only strengthen, not reverse, the verdict.

Mechanism evidence (the most informative part): (i) **GEPA genuinely learned domain
content** — the 8 103-char champion prompt contains substantive rules distilled from
validation failures (SMILES output conventions, molar-mass product-selection proce-
dures, first-order-TDPT/Gaussian-pulse regime qualifiers, "no prose in symbolic
answers"). Yet holdout accuracy is problem-identical to z0 (same 3 hits): prompt-level
domain rules cannot cross the "the model cannot solve the problem" capability wall —
**the reflective pathway's ceiling is the model's prior knowledge, not the search**.
(ii) **ESPO's internal stability currency outputs "do nothing" when signal ≈ noise**:
its best candidate (.512 on val-8) beat r0 (.453) but failed the bootstrap-75%
stability test on all four → champion is the original prompt — consistent with F9,
correct behavior. (iii) **A live case of single-point val overfitting**: mid-search
GEPA converged to an 8 103-char prompt scoring 1.000 on the 2-problem validation
minibatch by echoing that problem's near-full text. Had selection stayed on val[:2]
(the compressed form of GEPA's original protocol under our cost budget), that would
have shipped as champion. Protocol v2 (search on minibatch, **select on full val[:8]**)
caught it — a second empirical instance of F10's small-val/holdout asymmetry lesson,
this time with reflective content literally absorbed into the prompt. (The revision
happened before any gepa/espo holdout number existed; it is a mechanism-bug fix, not
outcome-driven; pre-registered verdict criteria unchanged.)

Reliability engineering by-product: HLE full-thinking costs 60–640 s/problem and
exposed three systemic defects in long-run harnesses — (a) tenacity retrying on a
keep-alive connection whose first response packet the server silently drops (observed:
0.72 s CPU across 5.25 h wall, zero response headers), fixed by stateless per-request
connections (`_fresh_http_call`); (b) httpx read timeouts not firing on half-open
proxy-CONNECT sockets, fixed by thread-level hard timeouts with detached executors;
(c) no per-sample checkpointing, fixed by streamed append + resume. None of these
affects F7-F10 conclusions (financial problems take <10 s and never entered the
pathological regime); all three become prerequisite assets for the multi-model round.

**Implication for the paper's claims.** Merging F11 with F1-F10 yields the complete
headroom taxonomy — the discriminator for APO payoff is not whether headroom exists
but **what kind**: (1) physical-truncation headroom (F10) is unreachable by any
prompt; (2) knowledge-type headroom (F11) is unreachable by any prompt pathway
(ability lives in weights; prompts can only reshuffle what the model already knows);
(3) protocol/format headroom (F1-F9 saturated side + the weak-model simulation
positive) is reachable and is exactly where structured genomes show their mechanism
advantage (simulation +0.05). The financial battleground belongs to (3) but is
saturated on a strong model — the self-consistent explanation of F1-F9's nulls. This
reframes "APC vs X" into a prior judgement of **when no APO can win**: deployment
procedure AutoAPC-Select (F8) gains a layer zero — run a headroom-type diagnostic
first; if headroom is knowledge-type, the correct action is to change model or add
retrieval, not to optimize the prompt.

## 4. Limitations

1. Real-model phase is single-model (whitelist key); the seed axis is covered
   (§3.5: 3 search seeds, 4 same-day z0 re-measures, quantified noise band
   ±.007 and day-drift .028 — any "gain" within ±.01 is judged noise).
   Multi-model scripts are ready (`--model <id>` + three `.env` lines).
   Ground truth is
   author-authored (rule-judge).
2. Task coverage: four simulation tasks + three external math benchmarks; open
   instruction tasks without unique answers still missing (judge-dependent).
3. PGAM pooled null on the simulator; confirmation requires genuinely
   multimodal real tasks with heterogeneous profiles.
4. SHA is task-structure-dependent: −0.0235 cost on unit-locus tasks, null on
   financial; not a universal accelerator.
5. SEPO head-to-head deferred to submission: the structured-editing direct competitor SEPO (arXiv 2608.28067) has no real-API run. F11's mechanism evidence (reflection learned domain content rules, transferred zero) extends the same argument to every editing-granularity variant: under knowledge-type headroom SEPO is predicted to tie, under the saturated financial regime to be null — both cells' discriminative power is already consumed by the GEPA+ESPO four-pathway coverage. The only increment left is the weak-model + protocol-pressure cell (needs the second key, see #1).
6. Judge reliability quantified in F4/F6: ordering conclusions hold under a
   fixed judge; absolute scores are not human-quality scores; independent
   third-party judge and 50–100-case human spot-check still missing.

## 5. Reproducibility

See `paper-apc.md` §5 checklist (all green): one-command simulation suite,
41 unit tests, 13 Java service tests, external-benchmark judge regression
(`test_external_judge.py`), logged per-arm predictions for offline re-judging
(`rejudge_external.py`), datasets + seeds in-repo, zero secrets (ggshield +
pattern double-check).

## 6. Related Work (details in `docs/literature/`)

Optimizers: APE/OPRO/APO–ProTeGi → PromptBreeder/EvoPrompt → PromptWizard/
GEPA head-to-head done (§3.5) → DSPy–MIPROv2. Profiles and
routing: HELM/FLASK/FrugalGPT/RouteLLM. Structure and transfer: Sclar format
sensitivity; soft prompts (non-portable) vs discrete genomes (re-compilable).
2026 frontier: Instruction Stacking Collapse (2608.02639) — compilation value is
capability-dependent, same axis as our F2; PromptBridge (2512.01420) — NL
carriers drift across models, complementary to our F3 (fragility lives in the
carrier, not the representation); DUALFIX (2607.05121) — rule-based evolution
resurges: rules belong in the genome, not free text; CAPO (2608.16068) —
constraint-aware optimization, baseline candidate for the constraint dimension;
Atlas (2603.15666).
2026Q3 GEPA-successor landscape (`docs/literature/frontier-2026.md`, 50-entry
snapshot): the structural camp — SEPO (2608.28067, typed-unit local edits with
edit-effect lineage; the direct genome-space competitor, mandatory submission-time
baseline), SAPO segment-wise (2608.11219), PCO codebooks (2605.28360),
control/data-flow separation (2609.00621, EMNLP 2026 Findings); GEPA-pathology
cluster — ESPO (2609.04197, EMNLP 2026 main, bootstrap stability selection,
mandatory baseline), NPO (2608.27266: complex search unnecessary under strong
teachers — APC's reply: genome value is auditability/transfer/gating, not search
size), MAGE (2607.11944: coupled-optimizer variance amplification; fixed good
prompts beat all reflective optimizers at low data — in direct dialogue with our
F1 null), p1 (2604.08801: response-variance dominance as the failure criterion —
the theoretical language for our F7 regime); failure modes — PROCTOR (2609.02246)
and RLMOpt (2608.10471: GEPA underperformed its own seed in 2/11 runs; gains =
f(seed headroom), neighboring our rule-root bimodality); reliability precedents
2608.00705 / 2608.08239 (COLM 2026) / 2408.04667 / 2606.26185 / 2607.24372 /
2604.12951 (F7 clause chain); theoretical cousins Textual Bayes (2506.10060) and
best-feasible bandits (2605.14553, ICLR 2026). GEPA itself is now ICLR 2026 Oral
(v2, 2026-02-14).
