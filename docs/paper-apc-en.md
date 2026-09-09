# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts

*Working English manuscript v0.1 — sections with stable content translated from
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
negative. Multi-model validation of PGAM awaits additional API credentials.

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
2 of 3 tasks ≤ +0.001). The same principle governs migration: seeds must pass the
holdout test before being carried (KR-6).

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
| z0 (same-batch reference) | .6983 | — |
| APC-safe champion | .7008 | +.0025 (within the noise band) |
| GEPA champion | .7014 | +.0024 (within the noise band) |
| APC-full champion | .6082 | **−.090 (a real liability)** |

Seed stability of the searches (each seed carries its own within-day baseline, hence
naturally paired): APC-full s42/s43 = .5704/.5705 (Δ.0001), while s44 collapses to
.1334 (val .1452, locked into the wrong basin) — **rule-root mutation exhibits seed
fragility**: beyond the deficit (−.09) there is also bimodal risk. APC-safe's three
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
of which free-text mutation provides. (iii) **Methodological contribution**: we
propose the paired same-day re-evaluation protocol — a reasoning-model evaluation
paper that does not control day-drift risks having every conclusion inside ±.03
flip; this paper's main four-arm table was run in one night under one protocol, and
the GEPA/reeval batches are paired same-day supplementary measurements.

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
5. Judge reliability quantified in F4/F6: ordering conclusions hold under a
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
