# APC: Profile-Guided Compilation and Evolution of Model-Specific Prompts

*Working English manuscript v0.1 — sections with stable content translated from
`paper-apc.md` (Chinese master). Experiment numbers cited here are final as of
v0.2; GEPA baseline (§3.5) pending. Venue target: EMNLP/NeurIPS-style.*

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

> Four tasks (dev 40 / validation 30 / holdout 30; a perturbation set for the
> financial task), three deterministic simulated model profiles (glm/qwen/gpt),
> seeds fixed per comparison (crc32-derived), paired bootstrap 95% CIs; task D
> additionally carries a no-halving control. Real-model phase: qwen3.8-flash,
> temp=0, identical rule-judge, dev5/val8/hold20 shared protocol.

*Full simulation tables and the real-model findings F1–F6 live in the Chinese
master `paper-apc.md` §3; the numbers are:*

- 3.2a–d (simulation): search − zero-shot significant on all four tasks;
  evolution − random +0.003 on financial, −0.013 on constraint-following (SHA
  trade-off, quantified, no free lunch); PGAM pooled null (+0.0004, n=42).
- 3.3 (migration): recover = 1.0000 [0.9976, 1.0025]; adopt 12/12; flat decay
  matrix is a property of the simulator, honestly flagged.
- 3.4 (real, qwen3.8-flash): F1 accuracy+robustness double saturation;
  F2 rule prior is a net liability, base-root search is a lossless floor;
  F3 zero-adapt transfer is the strongest policy (+0.20~+0.79 vs cold re-search);
  F4 rule-vs-self-LLM judge disagreement quantified (0.21 vs 0.83 on accuracy);
  F5 perturbation robustness |drop| ≤ 0.013 for all four genomes;
  F6 external benchmarks (GSM8K/MATH-L5/AIME24+25): all six champion-vs-base
  deltas within 1σ, three arms perfect on AIME — transfer and saturation both
  reproduce on author-unseen tasks.
- 3.5 (GEPA baseline, pending): reflective-Pareto official baseline under the
  identical protocol and rollout-aligned budget.

## 4. Limitations

1. Real-model phase is single-model, single-seed (whitelist key); scripts ready
   for multi-model (`--model <id>` + three `.env` lines). Ground truth is
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
DSPy–MIPROv2 → GEPA (strongest baseline, head-to-head in §3.5). Profiles and
routing: HELM/FLASK/FrugalGPT/RouteLLM. Structure and transfer: Sclar format
sensitivity; soft prompts (non-portable) vs discrete genomes (re-compilable).
2026 frontier: Instruction Stacking Collapse (2608.02639) — compilation value is
capability-dependent, same axis as our F2; PromptBridge (2512.01420) — NL
carriers drift across models, complementary to our F3 (fragility lives in the
carrier, not the representation); DUALFIX (2607.05121) — rule-based evolution
resurges: rules belong in the genome, not free text; CAPO (2608.16068) —
constraint-aware optimization, baseline candidate for the constraint dimension;
Atlas (2603.15666).
