# ARR Responsible NLP Checklist — prepared answers (fill at OpenReview submission)

## Are falsifiable claims supported by evidence? — YES (detail per claim)

1. **Structured prompt search on strong reasoners is statistically null.**
   Evidence: paired same-day 4-arm runs (Table 2), four optimizer families incl. official
   GEPA/ESPO reproductions (Sec 5.4), confirmatory n=100 McNemar p=.219 (Sec 5.5).
   Caveat stated in paper: gpt-6 search cells withdrawn; qwen null established on two providers.
2. **Profile-rule prior is a net liability.** Evidence: dev-deficit 0.2–0.53, 3-seed robust,
   bimodal collapse .5704/.5705/.1334/.1318 (Sec 5.2).
3. **Headroom-type taxonomy.** Evidence: F10 token-starve (six pathways at judge floor),
   F11 90-problem reachable-headroom tie, GPQA n=190 tie + zero-acceptance (Sec 5.5).
4. **GEPA cross-model divergence + judge-gaming decomposition.** Evidence: .8914 vs paired
   z0 .7039; champion text forensic quotes; stripped-replica (.8441→.8092 archived recheck);
   gated ESPO null .7021 (Sec 6.1). Caveat: single-seed; search trajectory possibly backend-mixed.
5. **PGAM prior transfer failure.** Evidence: .5960 vs uniform .6336, same budget/seed (Sec 6.2).
6. **Zero-adaptation transfer.** Evidence: t0 ≥ target z0 both directions; gpt-6 t0/cold clean
   cells +.2107 (Table 3).
7. **Deployment gate.** Evidence: 7 replay+blind groups, 5/7 exact-oracle, zero-regret rescue
   (Sec 7).
8. **Evaluation-infrastructure forensics.** Evidence: response-model audit stream
   (artifacts/backend_audit/), 74:6 one-way service-drift flips + 4/4 live-probe recovery
   (Sec 6.3).

## Limitations section — present (Sec 8). Ethics — present (Sec 9).

## Human subjects / data — No human subjects; no PII; benchmarks used per their licenses
(GSM8K/MIT, MATH/MIT, AIME/public, GPQA-Diamond/CC-BY-4.0, HLE-Exact per terms).

## Compute — CPU workstation + public inference APIs (DashScope qwen3.8-flash; gpt-6 via
OpenAI-compatible gateway). No training. Budget currency is task rollouts (8/arm search;
48-rollout baselines), matching GEPA's accounting.

## Code/data — Deterministic simulator + all real-API harness scripts + row-level streams +
per-call backend audit shipped in the supplementary repo archive; judge regression suite
(27 pos / 4 neg) with full prediction traces; three-way number-consistency audit script
(35 values × 3 artifacts).

## Risks — Model safety: none beyond standard text-in/text-out use. Misuse: the paper's
judge-gaming forensics is defensive (documents the risk of unconditional adoption and the
mitigation protocol); no attack tooling released beyond the description needed to evaluate
the claim.

## Notes for OpenReview forms
- Area: Resources and Evaluation / LLMs (prompt optimization & evaluation reliability)
- Suggest reviewers: prompt-optimization (DSPy/GEPA lineage), LLM evaluation reliability
- Withdrawn cells (gpt-6 search arms; contract apc-safe; transfer-ws) are marked † in-text
  and are NOT claimed — do not check "all cells multi-seed" boxes; the Limitations section
  enumerates this explicitly.
- ARR Oct 12, 2026 deadline; commitment NAACL 2027 (Dec 23) or COLING 2027 (Dec 23).
