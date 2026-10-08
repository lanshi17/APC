#!/usr/bin/env python
"""Is the "high-variance judge" regime reachable at all with the available models?

Scenario A's pre-registered design assumed a judge whose repeated scores on the
same (prompt, completion) pair vary by more than 2.0 on the 0-10 scale.  With the
primary judge the measured spread was ~0.5, so the scenario could not test the
hypothesis it was built for.  This probe repeats the measurement for every
available judge model, on a fixed subset of the Scenario A groups, and reports
the per-arm standard deviation of repeated draws.

    python -m jev_rsi.corpus_extended.probe_judge_variance \
        --models qwen3.8-flash qwen3.8-27b --groups 5 --draws 6
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from ..llm import LLMClient, write_json
from .build_scenario_a import (JUDGE_TEMPERATURE, SCENARIO, judge_once,
                               load_arms, load_tasks)

MODELS = ["qwen3.8-flash", "qwen3.8-27b"]


def _task_of(tasks: dict, gid: str) -> dict:
    """``load_tasks`` returns {tid: task} with tid == the group id."""
    return tasks[gid]


def probe(models: List[str], groups: int, draws: int, workers: int = 6) -> dict:
    tasks = load_tasks()
    arms = load_arms()
    gids = sorted(arms)[:groups]
    client = LLMClient()
    out: Dict[str, dict] = {}

    for model in models:
        per_arm: Dict[str, List[float]] = {}
        def run(job):
            gid, arm = job
            task = _task_of(tasks, gid)
            # seed varies with the draw index, exactly as in the main protocol
            scores = [judge_once(client, model, task, arm["completion"], draw=d,
                                 temperature=JUDGE_TEMPERATURE)
                      for d in range(draws)]
            return f"{gid}/{arm['id']}", scores

        jobs = [(gid, a) for gid in gids for a in arms[gid]]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, scores in ex.map(run, jobs):
                per_arm[key] = scores
                print(f"  [{model}] {key:38s} mean={statistics.fmean(scores):5.2f} "
                      f"sd={statistics.pstdev(scores):4.2f}")

        sds = [statistics.pstdev(v) for v in per_arm.values()]
        out[model] = {
            "n_arms": len(sds),
            "draws": draws,
            "temperature": JUDGE_TEMPERATURE,
            "mean_arm_std_10": round(statistics.fmean(sds), 4),
            "median_arm_std_10": round(statistics.median(sds), 4),
            "max_arm_std_10": round(max(sds), 4),
            "n_arms_std_gt_2": sum(1 for s in sds if s > 2.0),
            "frac_arms_std_gt_2": round(sum(1 for s in sds if s > 2.0) / len(sds), 4),
            "per_arm": {k: [round(x, 3) for x in v] for k, v in per_arm.items()},
        }
        print(f"  -> {model}: mean arm sd (0-10) = {out[model]['mean_arm_std_10']}, "
              f"max = {out[model]['max_arm_std_10']}, "
              f"arms with sd>2.0: {out[model]['n_arms_std_gt_2']}/{len(sds)}")

    payload = {
        "protocol": (f"{draws} independent judge draws at temperature "
                     f"{JUDGE_TEMPERATURE} on identical (prompt, completion) pairs; "
                     f"first {groups} Scenario A groups"),
        "group_ids": gids,
        "models": out,
    }
    write_json(SCENARIO / "judge_variance_probe.json", payload)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    probe(args.models, args.groups, args.draws, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
