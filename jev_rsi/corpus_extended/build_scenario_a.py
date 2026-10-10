"""Scenario A -- high-variance judge groups (P3 plan, Week 2).

20 creative-writing tasks (5 kinds x 4), each with 10 prompt strategies:

    zero_shot, few_shot_2ex, cot, persona_author, persona_child,
    step_by_step, creative_mode, constrained, temperature_high, temperature_low

Val     = a *single* judge drawing (0-10)
Holdout = the mean of *five independent* judge drawings (0-10)

Both judges see exactly the same prompt; they differ only in sampling.  The
per-arm spread of the five drawings is the judge-variance signal the plan asks
for (``std > 2.0`` on the 0-10 scale marks a high-variance group).

Scores are normalised to 0-1 before they enter the harness, because the gate's
noise band (``NOISE_BAND = 0.007``) and its threshold rules are defined on the
0-1 scale used by the frozen APC corpus.  The raw 0-10 numbers are kept in
``judge_raw``.

Usage
-----
    python -m jev_rsi.corpus_extended.build_scenario_a --stage tasks
    python -m jev_rsi.corpus_extended.build_scenario_a --stage arms
    python -m jev_rsi.corpus_extended.build_scenario_a --stage scores
    python -m jev_rsi.corpus_extended.build_scenario_a            # all three
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..llm import PRIMARY_MODEL, LLMClient, extract_score, write_json

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "scenario_a"

NO_THINK = {"enable_thinking": False}

STRATEGIES = ["zero_shot", "few_shot_2ex", "cot", "persona_author", "persona_child",
              "step_by_step", "creative_mode", "constrained", "temperature_high",
              "temperature_low"]
STRATEGY_TEMPERATURE = {"temperature_low": 0.1, "temperature_high": 1.5}
GEN_TEMPERATURE = 0.9
JUDGE_TEMPERATURE = 1.0
N_HOLDOUT_JUDGE_DRAWS = 5

FEW_SHOT_EXAMPLES = """Example 1
Task: Write a short scene about a rainy bus stop.
Response: The 7:15 had already given up. Rain drummed the tin shelter like an
impatient audience, and the woman with the cello case pressed her back to the
glass, guarding the instrument the way you guard a sleeping child.

Example 2
Task: Write a fable whose moral is that patience outlasts strength.
Response: The oak mocked the reed for bending in every wind, until the storm
came that broke the oak and left the reed standing in the morning, still green,
still humming with the river.
"""


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
TASKS: Dict[str, dict] = {
    "creative_story_01": {
        "type": "story_continuation",
        "prompt": "The old lighthouse keeper noticed something unusual in the fog that morning.",
        "target_length": 200,
        "criteria": "creativity, coherence, engagement",
    },
    "creative_story_02": {
        "type": "story_continuation",
        "prompt": "Every time the train passed the abandoned station, Mira counted one more suitcase on the platform.",
        "target_length": 200,
        "criteria": "creativity, coherence, engagement",
    },
    "creative_story_03": {
        "type": "story_continuation",
        "prompt": "The recipe card was written in a hand that had been dead for forty years.",
        "target_length": 200,
        "criteria": "creativity, coherence, engagement",
    },
    "creative_story_04": {
        "type": "story_continuation",
        "prompt": "On the last day of the drought, the well gave back something that was not water.",
        "target_length": 200,
        "criteria": "creativity, coherence, engagement",
    },
    "creative_poem_01": {
        "type": "poem_sonnet",
        "prompt": "Write a sonnet about the sea remembering a drowned city.",
        "target_length": 14,
        "criteria": "imagery, rhythm, emotional resonance",
    },
    "creative_poem_02": {
        "type": "poem_sonnet",
        "prompt": "Write a sonnet about a library closing for the last time.",
        "target_length": 14,
        "criteria": "imagery, rhythm, emotional resonance",
    },
    "creative_poem_03": {
        "type": "poem_sonnet",
        "prompt": "Write a sonnet about a city seen from a night train.",
        "target_length": 14,
        "criteria": "imagery, rhythm, emotional resonance",
    },
    "creative_poem_04": {
        "type": "poem_sonnet",
        "prompt": "Write a sonnet about an unfinished conversation with a parent.",
        "target_length": 14,
        "criteria": "imagery, rhythm, emotional resonance",
    },
    "creative_dialogue_01": {
        "type": "dialogue",
        "prompt": "Two strangers share a hospital waiting room at 3 a.m. One of them knows something the other does not.",
        "target_length": 200,
        "criteria": "character voice, subtext, naturalness",
    },
    "creative_dialogue_02": {
        "type": "dialogue",
        "prompt": "A retired teacher and a former student meet in a supermarket twenty years later.",
        "target_length": 200,
        "criteria": "character voice, subtext, naturalness",
    },
    "creative_dialogue_03": {
        "type": "dialogue",
        "prompt": "Two colleagues disagree about whether to report a mistake that no one has noticed yet.",
        "target_length": 200,
        "criteria": "character voice, subtext, naturalness",
    },
    "creative_dialogue_04": {
        "type": "dialogue",
        "prompt": "An astronaut and a mission controller argue during the last three minutes of a launch window.",
        "target_length": 200,
        "criteria": "character voice, subtext, naturalness",
    },
    "creative_scene_01": {
        "type": "scene_description",
        "prompt": "A fish market at dawn in a coastal town, just after a storm.",
        "target_length": 150,
        "criteria": "sensory detail, atmosphere, precision",
    },
    "creative_scene_02": {
        "type": "scene_description",
        "prompt": "An empty Olympic swimming pool in winter, twenty years after the games.",
        "target_length": 150,
        "criteria": "sensory detail, atmosphere, precision",
    },
    "creative_scene_03": {
        "type": "scene_description",
        "prompt": "A rooftop garden in a city during the first hour of a blackout.",
        "target_length": 150,
        "criteria": "sensory detail, atmosphere, precision",
    },
    "creative_scene_04": {
        "type": "scene_description",
        "prompt": "A country railway station at midnight in freezing fog.",
        "target_length": 150,
        "criteria": "sensory detail, atmosphere, precision",
    },
    "creative_fable_01": {
        "type": "fable",
        "prompt": "Write a short fable whose moral is that generosity is a kind of memory.",
        "target_length": 180,
        "criteria": "clarity of moral, narrative economy, charm",
    },
    "creative_fable_02": {
        "type": "fable",
        "prompt": "Write a short fable whose moral is that the loudest voice is rarely the wisest.",
        "target_length": 180,
        "criteria": "clarity of moral, narrative economy, charm",
    },
    "creative_fable_03": {
        "type": "fable",
        "prompt": "Write a short fable whose moral is that a map is not the country.",
        "target_length": 180,
        "criteria": "clarity of moral, narrative economy, charm",
    },
    "creative_fable_04": {
        "type": "fable",
        "prompt": "Write a short fable whose moral is that borrowed courage still counts.",
        "target_length": 180,
        "criteria": "clarity of moral, narrative economy, charm",
    },
}

FINAL_MARKER = "###FINAL###"


def stable_seed(text: str) -> int:
    """Process-independent seed (``hash()`` is randomised per interpreter)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


def strategy_prompt(strategy: str, task: dict) -> str:
    p = task["prompt"]
    crit = task["criteria"]
    n = task["target_length"]
    if strategy == "zero_shot":
        return f"{p}\n\nWrite the piece. Aim for roughly {n} words."
    if strategy == "few_shot_2ex":
        return (f"Here are two examples of the kind of writing wanted.\n\n{FEW_SHOT_EXAMPLES}\n"
                f"Now do the same for this task.\nTask: {p}\n\nResponse:")
    if strategy == "cot":
        return (f"{p}\n\nFirst reason briefly about what would make this piece succeed "
                f"on {crit}. Then write the final piece after the marker {FINAL_MARKER}. "
                f"Aim for roughly {n} words.")
    if strategy == "persona_author":
        return (f"You are a celebrated literary author known for precise, vivid prose. "
                f"{p}\n\nWrite the piece. Aim for roughly {n} words.")
    if strategy == "persona_child":
        return (f"Write in the voice of a curious eight-year-old telling this to a friend. "
                f"{p}\n\nAim for roughly {n} words.")
    if strategy == "step_by_step":
        return (f"{p}\n\nWork in stages: (1) brainstorm three possible directions, "
                f"(2) choose the strongest, (3) write it. Put the finished piece after "
                f"the marker {FINAL_MARKER}. Aim for roughly {n} words.")
    if strategy == "creative_mode":
        return (f"{p}\n\nBe as imaginative and surprising as you can while staying coherent. "
                f"Aim for roughly {n} words.")
    if strategy == "constrained":
        return (f"{p}\n\nHard constraints: exactly three paragraphs, each 3-5 sentences; "
                f"no exclamation marks; the final sentence must be under eight words. "
                f"Aim for roughly {n} words.")
    if strategy in ("temperature_high", "temperature_low"):
        return f"{p}\n\nWrite the piece. Aim for roughly {n} words."
    raise ValueError(strategy)


def extract_piece(text: str) -> str:
    if FINAL_MARKER in text:
        return text.split(FINAL_MARKER, 1)[1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def write_tasks() -> dict:
    payload = {"provenance": {"authored": "hand-written task bank (no API calls)",
                              "n_tasks": len(TASKS),
                              "types": sorted({t["type"] for t in TASKS.values()})},
               "tasks": TASKS}
    write_json(SCENARIO / "tasks.json", payload)
    print(f"wrote {SCENARIO/'tasks.json'} ({len(TASKS)} tasks)")
    return TASKS


def load_tasks() -> dict:
    path = SCENARIO / "tasks.json"
    if not path.exists():
        write_tasks()
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("tasks", data)


def generate_arms(model: str = PRIMARY_MODEL, workers: int = 8) -> dict:
    tasks = load_tasks()
    client = LLMClient()
    jobs: List[Tuple[str, str, str, float]] = []
    for tid, task in tasks.items():
        for s in STRATEGIES:
            jobs.append((tid, s, strategy_prompt(s, task),
                         STRATEGY_TEMPERATURE.get(s, GEN_TEMPERATURE)))

    def run(job):
        tid, s, prompt, temp = job
        rec = client.chat([{"role": "user", "content": prompt}], model=model,
                          temperature=temp, max_tokens=900,
                          seed=stable_seed(f"{tid}/{s}"),
                          tag=f"scenario_a/gen/{s}", extra=NO_THINK)
        return (tid, s, extract_piece(rec["text"]), prompt, temp,
                rec["usage"], rec["model_returned"])

    results: Dict[str, List[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (tid, s, text, prompt, temp, usage, backend) in enumerate(
                ex.map(run, jobs), 1):
            results.setdefault(tid, []).append({
                "id": s, "name": s, "strategy": s, "completion": text,
                "prompt": prompt, "temperature": temp,
                "usage": usage, "model_returned": backend,
            })
            print(f"  [{i:>3}/{len(jobs)}] {tid:<20} {s:<16} {len(text.split()):>4}w")

    for tid in results:
        results[tid].sort(key=lambda a: STRATEGIES.index(a["id"]))
    write_json(SCENARIO / "arms.json", {
        "provenance": {
            "generator_model": model, "temperature": GEN_TEMPERATURE,
            "enable_thinking": False, "strategies": STRATEGIES,
            "n_generations": len(jobs), "n_llm_calls": client.n_llm_calls,
            "in_prompt_reuse": ("arms.json repeats the strategy prompt per arm so the "
                                "artifact is self-contained"),
        },
        "groups": results,
    })
    print(f"wrote {SCENARIO/'arms.json'} ({len(jobs)} completions)")
    return results


def load_arms() -> dict:
    data = json.loads((SCENARIO / "arms.json").read_text(encoding="utf-8"))
    return data.get("groups", data)


JUDGE_PROMPT = """Rate the following creative writing on a scale of 0 to 10.

Task: {prompt}
Criteria to reward: {criteria}

Response to rate:
\"\"\"
{completion}
\"\"\"

Judge holistically against the criteria. 0 is unusable, 5 is competent but
forgettable, 10 is exceptional. Reply with JSON only: {{"score": <number>}}"""


def judge_once(client: LLMClient, model: str, task: dict, completion: str,
               draw: int, temperature: float = JUDGE_TEMPERATURE) -> float:
    prompt = JUDGE_PROMPT.format(prompt=task["prompt"], criteria=task["criteria"],
                                 completion=completion)
    obj, rec = client.chat_json(prompt, model=model, temperature=temperature,
                                max_tokens=200, seed=1000 + draw,
                                tag=f"scenario_a/judge/draw{draw}")
    score = extract_score(obj)
    if score is None:
        raise ValueError(f"bad judge payload: {obj!r}")
    return max(0.0, min(10.0, score))


def score_arms(model: str = PRIMARY_MODEL, workers: int = 8) -> dict:
    tasks = load_tasks()
    arms = load_arms()
    client = LLMClient()

    def run(item):
        tid, arm = item
        task = tasks[tid]
        val = judge_once(client, model, task, arm["completion"], draw=0)
        runs = [judge_once(client, model, task, arm["completion"], draw=d)
                for d in range(1, N_HOLDOUT_JUDGE_DRAWS + 1)]
        mean = statistics.fmean(runs)
        return tid, arm["id"], val, runs, mean, statistics.pstdev(runs)

    jobs = [(tid, arm) for tid, g in arms.items() for arm in g]
    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (tid, aid, val, runs, mean, sd) in enumerate(ex.map(run, jobs), 1):
            task = tasks[tid]
            rec = out.setdefault(tid, {
                "scenario": "A", "task": "creative_writing",
                "kind": task["type"], "prompt": task["prompt"],
                "criteria": task["criteria"], "target_length": task["target_length"],
                "arms": [],
            })
            rec["arms"].append({
                "id": aid, "name": aid, "strategy": aid,
                "completion": next(a["completion"] for a in arms[tid] if a["id"] == aid),
                "val_score": round(val / 10.0, 6),
                "holdout_score": round(mean / 10.0, 6),
                "judge_raw": {"val_10": val, "holdout_runs_10": runs,
                              "holdout_mean_10": round(mean, 4),
                              "holdout_std_10": round(sd, 4)},
            })
            print(f"  [{i:>3}/{len(jobs)}] {tid:<20} {aid:<16} "
                  f"val={val:4.1f} mean={mean:5.2f} sd={sd:4.2f}")

    high = 0
    for tid, g in out.items():
        g["arms"].sort(key=lambda a: STRATEGIES.index(a["id"]))
        sds = [a["judge_raw"]["holdout_std_10"] for a in g["arms"]]
        g["judge_variance_10"] = {"mean_arm_std": round(statistics.fmean(sds), 4),
                                  "max_arm_std": round(max(sds), 4)}
        if g["judge_variance_10"]["mean_arm_std"] > 2.0:
            high += 1

    payload = {
        "provenance": {
            "judge_model": model, "judge_temperature": JUDGE_TEMPERATURE,
            "n_holdout_draws": N_HOLDOUT_JUDGE_DRAWS,
            "protocol": ("val = one judge drawing; holdout = mean of 5 independent "
                         "drawings at the same prompt and temperature"),
            "scale_note": ("val_score/holdout_score are normalised to 0-1 for the gate; "
                           "judge_raw keeps the 0-10 numbers and the per-draw spread"),
            "n_llm_calls": client.n_llm_calls,
        },
        "high_variance_threshold_10": 2.0,
        "n_high_variance_groups": high,
        "groups": out,
    }
    write_json(SCENARIO / "scores.json", payload)
    print(f"wrote {SCENARIO/'scores.json'}  high-variance groups (mean arm std > 2.0): "
          f"{high}/{len(out)}")
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["tasks", "arms", "scores", "all"], default="all")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    if args.stage in ("tasks", "all"):
        write_tasks()
    if args.stage in ("arms", "all"):
        generate_arms(args.model, args.workers)
    if args.stage in ("scores", "all"):
        score_arms(args.model, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
