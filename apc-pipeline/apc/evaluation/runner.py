from __future__ import annotations

import json
import uuid
from pathlib import Path

from apc.core.prompt import CompiledPrompt
from apc.core.task_spec import TaskSpec
from apc.core.trial import TrialResult
from apc.evaluation.checker import RuleBasedChecker
from apc.evaluation.dataset import Dataset, load_dataset
from apc.evaluation.judge import BaseJudge, RuleBasedJudge
from apc.evaluation.scorer import TrialScorer
from apc.models.base import BaseModelClient

_REPO_ROOT = Path(__file__).parents[3]


class EvaluationRunner:
    """编译后 Prompt + 数据集 → 逐样本执行 → TrialResult + 原始输出落盘（FR-5）。"""

    def __init__(self, client: BaseModelClient, judge: BaseJudge | None = None,
                 checker: RuleBasedChecker | None = None, artifacts_dir: str | Path | None = None):
        self.client = client
        self.judge = judge or RuleBasedJudge()
        self.checker = checker or RuleBasedChecker()
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else _REPO_ROOT / "artifacts"
        self.outputs_dir = self.artifacts_dir / "outputs"

    def evaluate(
        self,
        task_spec: TaskSpec,
        compiled_prompt: CompiledPrompt,
        dataset: Dataset | str | Path,
        *,
        dataset_id: str | None = None,
        temperature: float = 0.0,
        limit: int | None = None,
        robustness_override: float | None = None,
        parent_genome_id: str | None = None,
        mutation_note: str | None = None,
        save_outputs: bool = True,
    ) -> TrialResult:
        ds = dataset if isinstance(dataset, Dataset) else load_dataset(dataset, dataset_id)
        trial_id = f"trial_{uuid.uuid4().hex[:12]}"
        case_results: list[dict] = []
        raw_lines: list[str] = []

        samples = ds.samples if limit is None else ds.samples[:limit]
        for i, sample in enumerate(samples):
            document = self._document(sample)
            prompt_text = compiled_prompt.prompt_text.replace("{{input}}", document)
            call = self.client.complete(prompt_text, temperature=temperature)

            rule = self.checker.check(call.text, task_spec)
            j = self.judge.judge(document, sample.get("expected", {}), call.text, task_spec)
            case = {
                "sample_id": ds.sample_id(i),
                "format_score": rule.get("format_score", 0.0),
                "constraint_score": rule.get("constraint_score", 0.0),
                "accuracy": round(float(j.get("accuracy", 0.0)), 4),
                "instruction_following": round(float(j.get("constraint_following", 0.0)), 4),
                "latency_ms": call.latency_ms,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "output": call.text,
            }
            if self.checker.format_error(rule):
                case["format_error"] = rule.get("format_error", "schema_violation")
            case_results.append(case)
            raw_lines.append(json.dumps({
                "sample_id": case["sample_id"], "output": call.text, "rule": rule,
                "judge": j, "usage": {"input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
                                      "latency_ms": call.latency_ms, "retries": call.retries},
            }, ensure_ascii=False))

        outputs_file = None
        if save_outputs:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            outputs_file = self.outputs_dir / f"{trial_id}.jsonl"
            outputs_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

        return TrialScorer().score(
            task_spec, case_results,
            trial_id=trial_id,
            model_id=self.client.model_id,
            genome_id=compiled_prompt.genome_id,
            prompt_id=compiled_prompt.prompt_id,
            dataset_id=ds.dataset_id,
            dataset_version=ds.version,
            judge_id=self.judge.judge_id,
            temperature=temperature,
            model_version=self.client.model_version,
            parent_genome_id=parent_genome_id,
            mutation_note=mutation_note,
            robustness_override=robustness_override,
            metadata={
                "prompt_token_estimate": compiled_prompt.token_estimate,
                "outputs_file": str(outputs_file) if outputs_file else None,
                "sample_count": len(samples),
            },
        )

    @staticmethod
    def _document(sample: dict) -> str:
        raw = sample.get("input", sample)
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            if "document" in raw:
                return str(raw["document"])
            return "\n".join(str(v) for v in raw.values())
        return str(raw)
