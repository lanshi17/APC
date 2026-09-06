"""仓储函数：核心对象 → 五类记录（幂等 upsert + 追溯查询）。"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.prompt import CompiledPrompt
from apc.core.task_spec import TaskSpec
from apc.core.trial import TrialResult
from apc.models.factory import load_model_config
from apc.storage.database import (
    CompiledPromptRecord,
    GenomeRecord,
    ModelRecord,
    TaskRecord,
    TrialRecord,
)


def _now():
    return datetime.now(timezone.utc)


def save_task(session: Session, spec: TaskSpec) -> TaskRecord:
    obj = session.query(TaskRecord).filter_by(task_id=spec.task_id, version=spec.version).first()
    if obj is None:
        obj = TaskRecord(task_id=spec.task_id, version=spec.version, created_at=_now())
        session.add(obj)
    obj.objective = spec.objective
    obj.spec_json = spec.model_dump(mode="json")
    session.commit()
    return obj


def save_model_profile(session: Session, profile: ModelProfile, provider: str = "") -> ModelRecord:
    obj = session.query(ModelRecord).filter_by(model_id=profile.model_id).first()
    if obj is None:
        obj = ModelRecord(model_id=profile.model_id, created_at=_now())
        session.add(obj)
    obj.provider = provider or load_model_config(profile.model_id).get("provider", "unknown")
    obj.capability_json = profile.capability.model_dump()
    obj.profile_version = profile.profile_version
    session.commit()
    return obj


def save_genome(session: Session, genome: PromptGenome, task_id: str,
                is_baseline: bool = False) -> GenomeRecord:
    obj = session.query(GenomeRecord).filter_by(genome_id=genome.genome_id).first()
    if obj is None:
        obj = GenomeRecord(genome_id=genome.genome_id, created_at=_now())
        session.add(obj)
    obj.parent_genome_id = genome.parent_genome_id
    obj.task_id = task_id
    obj.mutation_note = genome.mutation_note
    obj.is_baseline = is_baseline
    obj.genome_json = genome.model_dump(mode="json")
    session.commit()
    return obj


def save_compiled_prompt(session: Session, prompt: CompiledPrompt) -> CompiledPromptRecord:
    obj = session.query(CompiledPromptRecord).filter_by(prompt_id=prompt.prompt_id).first()
    if obj is None:
        obj = CompiledPromptRecord(prompt_id=prompt.prompt_id, created_at=_now())
        session.add(obj)
    obj.genome_id = prompt.genome_id
    obj.task_id = prompt.task_id
    obj.model_id = prompt.model_id
    obj.prompt_text = prompt.prompt_text
    obj.token_estimate = prompt.token_estimate or 0
    obj.metadata_json = prompt.metadata
    session.commit()
    return obj


def save_trial(session: Session, trial: TrialResult, phase: str | None = None) -> TrialRecord:
    obj = session.query(TrialRecord).filter_by(trial_id=trial.trial_id).first()
    if obj is None:
        obj = TrialRecord(trial_id=trial.trial_id, created_at=_now())
        session.add(obj)
    obj.task_id = trial.task_id
    obj.model_id = trial.model_id
    obj.genome_id = trial.genome_id
    obj.prompt_id = trial.prompt_id
    obj.dataset_id = trial.dataset_id
    obj.dataset_version = trial.dataset_version
    obj.judge_id = trial.judge_id
    obj.score = trial.score
    obj.accuracy = trial.accuracy
    obj.instruction_following = trial.instruction_following
    obj.format_score = trial.format_score
    obj.constraint_score = trial.constraint_score
    obj.robustness = trial.robustness
    obj.efficiency_score = trial.efficiency_score
    obj.format_error_rate = trial.format_error_rate
    obj.parent_genome_id = trial.parent_genome_id
    obj.mutation_note = trial.mutation_note
    obj.avg_latency_ms = trial.avg_latency_ms
    obj.total_input_tokens = trial.total_input_tokens
    obj.total_output_tokens = trial.total_output_tokens
    obj.phase = phase
    obj.metadata_json = trial.metadata
    session.commit()
    return obj


def get_genome_lineage(session: Session, genome_id: str, max_depth: int = 64) -> list[GenomeRecord]:
    """从任一 genome 沿 parent_genome_id 回溯完整血统（KR-3 追溯）。"""
    chain: list[GenomeRecord] = []
    current_id: str | None = genome_id
    for _ in range(max_depth):
        if not current_id:
            break
        rec = session.query(GenomeRecord).filter_by(genome_id=current_id).first()
        if rec is None:
            break
        chain.append(rec)
        current_id = rec.parent_genome_id
    return chain


def get_trial(session: Session, trial_id: str) -> TrialRecord | None:
    return session.query(TrialRecord).filter_by(trial_id=trial_id).first()


def list_trials(session: Session, task_id: str | None = None, model_id: str | None = None,
                phase: str | None = None) -> list[TrialRecord]:
    q = session.query(TrialRecord)
    if task_id:
        q = q.filter_by(task_id=task_id)
    if model_id:
        q = q.filter_by(model_id=model_id)
    if phase:
        q = q.filter_by(phase=phase)
    return q.order_by(TrialRecord.created_at.desc()).all()
