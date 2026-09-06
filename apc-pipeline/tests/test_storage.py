"""存储层测试：五类记录 round trip + 血统链（KR-3）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.prompt import CompiledPrompt
from apc.storage import repository as repo
from apc.storage.database import get_session

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def session(tmp_path):
    return get_session(f"sqlite:///{tmp_path / 'test.db'}")


def test_task_model_genome_roundtrip(session, task_spec, base_genome):
    repo.save_task(session, task_spec)
    repo.save_model_profile(session, ModelProfile(model_id="glm"), provider="zhipu")
    repo.save_genome(session, base_genome, task_spec.task_id, is_baseline=True)
    row = session.query(repo.GenomeRecord).filter_by(genome_id=base_genome.genome_id).one()
    assert row.genome_json["genome_id"] == base_genome.genome_id
    assert row.is_baseline
    model = session.query(repo.ModelRecord).filter_by(model_id="glm").one()
    assert model.provider == "zhipu"


def test_lineage_chain(session, base_genome, task_spec):
    repo.save_genome(session, base_genome, task_spec.task_id)
    child = PromptGenome.model_validate({**base_genome.model_dump(mode="json"),
                                        "genome_id": "genome_child000001",
                                        "parent_genome_id": base_genome.genome_id,
                                        "mutation_note": "output.strictness: high -> low"})
    grand = PromptGenome.model_validate({**base_genome.model_dump(mode="json"),
                                        "genome_id": "genome_grand0000001",
                                        "parent_genome_id": "genome_child000001",
                                        "mutation_note": "tone.formality: high -> medium"})
    repo.save_genome(session, child, task_spec.task_id)
    repo.save_genome(session, grand, task_spec.task_id)
    chain = repo.get_genome_lineage(session, "genome_grand0000001")
    assert [g.genome_id for g in chain] == ["genome_grand0000001", "genome_child000001",
                                            base_genome.genome_id]


def test_compiled_prompt_and_trial_roundtrip(session, task_spec, base_genome):
    cp = CompiledPrompt(task_id=task_spec.task_id, genome_id=base_genome.genome_id,
                        model_id="glm", prompt_text="PROMPT", token_estimate=5)
    repo.save_compiled_prompt(session, cp)
    got = session.query(repo.CompiledPromptRecord).filter_by(prompt_id=cp.prompt_id).one()
    assert got.prompt_text == "PROMPT" and got.token_estimate == 5

    from apc.core.trial import TrialResult

    trial = TrialResult(trial_id="trial_test0000001", task_id=task_spec.task_id, model_id="glm",
                        genome_id=base_genome.genome_id, prompt_id=cp.prompt_id,
                        dataset_id="dev", dataset_version="abc", score=0.7, accuracy=0.6,
                        instruction_following=0.8, format_score=1.0, constraint_score=1.0,
                        robustness=1.0, efficiency_score=1.0, format_error_rate=0.0,
                        judge_id="rule_based_v1", case_results=[], metadata={})
    repo.save_trial(session, trial, phase="eval")
    got_trial = repo.get_trial(session, "trial_test0000001")
    assert got_trial.phase == "eval" and got_trial.score == 0.7
    assert len(repo.list_trials(session, task_id=task_spec.task_id)) == 1


def test_save_is_idempotent(session, base_genome, task_spec):
    repo.save_genome(session, base_genome, task_spec.task_id)
    repo.save_genome(session, base_genome, task_spec.task_id)
    assert session.query(repo.GenomeRecord).count() == 1
