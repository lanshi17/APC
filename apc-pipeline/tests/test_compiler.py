"""编译器测试：genome 不可变性、规则开关、片段装配。"""
from __future__ import annotations

from apc.compiler.renderer import DefaultPromptCompiler
from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile


def test_compile_does_not_mutate_genome(base_genome, task_spec):
    before = base_genome.model_dump_json()
    profile = ModelProfile(model_id="glm")
    DefaultPromptCompiler().compile(base_genome, task_spec, profile)
    assert base_genome.model_dump_json() == before


def test_prompt_and_genome_id_formats(base_genome, task_spec):
    profile = ModelProfile(model_id="glm")
    cp = DefaultPromptCompiler().compile(base_genome, task_spec, profile)
    assert cp.prompt_id.startswith("prompt_") and len(cp.prompt_id) == 19
    assert cp.genome_id == base_genome.genome_id
    assert cp.task_id == task_spec.task_id


def test_apply_rules_true_disables_examples_for_low_benefit(base_genome, task_spec):
    profile = ModelProfile(model_id="glm")  # few_shot_benefit 默认 0 → 规则禁用
    profile.capability.few_shot_benefit = 0.0
    genome = PromptGenome.model_validate(base_genome.model_dump())
    genome.examples.enabled = True
    genome.examples.count = 2  # 显式给出数量（渲染语义：enabled + count>0 才渲染）
    cp = DefaultPromptCompiler().compile(genome, task_spec, profile, apply_rules=True)
    assert "示例" not in cp.prompt_text
    cp2 = DefaultPromptCompiler().compile(genome, task_spec, profile, apply_rules=False)
    assert "示例" in cp2.prompt_text


def test_json_schema_embedding(base_genome, task_spec):
    profile = ModelProfile(model_id="glm")
    cp = DefaultPromptCompiler().compile(base_genome, task_spec, profile)
    assert '"metrics"' in cp.prompt_text and '"summary"' in cp.prompt_text
    assert "不要添加 schema 之外的任何额外字段" in cp.prompt_text


def test_input_placeholder_present(base_genome, task_spec):
    profile = ModelProfile(model_id="glm")
    cp = DefaultPromptCompiler().compile(base_genome, task_spec, profile)
    assert "{{input}}" in cp.prompt_text
    assert cp.token_estimate > 0
