from __future__ import annotations
from copy import deepcopy
from pydantic import BaseModel
from apc.core.model_profile import ModelProfile
from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec

class CapabilityDelta(BaseModel):
    source_model_id: str
    target_model_id: str
    diffs: dict[str,float]

class CapabilityDeltaCalculator:
    def calculate(self, source: ModelProfile, target: ModelProfile) -> CapabilityDelta:
        s=source.capability.model_dump(); t=target.capability.model_dump()
        return CapabilityDelta(source_model_id=source.model_id, target_model_id=target.model_id, diffs={k: t[k]-s[k] for k in s})

class MigrationMutator:
    def apply_delta(self, genome: PromptGenome, delta: CapabilityDelta) -> PromptGenome:
        g=deepcopy(genome); d=delta.diffs
        if d.get("few_shot_benefit",0)>0.15:
            g.examples.enabled=True; g.examples.count=max(g.examples.count,2)
        if d.get("json_reliability",0)<-0.05:
            g.output.strictness="high"; g.output.include_schema_in_prompt=True; g.output.forbid_extra_fields=True
        if d.get("schema_strictness",0)>0.05:
            g.output.strictness="medium"; g.output.include_schema_in_prompt=False
        if d.get("self_verification_benefit",0)>0.10:
            g.verification.enabled=True; g.verification.type="constraint_check"
        if d.get("reasoning",0)>0.10: g.reasoning.strategy="hidden_analysis"
        return g

class PromptMigrationPipeline:
    def __init__(self, compiler, evaluator, optimizer, delta_calculator=None, migration_mutator=None):
        self.compiler=compiler; self.evaluator=evaluator; self.optimizer=optimizer
        self.delta_calculator=delta_calculator or CapabilityDeltaCalculator()
        self.migration_mutator=migration_mutator or MigrationMutator()
    def migrate(self, task_spec: TaskSpec, source_profile: ModelProfile, target_profile: ModelProfile, source_best_genome: PromptGenome):
        delta=self.delta_calculator.calculate(source_profile, target_profile)
        seed=self.migration_mutator.apply_delta(source_best_genome, delta)
        return self.optimizer.optimize(task_spec, target_profile, [seed], generations=3, population_size=20)
