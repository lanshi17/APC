from apc.core.genome import PromptGenome
from apc.core.model_profile import ModelProfile

class CompilerRules:
    @staticmethod
    def adjust_examples(genome: PromptGenome, profile: ModelProfile) -> PromptGenome:
        v = profile.capability.few_shot_benefit
        if v > 0.25:
            genome.examples.enabled = True
            if genome.examples.count < 2: genome.examples.count = 2
        elif v < 0.08:
            genome.examples.enabled = False
            genome.examples.count = 0
        return genome
    @staticmethod
    def adjust_output(genome: PromptGenome, profile: ModelProfile) -> PromptGenome:
        v = profile.capability.json_reliability
        if v < 0.90:
            genome.output.strictness = "high"
            genome.output.include_schema_in_prompt = True
            genome.output.forbid_extra_fields = True
        elif v > 0.96:
            genome.output.strictness = "medium"
            genome.output.include_schema_in_prompt = False
        return genome
    @staticmethod
    def adjust_reasoning(genome: PromptGenome, profile: ModelProfile) -> PromptGenome:
        if profile.capability.reasoning > 0.90 and profile.capability.self_verification_benefit > 0.25:
            genome.reasoning.strategy = "hidden_analysis"
            genome.verification.enabled = True
        return genome
    @classmethod
    def apply(cls, genome: PromptGenome, profile: ModelProfile) -> PromptGenome:
        genome = cls.adjust_examples(genome, profile)
        genome = cls.adjust_output(genome, profile)
        genome = cls.adjust_reasoning(genome, profile)
        return genome
