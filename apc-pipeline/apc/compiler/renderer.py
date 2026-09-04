from __future__ import annotations
import json, uuid
from copy import deepcopy
from apc.core.genome import PromptGenome
from apc.core.task_spec import TaskSpec
from apc.core.model_profile import ModelProfile
from apc.core.prompt import CompiledPrompt
from apc.compiler.rules import CompilerRules

class PromptSectionRenderer:
    @staticmethod
    def render_goal(task_spec: TaskSpec, genome: PromptGenome) -> str:
        return f"任务目标：\n{task_spec.objective.strip()}"
    @staticmethod
    def render_constraints(task_spec: TaskSpec, genome: PromptGenome) -> str:
        constraints = task_spec.constraints[: genome.constraints.max_count]
        lines = ["约束条件："]
        for i, c in enumerate(constraints, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)
    @staticmethod
    def render_output_schema(task_spec: TaskSpec, genome: PromptGenome) -> str:
        if not genome.output.include_schema_in_prompt:
            return "请严格按照任务要求的输出结构返回结果。"
        schema_text = json.dumps(task_spec.output.schema_, ensure_ascii=False, indent=2) if task_spec.output.schema_ else "{}"
        return f"输出要求：\n只输出 JSON，不要输出解释。\nJSON 必须符合以下 schema：\n{schema_text}\n不要添加额外字段。"

class DefaultPromptCompiler:
    def compile(self, genome: PromptGenome, task_spec: TaskSpec, model_profile: ModelProfile) -> CompiledPrompt:
        g = deepcopy(genome)
        g = CompilerRules.apply(g, model_profile)
        sections: list[str] = []
        if g.role.enabled:
            sections.append(f"你是一名{g.role.style}，请以专业方式完成任务。")
        sections.append(PromptSectionRenderer.render_goal(task_spec, g))
        if g.constraints.placement == "after_goal":
            sections.append(PromptSectionRenderer.render_constraints(task_spec, g))
        if g.examples.enabled:
            sections.append(self.render_examples(g))
        sections.append("输入内容如下：\n<document>\n{{input}}\n</document>")
        if g.reasoning.strategy != "none":
            sections.append(self.render_reasoning(g))
        if g.verification.enabled:
            sections.append(self.render_verification(g))
        sections.append(PromptSectionRenderer.render_output_schema(task_spec, g))
        prompt_text = self.assemble_sections(sections, g)
        return CompiledPrompt(
            prompt_id=str(uuid.uuid4()),
            genome_id=g.task_id,
            model_id=model_profile.model_id,
            prompt_text=prompt_text,
            token_estimate=len(prompt_text)//4,
            metadata={"genome_version": g.genome_version},
        )
    def render_examples(self, genome: PromptGenome) -> str:
        return "示例：\n输入：...\n输出：..."
    def render_reasoning(self, genome: PromptGenome) -> str:
        return {
            "hidden_analysis": "请先在内部分析输入内容，但不要输出分析过程，只输出最终结果。",
            "brief_plan": "请先简要规划完成任务的步骤，然后输出最终结果。",
            "structured_checklist": "请按照检查清单方式确认关键约束后再输出最终结果。",
            "decompose_then_answer": "请先分解任务，再逐步作答。",
        }.get(genome.reasoning.strategy, "")
    def render_verification(self, genome: PromptGenome) -> str:
        return {
            "constraint_check": "在输出前，请检查是否满足所有约束条件。",
            "source_grounding_check": "在输出前，请确认所有结论均来自输入材料。",
            "format_check": "在输出前，请检查输出格式是否符合要求。",
        }.get(genome.verification.type, "")
    def assemble_sections(self, sections: list[str], genome: PromptGenome) -> str:
        if genome.layout.delimiter == "xml":
            return "\n\n".join(f"<section_{i}>\n{s}\n</section_{i}>" for i, s in enumerate(sections, 1))
        return "\n\n".join(sections)
