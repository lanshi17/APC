from __future__ import annotations

import json
import re
from copy import deepcopy

from apc.compiler.rules import CompilerRules
from apc.core.genome import GENE_SECTION_NAMES, PromptGenome
from apc.core.model_profile import ModelProfile
from apc.core.prompt import CompiledPrompt
from apc.core.task_spec import TaskSpec

_SECTION_TITLES = {
    "role": "角色",
    "goal": "任务目标",
    "instructions": "执行说明",
    "constraints": "约束条件",
    "examples": "示例",
    "reasoning_instruction": "推理要求",
    "input": "输入内容",
    "verification": "校验要求",
    "output_format": "输出要求",
}


class PromptSectionRenderer:
    @staticmethod
    def render_role(genome: PromptGenome) -> str:
        authority = {"low": "在需澄清时提出疑问", "medium": "基于材料给出专业判断", "high": "以权威专业口径给出结论"}[
            genome.role.authority_level]
        style_name = {"expert": "领域专家", "assistant": "专业助理", "auditor": "审计师", "analyst": "资深分析师"}[
            genome.role.style]
        return f"你是一名{style_name}，{authority}。"

    @staticmethod
    def render_goal(task_spec: TaskSpec, genome: PromptGenome) -> str:
        goal = task_spec.objective.strip()
        if genome.goal.explicitness == "high":
            return f"任务目标（必须完成）：\n{goal}"
        if genome.goal.explicitness == "medium":
            return f"任务目标：\n{goal}"
        return goal

    @staticmethod
    def render_instructions(task_spec: TaskSpec, genome: PromptGenome) -> str:
        """按 instructions 基因的风格/粒度/分解生成执行说明。"""
        g = genome.instructions
        steps = ["仔细阅读输入材料，定位与任务目标相关的信息"]
        if g.granularity == "fine":
            steps += ["提取关键数据与变化，并核对是否可在输入中找到依据", "对照每条约束自查结论"]
        steps.append("完成任务目标所述的分析")
        steps.append("严格按「输出要求」规定的结构返回结果")
        if g.granularity == "coarse":
            steps = [steps[0], "完成任务目标所述分析并按规定结构返回"]
        if g.step_decomposition:
            if g.style == "imperative":
                return "执行说明：\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
            if g.style == "declarative":
                return "执行说明：\n本任务按以下步骤完成：\n" + "；".join(steps) + "。"
            return "执行说明：\n你可以先" + "，然后".join(steps[:-1]) + "，最后" + steps[-1] + "。"
        text = steps[0] + "，" + steps[-1]
        if g.style == "declarative":
            return f"执行说明：\n本任务需要{text}。"
        if g.style == "conversational":
            return f"执行说明：\n你可以先{steps[0]}，然后{steps[-1]}。"
        return f"执行说明：\n请先{steps[0]}，然后{steps[-1]}。"

    @staticmethod
    def render_constraints(task_spec: TaskSpec, genome: PromptGenome) -> str:
        constraints = task_spec.constraints[: genome.constraints.max_count]
        if not constraints:
            return ""
        prefix = {"high": "以下约束必须逐条满足", "medium": "请遵守以下约束", "low": "参考约束"}[
            genome.constraints.explicitness]
        lines = [f"{prefix}："] + [f"{i}. {c}" for i, c in enumerate(constraints, 1)]
        return "\n".join(lines)

    @staticmethod
    def render_examples(task_spec: TaskSpec, genome: PromptGenome) -> str:
        count = max(genome.examples.count, 2 if genome.examples.enabled else 0)
        if count <= 0:
            return ""
        examples = task_spec.examples[:count]
        if not examples:
            schema = task_spec.output.schema_ or {}
            skeleton = {k: ("<按输入填写>" if v == "string" else "[]" if isinstance(v, dict) and v.get("type") == "array" else 0.9)
                        for k, v in schema.items()}
            examples = [{"input": "<输入材料摘要>", "output": json.dumps(skeleton, ensure_ascii=False)}]
        lines = ["示例（仅演示输出结构，数据以实际输入为准）："]
        for i, ex in enumerate(examples, 1):
            lines.append(f"示例{i}：\n输入：{ex.get('input', '')}")
            if genome.examples.format == "input_reasoning_output" and ex.get("reasoning"):
                lines.append(f"分析：{ex['reasoning']}")
            lines.append(f"输出：{ex.get('output', '')}")
        return "\n".join(lines)

    @staticmethod
    def render_reasoning(genome: PromptGenome) -> str:
        return {
            "hidden_analysis": "请先在内部完整分析输入内容，但不要输出分析过程，只输出最终结果。",
            "brief_plan": "请先用一两句话规划完成任务的步骤，然后输出最终结果。",
            "structured_checklist": "请按检查清单逐项确认关键约束后再输出最终结果。",
            "decompose_then_answer": "请先将任务分解为子问题，逐步解答后输出最终结果。",
        }.get(genome.reasoning.strategy, "")

    @staticmethod
    def render_verification(genome: PromptGenome) -> str:
        return {
            "constraint_check": "在输出前，请逐条核对是否满足所有约束条件，不满足则修正后再输出。",
            "self_consistency": "在输出前，请重新推导一遍并核对两次结果是否一致，不一致则以更谨慎者为准。",
            "source_grounding_check": "在输出前，请确认所有结论均能在输入材料中找到依据，无法确认的内容明确说明。",
            "format_check": "在输出前，请检查输出格式是否完全符合输出要求。",
        }.get(genome.verification.type, "")

    @staticmethod
    def render_output(task_spec: TaskSpec, genome: PromptGenome) -> str:
        o = genome.output
        if task_spec.output.type != "json":
            return f"输出要求：\n以{task_spec.output.type}格式返回任务结果。"
        schema = task_spec.output.schema_
        lines = ["输出要求：", "只输出 JSON，不要输出任何解释、前后缀或代码块标记。"]
        if o.strictness != "low" and o.include_schema_in_prompt:
            lines.append("JSON 必须符合以下 schema：\n" + json.dumps(schema, ensure_ascii=False, indent=2))
        elif o.include_schema_in_prompt:
            fields = ", ".join(schema.keys()) if schema else ""
            lines.append(f"JSON 需包含以下字段：{fields}。")
        if o.forbid_extra_fields:
            lines.append("不要添加 schema 之外的任何额外字段。")
        if o.strictness == "high":
            lines.append("输出前请自查：JSON 可被直接解析、字段完整、无额外字段。")
        return "\n".join(lines)


class DefaultPromptCompiler:
    """Genome + TaskSpec + ModelProfile → CompiledPrompt（原始 Genome 不被原地修改）。"""

    def compile(self, genome: PromptGenome, task_spec: TaskSpec, model_profile: ModelProfile,
                apply_rules: bool = True) -> CompiledPrompt:
        g = deepcopy(genome)
        if apply_rules:
            g = CompilerRules.apply(g, model_profile)
        sections = self._build_sections(g, task_spec)
        sections = self._apply_placements(sections, g)
        prompt_text = self.assemble_sections(sections, g)
        return CompiledPrompt(
            task_id=task_spec.task_id,
            genome_id=g.genome_id,
            model_id=model_profile.model_id,
            prompt_text=prompt_text,
            token_estimate=max(1, len(prompt_text) // 3),
            metadata={
                "genome_version": g.genome_version,
                "source_genome_id": genome.genome_id,
                "delimiter": g.layout.delimiter,
                "sections": list(sections.keys()),
                "model_profile_version": model_profile.profile_version,
            },
        )

    def _build_sections(self, g: PromptGenome, spec: TaskSpec) -> dict[str, str]:
        s: dict[str, str] = {}
        if g.role.enabled:
            s["role"] = PromptSectionRenderer.render_role(g)
        s["goal"] = PromptSectionRenderer.render_goal(spec, g)
        s["instructions"] = PromptSectionRenderer.render_instructions(spec, g)
        cons = PromptSectionRenderer.render_constraints(spec, g)
        if cons:
            s["constraints"] = cons
        if g.examples.enabled:
            ex = PromptSectionRenderer.render_examples(spec, g)
            if ex:
                s["examples"] = ex
        if g.reasoning.strategy != "none":
            s["reasoning_instruction"] = PromptSectionRenderer.render_reasoning(g)
        s["input"] = "输入内容如下：\n<document>\n{{input}}\n</document>"
        if g.verification.enabled:
            v = PromptSectionRenderer.render_verification(g)
            if v:
                s["verification"] = v
        s["output_format"] = PromptSectionRenderer.render_output(spec, g)
        return s

    def _apply_placements(self, sections: dict[str, str], g: PromptGenome) -> dict[str, str]:
        """按 layout.section_order 排序，再应用 goal/constraints 的 placement 覆盖。"""
        order = [name for name in g.layout.section_order if name in sections]
        for name in sections:
            if name not in order:
                order.append(name)
        if "constraints" in order and g.constraints.placement == "top":
            order.remove("constraints")
            order.insert(0, "constraints")
        if "constraints" in order and g.constraints.placement == "bottom":
            order.remove("constraints")
            order.append("constraints")
        if "goal" in order and g.goal.placement == "after_context" and "input" in order:
            order.remove("goal")
            order.insert(order.index("input"), "goal")
        if "goal" in order and g.goal.placement == "bottom":
            order.remove("goal")
            order.append("goal")
        return {name: sections[name] for name in order}

    def assemble_sections(self, sections: dict[str, str], genome: PromptGenome) -> str:
        d = genome.layout.delimiter
        if d == "xml":
            return "\n\n".join(f"<{name}>\n{body}\n</{name}>" for name, body in sections.items())
        if d == "markdown":
            return "\n\n".join(f"## {_SECTION_TITLES.get(name, name)}\n{body}" for name, body in sections.items())
        if d == "yaml":
            return "\n\n".join(f"{name}: |\n" + "\n".join("  " + ln for ln in body.splitlines())
                               for name, body in sections.items())
        return "\n\n".join(f"[{_SECTION_TITLES.get(name, name)}]\n{body}" for name, body in sections.items())
