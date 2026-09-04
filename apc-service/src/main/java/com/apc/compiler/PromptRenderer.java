package com.apc.compiler;

import com.apc.domain.CompiledPrompt;
import com.apc.domain.ModelProfile;
import com.apc.domain.PromptGenome;
import com.apc.domain.TaskSpec;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

@Component
public class PromptRenderer {

    private final CompilerRules rules;
    private final ObjectMapper objectMapper;
    public PromptRenderer(CompilerRules rules, ObjectMapper objectMapper){this.rules=rules; this.objectMapper=objectMapper;}

    public CompiledPrompt compile(PromptGenome genome, TaskSpec spec, ModelProfile profile) {
        // deep copy via jackson to avoid mutating input
        PromptGenome g = objectMapper.convertValue(objectMapper.convertValue(genome, Map.class), PromptGenome.class);
        g = rules.apply(g, profile);

        List<String> sections = new ArrayList<>();

        if (g.getRole().isEnabled()) {
            sections.add("你是一名" + g.getRole().getStyle() + "，请以专业方式完成任务。");
        }
        sections.add(renderGoal(spec, g));
        if ("after_goal".equals(g.getConstraints().getPlacement())) {
            sections.add(renderConstraints(spec, g));
        }
        if (g.getExamples().isEnabled()) {
            sections.add(renderExamples(g));
        }
        sections.add("输入内容如下：\n<document>\n{{input}}\n</document>");
        if (!"none".equals(g.getReasoning().getStrategy())) {
            sections.add(renderReasoning(g));
        }
        if (g.getVerification().isEnabled()) {
            sections.add(renderVerification(g));
        }
        sections.add(renderOutputSchema(spec, g));

        String promptText = assemble(sections, g);

        return CompiledPrompt.builder()
                .promptId(UUID.randomUUID().toString())
                .genomeId(g.getTaskId())
                .modelId(profile.getModelId())
                .promptText(promptText)
                .tokenEstimate(promptText.length() / 4)
                .metadata(Map.of("genomeVersion", g.getGenomeVersion()))
                .build();
    }

    private String renderGoal(TaskSpec spec, PromptGenome g) {
        return "任务目标：\n" + (spec.getObjective() != null ? spec.getObjective().strip() : "");
    }

    private String renderConstraints(TaskSpec spec, PromptGenome g) {
        List<String> cs = spec.getConstraints() == null ? List.of() : spec.getConstraints();
        int max = g.getConstraints().getMaxCount();
        List<String> sliced = cs.stream().limit(max).collect(Collectors.toList());
        StringBuilder sb = new StringBuilder("约束条件：");
        for (int i = 0; i < sliced.size(); i++) sb.append("\n").append(i + 1).append(". ").append(sliced.get(i));
        return sb.toString();
    }

    private String renderExamples(PromptGenome g) {
        return "示例：\n输入：...\n输出：...";
    }

    private String renderReasoning(PromptGenome g) {
        return switch (g.getReasoning().getStrategy()) {
            case "hidden_analysis" -> "请先在内部分析输入内容，但不要输出分析过程，只输出最终结果。";
            case "brief_plan" -> "请先简要规划完成任务的步骤，然后输出最终结果。";
            case "structured_checklist" -> "请按照检查清单方式确认关键约束后再输出最终结果。";
            case "decompose_then_answer" -> "请先分解任务，再逐步作答。";
            default -> "";
        };
    }

    private String renderVerification(PromptGenome g) {
        return switch (g.getVerification().getType()) {
            case "constraint_check" -> "在输出前，请检查是否满足所有约束条件。";
            case "source_grounding_check" -> "在输出前，请确认所有结论均来自输入材料。";
            case "format_check" -> "在输出前，请检查输出格式是否符合要求。";
            default -> "";
        };
    }

    private String renderOutputSchema(TaskSpec spec, PromptGenome g) {
        if (!g.getOutput().isIncludeSchemaInPrompt()) return "请严格按照任务要求的输出结构返回结果。";
        try {
            String schemaText = spec.getOutput() != null && spec.getOutput().getSchema() != null
                    ? objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(spec.getOutput().getSchema())
                    : "{}";
            return "输出要求：\n只输出 JSON，不要输出解释。\nJSON 必须符合以下 schema：\n" + schemaText + "\n不要添加额外字段。";
        } catch (Exception e) {
            return "请严格按照 JSON 输出。";
        }
    }

    private String assemble(List<String> sections, PromptGenome g) {
        if ("xml".equals(g.getLayout().getDelimiter())) {
            List<String> wrapped = new ArrayList<>();
            for (int i = 0; i < sections.size(); i++) wrapped.add("<section_" + (i + 1) + ">\n" + sections.get(i) + "\n</section_" + (i + 1) + ">");
            return String.join("\n\n", wrapped);
        }
        return String.join("\n\n", sections);
    }
}
