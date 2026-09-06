package com.apc;

import com.apc.compiler.CompilerRules;
import com.apc.compiler.PromptRenderer;
import com.apc.domain.CompiledPrompt;
import com.apc.domain.ModelProfile;
import com.apc.domain.PromptGenome;
import com.apc.domain.TaskSpec;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class PromptRendererTest {

    private final PromptRenderer renderer = new PromptRenderer(new CompilerRules(), new ObjectMapper());

    private static TaskSpec spec() {
        TaskSpec s = new TaskSpec();
        s.setTaskId("t1");
        s.setName("test task");
        s.setObjective("完成测试目标");
        s.setConstraints(List.of("约束一", "约束二"));
        s.getOutput().setType("json");
        s.getOutput().setStrict(true);
        s.getOutput().setSchema(Map.of("response", "string", "confidence", "number"));
        return s;
    }

    private static ModelProfile neutralProfile() {
        ModelProfile p = new ModelProfile();
        p.setModelId("mock");
        p.getCapability().setFewShotBenefit(0.1);
        p.getCapability().setJsonReliability(0.93);
        return p;
    }

    @Test
    void compiledPromptKeepsInputPlaceholderAndMetadata() {
        CompiledPrompt cp = renderer.compile(new PromptGenome(), spec(), neutralProfile());
        assertThat(cp.getPromptText()).contains("{{input}}");
        assertThat(cp.getPromptText()).contains("完成测试目标");
        assertThat(cp.getModelId()).isEqualTo("mock");
        assertThat(cp.getTokenEstimate()).isPositive();
        assertThat(cp.getPromptId()).isNotBlank();
    }

    @Test
    void rulesDisableExamplesSectionForLowBenefitProfile() {
        PromptGenome g = new PromptGenome();
        g.getExamples().setEnabled(true);
        g.getExamples().setCount(2);
        ModelProfile p = neutralProfile();
        p.getCapability().setFewShotBenefit(0.0);
        CompiledPrompt cp = renderer.compile(g, spec(), p);
        assertThat(cp.getPromptText()).doesNotContain("示例");
    }

    @Test
    void compileDoesNotMutateInputGenome() throws Exception {
        ObjectMapper om = new ObjectMapper();
        PromptGenome g = new PromptGenome();
        g.getExamples().setEnabled(true);
        String before = om.writeValueAsString(g);
        ModelProfile p = neutralProfile();
        p.getCapability().setFewShotBenefit(0.0);
        renderer.compile(g, spec(), p);
        assertThat(om.writeValueAsString(g)).isEqualTo(before);
    }

    @Test
    void genomeJsonRoundTrip() throws Exception {
        ObjectMapper om = new ObjectMapper();
        PromptGenome g = new PromptGenome();
        g.setTaskId("t1");
        g.getReasoning().setStrategy("hidden_analysis");
        PromptGenome back = om.readValue(om.writeValueAsString(g), PromptGenome.class);
        assertThat(back.getTaskId()).isEqualTo("t1");
        assertThat(back.getReasoning().getStrategy()).isEqualTo("hidden_analysis");
    }
}
