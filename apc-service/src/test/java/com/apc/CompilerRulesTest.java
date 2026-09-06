package com.apc;

import com.apc.compiler.CompilerRules;
import com.apc.domain.ModelProfile;
import com.apc.domain.PromptGenome;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class CompilerRulesTest {

    private final CompilerRules rules = new CompilerRules();

    private static ModelProfile profile(double fewShot, double jsonRel, double reasoning, double selfVerif) {
        ModelProfile p = new ModelProfile();
        p.setModelId("test");
        p.getCapability().setFewShotBenefit(fewShot);
        p.getCapability().setJsonReliability(jsonRel);
        p.getCapability().setReasoning(reasoning);
        p.getCapability().setSelfVerificationBenefit(selfVerif);
        return p;
    }

    @Test
    void enablesExamplesWhenFewShotBenefitHigh() {
        PromptGenome g = new PromptGenome();
        rules.apply(g, profile(0.5, 0.95, 0.0, 0.0));
        assertThat(g.getExamples().isEnabled()).isTrue();
        assertThat(g.getExamples().getCount()).isGreaterThanOrEqualTo(2);
    }

    @Test
    void disablesExamplesWhenFewShotBenefitLow() {
        PromptGenome g = new PromptGenome();
        g.getExamples().setEnabled(true);
        g.getExamples().setCount(3);
        rules.apply(g, profile(0.0, 0.95, 0.0, 0.0));
        assertThat(g.getExamples().isEnabled()).isFalse();
        assertThat(g.getExamples().getCount()).isZero();
    }

    @Test
    void forcesStrictJsonWhenReliabilityLow() {
        PromptGenome g = new PromptGenome();
        g.getOutput().setStrictness("low");
        g.getOutput().setIncludeSchemaInPrompt(false);
        rules.apply(g, profile(0.5, 0.5, 0.0, 0.0));
        assertThat(g.getOutput().getStrictness()).isEqualTo("high");
        assertThat(g.getOutput().isIncludeSchemaInPrompt()).isTrue();
        assertThat(g.getOutput().isForbidExtraFields()).isTrue();
    }

    @Test
    void relaxesSchemaWhenReliabilityVeryHigh() {
        PromptGenome g = new PromptGenome();
        rules.apply(g, profile(0.5, 0.99, 0.0, 0.0));
        assertThat(g.getOutput().getStrictness()).isEqualTo("medium");
        assertThat(g.getOutput().isIncludeSchemaInPrompt()).isFalse();
    }

    @Test
    void enablesHiddenReasoningForStrongReasoners() {
        PromptGenome g = new PromptGenome();
        rules.apply(g, profile(0.0, 0.95, 0.95, 0.5));
        assertThat(g.getReasoning().getStrategy()).isEqualTo("hidden_analysis");
        assertThat(g.getVerification().isEnabled()).isTrue();
    }

    @Test
    void leavesReasoningAloneForWeakReasoners() {
        PromptGenome g = new PromptGenome();
        rules.apply(g, profile(0.0, 0.95, 0.5, 0.0));
        assertThat(g.getReasoning().getStrategy()).isEqualTo("none");
    }
}
