package com.apc.compiler;

import com.apc.domain.ModelProfile;
import com.apc.domain.PromptGenome;
import org.springframework.stereotype.Component;

@Component
public class CompilerRules {

    public PromptGenome apply(PromptGenome genome, ModelProfile profile) {
        genome = adjustExamples(genome, profile);
        genome = adjustOutput(genome, profile);
        genome = adjustReasoning(genome, profile);
        return genome;
    }

    private PromptGenome adjustExamples(PromptGenome g, ModelProfile p) {
        double benefit = p.getCapability().getFewShotBenefit();
        if (benefit > 0.25) {
            g.getExamples().setEnabled(true);
            if (g.getExamples().getCount() < 2) g.getExamples().setCount(2);
        } else if (benefit < 0.08) {
            g.getExamples().setEnabled(false);
            g.getExamples().setCount(0);
        }
        return g;
    }

    private PromptGenome adjustOutput(PromptGenome g, ModelProfile p) {
        double jsonRel = p.getCapability().getJsonReliability();
        if (jsonRel < 0.90) {
            g.getOutput().setStrictness("high");
            g.getOutput().setIncludeSchemaInPrompt(true);
            g.getOutput().setForbidExtraFields(true);
        } else if (jsonRel > 0.96) {
            g.getOutput().setStrictness("medium");
            g.getOutput().setIncludeSchemaInPrompt(false);
        }
        return g;
    }

    private PromptGenome adjustReasoning(PromptGenome g, ModelProfile p) {
        if (p.getCapability().getReasoning() > 0.90 && p.getCapability().getSelfVerificationBenefit() > 0.25) {
            g.getReasoning().setStrategy("hidden_analysis");
            g.getVerification().setEnabled(true);
        }
        return g;
    }
}
