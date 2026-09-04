package com.apc.domain;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import java.util.*;

public class TaskSpec {
    @NotBlank private String taskId;
    @NotBlank private String name;
    private String version = "1.0.0";
    private String objective;
    private Map<String, Object> input = new HashMap<>();
    private OutputSpec output = new OutputSpec();
    private List<String> constraints = new ArrayList<>();
    private ReasoningSpec reasoning = new ReasoningSpec();
    private QualityWeights qualityWeights = new QualityWeights();
    private CostLimits costLimits = new CostLimits();
    private Map<String, Object> evaluation = new HashMap<>();

    public TaskSpec() {}
    public String getTaskId(){return taskId;} public void setTaskId(String v){taskId=v;}
    public String getName(){return name;} public void setName(String v){name=v;}
    public String getVersion(){return version;} public void setVersion(String v){version=v;}
    public String getObjective(){return objective;} public void setObjective(String v){objective=v;}
    public Map<String,Object> getInput(){return input;} public void setInput(Map<String,Object> v){input=v;}
    public OutputSpec getOutput(){return output;} public void setOutput(OutputSpec v){output=v;}
    public List<String> getConstraints(){return constraints;} public void setConstraints(List<String> v){constraints=v;}
    public ReasoningSpec getReasoning(){return reasoning;} public void setReasoning(ReasoningSpec v){reasoning=v;}
    public QualityWeights getQualityWeights(){return qualityWeights;} public void setQualityWeights(QualityWeights v){qualityWeights=v;}
    public CostLimits getCostLimits(){return costLimits;} public void setCostLimits(CostLimits v){costLimits=v;}
    public Map<String,Object> getEvaluation(){return evaluation;} public void setEvaluation(Map<String,Object> v){evaluation=v;}

    public static class OutputSpec {
        private String type = "json";
        private boolean strict;
        @JsonProperty("schema") @JsonAlias("schema_")
        private Map<String, Object> schema;
        public OutputSpec() {}
        public String getType(){return type;} public void setType(String v){type=v;}
        public boolean isStrict(){return strict;} public void setStrict(boolean v){strict=v;}
        public Map<String,Object> getSchema(){return schema;} public void setSchema(Map<String,Object> v){schema=v;}
    }
    public static class ReasoningSpec {
        private boolean required = false;
        private String visibility = "hidden";
        public ReasoningSpec() {}
        public boolean isRequired(){return required;} public void setRequired(boolean v){required=v;}
        public String getVisibility(){return visibility;} public void setVisibility(String v){visibility=v;}
    }
    public static class QualityWeights {
        private double accuracy = 0.5;
        private double instructionFollowing = 0.2;
        private double format = 0.15;
        private double robustness = 0.1;
        private double efficiency = 0.05;
        public QualityWeights() {}
        public double getAccuracy(){return accuracy;} public void setAccuracy(double v){accuracy=v;}
        public double getInstructionFollowing(){return instructionFollowing;} public void setInstructionFollowing(double v){instructionFollowing=v;}
        public double getFormat(){return format;} public void setFormat(double v){format=v;}
        public double getRobustness(){return robustness;} public void setRobustness(double v){robustness=v;}
        public double getEfficiency(){return efficiency;} public void setEfficiency(double v){efficiency=v;}
    }
    public static class CostLimits {
        private Integer maxInputTokens;
        private Integer maxOutputTokens;
        private Integer maxLatencyMs;
        public CostLimits() {}
        public Integer getMaxInputTokens(){return maxInputTokens;} public void setMaxInputTokens(Integer v){maxInputTokens=v;}
        public Integer getMaxOutputTokens(){return maxOutputTokens;} public void setMaxOutputTokens(Integer v){maxOutputTokens=v;}
        public Integer getMaxLatencyMs(){return maxLatencyMs;} public void setMaxLatencyMs(Integer v){maxLatencyMs=v;}
    }
}
