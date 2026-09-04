package com.apc.storage;

import jakarta.persistence.*;
import java.time.Instant;

@Entity @Table(name = "task_spec")
class TaskSpecEntity {
    @Id private String id;
    private String name;
    private String version;
    @Column(columnDefinition = "TEXT") private String specJson;
    private Instant createdAt;
    public TaskSpecEntity(){}
    public String getId(){return id;} public void setId(String v){id=v;}
    public String getName(){return name;} public void setName(String v){name=v;}
    public String getVersion(){return version;} public void setVersion(String v){version=v;}
    public String getSpecJson(){return specJson;} public void setSpecJson(String v){specJson=v;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant v){createdAt=v;}
}

@Entity @Table(name = "model_info")
class ModelEntity {
    @Id private String id;
    private String provider;
    private String name;
    private String version;
    @Column(columnDefinition = "TEXT") private String apiConfig;
    private Instant createdAt;
    public ModelEntity(){}
    public String getId(){return id;} public void setId(String v){id=v;}
    public String getProvider(){return provider;} public void setProvider(String v){provider=v;}
    public String getName(){return name;} public void setName(String v){name=v;}
    public String getVersion(){return version;} public void setVersion(String v){version=v;}
    public String getApiConfig(){return apiConfig;} public void setApiConfig(String v){apiConfig=v;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant v){createdAt=v;}
}

@Entity @Table(name = "prompt_genome")
class GenomeEntity {
    @Id private String id;
    private String taskId;
    private String version;
    @Column(columnDefinition = "TEXT") private String genomeJson;
    private String parentId;
    private String origin;
    private Instant createdAt;
    public GenomeEntity(){}
    public String getId(){return id;} public void setId(String v){id=v;}
    public String getTaskId(){return taskId;} public void setTaskId(String v){taskId=v;}
    public String getVersion(){return version;} public void setVersion(String v){version=v;}
    public String getGenomeJson(){return genomeJson;} public void setGenomeJson(String v){genomeJson=v;}
    public String getParentId(){return parentId;} public void setParentId(String v){parentId=v;}
    public String getOrigin(){return origin;} public void setOrigin(String v){origin=v;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant v){createdAt=v;}
}

@Entity @Table(name = "compiled_prompt")
class CompiledPromptEntity {
    @Id private String id;
    private String genomeId;
    private String modelId;
    @Column(columnDefinition = "TEXT") private String promptText;
    private String templateVersion;
    private Integer tokenEstimate;
    private Instant createdAt;
    public CompiledPromptEntity(){}
    public String getId(){return id;} public void setId(String v){id=v;}
    public String getGenomeId(){return genomeId;} public void setGenomeId(String v){genomeId=v;}
    public String getModelId(){return modelId;} public void setModelId(String v){modelId=v;}
    public String getPromptText(){return promptText;} public void setPromptText(String v){promptText=v;}
    public String getTemplateVersion(){return templateVersion;} public void setTemplateVersion(String v){templateVersion=v;}
    public Integer getTokenEstimate(){return tokenEstimate;} public void setTokenEstimate(Integer v){tokenEstimate=v;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant v){createdAt=v;}
}

@Entity @Table(name = "trial")
class TrialEntity {
    @Id private String id;
    private String taskId;
    private String modelId;
    private String genomeId;
    private String promptId;
    private Double score;
    private Double accuracy;
    private Double constraintScore;
    private Double formatScore;
    private Double robustness;
    private Double efficiencyScore;
    private Integer totalTokens;
    private Integer latencyMs;
    private Instant createdAt;
    public TrialEntity(){}
    public String getId(){return id;} public void setId(String v){id=v;}
    public String getTaskId(){return taskId;} public void setTaskId(String v){taskId=v;}
    public String getModelId(){return modelId;} public void setModelId(String v){modelId=v;}
    public String getGenomeId(){return genomeId;} public void setGenomeId(String v){genomeId=v;}
    public String getPromptId(){return promptId;} public void setPromptId(String v){promptId=v;}
    public Double getScore(){return score;} public void setScore(Double v){score=v;}
    public Double getAccuracy(){return accuracy;} public void setAccuracy(Double v){accuracy=v;}
    public Double getConstraintScore(){return constraintScore;} public void setConstraintScore(Double v){constraintScore=v;}
    public Double getFormatScore(){return formatScore;} public void setFormatScore(Double v){formatScore=v;}
    public Double getRobustness(){return robustness;} public void setRobustness(Double v){robustness=v;}
    public Double getEfficiencyScore(){return efficiencyScore;} public void setEfficiencyScore(Double v){efficiencyScore=v;}
    public Integer getTotalTokens(){return totalTokens;} public void setTotalTokens(Integer v){totalTokens=v;}
    public Integer getLatencyMs(){return latencyMs;} public void setLatencyMs(Integer v){latencyMs=v;}
    public Instant getCreatedAt(){return createdAt;} public void setCreatedAt(Instant v){createdAt=v;}
}
