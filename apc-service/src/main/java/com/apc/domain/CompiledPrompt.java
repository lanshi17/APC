package com.apc.domain;

import java.util.Map;
import java.util.HashMap;

public class CompiledPrompt {
    private String promptId;
    private String genomeId;
    private String modelId;
    private String promptText;
    private String templateVersion = "1.0";
    private Integer tokenEstimate;
    private Map<String, Object> metadata = new HashMap<>();

    public CompiledPrompt(){}
    public String getPromptId(){return promptId;} public void setPromptId(String v){promptId=v;}
    public String getGenomeId(){return genomeId;} public void setGenomeId(String v){genomeId=v;}
    public String getModelId(){return modelId;} public void setModelId(String v){modelId=v;}
    public String getPromptText(){return promptText;} public void setPromptText(String v){promptText=v;}
    public String getTemplateVersion(){return templateVersion;} public void setTemplateVersion(String v){templateVersion=v;}
    public Integer getTokenEstimate(){return tokenEstimate;} public void setTokenEstimate(Integer v){tokenEstimate=v;}
    public Map<String,Object> getMetadata(){return metadata;} public void setMetadata(Map<String,Object> v){metadata=v;}

    public static Builder builder(){return new Builder();}
    public static class Builder {
        private CompiledPrompt obj=new CompiledPrompt();
        public Builder promptId(String v){obj.promptId=v; return this;}
        public Builder genomeId(String v){obj.genomeId=v; return this;}
        public Builder modelId(String v){obj.modelId=v; return this;}
        public Builder promptText(String v){obj.promptText=v; return this;}
        public Builder templateVersion(String v){obj.templateVersion=v; return this;}
        public Builder tokenEstimate(Integer v){obj.tokenEstimate=v; return this;}
        public Builder metadata(Map<String,Object> v){obj.metadata=v; return this;}
        public CompiledPrompt build(){return obj;}
    }
}
