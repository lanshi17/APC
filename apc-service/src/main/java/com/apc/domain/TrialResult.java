package com.apc.domain;

import java.util.*;

public class TrialResult {
    private String trialId, taskId, modelId, genomeId, promptId, datasetId;
    private double score, accuracy, instructionFollowing, formatScore, constraintScore, robustness, efficiencyScore;
    private int totalInputTokens, totalOutputTokens, latencyMs;
    private double variance;
    private List<Map<String,Object>> caseResults = new ArrayList<>();
    public TrialResult(){}
    public String getTrialId(){return trialId;} public void setTrialId(String v){trialId=v;}
    public String getTaskId(){return taskId;} public void setTaskId(String v){taskId=v;}
    public String getModelId(){return modelId;} public void setModelId(String v){modelId=v;}
    public String getGenomeId(){return genomeId;} public void setGenomeId(String v){genomeId=v;}
    public String getPromptId(){return promptId;} public void setPromptId(String v){promptId=v;}
    public String getDatasetId(){return datasetId;} public void setDatasetId(String v){datasetId=v;}
    public double getScore(){return score;} public void setScore(double v){score=v;}
    public double getAccuracy(){return accuracy;} public void setAccuracy(double v){accuracy=v;}
    public double getInstructionFollowing(){return instructionFollowing;} public void setInstructionFollowing(double v){instructionFollowing=v;}
    public double getFormatScore(){return formatScore;} public void setFormatScore(double v){formatScore=v;}
    public double getConstraintScore(){return constraintScore;} public void setConstraintScore(double v){constraintScore=v;}
    public double getRobustness(){return robustness;} public void setRobustness(double v){robustness=v;}
    public double getEfficiencyScore(){return efficiencyScore;} public void setEfficiencyScore(double v){efficiencyScore=v;}
    public int getTotalInputTokens(){return totalInputTokens;} public void setTotalInputTokens(int v){totalInputTokens=v;}
    public int getTotalOutputTokens(){return totalOutputTokens;} public void setTotalOutputTokens(int v){totalOutputTokens=v;}
    public int getLatencyMs(){return latencyMs;} public void setLatencyMs(int v){latencyMs=v;}
    public double getVariance(){return variance;} public void setVariance(double v){variance=v;}
    public List<Map<String,Object>> getCaseResults(){return caseResults;} public void setCaseResults(List<Map<String,Object>> v){caseResults=v;}
}
