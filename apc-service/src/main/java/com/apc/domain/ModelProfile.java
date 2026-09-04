package com.apc.domain;

import java.util.Map;
import java.util.HashMap;

public class ModelProfile {
    private String modelId;
    private String profileVersion = "v1";
    private CapabilityVector capability = new CapabilityVector();
    private BehaviorVector behavior = new BehaviorVector();
    private Map<String, Object> rawProbeResults = new HashMap<>();

    public ModelProfile(){}
    public String getModelId(){return modelId;} public void setModelId(String v){modelId=v;}
    public String getProfileVersion(){return profileVersion;} public void setProfileVersion(String v){profileVersion=v;}
    public CapabilityVector getCapability(){return capability;} public void setCapability(CapabilityVector v){capability=v;}
    public BehaviorVector getBehavior(){return behavior;} public void setBehavior(BehaviorVector v){behavior=v;}
    public Map<String,Object> getRawProbeResults(){return rawProbeResults;} public void setRawProbeResults(Map<String,Object> v){rawProbeResults=v;}

    public static class CapabilityVector {
        private double instructionFollowing=0, multiConstraintFollowing=0, longContext=0, jsonReliability=0, schemaStrictness=0, tableUnderstanding=0, math=0, reasoning=0, informationExtraction=0, fewShotBenefit=0, selfVerificationBenefit=0, toolUsage=0, robustnessToDistraction=0, chineseSemantic=0, safetyBoundary=0;
        public CapabilityVector(){}
        public double getInstructionFollowing(){return instructionFollowing;} public void setInstructionFollowing(double v){instructionFollowing=v;}
        public double getMultiConstraintFollowing(){return multiConstraintFollowing;} public void setMultiConstraintFollowing(double v){multiConstraintFollowing=v;}
        public double getLongContext(){return longContext;} public void setLongContext(double v){longContext=v;}
        public double getJsonReliability(){return jsonReliability;} public void setJsonReliability(double v){jsonReliability=v;}
        public double getSchemaStrictness(){return schemaStrictness;} public void setSchemaStrictness(double v){schemaStrictness=v;}
        public double getTableUnderstanding(){return tableUnderstanding;} public void setTableUnderstanding(double v){tableUnderstanding=v;}
        public double getMath(){return math;} public void setMath(double v){math=v;}
        public double getReasoning(){return reasoning;} public void setReasoning(double v){reasoning=v;}
        public double getInformationExtraction(){return informationExtraction;} public void setInformationExtraction(double v){informationExtraction=v;}
        public double getFewShotBenefit(){return fewShotBenefit;} public void setFewShotBenefit(double v){fewShotBenefit=v;}
        public double getSelfVerificationBenefit(){return selfVerificationBenefit;} public void setSelfVerificationBenefit(double v){selfVerificationBenefit=v;}
        public double getToolUsage(){return toolUsage;} public void setToolUsage(double v){toolUsage=v;}
        public double getRobustnessToDistraction(){return robustnessToDistraction;} public void setRobustnessToDistraction(double v){robustnessToDistraction=v;}
        public double getChineseSemantic(){return chineseSemantic;} public void setChineseSemantic(double v){chineseSemantic=v;}
        public double getSafetyBoundary(){return safetyBoundary;} public void setSafetyBoundary(double v){safetyBoundary=v;}
    }
    public static class BehaviorVector {
        private String verbosity="medium", prefixTendency="low", jsonPrefixNoise="low", overRefusal="low", hallucinationTendency="low", latency="medium", cost="medium";
        public BehaviorVector(){}
        public String getVerbosity(){return verbosity;} public void setVerbosity(String v){verbosity=v;}
        public String getPrefixTendency(){return prefixTendency;} public void setPrefixTendency(String v){prefixTendency=v;}
        public String getJsonPrefixNoise(){return jsonPrefixNoise;} public void setJsonPrefixNoise(String v){jsonPrefixNoise=v;}
        public String getOverRefusal(){return overRefusal;} public void setOverRefusal(String v){overRefusal=v;}
        public String getHallucinationTendency(){return hallucinationTendency;} public void setHallucinationTendency(String v){hallucinationTendency=v;}
        public String getLatency(){return latency;} public void setLatency(String v){latency=v;}
        public String getCost(){return cost;} public void setCost(String v){cost=v;}
    }
}
