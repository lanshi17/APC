package com.apc.domain;

import java.util.*;

public class PromptGenome {
    private String genomeVersion = "1.0";
    private String taskId;
    private RoleGene role = new RoleGene();
    private GoalGene goal = new GoalGene();
    private InstructionGene instructions = new InstructionGene();
    private ConstraintGene constraints = new ConstraintGene();
    private ExampleGene examples = new ExampleGene();
    private ReasoningGene reasoning = new ReasoningGene();
    private VerificationGene verification = new VerificationGene();
    private OutputGene output = new OutputGene();
    private StyleGene style = new StyleGene();
    private LayoutGene layout = new LayoutGene();
    private Map<String, List<Object>> searchSpace = new HashMap<>();

    public PromptGenome(){}
    public String getGenomeVersion(){return genomeVersion;} public void setGenomeVersion(String v){genomeVersion=v;}
    public String getTaskId(){return taskId;} public void setTaskId(String v){taskId=v;}
    public RoleGene getRole(){return role;} public void setRole(RoleGene v){role=v;}
    public GoalGene getGoal(){return goal;} public void setGoal(GoalGene v){goal=v;}
    public InstructionGene getInstructions(){return instructions;} public void setInstructions(InstructionGene v){instructions=v;}
    public ConstraintGene getConstraints(){return constraints;} public void setConstraints(ConstraintGene v){constraints=v;}
    public ExampleGene getExamples(){return examples;} public void setExamples(ExampleGene v){examples=v;}
    public ReasoningGene getReasoning(){return reasoning;} public void setReasoning(ReasoningGene v){reasoning=v;}
    public VerificationGene getVerification(){return verification;} public void setVerification(VerificationGene v){verification=v;}
    public OutputGene getOutput(){return output;} public void setOutput(OutputGene v){output=v;}
    public StyleGene getStyle(){return style;} public void setStyle(StyleGene v){style=v;}
    public LayoutGene getLayout(){return layout;} public void setLayout(LayoutGene v){layout=v;}
    public Map<String,List<Object>> getSearchSpace(){return searchSpace;} public void setSearchSpace(Map<String,List<Object>> v){searchSpace=v;}

    public static class RoleGene { private boolean enabled=false; private String style="expert"; private String authorityLevel="medium"; public RoleGene(){} public boolean isEnabled(){return enabled;} public void setEnabled(boolean v){enabled=v;} public String getStyle(){return style;} public void setStyle(String v){style=v;} public String getAuthorityLevel(){return authorityLevel;} public void setAuthorityLevel(String v){authorityLevel=v;} }
    public static class GoalGene { private String explicitness="high"; private String placement="top"; public GoalGene(){} public String getExplicitness(){return explicitness;} public void setExplicitness(String v){explicitness=v;} public String getPlacement(){return placement;} public void setPlacement(String v){placement=v;} }
    public static class InstructionGene { private String style="imperative"; private String granularity="medium"; private boolean stepDecomposition=false; public InstructionGene(){} public String getStyle(){return style;} public void setStyle(String v){style=v;} public String getGranularity(){return granularity;} public void setGranularity(String v){granularity=v;} public boolean isStepDecomposition(){return stepDecomposition;} public void setStepDecomposition(boolean v){stepDecomposition=v;} }
    public static class ConstraintGene { private String placement="after_goal"; private String explicitness="high"; private int maxCount=8; public ConstraintGene(){} public String getPlacement(){return placement;} public void setPlacement(String v){placement=v;} public String getExplicitness(){return explicitness;} public void setExplicitness(String v){explicitness=v;} public int getMaxCount(){return maxCount;} public void setMaxCount(int v){maxCount=v;} }
    public static class ExampleGene { private boolean enabled=false; private int count=0; private String selection="diverse"; private String format="input_output"; public ExampleGene(){} public boolean isEnabled(){return enabled;} public void setEnabled(boolean v){enabled=v;} public int getCount(){return count;} public void setCount(int v){count=v;} public String getSelection(){return selection;} public void setSelection(String v){selection=v;} public String getFormat(){return format;} public void setFormat(String v){format=v;} }
    public static class ReasoningGene { private String strategy="none"; private String visibility="hidden"; private String budget="medium"; public ReasoningGene(){} public String getStrategy(){return strategy;} public void setStrategy(String v){strategy=v;} public String getVisibility(){return visibility;} public void setVisibility(String v){visibility=v;} public String getBudget(){return budget;} public void setBudget(String v){budget=v;} }
    public static class VerificationGene { private boolean enabled=false; private String type="none"; private String position="before_final_output"; public VerificationGene(){} public boolean isEnabled(){return enabled;} public void setEnabled(boolean v){enabled=v;} public String getType(){return type;} public void setType(String v){type=v;} public String getPosition(){return position;} public void setPosition(String v){position=v;} }
    public static class OutputGene { private String format="json_schema"; private String strictness="high"; private boolean includeSchemaInPrompt=true; private boolean forbidExtraFields=true; public OutputGene(){} public String getFormat(){return format;} public void setFormat(String v){format=v;} public String getStrictness(){return strictness;} public void setStrictness(String v){strictness=v;} public boolean isIncludeSchemaInPrompt(){return includeSchemaInPrompt;} public void setIncludeSchemaInPrompt(boolean v){includeSchemaInPrompt=v;} public boolean isForbidExtraFields(){return forbidExtraFields;} public void setForbidExtraFields(boolean v){forbidExtraFields=v;} }
    public static class StyleGene { private String tone="professional"; private String verbosity="low"; private String language="zh"; public StyleGene(){} public String getTone(){return tone;} public void setTone(String v){tone=v;} public String getVerbosity(){return verbosity;} public void setVerbosity(String v){verbosity=v;} public String getLanguage(){return language;} public void setLanguage(String v){language=v;} }
    public static class LayoutGene { private List<String> sectionOrder = new ArrayList<>(List.of("goal","constraints","input","reasoning_instruction","output_format")); private String delimiter="xml"; public LayoutGene(){} public List<String> getSectionOrder(){return sectionOrder;} public void setSectionOrder(List<String> v){sectionOrder=v;} public String getDelimiter(){return delimiter;} public void setDelimiter(String v){delimiter=v;} }
}
